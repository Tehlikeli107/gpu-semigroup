"""
GPU Semigroup Counting Revolution
===================================
Extends the Counting Revolution to semigroups:
  - Magmas   (0 axioms): 29x amplification
  - Semigroups (1 axiom: associativity): ???
  - Groups   (3 axioms): 2.5-6x amplification
  - Rings    (4 axioms): 2x amplification

Hypothesis: amplification decreases monotonically with axiom count.
"""

import torch
import time
from itertools import permutations

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ==============================================================
# 1. GPU BATCH ASSOCIATIVITY CHECK
# ==============================================================

def check_associativity_gpu(ops: torch.Tensor) -> torch.Tensor:
    """
    ops: [B, n, n] int8 — multiplication tables
    Returns: [B] bool — which ops satisfy associativity
    """
    B, n, _ = ops.shape
    # For all (a, b, c): ops[a, ops[b, c]] == ops[ops[a, b], c]
    # a, b, c each range over n
    a_idx = torch.arange(n, device=DEVICE)
    b_idx = torch.arange(n, device=DEVICE)
    c_idx = torch.arange(n, device=DEVICE)

    # Build all (a, b, c) triples
    a, b, c = torch.meshgrid(a_idx, b_idx, c_idx, indexing='ij')
    a = a.reshape(-1)  # [n^3]
    b = b.reshape(-1)
    c = c.reshape(-1)

    # ops: [B, n, n]
    # lhs = ops[a, ops[b, c]]
    bc = ops[:, b, c]         # [B, n^3]
    lhs = ops[torch.arange(B, device=DEVICE).unsqueeze(1), a.unsqueeze(0), bc]  # [B, n^3]

    # rhs = ops[ops[a, b], c]
    ab = ops[:, a, b]         # [B, n^3]
    rhs = ops[torch.arange(B, device=DEVICE).unsqueeze(1), ab, c.unsqueeze(0)]  # [B, n^3]

    assoc = (lhs == rhs).all(dim=1)  # [B]
    return assoc


# ==============================================================
# 2. COUNTING INVARIANTS FOR SEMIGROUPS
# ==============================================================

def compute_semigroup_invariants(ops: torch.Tensor) -> dict:
    """
    ops: [B, n, n] int64
    Returns dict of tensors each [B]
    """
    B, n, _ = ops.shape
    elems = torch.arange(n, device=DEVICE)

    inv = {}

    # 1. Commutativity count: #{(a,b): ops[a,b] == ops[b,a]}
    inv['comm'] = (ops == ops.transpose(1, 2)).sum(dim=(1, 2)).float()

    # 2. Idempotent count: #{a: ops[a,a] == a}
    diag = ops[:, elems, elems]  # [B, n]
    inv['idemp'] = (diag == elems.unsqueeze(0)).sum(dim=1).float()

    # 3. Left identity count: #{e: ops[e, a] == a for all a}
    # ops[:, e, :] == elems for all e
    left_id_mask = (ops == elems.unsqueeze(0).unsqueeze(0))  # [B, n, n] — ops[b,a]==a
    inv['left_id'] = left_id_mask.all(dim=2).sum(dim=1).float()

    # 4. Right identity count: #{e: ops[a, e] == a for all a}
    right_id_mask = (ops == elems.unsqueeze(0).unsqueeze(2))  # [B, n, n] — ops[a,e]==a
    inv['right_id'] = right_id_mask.all(dim=1).sum(dim=1).float()

    # 5. Left zero count: #{z: ops[z, a] == z for all a}
    # ops[:, z, :] == z  => ops[:,z,:] == elems[z]
    # ops[b, z, a] == z for all a
    # left_zero[b, z] = (ops[b, z, :] == z).all()
    left_zero = (ops == elems.unsqueeze(0).unsqueeze(2)).all(dim=2)  # [B, n]
    inv['left_zero'] = left_zero.sum(dim=1).float()

    # 6. Right zero count: #{z: ops[a, z] == z for all a}
    right_zero = (ops == elems.unsqueeze(0).unsqueeze(1)).all(dim=1)  # [B, n]
    inv['right_zero'] = right_zero.sum(dim=1).float()

    # 7. Center size: #{a: ops[a,b]==ops[b,a] for all b}
    comm_all = (ops == ops.transpose(1, 2))  # [B, n, n]
    center = comm_all.all(dim=2)  # [B, n] — a commutes with all b
    inv['center'] = center.sum(dim=1).float()

    # 8. Square image size: |{a^2 : a in S}| = |{ops[a,a] : a}|
    sq_image = torch.zeros(B, n, device=DEVICE, dtype=torch.float)
    for b in range(B):
        sq_image[b, diag[b]] = 1.0
    inv['sq_image'] = sq_image.sum(dim=1)

    # 9. Nilpotent-like: #{a: ops[a,a] == 0} (zero element)
    inv['sq_zero'] = (diag == 0).sum(dim=1).float()

    # 10. Absorbing element count: #{z: ops[z,a]=z AND ops[a,z]=z for all a}
    absorb = left_zero & right_zero  # [B, n]
    inv['absorbing'] = absorb.sum(dim=1).float()

    return inv


# ==============================================================
# 3. BOOLEAN CLASSIFICATION
# ==============================================================

def boolean_class(ops: torch.Tensor) -> torch.Tensor:
    """
    Boolean features: (commutative, has_left_id, has_right_id, has_left_zero, has_right_zero, is_band)
    Returns [B, 6] bool tensor
    """
    B, n, _ = ops.shape
    elems = torch.arange(n, device=DEVICE)

    comm = (ops == ops.transpose(1, 2)).all(dim=(1, 2))
    diag = ops[:, elems, elems]

    left_id = (ops == elems.unsqueeze(0).unsqueeze(0)).all(dim=2).any(dim=1)
    right_id = (ops == elems.unsqueeze(0).unsqueeze(2)).all(dim=1).any(dim=1)
    left_zero = (ops == elems.unsqueeze(0).unsqueeze(2)).all(dim=2).any(dim=1)
    right_zero = (ops == elems.unsqueeze(0).unsqueeze(1)).all(dim=1).any(dim=1)
    is_band = (diag == elems.unsqueeze(0)).all(dim=1)

    return torch.stack([comm, left_id, right_id, left_zero, right_zero, is_band], dim=1)


# ==============================================================
# 4. ISOMORPHISM CLASSIFICATION (canonical form)
# ==============================================================

def canonical_form_cpu(op_np):
    """Compute canonical form of a semigroup operation table via relabeling."""
    n = op_np.shape[0]
    best = None
    for perm in permutations(range(n)):
        p = list(perm)
        # Relabel: new_op[i,j] = p_inv[op[p[i], p[j]]]
        p_inv = [0] * n
        for i, v in enumerate(p):
            p_inv[v] = i
        new_op = [[p_inv[op_np[p[i], p[j]]] for j in range(n)] for i in range(n)]
        flat = tuple(x for row in new_op for x in row)
        if best is None or flat < best:
            best = flat
    return best


def classify_iso_classes(ops_cpu, max_samples=None):
    """ops_cpu: list of n×n numpy arrays. Returns list of iso class ids."""
    iso_map = {}
    iso_ids = []
    for i, op in enumerate(ops_cpu):
        if max_samples and i >= max_samples:
            break
        cf = canonical_form_cpu(op)
        if cf not in iso_map:
            iso_map[cf] = len(iso_map)
        iso_ids.append(iso_map[cf])
    return iso_ids, iso_map


# ==============================================================
# 5. EXHAUSTIVE ENUMERATION FOR n=3
# ==============================================================

def run_n3_exhaustive():
    print("\n" + "="*60)
    print("n=3 EXHAUSTIVE SEMIGROUP COUNTING REVOLUTION")
    print("="*60)

    n = 3
    # All 3^9 = 19683 operations
    total = n ** (n * n)
    print(f"Total operations: {total:,}")

    # Generate all ops on GPU in batches
    BATCH = 4096
    all_semigroups = []
    t0 = time.time()

    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        batch_size = end - start
        indices = torch.arange(start, end, device=DEVICE)

        # Decode indices to n×n tables
        ops = torch.zeros(batch_size, n, n, dtype=torch.int64, device=DEVICE)
        idx = indices.clone()
        for row in range(n):
            for col in range(n):
                ops[:, row, col] = idx % n
                idx = idx // n

        # Check associativity
        assoc_mask = check_associativity_gpu(ops)
        semigroup_ops = ops[assoc_mask]

        if semigroup_ops.shape[0] > 0:
            all_semigroups.append(semigroup_ops)

    t1 = time.time()
    all_semigroups = torch.cat(all_semigroups, dim=0)
    print(f"Semigroup operations found: {all_semigroups.shape[0]:,} ({t1-t0:.2f}s)")

    # Classify iso classes
    t2 = time.time()
    ops_cpu = all_semigroups.cpu().numpy()
    iso_ids, iso_map = classify_iso_classes(ops_cpu)
    t3 = time.time()
    n_iso = len(iso_map)
    print(f"Iso classes: {n_iso} ({t3-t2:.2f}s)")
    print(f"Expected: 24 (OEIS A001423 — iso only; 18 if also quotienting anti-iso)")

    # Boolean classification
    bool_feats = boolean_class(all_semigroups)  # [N, 6]
    bool_tuples = [tuple(bool_feats[i].cpu().tolist()) for i in range(len(all_semigroups))]
    # Map each iso class to its boolean tuple
    iso_to_bool = {}
    for i, (iso_id, bt) in enumerate(zip(iso_ids, bool_tuples)):
        iso_to_bool[iso_id] = bt

    bool_classes = set(iso_to_bool.values())
    n_bool = len(bool_classes)
    print(f"Boolean classes: {n_bool}")

    # Counting invariants
    inv = compute_semigroup_invariants(all_semigroups)
    # Map each iso class to counting tuple
    iso_to_count = {}
    for i, iso_id in enumerate(iso_ids):
        count_tuple = tuple(inv[k][i].item() for k in sorted(inv.keys()))
        iso_to_count[iso_id] = count_tuple

    count_classes = set(iso_to_count.values())
    n_count = len(count_classes)
    print(f"Counting classes (10 invariants): {n_count}")

    amplification = n_count / n_bool if n_bool > 0 else 0
    print(f"Amplification: {amplification:.1f}x")

    # Show all 18 classes
    print(f"\nAll {n_iso} iso classes:")
    print(f"{'Class':>6} {'comm':>5} {'idemp':>6} {'lid':>4} {'rid':>4} {'lz':>4} {'rz':>4} "
          f"{'ctr':>4} {'sqim':>5} {'bool-comm':>10} {'bool-lid':>9}")
    for iso_id in sorted(iso_to_count.keys()):
        bt = iso_to_bool[iso_id]
        ct = iso_to_count[iso_id]
        keys = sorted(inv.keys())
        ct_dict = dict(zip(keys, ct))
        print(f"{iso_id:>6} {ct_dict.get('comm',0):>5.0f} {ct_dict.get('idemp',0):>6.0f} "
              f"{ct_dict.get('left_id',0):>4.0f} {ct_dict.get('right_id',0):>4.0f} "
              f"{ct_dict.get('left_zero',0):>4.0f} {ct_dict.get('right_zero',0):>4.0f} "
              f"{ct_dict.get('center',0):>4.0f} {ct_dict.get('sq_image',0):>5.0f} "
              f"{str(bt[0]):>10} {str(bt[1]):>9}")

    return n_bool, n_count, n_iso


# ==============================================================
# 6. n=4 SAMPLING + THROUGHPUT
# ==============================================================

def run_n4_sampling():
    """
    For n=4: P(random table is semigroup) = ~3492 / 4^16 ~ 8e-7
    Need ~1.25M trials to expect 1 semigroup. We run 5M for a good sample.
    At 17M tables/sec on GPU, this takes ~0.3 seconds.
    """
    print("\n" + "="*60)
    print("n=4 SEMIGROUP: THROUGHPUT + SAMPLING")
    print("="*60)

    n = 4
    BATCH = 65536  # larger batch for n=4

    total_tested = 0
    total_semigroups = 0
    iso_map = {}
    iso_count_map = {}
    iso_bool_map = {}

    t0 = time.time()
    TARGET = 5_000_000  # test 5M tables

    while total_tested < TARGET:
        ops = torch.randint(0, n, (BATCH, n, n), device=DEVICE, dtype=torch.int64)
        assoc = check_associativity_gpu(ops)
        sg_ops = ops[assoc]

        if sg_ops.shape[0] > 0:
            inv = compute_semigroup_invariants(sg_ops)
            bool_feats = boolean_class(sg_ops)
            sg_cpu = sg_ops.cpu().numpy()
            for b in range(sg_ops.shape[0]):
                cf = canonical_form_cpu(sg_cpu[b])
                if cf not in iso_map:
                    iso_id = len(iso_map)
                    iso_map[cf] = iso_id
                    count_tuple = tuple(inv[k][b].item() for k in sorted(inv.keys()))
                    bool_tuple = tuple(bool_feats[b].cpu().tolist())
                    iso_count_map[iso_id] = count_tuple
                    iso_bool_map[iso_id] = bool_tuple

        total_semigroups += sg_ops.shape[0]
        total_tested += BATCH

    t1 = time.time()
    elapsed = t1 - t0

    n_iso_found = len(iso_map)
    rate_pct = 100.0 * total_semigroups / total_tested if total_tested > 0 else 0
    print(f"Random tables tested: {total_tested:,}")
    print(f"Semigroups found:     {total_semigroups:,} ({rate_pct:.4f}%)")
    print(f"Iso classes found:    {n_iso_found} of 126")
    print(f"Throughput:           {total_tested/elapsed/1e6:.1f}M tables/sec")

    if n_iso_found > 0:
        bool_classes = set(iso_bool_map.values())
        count_classes = set(iso_count_map.values())
        n_bool = len(bool_classes)
        n_count = len(count_classes)
        print(f"\nBoolean classes (partial):  {n_bool}")
        print(f"Counting classes (partial): {n_count}")

    return n_iso_found, total_tested, elapsed


# ==============================================================
# 7. AXIOM HIERARCHY SUMMARY
# ==============================================================

def print_axiom_hierarchy(n3_bool, n3_count, n3_iso):
    print("\n" + "="*60)
    print("AXIOM HIERARCHY: COUNTING REVOLUTION AMPLIFICATION")
    print("="*60)
    print(f"{'Structure':<20} {'Axioms':>7} {'Bool classes':>13} {'Count classes':>14} {'Amplification':>14}")
    print("-"*70)
    print(f"{'Magmas (n=3)':<20} {'0':>7} {'114':>13} {'3,328':>14} {'29.2x':>14}")
    n3_amp = f"{n3_count/n3_bool:.1f}x" if n3_bool else "N/A"
    print(f"{'Semigroups (n=3)':<20} {'1':>7} {n3_bool:>13} {n3_count:>14} {n3_amp:>14}")
    print(f"{'Groups (n=8)':<20} {'3':>7} {'2':>13} {'5':>14} {'2.5x':>14}")
    print(f"{'Rings (n=4)':<20} {'4':>7} {'4':>13} {'8':>14} {'2.0x':>14}")
    print("-"*70)
    print("Principle: MORE axioms => LESS room for counting invariants")
    print("           => amplification decreases monotonically")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    # Section 1: n=3 exhaustive
    n3_bool, n3_count, n3_iso = run_n3_exhaustive()

    # Section 2: n=4 throughput + sampling
    n4_iso_found, n4_tested, n4_elapsed = run_n4_sampling()

    # Section 3: axiom hierarchy
    print_axiom_hierarchy(n3_bool, n3_count, n3_iso)

    print("\nDone.")
