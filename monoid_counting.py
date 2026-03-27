"""
n=3 Monoid Counting Revolution
Monoids = Semigroups + Identity element (2 axioms)
Fills the hierarchy between Semigroups (1 axiom) and Groups (3 axioms).
"""

import torch
import time
from itertools import permutations

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def check_associativity_gpu(ops):
    B, n, _ = ops.shape
    a_idx = torch.arange(n, device=DEVICE)
    b_idx = torch.arange(n, device=DEVICE)
    c_idx = torch.arange(n, device=DEVICE)
    a, b, c = torch.meshgrid(a_idx, b_idx, c_idx, indexing='ij')
    a = a.reshape(-1); b = b.reshape(-1); c = c.reshape(-1)
    bc = ops[:, b, c]
    lhs = ops[torch.arange(B, device=DEVICE).unsqueeze(1), a.unsqueeze(0), bc]
    ab = ops[:, a, b]
    rhs = ops[torch.arange(B, device=DEVICE).unsqueeze(1), ab, c.unsqueeze(0)]
    return (lhs == rhs).all(dim=1)

def check_identity_gpu(ops):
    """Has a two-sided identity element."""
    B, n, _ = ops.shape
    elems = torch.arange(n, device=DEVICE)
    # Left identity: ops[e, a] == a for all a
    left_id = (ops == elems.unsqueeze(0).unsqueeze(0)).all(dim=2)   # [B, n]
    # Right identity: ops[a, e] == a for all a
    right_id = (ops == elems.unsqueeze(0).unsqueeze(2)).all(dim=1)  # [B, n]
    # Two-sided: any e satisfying both
    both = (left_id & right_id).any(dim=1)  # [B]
    return both

def canonical_form_cpu(op_np):
    n = op_np.shape[0]
    best = None
    for perm in permutations(range(n)):
        p = list(perm)
        p_inv = [0] * n
        for i, v in enumerate(p):
            p_inv[v] = i
        new_op = [[p_inv[op_np[p[i], p[j]]] for j in range(n)] for i in range(n)]
        flat = tuple(x for row in new_op for x in row)
        if best is None or flat < best:
            best = flat
    return best

def compute_counting_invariants(ops):
    """
    Rich counting invariants for monoids.
    Key insight: 'has_identity' is always True for monoids -> zero info.
    Need element-level counting: orders, power structure.
    """
    B, n, _ = ops.shape
    elems = torch.arange(n, device=DEVICE)
    ops_cpu = ops.cpu().numpy()
    inv = {}

    # 1. Commutativity count
    inv['comm'] = (ops == ops.transpose(1, 2)).sum(dim=(1, 2)).float()

    # 2. Idempotent count: #{a: a^2 = a}
    diag = ops[:, elems, elems]
    inv['idemp'] = (diag == elems.unsqueeze(0)).sum(dim=1).float()

    # 3. Center size: #{a: ab=ba for all b}
    center = (ops == ops.transpose(1, 2)).all(dim=2)
    inv['center'] = center.sum(dim=1).float()

    # 4. Square image size: |{a^2}|
    sq_img = torch.zeros(B, device=DEVICE)
    for b in range(B):
        sq_img[b] = len(set(int(diag[b, i]) for i in range(n)))
    inv['sq_image'] = sq_img

    # 5. Element period histogram (via CPU) — KEY invariant for monoids
    # Period of a = smallest k > 0 such that a^(k+1) = a (index = 1 for groups)
    # Actually: smallest j >= 0, k >= 1 such that a^(j+k) = a^j (index, period)
    # Simplified: power_set size = |{a, a^2, ..., a^n}| per element
    power_set_sizes = torch.zeros(B, n, device='cpu')
    for b in range(B):
        op = ops_cpu[b]
        for a in range(n):
            seen = set()
            cur = a
            for _ in range(n + 1):
                if cur in seen:
                    break
                seen.add(cur)
                cur = op[cur, a]
            power_set_sizes[b, a] = len(seen)

    # Histogram: how many elements have each power-set size (1..n)
    for sz in range(1, n + 1):
        inv[f'power_sz_{sz}'] = (power_set_sizes == sz).sum(dim=1).float().to(DEVICE)

    # 6. Number of left-cancellable elements: #{a: (ab=ac => b=c)}
    lc = torch.zeros(B, device=DEVICE)
    for b in range(B):
        op = ops_cpu[b]
        count = 0
        for a in range(n):
            # Check: for all b, c: if op[a,b]=op[a,c] then b=c
            is_lc = True
            for bi in range(n):
                for ci in range(n):
                    if op[a, bi] == op[a, ci] and bi != ci:
                        is_lc = False
                        break
                if not is_lc:
                    break
            if is_lc:
                count += 1
        lc[b] = count
    inv['left_cancel'] = lc

    return inv

def boolean_class(ops):
    B, n, _ = ops.shape
    elems = torch.arange(n, device=DEVICE)
    comm = (ops == ops.transpose(1, 2)).all(dim=(1, 2))
    diag = ops[:, elems, elems]
    has_id = check_identity_gpu(ops)
    is_band = (diag == elems.unsqueeze(0)).all(dim=1)
    left_zero = (ops == elems.unsqueeze(0).unsqueeze(2)).all(dim=2).any(dim=1)
    return torch.stack([comm, has_id, is_band, left_zero], dim=1)

if __name__ == "__main__":
    print("n=3 MONOID COUNTING REVOLUTION")
    print("="*50)
    n = 3
    total = n ** (n * n)
    BATCH = 4096
    all_monoids = []
    t0 = time.time()

    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        batch_size = end - start
        indices = torch.arange(start, end, device=DEVICE)
        ops = torch.zeros(batch_size, n, n, dtype=torch.int64, device=DEVICE)
        idx = indices.clone()
        for row in range(n):
            for col in range(n):
                ops[:, row, col] = idx % n
                idx = idx // n

        assoc = check_associativity_gpu(ops)
        semigroups = ops[assoc]
        if semigroups.shape[0] > 0:
            monoid_mask = check_identity_gpu(semigroups)
            monoids = semigroups[monoid_mask]
            if monoids.shape[0] > 0:
                all_monoids.append(monoids)

    t1 = time.time()
    if all_monoids:
        all_monoids = torch.cat(all_monoids, dim=0)
    else:
        print("No monoids found!")
        exit()

    print(f"Monoid operations: {all_monoids.shape[0]} ({t1-t0:.2f}s)")

    # Iso classification
    ops_cpu = all_monoids.cpu().numpy()
    iso_map = {}
    iso_ids = []
    for op in ops_cpu:
        cf = canonical_form_cpu(op)
        if cf not in iso_map:
            iso_map[cf] = len(iso_map)
        iso_ids.append(iso_map[cf])

    n_iso = len(iso_map)
    print(f"Iso classes: {n_iso} (expected 7, OEIS A058129)")

    # Boolean
    bool_feats = boolean_class(all_monoids)
    iso_to_bool = {}
    for i, iso_id in enumerate(iso_ids):
        iso_to_bool[iso_id] = tuple(bool_feats[i].cpu().tolist())
    n_bool = len(set(iso_to_bool.values()))

    # Counting
    inv = compute_counting_invariants(all_monoids)
    iso_to_count = {}
    for i, iso_id in enumerate(iso_ids):
        ct = tuple(inv[k][i].item() for k in sorted(inv.keys()))
        iso_to_count[iso_id] = ct
    n_count = len(set(iso_to_count.values()))

    print(f"Boolean classes:  {n_bool}")
    print(f"Counting classes: {n_count}")
    print(f"Amplification:    {n_count/n_bool:.1f}x")

    print("\nAXIOM HIERARCHY (n=3):")
    print(f"{'Structure':<20} {'Axioms':>7} {'Bool':>6} {'Count':>7} {'Amp':>6}")
    print("-"*48)
    print(f"{'Magmas':<20} {'0':>7} {'114':>6} {'3328':>7} {'29.2x':>6}")
    print(f"{'Semigroups':<20} {'1':>7} {'16':>6} {'23':>7} {'1.4x':>6}")
    print(f"{'Monoids':<20} {'2':>7} {n_bool:>6} {n_count:>7} {n_count/n_bool:.1f}x")
    print(f"{'Groups':<20} {'3':>7} {'1':>6} {'1':>7} {'1.0x':>6}")
