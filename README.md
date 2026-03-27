# GPU Semigroup Counting Revolution

Extends the [Counting Revolution](https://github.com/Tehlikeli107/counting-revolution) from magmas to **semigroups**, completing the axiom hierarchy.

## Key Result: Dramatic Amplification Drop at First Axiom

| Structure | Axioms | Iso classes (n=3) | Bool classes | Count classes | Amplification |
|-----------|--------|-------------------|--------------|---------------|---------------|
| Magmas | 0 | 3,330 | 114 | 3,328 | **29.2x** |
| **Semigroups** | **1 (assoc)** | **24** | **16** | **23** | **1.4x** |
| Groups | 3 | 1 | 1 | 1 | **1.0x** |

**Universal Principle**: One axiom (associativity) reduces iso classes from 3,330 to 24 (99.3% reduction) and makes boolean already powerful. Counting adds only marginal new classification power.

Cross-order comparison:

| Structure | Order | Bool classes | Count classes | Amplification |
|-----------|-------|--------------|---------------|---------------|
| Magmas | n=3 | 114 | 3,328 | **29.2x** |
| Semigroups | n=3 | 16 | 23 | **1.4x** |
| Groups | n=8 | 2 | 5 | **2.5x** |
| Rings | n=4 | 4 | 8 | **2.0x** |

## Why the Drop is Dramatic

For n=3 magmas (no axioms): 3,330 iso classes → huge classification challenge → counting invariants shine.

For n=3 semigroups (associativity): only 24 iso classes remain. Boolean already captures 16 of these. The constraint collapses the space so much that there is little room left for counting to improve on.

**The first axiom is the most powerful** — associativity alone eliminates 99.3% of magma iso classes. Additional axioms (identity, inverse) eliminate progressively fewer.

## Semigroup Counting Invariants

For a semigroup (S, ×):

1. **Commutativity count**: #{(a,b): a×b = b×a}
2. **Idempotent count**: #{a: a² = a}
3. **Left identity count**: #{e: e×a = a for all a}
4. **Right identity count**: #{e: a×e = a for all a}
5. **Left zero count**: #{z: z×a = z for all a}
6. **Right zero count**: #{z: a×z = z for all a}
7. **Center size**: #{a: a×b = b×a for all b}
8. **Square image size**: |{a²: a ∈ S}|
9. **Squares-to-zero count**: #{a: a² = 0}
10. **Absorbing element count**: #{z: z×a = a×z = z for all a}

Boolean classification: "is it commutative?", "does it have identity?", "left/right zero?", "is it a band?"

## n=3 Complete Classification

All 24 iso classes (up to relabeling; 18 if also quotienting anti-isomorphisms):

| Class | comm | idemp | l-id | r-id | l-zero | r-zero | center | sq-img |
|-------|------|-------|------|------|--------|--------|--------|--------|
| 0 | 9 | 1 | 0 | 0 | 1 | 1 | 3 | 1 |
| 1 | 9 | 2 | 0 | 0 | 1 | 1 | 3 | 2 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 17 | 9 | 3 | 1 | 1 | 1 | 1 | 3 | 3 |

Boolean sees 16 classes. Counting + 10 invariants sees **23 classes** (1.4x amplification).

## GPU Throughput (RTX 4070 Laptop)

| Order | Tables/sec | Method |
|-------|-----------|--------|
| n=3 | 136M/sec | Exhaustive (all 19,683 tables) |
| n=4 | 8.8M/sec | Random sampling (P(semigroup) ≈ 8×10⁻⁷) |

vs libsemigroups / Smallsemi (GAP): CPU-only, single-threaded.

## Connection to Counting Revolution

Fourth algebraic structure in the framework:

1. **Magmas** → 29x amplification (no axioms, maximum freedom)
2. **Semigroups** → 1.4x amplification (1 axiom: associativity)
3. **Groups** → 2.5x–6x amplification (3 axioms)
4. **Rings** → 2x amplification (4 axioms)

The non-monotone Groups/Rings ordering is due to using different orders (n=8 for groups, n=4 for rings). The key result is the catastrophic drop from Magmas to Semigroups: one axiom destroys 99.3% of structure and 95% of counting power.

## Usage

```bash
pip install torch  # CUDA version
python gpu_semigroup_counting.py
```

---

*Part of the [Counting Revolution](https://github.com/Tehlikeli107/counting-revolution) project.*
