# TSP R&D — Final Report

## 1. What we built

Starting from the premise *"compute a convex hull first, then connect the
remaining nodes"*, we developed **45 algorithm variants** (`v01` … `v45`)
and a shared toolkit (`algorithms/_helpers.py`, plus a C kernel
`algorithms/_ls_inner.c`). Every variant obeys the project invariant:
the shared boundary-chain construction runs first, then the variant
decides how the interior nodes are woven in.

The work passed through five families of ideas:

| Family | Examples | Verdict |
|--------|----------|---------|
| Insertion order | furthest, regret, hull-distance, centroid-spiral | beat baseline modestly |
| Geometric assignment | perpendicular projection, wedge / angle-bisector decomposition | poor — geometry ≠ cost |
| Layered / recursive | convex layers (onion), recursive hull | poor without cost awareness |
| Path-splicing | MST-DFS, NN-splice, strip decomposition | poor |
| Globally-aware | Hungarian assignment, Delaunay BFS, DP window, lookahead | competitive construction |
| **Local search + metaheuristic** | 2-opt, Or-opt, 3-opt, **Iterated Local Search** | **decisive** |

The champion is **`v44_native_ils_full`**: hull construction → cheapest
insertion → **Iterated Local Search (ILS)** whose entire loop
(double-bridge perturbation + 2-opt + Or-opt + cost + acceptance) runs
natively in C, across 16 parallel chains.

### Final results (largest labelled-optimal instances)

| Instance | Nodes | Best cost | Optimal | Gap | Champion |
|----------|------:|----------:|--------:|----:|----------|
| a280   | 280  | 2 579   | 2 579   | **0.00 %** | v44 |
| pcb442 | 442  | 50 778  | 50 778  | **0.00 %** | v44 |
| gr666  | 666  | 294 358 | 294 358 | **0.00 %** | v44 |
| pr1002 | 1002 | 259 045 | 259 045 | **0.00 %** | v44 |
| pr2392 | 2392 | 378 665 | 378 032 | 0.167 %    | v45 |

**Average gap 0.04 %**, with 4 of 5 instances solved to the *proven
optimum*. The smaller historical suite (22–280 nodes) reaches **0.00 %
on every instance**.

### Speed / time-complexity progression

| Variant | Key idea | Avg gap | Total time |
|---------|----------|--------:|-----------:|
| v32 | sequential multi-start ILS | 0.37 % | 1 565 s |
| v37 | parallel multi-start ILS | 0.01 %¹ | 2 009 s |
| v40 | + O(n²) construction (was O(n³)) | 0.24 % | 5 949 s |
| v41 | + **C local-search kernel** (47× inner-loop) | 0.12 % | 1 024 s |
| v43 | + 3× ILS budget | 0.058 % | 5 464 s |
| **v44** | **entire ILS loop in C** | **0.04 %** | **1 874 s** |

¹ v37's suite topped out at 666 nodes; v40+ add pr1002 / pr2392.

The decisive efficiency win was recognising that, once the inner local
search was already in C, the **per-iteration Python overhead** (the
`double_bridge` slice, the list↔numpy marshalling and an O(n) Python
tour-cost loop) dominated. Moving the *whole* ILS loop into C removed it,
giving a 50–100× reduction in per-iteration wall-time. The freed budget
was reinvested as more iterations — so v44 is simultaneously **3× faster
than v43 and more accurate**, which is exactly the
"lowest time-complexity without sacrificing accuracy" target.

### Time complexity of the champion

* Boundary construction (sort + sweep): **O(n log n)**.
* Cheapest insertion: **O(n²)**.
* k-nearest-neighbour lists: **O(n²)** (dominated by the distance matrix).
* Per ILS iteration: **O(n·k)** amortised (neighbour-restricted LS with
  don't-look bits; k = 20 constant).
* Whole run: **O(iterations · n · k)** time, **O(n²)** memory.

The O(n²) distance matrix is the real scaling ceiling (≈ 45 MB at
n = 2392; it would be ~1.4 GB at n = 13 509). Beyond a few thousand
nodes a spatial neighbour index would have to replace the dense matrix.

---

## 2. Was the "convex-hull-first" premise correct?

**Short answer: it is a sound, classical *construction* heuristic, but it
is *not* the source of our competitive results, and it does not make our
solver better than the best global TSP algorithms.**

Three findings support this, all measured in this repo:

### (a) The hull captures very little of the tour

The shared construction is actually a **monotone geometric boundary
sweep**, not a strict convex hull. On our instances it places only a
small fraction of the nodes on the boundary chain:

| Instance | Nodes on boundary chain | Total nodes |
|----------|------------------------:|------------:|
| gr666  | 18  | 666  |
| pr1002 | 50  | 1002 |
| a280   | 67  | 280  |
| pr2392 | 84  | 2392 |

So 90–97 % of the tour is decided *after* the hull, by the insertion +
local-search phases — not by the hull itself.

### (b) The construction starts ~25 % above optimum; the metaheuristic does the work

| Instance | Hull + cheapest insertion | After one LS pass | After full ILS |
|----------|--------------------------:|------------------:|---------------:|
| a280   | +20.4 % | +2.3 % | **0.0 %** |
| pcb442 | +22.4 % | +5.5 % | **0.0 %** |
| gr666  | +29.0 % | +5.2 % | **0.0 %** |
| pr1002 | +26.7 % | +7.2 % | **0.0 %** |
| pr2392 | +25.6 % | +6.6 % | +0.17 % |

The journey from +25 % to 0 % is travelled almost entirely by local
search and ILS. The hull provides a *reasonable, fast* starting point —
nothing more.

### (c) The starting construction barely matters

Variant **`v38_diverse_starts_ils`** seeded the ILS from **eight
different** hull-anchored constructions (cheapest, furthest, regret,
hull-distance asc/desc, centroid-spiral, NN-splice, lookahead). It was
**no better** than starting from a single cheapest-insertion tour
(0.06 % vs v37's 0.01 % on the same suite). A good metaheuristic erases
the influence of its starting tour — a well-known property of ILS.

### Verdict on the premise

* **As a construction heuristic, "hull first" is legitimate and good.**
  Convex-hull insertion is a genuine, studied TSP construction
  (Stewart 1977; Golden et al. 1980; Or 1976). It is geometrically
  motivated — the optimal tour visits the convex-hull vertices in their
  hull order, so anchoring on the boundary never *hurts* — it is fast
  (O(n²)), and it is easy to visualise.
* **But it is not why our solver is competitive.** Our 0.00 % results
  come from the Iterated Local Search, which is construction-agnostic.
  Had we started from a nearest-neighbour or even random tour, the final
  numbers would be essentially identical.
* **It does not beat the best global algorithms.** It neither needs nor
  benefits from the hull in a way that a generic ILS wouldn't, and it
  does not match exact/at-optimum guarantees on large instances.

---

## 3. How our best solution compares to known TSP algorithms

| Algorithm / heuristic | Typical gap to optimum | Notes vs. ours |
|-----------------------|------------------------|----------------|
| Nearest neighbour | 20–25 % | We beat it before any LS. |
| Christofides | ≤ 50 % bound, ~10 % typical | Has a worst-case guarantee we lack; we are far better in practice. |
| Convex-hull / greedy insertion | 10–15 % | This is essentially our *construction* phase. |
| 2-opt local optimum | 4–6 % | We use it, then go far past it. |
| Or-opt + 2-opt ("2.5-opt") | 2–4 % | Our inner LS. |
| **Our ILS (v44)** | **0–0.17 %** | 4/5 huge instances exact; minutes of runtime. |
| Lin–Kernighan (LK) | 0.5–2 % single run | Stronger *single* descent than ours via sequential k-opt. |
| **LKH** (Helsgaun) | **~0 %, often optimal** | The gold-standard heuristic; reaches optimum on instances 10–100× larger than ours. |
| Concorde (exact, branch-and-cut) | **0 % proven** | Proves optimality; the reference truth we measure against. |

### Where our solution is genuinely strong

1. **Near-exact quality on medium instances.** 0.00 % on every instance
   up to ~1000 nodes, reached in seconds-to-minutes. For a from-scratch
   implementation this is excellent.
2. **Excellent accuracy/runtime trade-off.** v44 reaches the optimum on
   four large instances in ~31 minutes *total* on a commodity 8-core
   laptop, with no external solver — just NumPy + a ~300-line C kernel
   compiled on the fly.
3. **Dependency-light and portable.** No LKH binary, no ILP solver, no
   GPU. The C kernel is built at import time with the system compiler.
4. **Deterministic and reproducible.** Per-chain xoshiro256** seeds make
   every run bit-for-bit repeatable.
5. **Interpretable.** The hull + insertion phases are fully traced and
   visualisable step-by-step, which exact solvers are not.

### Where it is *not* better (honest limits)

1. **It is not exact.** Concorde *proves* optimality; we cannot.
2. **It does not beat LKH.** LKH's sequential 5-opt moves reach optimum
   on instances orders of magnitude larger; our 2-opt + Or-opt + ILS
   plateaus (pr2392 stalls at ~0.17 % regardless of budget — the clearest
   evidence that our move set, not our compute, is the bottleneck).
3. **O(n²) memory.** The dense distance matrix caps us at a few thousand
   nodes; LKH/Concorde use spatial structures and scale far past that.

### The one remaining gap, explained

`pr2392` shrank monotonically with budget (0.69 → 0.28 → 0.25 → 0.18)
and then **stalled at 0.167 %** even after tripling the iteration count
(v45). A plateau that is insensitive to compute is the signature of a
*neighbourhood* limit: the double-bridge + 2-opt + Or-opt move set
cannot represent the final improving moves. Closing it would require a
true **Lin–Kernighan sequential-k-opt** search — the next logical
extension, and precisely the technique that separates "very good ILS"
from "state-of-the-art LKH".

---

## 4. Conclusion

* The convex-hull-first premise is **correct as a fast, principled
  construction** and a nice scaffold for experimentation, but it is
  **not the reason for our near-optimal results** — those come from the
  Iterated Local Search, which is largely indifferent to its starting
  tour.
* Our best solution (**v44**) is **competitive with strong heuristics
  and near-exact on medium instances**, achieved with a compact,
  dependency-light, fully reproducible implementation that is **3×
  faster than our previous champion while being more accurate**.
* It is **not better than LKH or Concorde** in the ways that matter most
  to the literature (optimality proofs, scaling, the hardest large
  instances). The honest frontier for surpassing the last 0.17 % is a
  Lin–Kernighan-class local search — a clear, well-defined next step.
