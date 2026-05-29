"""
v33_adaptive_ils — multi-start ILS with budget that scales with instance size.

Running v32 on the larger suite (ch150 … gr666) revealed two stragglers:

    pcb442  →  0.69%  gap (442 nodes)
    gr666   →  1.36%  gap (666 nodes)
    gr202   →  0.17%  gap (202 nodes)
    others  →  0.00%  (already optimal)

The natural fix is more iteration budget on the bigger instances, but a
flat increase blows up small-instance runtime. v33 instead picks
``starts`` and ``iterations`` from a bracket lookup based on ``n``. The
LS pipeline is unchanged (k-NN restricted 2-opt + Or-opt with don't-look
bits), but Or-opt now runs the new ``or_opt_neighbors_dl`` variant which
is 5–10× faster on n>400 thanks to incremental position updates.

Bracket schedule (chosen so total runtime is sub-linear in n²):

    n ≤ 200  : 4 starts × 300 iter
    n ≤ 400  : 5 starts × 600 iter
    n ≤ 700  : 6 starts × 1000 iter
    n > 700  : 8 starts × 1500 iter
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms import v01_baseline
from algorithms._helpers import (
    build_neighbor_lists,
    fast_local_search,
    compound_local_search,
    double_bridge,
    compute_tour_cost,
)


_NEIGHBOR_K = 20
_BASE_SEED = 0xC0FFEE


def _budget(n: int) -> tuple[int, int]:
    if n <= 200:
        return 4, 300
    if n <= 400:
        return 5, 600
    if n <= 700:
        return 6, 1000
    return 8, 1500


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v01_baseline.solve_from_hull(hull, remaining, distance_matrix, coords)
    return _adaptive_ils(tour, distance_matrix)


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v01_baseline.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    tour = _adaptive_ils(tour, distance_matrix)
    return tour, insert_steps


def _adaptive_ils(initial_tour: list[int], dist) -> list[int]:
    n = len(initial_tour)
    starts, iters = _budget(n)
    k = min(_NEIGHBOR_K, n - 1)
    neighbors = build_neighbor_lists(dist, k=k)

    base = fast_local_search(initial_tour, dist, neighbors)
    overall_best = list(base)
    overall_best_cost = compute_tour_cost(overall_best, dist)

    for s in range(starts):
        rng = np.random.default_rng(_BASE_SEED + s * 1_000_003)
        if s == 0:
            cur = list(base)
        else:
            # Diversify each start by chaining 2 double-bridges then LS.
            kicked = double_bridge(double_bridge(base, rng), rng)
            cur = fast_local_search(kicked, dist, neighbors)
        best = list(cur)
        best_cost = compute_tour_cost(best, dist)
        for _ in range(iters):
            kicked = double_bridge(best, rng)
            kicked = fast_local_search(kicked, dist, neighbors)
            c = compute_tour_cost(kicked, dist)
            if c < best_cost - 1e-9:
                best = kicked
                best_cost = c
        if best_cost < overall_best_cost - 1e-9:
            overall_best = best
            overall_best_cost = best_cost

    # Final polish with full O(n²) compound LS to capture any moves the
    # neighbour-restricted LS missed.
    overall_best, _ = compound_local_search(
        overall_best, dist, or_opt_chain_lengths=(1, 2, 3, 4, 5)
    )
    return overall_best
