"""
v32_multistart_ils — multi-start ILS to close the last % gap.

v31 (neighbour-restricted ILS, 300 iterations) drove every instance to
optimum *except* ``tsp225`` (1.61%). The pattern that holdout points
to is the classic problem with single-chain ILS: at some point all
kicks have been tried and the search becomes a deterministic loop
around the current basin.

The standard remedy is **multi-start**: run several *independent* ILS
chains from different random seeds and keep the best tour ever seen
across all chains. The diversity of starting points gives each chain
access to a different fraction of the search space; the union is
much closer to the global optimum.

We use **4 chains × 300 iterations** = 1200 total iterations per
instance, but with quadratically diversified perturbation paths. The
runtime is ~4× v31's, mostly on the larger instances. The construction
+ LS pipeline is identical to v31 so the per-iteration cost is the
same.
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


_STARTS = 4
_ITERATIONS = 300
_NEIGHBOR_K = 20
_BASE_SEED = 0xC0FFEE


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v01_baseline.solve_from_hull(hull, remaining, distance_matrix, coords)
    return _multistart_ils(tour, distance_matrix)


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v01_baseline.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    tour = _multistart_ils(tour, distance_matrix)
    return tour, insert_steps


def _multistart_ils(initial_tour: list[int], dist) -> list[int]:
    k = min(_NEIGHBOR_K, len(initial_tour) - 1)
    neighbors = build_neighbor_lists(dist, k=k)

    base = fast_local_search(initial_tour, dist, neighbors)
    overall_best = list(base)
    overall_best_cost = compute_tour_cost(overall_best, dist)

    for s in range(_STARTS):
        rng = np.random.default_rng(_BASE_SEED + s * 1_000_003)
        # Seed each chain with a *different* perturbation of the initial LS
        # local minimum — diversifies the starting points.
        cur = list(base) if s == 0 else fast_local_search(
            double_bridge(double_bridge(base, rng), rng), dist, neighbors
        )
        best = list(cur)
        best_cost = compute_tour_cost(best, dist)
        for _ in range(_ITERATIONS):
            kicked = double_bridge(best, rng)
            kicked = fast_local_search(kicked, dist, neighbors)
            c = compute_tour_cost(kicked, dist)
            if c < best_cost - 1e-9:
                best = kicked
                best_cost = c
        if best_cost < overall_best_cost - 1e-9:
            overall_best = best
            overall_best_cost = best_cost

    # Polish with full O(n²) compound LS at the end.
    overall_best, _ = compound_local_search(
        overall_best, dist, or_opt_chain_lengths=(1, 2, 3, 4, 5)
    )
    return overall_best
