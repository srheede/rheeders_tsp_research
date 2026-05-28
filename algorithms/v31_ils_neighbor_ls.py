"""
v31_ils_neighbor_ls — ILS with neighbour-restricted LS for many more kicks.

v28 showed that ILS smashes the optimality gap (5.92% → 0.51%) but each
iteration is dominated by the LS step, which is O(n²) for 2-opt and
O(n²·L) for Or-opt. For n=280 each iteration takes ~6 seconds → 30
iterations costs 3 minutes.

By restricting each LS pass to **k-nearest-neighbour edges** (k=20)
plus standard don't-look-bits, each pass becomes O(n·k) — empirically
10-15× faster — at minimal quality cost. The freed budget pays for
*many more* ILS kicks, which is the real driver of solution quality.

Concretely:

  * Construction: v01 cheapest insertion (kept simple — ILS reshapes
    everything anyway).
  * Pre-compute one nearest-neighbour list per node.
  * Initial fast_local_search to convergence.
  * **300 ILS iterations** (10× v28), each = double-bridge kick +
    fast_local_search.
  * Final compound_local_search using the full O(n²) routines to
    polish off any neighbour-list-missed moves.
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


_ITERATIONS = 300
_NEIGHBOR_K = 20
_RNG_SEED = 0xC0FFEE


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v01_baseline.solve_from_hull(hull, remaining, distance_matrix, coords)
    return _ils_with_neighbors(tour, distance_matrix)


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v01_baseline.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    tour = _ils_with_neighbors(tour, distance_matrix)
    return tour, insert_steps  # ILS kicks are not represented as TraceSteps


def _ils_with_neighbors(tour: list[int], dist) -> list[int]:
    k = min(_NEIGHBOR_K, len(tour) - 1)
    neighbors = build_neighbor_lists(dist, k=k)
    tour = fast_local_search(tour, dist, neighbors)

    best = list(tour)
    best_cost = compute_tour_cost(best, dist)
    rng = np.random.default_rng(_RNG_SEED)

    for _ in range(_ITERATIONS):
        k_t = double_bridge(best, rng)
        k_t = fast_local_search(k_t, dist, neighbors)
        c = compute_tour_cost(k_t, dist)
        if c < best_cost - 1e-9:
            best = k_t
            best_cost = c

    # Polish with the full O(n²) compound search to capture any
    # improvement the neighbour-restricted LS missed.
    best, _ = compound_local_search(best, dist, or_opt_chain_lengths=(1, 2, 3, 4, 5))
    return best
