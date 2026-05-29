"""
v36_parallel_ils_deep — parallel ILS with a deeper LS.

v35 closed the gap on the larger suite to 0.01% avg, with two stragglers:

    pcb442 → 0.01%   ( 6 over a 50778 optimum)
    gr666  → 0.05%   (155 over a 294358 optimum)

To eliminate them, v36 layers three small but compounding improvements:

  1. Wider neighbour list (k=25 up from k=20).  Larger k discovers a
     few more improving moves per pass, especially on n>500 instances
     where the optimal tour edges sometimes reach beyond the 20th
     nearest neighbour.

  2. Longer Or-opt chains (1..7 up from 1..5).  Empirically a few of
     the gr666 local-minimum gaps are closed by 6- or 7-node
     relocations that 1..5 misses.

  3. Polish *every* chain's best with the full O(n²) compound LS,
     then keep the global minimum.  v35 only polished the single best,
     so a slightly-worse chain minimum that happened to sit closer to
     the true optimum could go unrecovered.

The iteration budget is increased modestly because the per-iter cost
is higher (k=25 vs 20).
"""

from __future__ import annotations

import multiprocessing as mp
import os
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


_NEIGHBOR_K = 25
_OR_OPT_LENGTHS = (1, 2, 3, 4, 5, 6, 7)
_BASE_SEED = 0xC0FFEE


def _budget(n: int) -> tuple[int, int]:
    if n <= 200:
        return 8, 500
    if n <= 400:
        return 8, 1500
    if n <= 700:
        return 8, 3500
    return 12, 5000


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v01_baseline.solve_from_hull(hull, remaining, distance_matrix, coords)
    return _parallel_ils_deep(tour, distance_matrix)


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v01_baseline.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    tour = _parallel_ils_deep(tour, distance_matrix)
    return tour, insert_steps


def _run_chain(args) -> tuple[list[int], float]:
    base_tour, dist, neighbors, iters, seed = args
    rng = np.random.default_rng(seed)
    if (seed % 1_000_003) != 0:
        kicked = double_bridge(double_bridge(base_tour, rng), rng)
        cur = fast_local_search(kicked, dist, neighbors, _OR_OPT_LENGTHS)
    else:
        cur = list(base_tour)
    best = list(cur)
    best_cost = compute_tour_cost(best, dist)

    for _ in range(iters):
        kicked = double_bridge(best, rng)
        kicked = fast_local_search(kicked, dist, neighbors, _OR_OPT_LENGTHS)
        c = compute_tour_cost(kicked, dist)
        if c < best_cost - 1e-9:
            best = kicked
            best_cost = c
    return best, best_cost


def _polish(args) -> tuple[list[int], float]:
    """Polish a single chain result with full O(n²) compound LS."""
    tour, dist = args
    polished, _ = compound_local_search(
        list(tour), dist, or_opt_chain_lengths=_OR_OPT_LENGTHS
    )
    return polished, compute_tour_cost(polished, dist)


def _parallel_ils_deep(initial_tour: list[int], dist) -> list[int]:
    n = len(initial_tour)
    starts, iters = _budget(n)
    k = min(_NEIGHBOR_K, n - 1)
    neighbors = build_neighbor_lists(dist, k=k)

    base = fast_local_search(initial_tour, dist, neighbors, _OR_OPT_LENGTHS)

    n_workers = min(starts, os.cpu_count() or 4)
    payloads = [
        (base, dist, neighbors, iters, _BASE_SEED + s * 1_000_003)
        for s in range(starts)
    ]

    if n_workers <= 1 or n <= 50:
        results = [_run_chain(p) for p in payloads]
    else:
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            results = pool.map(_run_chain, payloads)

    # Polish every chain's best, in parallel, then take the global min.
    polish_payloads = [(t, dist) for t, _ in results]
    if n_workers <= 1 or n <= 50:
        polished = [_polish(p) for p in polish_payloads]
    else:
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            polished = pool.map(_polish, polish_payloads)

    overall_best, _ = min(polished, key=lambda x: x[1])
    return overall_best
