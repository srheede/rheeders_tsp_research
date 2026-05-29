"""
v35_parallel_ils_pure — parallel multi-start ILS with pure double-bridge.

Diagnostic to v34's quality regression on pcb442 and gr666: this is
exactly v33's algorithm, but each ILS chain runs in its own worker
process. No exotic kick variations, no no-improve restart. The only
difference vs v33 is wall-clock parallelism, which lets us afford more
iterations within the same time budget.

For n=666 the budget is 8 × 2500 iter (versus v33's 6 × 1000).
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


_NEIGHBOR_K = 20
_BASE_SEED = 0xC0FFEE  # same seed family as v33


def _budget(n: int) -> tuple[int, int]:
    if n <= 200:
        return 8, 500
    if n <= 400:
        return 8, 1200
    if n <= 700:
        return 8, 2500
    return 12, 4000


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v01_baseline.solve_from_hull(hull, remaining, distance_matrix, coords)
    return _parallel_ils_pure(tour, distance_matrix)


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v01_baseline.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    tour = _parallel_ils_pure(tour, distance_matrix)
    return tour, insert_steps


def _run_chain(args) -> tuple[list[int], float]:
    base_tour, dist, neighbors, iters, seed = args
    rng = np.random.default_rng(seed)
    if seed % 1_000_003 != 0:
        # Diversify each non-zero start with two double-bridges + LS.
        kicked = double_bridge(double_bridge(base_tour, rng), rng)
        cur = fast_local_search(kicked, dist, neighbors)
    else:
        cur = list(base_tour)
    best = list(cur)
    best_cost = compute_tour_cost(best, dist)

    for _ in range(iters):
        kicked = double_bridge(best, rng)
        kicked = fast_local_search(kicked, dist, neighbors)
        c = compute_tour_cost(kicked, dist)
        if c < best_cost - 1e-9:
            best = kicked
            best_cost = c
    return best, best_cost


def _parallel_ils_pure(initial_tour: list[int], dist) -> list[int]:
    n = len(initial_tour)
    starts, iters = _budget(n)
    k = min(_NEIGHBOR_K, n - 1)
    neighbors = build_neighbor_lists(dist, k=k)

    base = fast_local_search(initial_tour, dist, neighbors)

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

    overall_best, _ = min(results, key=lambda x: x[1])

    overall_best, _ = compound_local_search(
        list(overall_best), dist, or_opt_chain_lengths=(1, 2, 3, 4, 5)
    )
    return overall_best
