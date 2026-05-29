"""
v40_huge_parallel_ils — v37's parallel ILS scaled to huge instances.

Two fixes vs v37 needed for n ≥ 1000:

  * **Fast construction.** v01_baseline (cheapest-insertion) is O(n³) —
    on pr1002 it took 123 s, on pr2392 it would be over 25 min. v40
    swaps in ``fast_cheapest_insertion`` (O(n²) — sequential cheapest
    insertion in arbitrary order). The starting-tour quality is
    slightly worse, but LS converges to the same basin.

  * **Re-tuned per-n iteration budget.** With per-iter time scaling
    roughly with n on Euclidean instances, a flat 8000 ILS iter would
    take 90 + min on pr2392 even after parallelising 16 chains. v40
    annealed the budget: more chains, fewer iter as n grows.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import numpy as np

from algorithms.protocol import TraceStep
from algorithms import v01_baseline
from algorithms._helpers import (
    fast_cheapest_insertion,
    build_neighbor_lists,
    fast_local_search,
    compound_local_search,
    double_bridge,
    compute_tour_cost,
)


_NEIGHBOR_K = 20
_OR_OPT_LENGTHS = (1, 2, 3, 4, 5)
_BASE_SEED = 0xC0FFEE


def _budget(n: int) -> tuple[int, int]:
    if n <= 200:
        return 8, 800
    if n <= 400:
        return 8, 2000
    if n <= 700:
        return 12, 4500
    if n <= 1200:
        return 16, 4000
    if n <= 2000:
        return 16, 3000
    return 16, 2000


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    n_total = len(hull) + len(remaining)
    if n_total <= 600:
        # Small enough that v01's better starting tour is worth the cost.
        tour = v01_baseline.solve_from_hull(
            hull, remaining, distance_matrix, coords
        )
    else:
        tour = fast_cheapest_insertion(hull, remaining, distance_matrix)
    return _parallel_ils_xl(tour, distance_matrix)


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v01_baseline.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    final_tour = _parallel_ils_xl(tour, distance_matrix)
    return final_tour, insert_steps


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
    tour, dist = args
    polished, _ = compound_local_search(
        list(tour), dist, or_opt_chain_lengths=_OR_OPT_LENGTHS
    )
    return polished, compute_tour_cost(polished, dist)


def _parallel_ils_xl(initial_tour: list[int], dist) -> list[int]:
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

    # Skip the full O(n²) compound polish for huge instances — too slow
    # and the chain results are already locally optimal at fast_LS depth.
    if n > 1500:
        overall_best, _ = min(results, key=lambda x: x[1])
        return overall_best

    polish_payloads = [(t, dist) for t, _ in results]
    if n_workers <= 1 or n <= 50:
        polished = [_polish(p) for p in polish_payloads]
    else:
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            polished = pool.map(_polish, polish_payloads)

    overall_best, _ = min(polished, key=lambda x: x[1])
    return overall_best
