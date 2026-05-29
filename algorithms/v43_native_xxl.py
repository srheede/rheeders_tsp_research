"""
v43_native_xxl — v42 with 2-3× the ILS budget.

After the C-kernel speedup (v41 / v42), per-iteration cost on pr2392 is
≈ 25 ms — a fraction of what it was — so we can afford to triple the
iteration count on the holdouts (pr1002, pr2392) and still keep the
total under an hour.

Schedule (per chain, parallel across cores):

    n ≤ 200   :  8 × 12 000
    n ≤ 400   :  8 × 40 000
    n ≤ 700   : 12 × 120 000
    n ≤ 1200  : 16 × 200 000
    n ≤ 2000  : 16 × 100 000
    n > 2000  : 16 × 60 000

3-opt polish at the end (variants 4/5/6) is unchanged.
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
    compound_local_search,
    three_opt_neighbors,
    double_bridge,
    compute_tour_cost,
)
from algorithms._ls_native import fast_local_search_c


_NEIGHBOR_K = 20
_OR_OPT_LENGTHS = (1, 2, 3, 4, 5)
_BASE_SEED = 0xC0FFEE


def _budget(n: int) -> tuple[int, int]:
    if n <= 200:
        return 8, 12000
    if n <= 400:
        return 8, 40000
    if n <= 700:
        return 12, 120000
    if n <= 1200:
        return 16, 200000
    if n <= 2000:
        return 16, 100000
    return 16, 60000


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    n_total = len(hull) + len(remaining)
    if n_total <= 600:
        tour = v01_baseline.solve_from_hull(
            hull, remaining, distance_matrix, coords
        )
    else:
        tour = fast_cheapest_insertion(hull, remaining, distance_matrix)
    return _parallel_ils_xxl(tour, distance_matrix)


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v01_baseline.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    final_tour = _parallel_ils_xxl(tour, distance_matrix)
    return final_tour, insert_steps


def _run_chain(args) -> tuple[list[int], float]:
    base_tour, dist, neighbors, iters, seed = args
    rng = np.random.default_rng(seed)
    if (seed % 1_000_003) != 0:
        kicked = double_bridge(double_bridge(base_tour, rng), rng)
        cur = fast_local_search_c(kicked, dist, neighbors, _OR_OPT_LENGTHS)
    else:
        cur = list(base_tour)
    best = list(cur)
    best_cost = compute_tour_cost(best, dist)

    for _ in range(iters):
        kicked = double_bridge(best, rng)
        kicked = fast_local_search_c(kicked, dist, neighbors, _OR_OPT_LENGTHS)
        c = compute_tour_cost(kicked, dist)
        if c < best_cost - 1e-9:
            best = kicked
            best_cost = c
    return best, best_cost


def _polish_3opt(args) -> tuple[list[int], float]:
    tour, dist, neighbors = args
    tour = three_opt_neighbors(tour, dist, neighbors)
    tour = fast_local_search_c(tour, dist, neighbors, _OR_OPT_LENGTHS)
    tour = three_opt_neighbors(tour, dist, neighbors)
    return tour, compute_tour_cost(tour, dist)


def _polish_full(args) -> tuple[list[int], float]:
    tour, dist = args
    polished, _ = compound_local_search(
        list(tour), dist, or_opt_chain_lengths=_OR_OPT_LENGTHS
    )
    return polished, compute_tour_cost(polished, dist)


def _parallel_ils_xxl(initial_tour: list[int], dist) -> list[int]:
    n = len(initial_tour)
    starts, iters = _budget(n)
    k = min(_NEIGHBOR_K, n - 1)
    neighbors = build_neighbor_lists(dist, k=k)

    base = fast_local_search_c(initial_tour, dist, neighbors, _OR_OPT_LENGTHS)

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

    polish_payloads = [(t, dist, neighbors) for t, _ in results]
    if n_workers <= 1 or n <= 50:
        polished = [_polish_3opt(p) for p in polish_payloads]
    else:
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            polished = pool.map(_polish_3opt, polish_payloads)

    if n <= 1500:
        full_payloads = [(t, dist) for t, _ in polished]
        if n_workers <= 1 or n <= 50:
            polished_full = [_polish_full(p) for p in full_payloads]
        else:
            with mp.get_context("spawn").Pool(processes=n_workers) as pool:
                polished_full = pool.map(_polish_full, full_payloads)
        polished = polished_full

    overall_best, _ = min(polished, key=lambda x: x[1])
    return overall_best
