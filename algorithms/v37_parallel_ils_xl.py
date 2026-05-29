"""
v37_parallel_ils_xl — v35's pure parallel ILS with bigger budget and
per-chain polish.

Findings so far on the larger suite (ch150 … gr666):

    v33  (sequential, 0.04% avg, gr666 0.16%)
    v34  (parallel + extra kicks, 0.10% avg — kicks hurt)
    v35  (parallel + pure double-bridge, 0.01% avg, gr666 0.05%)
    v36  (parallel, k=25, Or-opt 1..7, 0.07% avg — *worse* than v35)

The k=25 / Or-opt 1..7 widening in v36 explored different local minima
that turned out to be a bit shallower on average, so v37 reverts to the
v35 LS parameterisation (k=20, Or-opt 1..5) and instead invests the
extra time in:

  • bigger ILS budget per chain (≈ 2× v35 on the holdouts);
  • polishing *every* chain's best with the full O(n²) compound LS, so
    a slightly worse chain optimum that's actually closer to the global
    optimum can win after polish.
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
_OR_OPT_LENGTHS = (1, 2, 3, 4, 5)
_BASE_SEED = 0xC0FFEE


def _budget(n: int) -> tuple[int, int]:
    if n <= 200:
        return 8, 800
    if n <= 400:
        return 8, 2000
    if n <= 700:
        return 12, 4500
    return 16, 8000


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v01_baseline.solve_from_hull(hull, remaining, distance_matrix, coords)
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
    tour = _parallel_ils_xl(tour, distance_matrix)
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

    polish_payloads = [(t, dist) for t, _ in results]
    if n_workers <= 1 or n <= 50:
        polished = [_polish(p) for p in polish_payloads]
    else:
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            polished = pool.map(_polish, polish_payloads)

    overall_best, _ = min(polished, key=lambda x: x[1])
    return overall_best
