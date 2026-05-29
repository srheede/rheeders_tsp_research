"""
v34_parallel_ils — parallel multi-start ILS.

v33 brought the larger suite from 0.37% to 0.04% avg gap, with two
holdouts:

    pcb442  →  0.09%  (8 / 1000 iter ≈ 50 822 vs opt 50 778)
    gr666   →  0.16%  (1000 iter ≈ 294 838 vs opt 294 358)

The remaining gap is tiny — almost certainly within reach of a few
thousand more ILS iterations. Doing them sequentially is too slow, so
v34 runs the independent ILS chains across worker processes
(``multiprocessing.Pool``). On an 8-core box this is ~7× wall-clock
faster, allowing roughly an order of magnitude more total iterations
in the same time.

Kick diversity is increased by *occasionally* (≈ 1 in 8 iterations)
swapping the standard double-bridge for a chained
double-bridge-twice — a much stronger escape from deep basins.
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
_BASE_SEED = 0xBEEF


def _budget(n: int) -> tuple[int, int]:
    """(starts, iterations_per_start) chosen for runtime ~5–15 min/instance
    when the starts run in parallel on an 8-core machine."""
    if n <= 200:
        return 8, 400
    if n <= 400:
        return 8, 800
    if n <= 700:
        return 8, 1500
    return 12, 2500


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v01_baseline.solve_from_hull(hull, remaining, distance_matrix, coords)
    return _parallel_ils(tour, distance_matrix)


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v01_baseline.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    tour = _parallel_ils(tour, distance_matrix)
    return tour, insert_steps


def _run_chain(args) -> tuple[list[int], float]:
    """Worker entry point — one ILS chain.

    Receives a fully self-contained payload so the worker doesn't need
    any shared state besides what's pickled here.
    """
    base_tour, dist, neighbors, iters, seed = args
    rng = np.random.default_rng(seed)
    if seed % 7 == 0:
        # Diversify the starting point of every 7th chain.
        cur = double_bridge(double_bridge(base_tour, rng), rng)
    else:
        cur = list(base_tour)
    cur = fast_local_search(cur, dist, neighbors)
    best = list(cur)
    best_cost = compute_tour_cost(best, dist)

    no_improve = 0
    for it in range(iters):
        # Mix kick types: every 8th iteration use a stronger 2× double-bridge.
        if it % 8 == 7:
            kicked = double_bridge(double_bridge(best, rng), rng)
        else:
            kicked = double_bridge(best, rng)
        kicked = fast_local_search(kicked, dist, neighbors)
        c = compute_tour_cost(kicked, dist)
        if c < best_cost - 1e-9:
            best = kicked
            best_cost = c
            no_improve = 0
        else:
            no_improve += 1
            # If we've gone a long stretch without any improvement,
            # restart from the current best with a much stronger kick.
            if no_improve >= max(150, iters // 10):
                kicked = best
                for _ in range(4):
                    kicked = double_bridge(kicked, rng)
                kicked = fast_local_search(kicked, dist, neighbors)
                c2 = compute_tour_cost(kicked, dist)
                if c2 < best_cost - 1e-9:
                    best = kicked
                    best_cost = c2
                no_improve = 0
    return best, best_cost


def _parallel_ils(initial_tour: list[int], dist) -> list[int]:
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

    overall_best, overall_best_cost = min(results, key=lambda x: x[1])

    # Final polish with the full O(n²) compound LS — captures any moves
    # the neighbour-restricted LS might have missed.
    overall_best, _ = compound_local_search(
        list(overall_best), dist, or_opt_chain_lengths=(1, 2, 3, 4, 5)
    )
    return overall_best
