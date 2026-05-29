"""
v38_diverse_starts_ils — parallel ILS seeded from diverse hull-anchored
initial constructions.

v37 closed pcb442 to optimum but gr666 stuck at exactly 294513 (155 over)
no matter how many parallel ILS iterations we ran.  All chains converged
to the same local minimum — strong evidence that the basin landed in by
the v01-cheapest construction is a *trap*: 2-opt + Or-opt + double-bridge
perturbation cannot reach the true optimum from there.

The fix is starting-state diversity.  Each parallel chain in v38 gets
its **own initial hull-anchored construction**:

    v01  — cheapest insertion (the original baseline)
    v02  — furthest-insertion
    v03  — regret insertion
    v04  — hull-distance-descending
    v05  — hull-distance-ascending
    v13  — centroid-spiral inward
    v15  — nearest-neighbour path splice
    v17  — lookahead k-cheapest insertion

Each is locally-optimal in its own neighbourhood.  Running ILS from each
independently lets the global ``min`` operation pick the best.

The hull is built once (shared) per the project's invariant.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import numpy as np

from algorithms.protocol import TraceStep
from algorithms.convex_hull import build_hull
from algorithms import (
    v01_baseline,
    v02_furthest_insertion,
    v03_regret_insertion,
    v04_hull_distance_descending,
    v05_hull_distance_ascending,
    v13_centroid_spiral_inward,
    v15_nn_path_splice,
    v17_lookahead_k_cheapest,
)
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


_INIT_SOLVERS = (
    ("v01", v01_baseline.solve_from_hull),
    ("v02", v02_furthest_insertion.solve_from_hull),
    ("v03", v03_regret_insertion.solve_from_hull),
    ("v04", v04_hull_distance_descending.solve_from_hull),
    ("v05", v05_hull_distance_ascending.solve_from_hull),
    ("v13", v13_centroid_spiral_inward.solve_from_hull),
    ("v15", v15_nn_path_splice.solve_from_hull),
    ("v17", v17_lookahead_k_cheapest.solve_from_hull),
)


def _budget(n: int) -> int:
    """Iterations per chain — there's already 8 chains × 8 starting tours."""
    if n <= 200:
        return 400
    if n <= 400:
        return 1200
    if n <= 700:
        return 3000
    return 5000


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    initial_tours = _build_initial_tours(hull, remaining, distance_matrix, coords)
    return _run_diverse_ils(initial_tours, distance_matrix)


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    # Use the v01 baseline trace for the visualizer (the others would
    # produce qualitatively similar trace timelines and we don't want to
    # multiply the trace size by 8).
    tour, insert_steps = v01_baseline.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    initial_tours = _build_initial_tours(hull, remaining, distance_matrix, coords)
    final_tour = _run_diverse_ils(initial_tours, distance_matrix)
    return final_tour, insert_steps


def _build_initial_tours(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None,
) -> list[list[int]]:
    out = []
    for name, solver in _INIT_SOLVERS:
        try:
            tour = solver(hull, remaining, distance_matrix, coords)
        except Exception:
            continue
        if tour and len(tour) == len(hull) + len(remaining):
            out.append(tour)
    return out


def _run_chain(args) -> tuple[list[int], float]:
    init_tour, dist, neighbors, iters, seed = args
    rng = np.random.default_rng(seed)
    cur = fast_local_search(init_tour, dist, neighbors, _OR_OPT_LENGTHS)
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


def _run_diverse_ils(initial_tours: list[list[int]], dist) -> list[int]:
    n = len(initial_tours[0])
    iters = _budget(n)
    k = min(_NEIGHBOR_K, n - 1)
    neighbors = build_neighbor_lists(dist, k=k)

    payloads = [
        (init, dist, neighbors, iters, _BASE_SEED + s * 1_000_003)
        for s, init in enumerate(initial_tours)
    ]
    n_workers = min(len(payloads), os.cpu_count() or 4)

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
