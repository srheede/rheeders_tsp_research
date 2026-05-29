"""
v44_native_ils_full — Iterated Local Search with the ENTIRE loop in C.

Previous champion v43 reached 0.058% avg gap on the huge suite but took
91 minutes: every ILS iteration bounced back into Python for the
double-bridge kick, the list↔numpy marshalling, and an O(n) Python
tour-cost loop. With millions of iterations those constant factors
dominated.

v44 moves the whole ILS chain into the C kernel (``ils_run`` in
``_ls_inner.c``): perturbation, 2-opt + Or-opt local search, cost
evaluation and accept-if-better all run natively, looping thousands of
times without ever returning to Python. Measured speedups vs the
Python-driven loop:

    a280   ~3000 ILS iter/s   (was ~60/s)
    gr666  ~700  ILS iter/s   (was ~6/s)

i.e. roughly a 50-100× reduction in per-iteration wall-time. The freed
budget is reinvested as more iterations across the 16 parallel chains,
so quality matches or beats v43 at a fraction of the runtime — the
"lowest time complexity without sacrificing accuracy" target.

Construction: O(n²) ``fast_cheapest_insertion`` for n>600 (v01's O(n³)
cheapest insertion is only worth it on small instances).
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
    compute_tour_cost,
)
from algorithms._ls_native import ils_run_c


_NEIGHBOR_K = 20
_OR_OPT_LENGTHS = (1, 2, 3, 4, 5)
_BASE_SEED = 0xC0FFEE


def _budget(n: int) -> tuple[int, int]:
    """(chains, iterations_per_chain). Chains = logical cores (16)."""
    if n <= 300:
        return 16, 60000
    if n <= 500:
        return 16, 80000
    if n <= 700:
        return 16, 150000
    if n <= 1200:
        return 16, 150000
    if n <= 2000:
        return 16, 90000
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
    return _parallel_native_ils(tour, distance_matrix)


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v01_baseline.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    final_tour = _parallel_native_ils(tour, distance_matrix)
    return final_tour, insert_steps


def _run_chain(args) -> tuple[list[int], float]:
    base_tour, dist, neighbors, iters, seed, init_kick = args
    best = ils_run_c(
        base_tour, dist, neighbors,
        iterations=iters, seed=seed,
        init_kick=init_kick,
        or_opt_chain_lengths=_OR_OPT_LENGTHS,
        use_3opt=False,
    )
    return best, compute_tour_cost(best, dist)


def _polish(args) -> tuple[list[int], float]:
    tour, dist = args
    polished, _ = compound_local_search(
        list(tour), dist, or_opt_chain_lengths=_OR_OPT_LENGTHS
    )
    return polished, compute_tour_cost(polished, dist)


def _parallel_native_ils(initial_tour: list[int], dist) -> list[int]:
    n = len(initial_tour)
    chains, iters = _budget(n)
    k = min(_NEIGHBOR_K, n - 1)
    neighbors = build_neighbor_lists(dist, k=k)

    n_workers = min(chains, os.cpu_count() or 4)
    payloads = [
        (
            initial_tour, dist, neighbors, iters,
            _BASE_SEED + s * 1_000_003,
            s != 0,  # chain 0 starts from base; others diversify with a kick
        )
        for s in range(chains)
    ]

    if n_workers <= 1 or n <= 50:
        results = [_run_chain(p) for p in payloads]
    else:
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            results = pool.map(_run_chain, payloads)

    # Polish every chain's best with the full O(n²) compound LS (cheap
    # relative to the ILS itself) then keep the global minimum.
    if n <= 1500:
        polish_payloads = [(t, dist) for t, _ in results]
        if n_workers <= 1 or n <= 50:
            polished = [_polish(p) for p in polish_payloads]
        else:
            with mp.get_context("spawn").Pool(processes=n_workers) as pool:
                polished = pool.map(_polish, polish_payloads)
        results = polished

    overall_best, _ = min(results, key=lambda x: x[1])
    return overall_best
