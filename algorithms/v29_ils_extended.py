"""
v29_ils_extended — push ILS toward zero gap.

v28 already drove the average gap to 0.51% on 30 iterations. The
instances that still have non-trivial gaps (tsp225 2.76%, eil76 1.12%,
a280 1.05%, ch130 0.77%) point at two limitations:

  1. **Insufficient kick budget** on the larger instances — the search
     space is just too big for 30 random double-bridges.
  2. **LS strength** — Or-opt(1..5) + 2-opt cannot break some 3-opt-
     reachable local minima.

v29 addresses both:

  * **Construction**: regret-insertion (v03) instead of v01 — gives the
    ILS a better starting basin.
  * **Inner LS**: Or-opt(1..5) + 2-opt to convergence (same as v28).
  * **Outer 3-opt finishing pass**: applied *once* after the last
    accepted improvement, before returning. This rescues any 3-opt-
    only improving move that the ILS loop missed.
  * **More iterations**: 60 (double of v28). Kicks are reproducible
    via a fixed RNG seed so results are deterministic.

The expected cost is ~2× v28's runtime (mostly on tsp225 and a280)
in exchange for ~0.3% extra gap reduction.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms import v03_regret_insertion
from algorithms._helpers import (
    compound_local_search,
    or_opt,
    three_opt,
    double_bridge,
    compute_tour_cost,
)


_ITERATIONS = 60
_RNG_SEED = 0xC0FFEE


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v03_regret_insertion.solve_from_hull(
        hull, remaining, distance_matrix, coords
    )
    tour = _full_search(tour, distance_matrix)
    return tour


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v03_regret_insertion.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    tour, opt_steps = _full_search_traced(tour, distance_matrix)
    return tour, insert_steps + opt_steps


def _full_search(tour: list[int], dist) -> list[int]:
    tour, _ = or_opt(tour, dist, chain_lengths=(1, 2, 3, 4, 5))
    tour, _ = compound_local_search(tour, dist)
    tour = _ils_loop(tour, dist)
    tour, _ = three_opt(tour, dist)
    return tour


def _full_search_traced(
    tour: list[int],
    dist,
) -> tuple[list[int], list[TraceStep]]:
    steps: list[TraceStep] = []
    tour, s1 = or_opt(tour, dist, chain_lengths=(1, 2, 3, 4, 5))
    steps.extend(s1)
    tour, s2 = compound_local_search(tour, dist)
    steps.extend(s2)
    tour, s3 = _ils_loop_traced(tour, dist)
    steps.extend(s3)
    tour, s4 = three_opt(tour, dist)
    steps.extend(s4)
    return tour, steps


def _ils_loop(tour: list[int], dist) -> list[int]:
    best_tour = list(tour)
    best_cost = compute_tour_cost(best_tour, dist)
    rng = np.random.default_rng(_RNG_SEED)
    for _ in range(_ITERATIONS):
        kicked = double_bridge(best_tour, rng)
        kicked, _ = or_opt(kicked, dist, chain_lengths=(1, 2, 3, 4, 5))
        kicked, _ = compound_local_search(kicked, dist)
        new_cost = compute_tour_cost(kicked, dist)
        if new_cost < best_cost - 1e-9:
            best_tour = kicked
            best_cost = new_cost
    return best_tour


def _ils_loop_traced(
    tour: list[int],
    dist,
) -> tuple[list[int], list[TraceStep]]:
    best_tour = list(tour)
    best_cost = compute_tour_cost(best_tour, dist)
    rng = np.random.default_rng(_RNG_SEED)
    steps: list[TraceStep] = []
    for it in range(_ITERATIONS):
        kicked = double_bridge(best_tour, rng)
        kicked, _ = or_opt(kicked, dist, chain_lengths=(1, 2, 3, 4, 5))
        kicked, _ = compound_local_search(kicked, dist)
        new_cost = compute_tour_cost(kicked, dist)
        if new_cost < best_cost - 1e-9:
            best_tour = kicked
            best_cost = new_cost
            steps.append(TraceStep(
                node=best_tour[0],
                inserted_after=best_tour[-1],
                removed_edge=None,
                new_edges=[],
                description=(
                    f"ILS iter {it}: kick accepted, cost={new_cost:.2f}"
                ),
            ))
    return best_tour, steps
