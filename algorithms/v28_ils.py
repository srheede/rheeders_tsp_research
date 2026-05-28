"""
v28_ils — Iterated Local Search with double-bridge kicks.

Local search alone (Or-opt + 2-opt, even with 3-opt) gets stuck in a
local minimum. The classical escape mechanism is **Iterated Local
Search**:

  1. Build an initial tour ``T*`` via construction + full LS.
  2. Loop K times:
       a. Perturb ``T*`` with a 4-opt **double-bridge** move that no
          2-opt / 3-opt can reverse in a single step.
       b. Apply LS again to converge to a (possibly different) local
          minimum ``T'``.
       c. If ``cost(T') < cost(T*)`` accept ``T' → T*``.
  3. Return the best tour ever seen.

The double-bridge move splits the tour into 4 segments at 3 random cut
points and reconnects them in the order ``A + C + B + D``. The
resulting tour is significantly different from ``T*`` but preserves the
local structure of most edges, so the next LS round converges quickly
and tends to land in a *different* basin from the one we just escaped.

We use **30 iterations** with a fixed PRNG seed so results are
reproducible. The accept-best strategy is used (no acceptance of
worse tours) since the goal here is solution quality, not exploration
diversity. Local search is the (Or-opt 1..5) + 2-opt loop from v25/v26.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms import v01_baseline
from algorithms._helpers import (
    compound_local_search,
    double_bridge,
    compute_tour_cost,
    or_opt,
)


_ITERATIONS = 30
_RNG_SEED = 0xC0FFEE


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v01_baseline.solve_from_hull(hull, remaining, distance_matrix, coords)
    tour, _ = or_opt(tour, distance_matrix, chain_lengths=(1, 2, 3, 4, 5))
    tour, _ = compound_local_search(tour, distance_matrix)
    return _ils(tour, distance_matrix, trace=False)[0]


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v01_baseline.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    tour, or_steps = or_opt(tour, distance_matrix, chain_lengths=(1, 2, 3, 4, 5))
    tour, cls_steps = compound_local_search(tour, distance_matrix)
    tour, ils_steps = _ils(tour, distance_matrix, trace=True)
    return tour, insert_steps + or_steps + cls_steps + ils_steps


def _ils(
    tour: list[int],
    dist,
    trace: bool,
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
            if trace:
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
