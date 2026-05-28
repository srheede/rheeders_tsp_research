"""
v24_v03_plus_or_opt — champion construction + champion post-processing.

The benchmark sweep revealed two clear winners:

  * construction-only: v03 regret_insertion (4.31% avg gap)
  * post-processing : v23 v01+Or-opt        (2.03% avg gap)

v24 combines them: run regret-insertion to build a high-quality starting
tour, then apply Or-opt (chain lengths 1, 2, 3) to convergence. The
regret-insertion construction is itself cost-aware so Or-opt has fewer
"obvious" misplacements to fix; the question this variant answers is
whether the marginal gain from the regret start over v01 survives the
Or-opt's flattening effect.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms import v03_regret_insertion
from algorithms._helpers import or_opt


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v03_regret_insertion.solve_from_hull(
        hull, remaining, distance_matrix, coords
    )
    tour, _ = or_opt(tour, distance_matrix)
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
    tour, opt_steps = or_opt(tour, distance_matrix)
    return tour, insert_steps + opt_steps
