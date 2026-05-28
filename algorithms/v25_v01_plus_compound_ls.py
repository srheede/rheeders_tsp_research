"""
v25_v01_plus_compound_ls — alternate 2-opt and Or-opt to joint convergence.

Each local search has a different neighbourhood:

  * 2-opt fixes edge **crossings** (un-twists the tour) but cannot move
    a node away from a basin of attraction.
  * Or-opt relocates **short chains** but cannot un-cross edges by itself.

Running them sequentially is a classical *Variable Neighbourhood
Descent* (VND): after Or-opt converges, the resulting tour may contain
new crossings that 2-opt can remove; after 2-opt converges, Or-opt may
find new beneficial relocations on the un-crossed tour; etc. We loop
until neither LS produces any further improvement.

Compared to v21 (2-opt only, 4.03%) and v23 (Or-opt only, 2.03%), the
joint VND should match or beat the better of the two on every instance.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms import v01_baseline
from algorithms._helpers import compound_local_search


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v01_baseline.solve_from_hull(hull, remaining, distance_matrix, coords)
    tour, _ = compound_local_search(tour, distance_matrix)
    return tour


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v01_baseline.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    tour, ls_steps = compound_local_search(tour, distance_matrix)
    return tour, insert_steps + ls_steps
