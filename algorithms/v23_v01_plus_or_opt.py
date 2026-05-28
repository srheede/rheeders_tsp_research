"""
v23_v01_plus_or_opt — baseline insertion + Or-opt segment relocation.

Or-opt is a restricted form of 3-opt: it considers moving a small chain
of consecutive tour nodes (length 1, 2, or 3) to a different position in
the tour, with optional reversal. It is *significantly* cheaper than full
3-opt and is often a better post-processing partner for insertion
heuristics than 2-opt because:

  * 2-opt only un-crosses edges; it cannot relocate a misplaced node
    cluster.
  * Or-opt can shift a tight cluster ("oops, I inserted this loop on
    the wrong side") to its natural neighbourhood.

Pairing v01 with Or-opt lets us isolate this complementary signal.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms import v01_baseline
from algorithms._helpers import or_opt


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v01_baseline.solve_from_hull(hull, remaining, distance_matrix, coords)
    tour, _ = or_opt(tour, distance_matrix)
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
    tour, opt_steps = or_opt(tour, distance_matrix)
    return tour, insert_steps + opt_steps
