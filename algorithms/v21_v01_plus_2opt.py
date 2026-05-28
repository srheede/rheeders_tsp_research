"""
v21_v01_plus_2opt — measure the ceiling of greedy construction + classic 2-opt.

This variant is a *post-processing* variant: it runs the v01 baseline
cheapest-insertion construction *unchanged* and then applies 2-opt until
no improving swap remains.

2-opt repeatedly examines pairs of non-adjacent edges (a-b) and (c-d) in
the current tour and replaces them with (a-c) and (b-d) (reversing the
segment between them) whenever doing so shortens the tour. It cannot
cross edges so the resulting tour stays valid, and on Euclidean
instances it removes all *self-intersections*.

Why this is interesting in the hull-anchored project: by isolating the
gain attributable purely to 2-opt on top of v01, we get a calibration
benchmark — any future hull-aware construction whose pre-2-opt cost is
already below this is worth keeping.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms import v01_baseline
from algorithms._helpers import two_opt


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v01_baseline.solve_from_hull(hull, remaining, distance_matrix, coords)
    tour, _ = two_opt(tour, distance_matrix)
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
    tour, opt_steps = two_opt(tour, distance_matrix)
    return tour, insert_steps + opt_steps
