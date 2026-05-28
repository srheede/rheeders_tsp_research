"""
v27_v01_plus_3opt — Or-opt+2-opt local minimum + sequential 3-opt.

After Or-opt and 2-opt converge there are still tours with no short
improving move but a beneficial 3-edge swap. The classic example is a
"zig-zag" where two adjacent triangles each look optimal locally but
swapping a 3-cycle of edges removes a long diagonal.

v27 layers a first-improvement sequential 3-opt on top of v26's
compound (Or-opt(1..5) + 2-opt) local minimum. Each pass considers
every triple of non-adjacent edges and accepts the first reconnection
(out of 7 non-trivial cases) that reduces tour length; the loop
continues until a full pass finds no improvement.

3-opt is O(n³) per pass so this is the most expensive variant so far —
on ``a280`` we expect tens of seconds.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms import v01_baseline
from algorithms._helpers import compound_local_search, three_opt, or_opt


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v01_baseline.solve_from_hull(hull, remaining, distance_matrix, coords)
    tour, _ = or_opt(tour, distance_matrix, chain_lengths=(1, 2, 3, 4, 5))
    tour, _ = compound_local_search(tour, distance_matrix)
    tour, _ = three_opt(tour, distance_matrix)
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
    tour, or_steps = or_opt(tour, distance_matrix, chain_lengths=(1, 2, 3, 4, 5))
    tour, cls_steps = compound_local_search(tour, distance_matrix)
    tour, three_steps = three_opt(tour, distance_matrix)
    return tour, insert_steps + or_steps + cls_steps + three_steps
