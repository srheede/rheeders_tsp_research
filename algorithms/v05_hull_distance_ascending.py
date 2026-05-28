"""
v05_hull_distance_ascending — insert "shallow" interior nodes first.

Sister algorithm of v04. The interior node with the smallest minimum
perpendicular distance to any hull edge is the *closest to the hull
boundary*. Those nodes are the most natural extensions of the hull and
should pop into place very cheaply — almost like extending the hull
chain itself. Once they are in, the "effective hull" shrinks and the
next-closest layer is inserted, and so on.

Why test ascending vs v04's descending? On hulls with deeply concave
interiors (e.g. ``a280``), descending may force long jumps for the
first inserted nodes; on tightly clustered instances ascending may
pile too many nodes onto the same hull edge before the structure
becomes clear.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms._helpers import (
    best_insertion_position,
    insert_node_at,
    min_distance_to_hull,
)


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour, _ = _solve(hull, remaining, distance_matrix, coords, trace=False)
    return tour


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    return _solve(hull, remaining, distance_matrix, coords, trace=True)


def _solve(
    hull: list[int],
    remaining: list[int],
    dist,
    coords: np.ndarray,
    trace: bool,
) -> tuple[list[int], list[TraceStep]]:
    if coords is None:
        raise ValueError("v05 requires coordinates.")
    tour = list(hull)
    steps: list[TraceStep] = []
    if not remaining:
        return tour, steps

    depths = {n: min_distance_to_hull(coords, hull, n) for n in remaining}
    order = sorted(remaining, key=lambda n: depths[n])

    for node in order:
        idx, delta = best_insertion_position(tour, node, dist)
        step = insert_node_at(tour, node, idx)
        if trace:
            step.description = (
                f"Hull-depth asc: node {node} depth={depths[node]:.2f} Δ={delta:.2f}"
            )
            steps.append(step)
    return tour, steps
