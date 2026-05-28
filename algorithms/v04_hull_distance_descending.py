"""
v04_hull_distance_descending — insert "deep" interior nodes first.

Compute, for every interior node, its minimum perpendicular distance to
any hull edge — i.e. how far inside the hull it lies. Sort interior
nodes by this distance **descending**: deepest first, shallowest last.
Each node is inserted into the current tour at the cheapest position
(same Δ-criterion as v01).

Why this is interesting in a hull-anchored project: the hull is fixed,
so interior nodes far from the hull boundary have the *fewest natural
neighbours* among the hull. Inserting them first forces the tour to
"reach into" the centre early, fixing the most awkward placements
before the easy ones consume the cheap edges.

Compare to v05 (ascending) — together they bracket the question:
"should hard cases be placed first or last?"
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
        raise ValueError("v04 requires coordinates.")
    tour = list(hull)
    steps: list[TraceStep] = []
    if not remaining:
        return tour, steps

    depths = {n: min_distance_to_hull(coords, hull, n) for n in remaining}
    order = sorted(remaining, key=lambda n: -depths[n])

    for node in order:
        idx, delta = best_insertion_position(tour, node, dist)
        step = insert_node_at(tour, node, idx)
        if trace:
            step.description = (
                f"Hull-depth desc: node {node} depth={depths[node]:.2f} Δ={delta:.2f}"
            )
            steps.append(step)
    return tour, steps
