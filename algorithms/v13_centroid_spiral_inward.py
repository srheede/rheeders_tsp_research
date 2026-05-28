"""
v13_centroid_spiral_inward — anchor interior structure from the center out.

Sort all interior nodes by their **distance to the hull's centroid**, in
**ascending** order, and insert them one-by-one using cheapest insertion.

Why this order? In Euclidean TSP, the optimal tour tends to weave from
the boundary inward and back out again as it traces the polygon. The
nodes deepest in the cluster (smallest centroid distance) are the
"hardest" — they get sandwiched between many other nodes and rotating
them is expensive. Inserting them *first* locks the central backbone
early, then subsequent (outer) interior nodes can wrap around it.

This is the geometric inverse of v04 (insert farthest-from-hull first).
The two together test which end of the centrality axis matters more.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms._helpers import best_insertion_position, insert_node_at


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
        raise ValueError("v13 requires coordinates.")
    tour = list(hull)
    steps: list[TraceStep] = []
    if not remaining:
        return tour, steps

    centroid = coords[hull].mean(axis=0)
    order = sorted(
        remaining,
        key=lambda n: np.linalg.norm(coords[n] - centroid),
    )
    for node in order:
        idx, delta = best_insertion_position(tour, node, dist)
        step = insert_node_at(tour, node, idx)
        if trace:
            r = float(np.linalg.norm(coords[node] - centroid))
            step.description = (
                f"Centroid-spiral-in: node {node} r={r:.2f} Δ={delta:.2f}"
            )
            steps.append(step)
    return tour, steps
