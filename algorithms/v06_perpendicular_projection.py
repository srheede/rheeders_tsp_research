"""
v06_perpendicular_projection — project each interior node onto the hull.

Every interior node is assigned to exactly one hull edge by the rule:

    edge*(n) = argmin_{(a,b) ∈ hull edges, foot(n,a,b) ∈ [a,b]}  ⊥-distance(n, ab)

In words: of all the hull edges whose perpendicular foot from ``n`` lies
inside the segment, pick the closest one. Nodes whose foot lands outside
every hull edge fall back to the edge whose nearer endpoint they are
closest to (clamped projection).

Once every interior node has its target hull edge, each edge ``(a, b)`` is
replaced by the mini-path ``a → n_1 → n_2 → … → n_k → b`` where
``n_1..n_k`` are the assigned nodes sorted by their projection parameter
``t ∈ [0, 1]`` along ``ab``. This produces a *coherent local ordering*
that any greedy insertion can only approximate.

The trace records one TraceStep per node in left-to-right commit order.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms._helpers import (
    perpendicular_distance_to_segment,
    distance_to_segment,
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
        raise ValueError("v06 requires coordinates.")

    n_hull = len(hull)
    # assignments[edge_index] = list of (t_param, node)
    assignments: list[list[tuple[float, int]]] = [[] for _ in range(n_hull)]

    for node in remaining:
        p = coords[node]
        best_edge = -1
        best_dist = float("inf")
        best_t = 0.0
        # First pass: only consider edges whose perpendicular foot is inside the segment.
        for i in range(n_hull):
            a = coords[hull[i]]
            b = coords[hull[(i + 1) % n_hull]]
            d, t = perpendicular_distance_to_segment(p, a, b)
            if 0.0 <= t <= 1.0 and d < best_dist:
                best_dist = d
                best_edge = i
                best_t = t
        # Fallback: no edge admits an in-segment foot — use clamped distance.
        if best_edge < 0:
            best_dist = float("inf")
            for i in range(n_hull):
                a = coords[hull[i]]
                b = coords[hull[(i + 1) % n_hull]]
                d, t = distance_to_segment(p, a, b)
                if d < best_dist:
                    best_dist = d
                    best_edge = i
                    best_t = t
        assignments[best_edge].append((best_t, node))

    # Sort each edge's bucket along the edge direction.
    for bucket in assignments:
        bucket.sort(key=lambda x: x[0])

    tour: list[int] = []
    steps: list[TraceStep] = []
    for i in range(n_hull):
        a = hull[i]
        b = hull[(i + 1) % n_hull]
        tour.append(a)
        prev = a
        for j, (t, node) in enumerate(assignments[i]):
            next_after = (
                assignments[i][j + 1][1] if j + 1 < len(assignments[i]) else b
            )
            tour.append(node)
            if trace:
                steps.append(TraceStep(
                    node=node,
                    inserted_after=prev,
                    removed_edge=(prev, next_after) if j > 0 else (a, b),
                    new_edges=[(prev, node), (node, next_after)],
                    description=(
                        f"Project onto edge ({a},{b}): node {node} at t={t:.2f}"
                    ),
                ))
            prev = node
    return tour, steps
