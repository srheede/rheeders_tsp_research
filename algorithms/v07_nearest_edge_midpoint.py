"""
v07_nearest_edge_midpoint — robust cousin of v06.

v06 picks the hull edge whose perpendicular foot is closest. On long /
narrow hulls (e.g. ``a280``) many interior nodes have feet that fall
*outside* every edge or have ties, so the assignment can be brittle.

v07 instead assigns each interior node to the hull edge whose **midpoint**
is closest. Midpoint distance is a much smoother, less geometric-edge-
case-prone criterion:

    edge*(n) = argmin_{(a,b)}  ‖n − ½(a + b)‖

After assignment, nodes assigned to the same edge are sorted along that
edge by their projection ``t`` and spliced in. The hope is that giving
up the strict ⊥-distance optimality (vs v06) trades a little geometric
purity for substantially better global tour quality on real instances.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms._helpers import distance_to_segment


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
        raise ValueError("v07 requires coordinates.")

    n_hull = len(hull)
    midpoints = np.array([
        0.5 * (coords[hull[i]] + coords[hull[(i + 1) % n_hull]])
        for i in range(n_hull)
    ])

    assignments: list[list[tuple[float, int]]] = [[] for _ in range(n_hull)]

    for node in remaining:
        p = coords[node]
        diffs = midpoints - p
        dists = np.linalg.norm(diffs, axis=1)
        edge_i = int(np.argmin(dists))
        # Sub-order along the chosen edge by projection parameter.
        a = coords[hull[edge_i]]
        b = coords[hull[(edge_i + 1) % n_hull]]
        _, t = distance_to_segment(p, a, b)
        assignments[edge_i].append((t, node))

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
                        f"Midpoint-assignment to edge ({a},{b}); "
                        f"node {node} at t={t:.2f}"
                    ),
                ))
            prev = node
    return tour, steps
