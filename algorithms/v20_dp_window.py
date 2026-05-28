"""
v20_dp_window — per-hull-edge Held-Karp DP for the optimal local sub-path.

The previous "rigid assignment" geometric variants (v06, v07, v09, v10)
all suffered because of *how* they ordered nodes inside a hull edge's
bucket — typically by projection ``t``, which is a poor proxy for tour
length. v20 fixes exactly that issue: it does the same nearest-edge
assignment as v07 (closest midpoint) but then runs **exact Held-Karp
dynamic programming** to find the optimal Hamiltonian sub-path through
the bucket with fixed endpoints ``h_i → h_{i+1}``.

Pipeline:

  1. Assign each interior node to its nearest hull edge by midpoint
     distance.
  2. For every hull edge with bucket size > 0:
       * if ≤ 11 nodes → exact DP, optimal sub-path
       * else → drop excess to the global cheapest-insertion fallback
         (rare for our test instances since the average bucket is
         ``≈ remaining / hull_size`` ≈ 5–10 even on ``a280``).
  3. Concatenate sub-paths to form the final tour.

This is the "best you can do without changing the hull-edge
assignment". If it still under-performs cheapest insertion, the lesson
is clear: the assignment to a single hull edge is the limiting factor,
not the within-bucket ordering.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms.v09_wedge_decomposition import _shortest_subpath, _DP_LIMIT


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
        raise ValueError("v20 requires coordinates.")
    n_hull = len(hull)
    if not remaining:
        return list(hull), []

    midpoints = np.array([
        0.5 * (coords[hull[i]] + coords[hull[(i + 1) % n_hull]])
        for i in range(n_hull)
    ])
    assignments: list[list[int]] = [[] for _ in range(n_hull)]
    for node in remaining:
        diffs = midpoints - coords[node]
        d = np.linalg.norm(diffs, axis=1)
        assignments[int(np.argmin(d))].append(node)

    tour: list[int] = []
    steps: list[TraceStep] = []
    for i in range(n_hull):
        a = hull[i]
        b = hull[(i + 1) % n_hull]
        tour.append(a)
        bucket = assignments[i]
        if not bucket:
            continue
        # Sub-path: optimal up to _DP_LIMIT, NN+2-opt otherwise.
        sub_path = _shortest_subpath(a, b, bucket, dist)
        # Drop endpoints — a was just appended, b will be appended next loop.
        for j, node in enumerate(sub_path[1:-1]):
            next_after = sub_path[j + 2]
            tour.append(node)
            if trace:
                steps.append(TraceStep(
                    node=node,
                    inserted_after=sub_path[j],
                    removed_edge=(sub_path[j], next_after) if j > 0 else (a, b),
                    new_edges=[(sub_path[j], node), (node, next_after)],
                    description=(
                        f"DP-window edge ({a},{b}): "
                        f"place node {node} via "
                        f"{'HK' if len(bucket) <= _DP_LIMIT else 'NN+2opt'} sub-path"
                    ),
                ))
    return tour, steps
