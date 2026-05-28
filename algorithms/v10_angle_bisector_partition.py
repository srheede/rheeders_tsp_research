"""
v10_angle_bisector_partition — local hull-vertex bisectors decide ownership.

Where v09 partitions the interior by rays from the global *centroid*, v10
partitions it by the **interior angle bisector** at each hull vertex. The
hull is a convex polygon: at every vertex ``h_i`` the bisector of the two
adjacent hull edges points into the interior and divides the local
neighbourhood into two cones — one "belongs to edge ``(h_{i-1}, h_i)``"
and the other "belongs to edge ``(h_i, h_{i+1})``".

A node ``n`` is assigned to edge ``(h_i, h_{i+1})`` iff:

  * from ``h_i``'s perspective ``n`` is on the side of bisector(``h_i``)
    that contains the direction toward ``h_{i+1}``;
  * from ``h_{i+1}``'s perspective ``n`` is on the side of
    bisector(``h_{i+1}``) that contains the direction toward ``h_i``.

This produces a partition that is *locally hull-aware* (the centroid is
not used). In flat or elongated hulls — where centroid-wedges of v09
degenerate — this often gives a better assignment.

Inside each edge's bucket we again build an optimal sub-path with fixed
endpoints (Held-Karp DP up to 11 nodes, else NN + 2-opt).
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms.v09_wedge_decomposition import _shortest_subpath


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
        raise ValueError("v10 requires coordinates.")
    if not remaining:
        return list(hull), []

    n_hull = len(hull)

    # For each hull vertex h_i, compute the inward-pointing bisector direction.
    # bisector at h_i = -(unit(h_i→h_{i-1}) + unit(h_i→h_{i+1}))/2  (inward).
    bisector = np.zeros((n_hull, 2))
    for i in range(n_hull):
        h = coords[hull[i]]
        prev_v = coords[hull[(i - 1) % n_hull]] - h
        next_v = coords[hull[(i + 1) % n_hull]] - h
        prev_u = prev_v / (np.linalg.norm(prev_v) + 1e-12)
        next_u = next_v / (np.linalg.norm(next_v) + 1e-12)
        # Inward bisector points opposite to the average of the outward edges.
        # On a convex hull the "interior bisector" is the +average of edge unit
        # vectors when those edges are oriented away from h.
        b = prev_u + next_u
        if np.linalg.norm(b) < 1e-9:
            # Degenerate: collinear neighbours — pick the perpendicular pointing
            # toward the polygon's centroid.
            centroid = coords[hull].mean(axis=0)
            inward = centroid - h
            b = inward / (np.linalg.norm(inward) + 1e-12)
        else:
            b /= np.linalg.norm(b)
        bisector[i] = b

    assignments: list[list[tuple[float, int]]] = [[] for _ in range(n_hull)]

    for node in remaining:
        p = coords[node]
        # For each edge (h_i, h_{i+1}), test the two bisector half-plane
        # conditions. We expect one or two candidates; tie-break by the
        # closer hull-edge midpoint.
        candidates = []
        for i in range(n_hull):
            j = (i + 1) % n_hull
            hi = coords[hull[i]]
            hj = coords[hull[j]]
            # Condition 1 at h_i: vector (n - h_i) and direction (h_j - h_i)
            # should be on the same side of bisector[i].
            ok_i = _same_side_of_bisector(
                p - hi, hj - hi, bisector[i]
            )
            ok_j = _same_side_of_bisector(
                p - hj, hi - hj, bisector[j]
            )
            if ok_i and ok_j:
                # Sub-order: projection along edge.
                ab = hj - hi
                ab_sq = float(ab @ ab)
                t = float((p - hi) @ ab / ab_sq) if ab_sq > 0 else 0.0
                mid = 0.5 * (hi + hj)
                candidates.append((float(np.linalg.norm(p - mid)), i, t))
        if not candidates:
            # Numerical edge case (e.g. node on the hull boundary or
            # collinear): fall back to nearest midpoint.
            mids = np.array([
                0.5 * (coords[hull[i]] + coords[hull[(i + 1) % n_hull]])
                for i in range(n_hull)
            ])
            i = int(np.argmin(np.linalg.norm(mids - p, axis=1)))
            ab = coords[hull[(i + 1) % n_hull]] - coords[hull[i]]
            ab_sq = float(ab @ ab)
            t = float((p - coords[hull[i]]) @ ab / ab_sq) if ab_sq > 0 else 0.0
            assignments[i].append((t, node))
        else:
            candidates.sort(key=lambda x: x[0])
            _, edge_i, t = candidates[0]
            assignments[edge_i].append((t, node))

    tour: list[int] = []
    steps: list[TraceStep] = []
    for i in range(n_hull):
        a = hull[i]
        b = hull[(i + 1) % n_hull]
        tour.append(a)
        bucket = sorted(assignments[i], key=lambda x: x[0])
        if not bucket:
            continue
        mids = [node for _, node in bucket]
        sub_path = _shortest_subpath(a, b, mids, dist)
        prev = a
        for j, node in enumerate(sub_path[1:-1]):
            next_after = sub_path[j + 2]
            tour.append(node)
            if trace:
                steps.append(TraceStep(
                    node=node,
                    inserted_after=prev,
                    removed_edge=(prev, next_after) if j > 0 else (a, b),
                    new_edges=[(prev, node), (node, next_after)],
                    description=(
                        f"Bisector-assignment edge ({a},{b}): place {node}"
                    ),
                ))
            prev = node
    return tour, steps


def _same_side_of_bisector(
    test_vec: np.ndarray,
    reference_vec: np.ndarray,
    bisector_vec: np.ndarray,
) -> bool:
    """Check whether ``test_vec`` and ``reference_vec`` lie on the same side of
    the line through the origin perpendicular to ``bisector_vec``.

    Equivalent to: signed projection of test_vec onto perpendicular(bisector_vec)
    has the same sign as that of reference_vec.
    """
    perp = np.array([-bisector_vec[1], bisector_vec[0]])
    s_test = float(test_vec @ perp)
    s_ref = float(reference_vec @ perp)
    return s_test * s_ref >= 0.0
