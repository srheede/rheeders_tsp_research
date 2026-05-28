"""
v12_recursive_hull_tsp — solve the interior as a nested hull problem.

The project's premise is that "convex hull is the right first step." If
true, the same recipe should apply to the interior: take the hull of the
interior, solve the residue inside *that* hull, etc., until the deepest
residue is small enough to solve exactly.

Algorithm:

  1. The shared outer hull is already given.
  2. Compute the **convex hull of the interior nodes**. That gives an
     inner closed polygon plus a (smaller) interior residue.
  3. Recursively apply the same procedure to the residue until the
     residue has ≤ 3 nodes (trivially solvable).
  4. Now collapse the recursion outward: starting from the innermost
     polygon, repeatedly merge each layer's tour into the next outer
     layer's tour with the optimal two-bridge splice (same merger as
     v11).

Difference vs v11 (onion peeling): v11 peels *all* hulls upfront without
solving the residue first; v12 ensures each residue is itself an optimal
nested-hull tour before merging.

For small instances this often produces tighter tours because each layer
is internally arranged before being committed to the polygon shape.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from algorithms.protocol import TraceStep
from algorithms._helpers import (
    merge_two_polygons,
    splice_chain_into_tour,
    best_insertion_position,
    insert_node_at,
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
        raise ValueError("v12 requires coordinates.")
    if not remaining:
        return list(hull), []

    inner_tour, _is_closed = _recursive_tour(remaining, coords, dist)

    if _is_closed:
        new_tour, steps = merge_two_polygons(hull, inner_tour, dist)
    elif len(inner_tour) == 1:
        idx, _ = best_insertion_position(hull, inner_tour[0], dist)
        new_tour = list(hull)
        step = insert_node_at(new_tour, inner_tour[0], idx)
        steps = [step]
    else:
        new_tour, steps = splice_chain_into_tour(hull, inner_tour, dist)
    return new_tour, steps if trace else []


def _recursive_tour(
    nodes: list[int],
    coords: np.ndarray,
    dist,
) -> tuple[list[int], bool]:
    """Build a tour over ``nodes`` using the nested-hull recipe.

    Returns (tour, is_closed). ``is_closed=True`` means the tour is a
    closed cycle (≥ 3 nodes), ``False`` means it is an open chain
    (1 or 2 nodes) and the caller should splice it accordingly.
    """
    n = len(nodes)
    if n == 0:
        return [], False
    if n == 1:
        return list(nodes), False
    if n == 2:
        return list(nodes), False
    if n == 3:
        return list(nodes), True

    # Compute the hull of these nodes.
    pts = coords[nodes]
    try:
        ch = ConvexHull(pts)
        hull_local = list(ch.vertices)
        hull_nodes = [nodes[i] for i in hull_local]
    except QhullError:
        # All collinear: treat as a chain along principal axis.
        proj = (pts - pts.mean(axis=0)) @ _principal_direction(pts)
        order = sorted(range(n), key=lambda i: proj[i])
        return [nodes[i] for i in order], False

    hull_set = set(hull_nodes)
    interior_nodes = [n_id for n_id in nodes if n_id not in hull_set]

    if not interior_nodes:
        return hull_nodes, True

    inner_tour, inner_closed = _recursive_tour(interior_nodes, coords, dist)
    if inner_closed:
        merged, _ = merge_two_polygons(hull_nodes, inner_tour, dist)
        return merged, True
    elif len(inner_tour) == 1:
        idx, _ = best_insertion_position(hull_nodes, inner_tour[0], dist)
        merged = list(hull_nodes)
        merged.insert(idx + 1, inner_tour[0])
        return merged, True
    else:
        merged, _ = splice_chain_into_tour(hull_nodes, inner_tour, dist)
        return merged, True


def _principal_direction(pts: np.ndarray) -> np.ndarray:
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, -1]
