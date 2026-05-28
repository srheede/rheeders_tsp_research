"""
v11_convex_layers_onion — peel concentric convex hulls and merge inward.

Repeatedly compute the convex hull of the remaining nodes, remove its
vertices, and continue until no nodes remain. The result is a stack of
concentric layers (the "convex layers" / "onion peeling" of the point
set). Each layer ``L_k`` is itself a closed polygon. We then merge
layers from outer to inner:

  - layer 0 is the project's shared hull tour ``hull`` (already given);
  - merge ``L_1`` into the current tour by finding the **optimal two-
    bridge splice**: pick an edge ``(a, b)`` in the current tour and an
    edge ``(c, d)`` in ``L_1`` to cut, replacing them with two new edges
    so the union becomes a single Hamiltonian cycle of minimum cost;
  - repeat for ``L_2``, ``L_3``, … each time against the latest tour.

Geometric intuition: a convex polygon "wants" to remain a polygon in the
optimal tour as much as possible (because reversing the order along a
convex chain only creates crossings). Merging by a single two-bridge
preserves the polygon shape of each layer almost entirely, sacrificing
only the two cut edges.

Singleton or 2-node residues are handled as special cases (chain splice
or cheapest single insertion).
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
        raise ValueError("v11 requires coordinates.")
    if not remaining:
        return list(hull), []

    layers = _convex_layers(remaining, coords)
    tour = list(hull)
    all_steps: list[TraceStep] = []
    for layer in layers:
        if len(layer) == 1:
            node = layer[0]
            idx, _ = best_insertion_position(tour, node, dist)
            step = insert_node_at(tour, node, idx)
            all_steps.append(step)
        elif len(layer) == 2:
            tour, steps = splice_chain_into_tour(tour, layer, dist)
            all_steps.extend(steps)
        else:
            tour, steps = merge_two_polygons(tour, layer, dist)
            all_steps.extend(steps)
    return tour, all_steps if trace else []


def _convex_layers(nodes: list[int], coords: np.ndarray) -> list[list[int]]:
    """Compute concentric convex layers of ``nodes`` using scipy.

    Each returned layer is a closed-polygon list of node IDs in CCW order.
    For interior residues that are collinear or under 3 points, the layer
    is returned as a 1- or 2-element list (handled specially by callers).
    """
    layers: list[list[int]] = []
    pool = list(nodes)
    while pool:
        if len(pool) >= 3:
            sub_pts = coords[pool]
            try:
                ch = ConvexHull(sub_pts)
                hull_local = list(ch.vertices)
                layer = [pool[i] for i in hull_local]
                # ConvexHull yields CCW for 2D. Keep that order.
                layers.append(layer)
                hull_set = set(layer)
                pool = [n for n in pool if n not in hull_set]
            except QhullError:
                # Degenerate (e.g. all collinear): treat residue as a chain
                # sorted along its principal axis.
                pts = coords[pool]
                proj = (pts - pts.mean(axis=0)) @ _principal_direction(pts)
                order = sorted(range(len(pool)), key=lambda i: proj[i])
                layers.append([pool[i] for i in order])
                pool = []
        elif len(pool) == 2:
            layers.append(list(pool))
            pool = []
        else:  # 1
            layers.append(list(pool))
            pool = []
    return layers


def _principal_direction(pts: np.ndarray) -> np.ndarray:
    """First PCA direction of ``pts``."""
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, -1]  # eigenvector of largest eigenvalue
