"""
Shared convex hull construction — fixed first step for all algorithm variants.

This implements the Rheeders geometric sweep (phase 1): nodes sorted by y are
walked to build an outer boundary chain. Nodes that cannot be placed on the
chain are deferred to the variant-specific insertion phase.

This module is NOT part of any variant; variants only implement what happens
after the hull and remaining-node set are produced.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import HullStep


def build_hull(coords: np.ndarray) -> tuple[list[int], list[int]]:
    """Build the outer hull chain and return nodes left for insertion.

    Returns
    -------
    hull:
        Ordered 0-based node indices forming the boundary chain (open path).
    remaining:
        Nodes not placed on the hull; passed to the variant insertion phase.
    """
    hull, remaining, _ = _build_hull_with_trace(coords, trace=False)
    return hull, remaining


def build_hull_traced(coords: np.ndarray) -> tuple[list[int], list[int], list[HullStep]]:
    """Like build_hull() but records one step per node added to the hull."""
    return _build_hull_with_trace(coords, trace=True)


def _build_hull_with_trace(
    coords: np.ndarray,
    trace: bool,
) -> tuple[list[int], list[int], list[HullStep]]:
    n = len(coords)
    sorted_by_y = sorted(range(n), key=lambda i: coords[i][1])

    hull: list[int] = []
    rest: list[int] = []
    steps: list[HullStep] = []
    left_limit = right_limit = 0.0

    def record_step(
        hull_before: list[int],
        node: int,
        action: str,
        description: str,
    ) -> None:
        if not trace:
            return
        removed, added = _edge_change_for_addition(hull_before, node, action)
        steps.append(HullStep(
            node=node,
            action=action,
            removed_edge=removed,
            new_edges=added,
            description=description,
        ))

    # First pass: y-sorted sweep
    for node in sorted_by_y:
        x = coords[node][0]
        if not hull:
            hull.append(node)
            left_limit = right_limit = x
            record_step([], node, "init", f"Start hull with node {node} (lowest y)")
        elif x <= left_limit:
            before = list(hull)
            hull.insert(0, node)
            left_limit = x
            record_step(before, node, "prepend", f"Add node {node} to front of hull")
        elif x >= right_limit:
            before = list(hull)
            hull.append(node)
            right_limit = x
            record_step(before, node, "append", f"Add node {node} to back of hull")
        else:
            rest.append(node)

    # Second pass: absorb rest nodes sorted by x
    rest_sorted = sorted(rest, key=lambda i: coords[i][0])
    remaining: list[int] = []
    rest2: list[int] = []
    limit_y = coords[hull[0]][1]

    for node in rest_sorted:
        if coords[node][1] >= limit_y:
            before = list(hull)
            hull.insert(0, node)
            limit_y = coords[node][1]
            left_limit = coords[node][0]
            record_step(before, node, "prepend", f"Add rest node {node} to front of hull")
        else:
            rest2.insert(0, node)

    limit_y = coords[hull[-1]][1]
    for node in rest2:
        if coords[node][1] >= limit_y and left_limit <= coords[node][0]:
            before = list(hull)
            hull.append(node)
            limit_y = coords[node][1]
            record_step(before, node, "append", f"Add rest node {node} to back of hull")
        else:
            remaining.append(node)

    return hull, remaining, steps


def _edge_change_for_addition(
    hull_before: list[int],
    node: int,
    action: str,
) -> tuple[tuple[int, int] | None, list[tuple[int, int]]]:
    """Compute the removed edge and new edges when adding ``node`` to the hull.

    The hull is an open path during construction, so each step only adds the
    single new link between ``node`` and the adjacent endpoint it joins.
    """
    if not hull_before:
        return None, []

    if action == "prepend":
        return None, [(node, hull_before[0])]

    if action == "append":
        return None, [(hull_before[-1], node)]

    return None, []
