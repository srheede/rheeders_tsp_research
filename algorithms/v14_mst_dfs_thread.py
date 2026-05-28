"""
v14_mst_dfs_thread — build a chain through interior, then splice into hull.

Phase 2:

  1. Compute the **minimum spanning tree** of the *interior* nodes only,
     using the existing Euclidean distance matrix.
  2. Pick a root (the interior node closest to the hull) and DFS the tree
     in pre-order. The DFS visit order is taken as a Hamiltonian path on
     the interior — this is the classical Christofides shortcut but
     restricted to interior nodes (because the hull already provides a
     known cycle around them).
  3. Try **all** hull edges as a potential splice point: replacing
     ``(a, b)`` by ``a → chain → b`` (or the reversed chain). Commit the
     splice that minimises tour length.

Key insight from the hull-anchored premise: a single optimal two-bridge
between the hull and an interior path is *much* better than committing
each interior node independently, because the path's internal cost is
already minimised by the MST shortcut and the only free parameter left
is the splice location.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms._helpers import best_chain_insertion, splice_chain_into_tour


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour, _ = _solve(hull, remaining, distance_matrix, trace=False)
    return tour


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    return _solve(hull, remaining, distance_matrix, trace=True)


def _solve(
    hull: list[int],
    remaining: list[int],
    dist,
    trace: bool,
) -> tuple[list[int], list[TraceStep]]:
    if not remaining:
        return list(hull), []
    if len(remaining) == 1:
        from algorithms._helpers import best_insertion_position, insert_node_at
        idx, _ = best_insertion_position(hull, remaining[0], dist)
        tour = list(hull)
        step = insert_node_at(tour, remaining[0], idx)
        return tour, [step] if trace else []

    # 1. MST of remaining nodes via Prim's algorithm.
    mst_adj = _prim_mst(remaining, dist)

    # 2. Pick root: interior node closest to ANY hull node.
    root = min(
        remaining,
        key=lambda n: min(dist[n][h] for h in hull),
    )

    # 3. DFS pre-order = Hamiltonian path candidate.
    chain = _dfs_preorder(root, mst_adj)

    # 4. Splice chain into hull at the cheapest position.
    new_tour, steps = splice_chain_into_tour(hull, chain, dist)
    return new_tour, steps if trace else []


def _prim_mst(nodes: list[int], dist) -> dict[int, list[int]]:
    """Prim's MST restricted to ``nodes``. Returns adjacency dict."""
    if not nodes:
        return {}
    in_tree = {nodes[0]}
    out = set(nodes[1:])
    adj: dict[int, list[int]] = {n: [] for n in nodes}
    # Cheapest connection from in_tree for each out-node:
    best_edge: dict[int, tuple[float, int]] = {
        n: (dist[nodes[0]][n], nodes[0]) for n in out
    }
    while out:
        # Pick the out-node with the cheapest connection.
        next_node = min(out, key=lambda n: best_edge[n][0])
        cost, parent = best_edge[next_node]
        adj[next_node].append(parent)
        adj[parent].append(next_node)
        out.remove(next_node)
        in_tree.add(next_node)
        # Update best_edge for remaining out-nodes.
        for n in out:
            d = dist[next_node][n]
            if d < best_edge[n][0]:
                best_edge[n] = (d, next_node)
    return adj


def _dfs_preorder(root: int, adj: dict[int, list[int]]) -> list[int]:
    """Iterative DFS pre-order, sorting children by distance ascending.

    The sorting heuristic helps shortcut quality because a child closer
    to its parent is more likely to be closer in the resulting chain.
    """
    visited = set()
    order: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        # Push children in reverse so the closest one is processed next.
        children = [c for c in adj[node] if c not in visited]
        # Stable order independent of input shuffling.
        children.sort(reverse=True)
        for c in children:
            stack.append(c)
    return order
