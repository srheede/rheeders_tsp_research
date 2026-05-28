"""
v19_delaunay_bfs — geometric-locality-guided insertion order.

The Delaunay triangulation of a 2-D point set captures **geometric
locality**: two points share a Delaunay edge iff there is some empty
circle that passes through both. In an optimal Euclidean TSP tour, each
node tends to be adjacent to one of its Delaunay neighbours — so
Delaunay neighbours are excellent insertion-position hints.

Algorithm:

  1. Compute the Delaunay triangulation of *all* nodes.
  2. Do a BFS through Delaunay edges starting from the hull-vertex
     "level 0". This produces an insertion order where each newly-
     visited interior node has at least one already-placed Delaunay
     parent in the tour.
  3. For each insertion: among the parent's tour neighbours and the
     parent itself, score a small set of *candidate* tour-insertion
     positions; otherwise (no useful parent info, e.g. isolated branch)
     fall back to global cheapest-insertion.

The hope is that we get cheapest-insertion-quality decisions for ~95%
of placements but with insertion candidates restricted to a constant-
size local neighbourhood — much faster in the heart of the algorithm,
and crucially *cluster-aware* in the way pure greedy isn't.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial import Delaunay, QhullError

from algorithms.protocol import TraceStep
from algorithms._helpers import best_insertion_position, insert_node_at


# Window of tour-positions around a parent's index to consider as candidates.
_LOCAL_WINDOW = 5


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
        raise ValueError("v19 requires coordinates.")
    tour = list(hull)
    steps: list[TraceStep] = []
    if not remaining:
        return tour, steps

    n = len(coords)
    try:
        tri = Delaunay(coords)
    except QhullError:
        # Degenerate point set — fall back to plain cheapest insertion.
        return _fallback_cheapest(tour, remaining, dist, trace)

    adj: list[set[int]] = [set() for _ in range(n)]
    for simplex in tri.simplices:
        a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)

    placed = set(hull)
    parent_of: dict[int, int] = {}
    queue = deque()
    for h in hull:
        for nb in adj[h]:
            if nb not in placed and nb in set(remaining):
                if nb not in parent_of:
                    parent_of[nb] = h
                    queue.append(nb)

    insert_order: list[int] = []
    remaining_set = set(remaining)
    while queue:
        node = queue.popleft()
        if node not in remaining_set:
            continue
        insert_order.append(node)
        remaining_set.remove(node)
        for nb in adj[node]:
            if nb in remaining_set and nb not in parent_of:
                parent_of[nb] = node
                queue.append(nb)
    # Any nodes not reached by Delaunay BFS (shouldn't happen if connected) —
    # append in arbitrary order.
    insert_order.extend(remaining_set)

    # Perform insertions guided by the parent's tour position.
    pos_of: dict[int, int] = {node: i for i, node in enumerate(tour)}

    for node in insert_order:
        parent = parent_of.get(node)
        candidate_edges = _candidate_edges_around(tour, pos_of, parent)
        best_delta = float("inf")
        best_edge = 0
        for edge_i in candidate_edges:
            a = tour[edge_i]
            b = tour[(edge_i + 1) % len(tour)]
            d = dist[a][node] + dist[node][b] - dist[a][b]
            if d < best_delta:
                best_delta = d
                best_edge = edge_i
        # If the parent-local window already includes a low-Δ position
        # we use it; otherwise fall back to the global cheapest position.
        global_idx, global_delta = best_insertion_position(tour, node, dist)
        if global_delta < best_delta - 1e-9:
            best_edge = global_idx
            best_delta = global_delta
        step = insert_node_at(tour, node, best_edge)
        if trace:
            step.description = (
                f"Delaunay-BFS: insert {node} "
                f"(parent={parent} Δ={best_delta:.2f})"
            )
            steps.append(step)
        # Update tour-position lookup.
        pos_of = {n_id: i for i, n_id in enumerate(tour)}

    return tour, steps


def _candidate_edges_around(
    tour: list[int],
    pos_of: dict[int, int],
    parent: int | None,
) -> list[int]:
    if parent is None or parent not in pos_of:
        return list(range(len(tour)))  # no info — full search
    n = len(tour)
    p = pos_of[parent]
    edges = set()
    for off in range(-_LOCAL_WINDOW, _LOCAL_WINDOW + 1):
        edges.add((p + off) % n)
    return list(edges)


def _fallback_cheapest(
    tour: list[int],
    remaining: list[int],
    dist,
    trace: bool,
) -> tuple[list[int], list[TraceStep]]:
    steps: list[TraceStep] = []
    rem = list(remaining)
    while rem:
        best_delta = float("inf")
        best_node = rem[0]
        best_edge = 0
        for node in rem:
            idx, d = best_insertion_position(tour, node, dist)
            if d < best_delta:
                best_delta = d
                best_node = node
                best_edge = idx
        step = insert_node_at(tour, best_node, best_edge)
        if trace:
            steps.append(step)
        rem.remove(best_node)
    return tour, steps
