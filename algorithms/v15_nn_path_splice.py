"""
v15_nn_path_splice — nearest-neighbour chain through interior, optimal hull splice.

Sister algorithm to v14, but a different way of building the interior
chain:

  1. Pick the interior node closest to the hull as the *start*.
  2. Build a Hamiltonian path through interior nodes using **nearest-
     neighbour expansion**: at each step go to the nearest unvisited
     interior node.
  3. Splice the path into the hull by enumerating all hull edges as
     potential cut points and choosing the (edge, orientation) that
     minimises tour length.

NN paths are often more "snake-like" than MST DFS shortcuts (which can
have long jumps when the MST has branching). On clustered instances
that often produces a cleaner chain to splice.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms._helpers import (
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
        idx, _ = best_insertion_position(hull, remaining[0], dist)
        tour = list(hull)
        step = insert_node_at(tour, remaining[0], idx)
        return tour, [step] if trace else []

    # Build NN path starting from the interior node closest to the hull.
    start = min(remaining, key=lambda n: min(dist[n][h] for h in hull))
    chain = _nn_path(start, remaining, dist)
    return splice_chain_into_tour(hull, chain, dist)


def _nn_path(start: int, nodes: list[int], dist) -> list[int]:
    pool = set(nodes)
    pool.remove(start)
    path = [start]
    while pool:
        last = path[-1]
        nxt = min(pool, key=lambda n: dist[last][n])
        pool.remove(nxt)
        path.append(nxt)
    return path
