"""
v02_furthest_insertion — anchor topology with the most-committed node first.

Classical "cheapest insertion" picks the *cheapest* node to insert at every
step. The intuition behind *furthest* insertion is the opposite: the node
that is currently farthest from the tour will be the most painful to insert
later, so we lock its position in early. Once placed, it "anchors" a piece
of the topology that subsequent (closer) nodes can safely cluster around.

At each step:
  1. node*  = argmax_{n in remaining} min_{t in tour} d(n, t)
  2. Insert node* at the position minimising
        Δ = d(prev, node*) + d(node*, next) − d(prev, next)

This is reported to consistently outperform cheapest insertion on Euclidean
TSP (especially clustered point sets) and is a natural starting point for
a hull-anchored construction.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms._helpers import best_insertion_position, insert_node_at


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour, _ = _furthest_insertion(hull, remaining, distance_matrix, trace=False)
    return tour


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    return _furthest_insertion(hull, remaining, distance_matrix, trace=True)


def _furthest_insertion(
    hull: list[int],
    remaining: list[int],
    dist,
    trace: bool,
) -> tuple[list[int], list[TraceStep]]:
    tour = list(hull)
    remaining = set(remaining)
    steps: list[TraceStep] = []

    # min_dist_to_tour[node] = min distance from `node` to any node in `tour`
    min_dist_to_tour: dict[int, float] = {
        node: min(dist[node][t] for t in tour) for node in remaining
    }

    while remaining:
        # Pick the remaining node farthest from the current tour.
        chosen = max(remaining, key=lambda n: min_dist_to_tour[n])
        idx, delta = best_insertion_position(tour, chosen, dist)
        step = insert_node_at(tour, chosen, idx)
        if trace:
            step.description = (
                f"Furthest-insertion: node {chosen} "
                f"(min-d-to-tour={min_dist_to_tour[chosen]:.2f}) Δ={delta:.2f}"
            )
            steps.append(step)
        remaining.remove(chosen)

        # Update min_dist_to_tour: the new tour node ``chosen`` could be the
        # closest existing-tour node for any of the remaining nodes.
        for n in remaining:
            d = dist[n][chosen]
            if d < min_dist_to_tour[n]:
                min_dist_to_tour[n] = d

    return tour, steps
