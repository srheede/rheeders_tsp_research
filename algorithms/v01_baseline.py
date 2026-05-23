"""
v01_baseline — cheapest greedy insertion after the shared hull step.

This variant implements only phase 2 of the original Rheeders algorithm:
each remaining node is inserted at the position that minimises

    Δ = d(prev, node) + d(node, next) − d(prev, next)

Hull construction is handled by ``algorithms.convex_hull`` and is identical
for every variant.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    """Insert all remaining nodes using cheapest insertion."""
    tour, _ = _cheapest_insertion(hull, remaining, distance_matrix, trace=False)
    return tour


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    """Cheapest insertion with TraceStep recording."""
    return _cheapest_insertion(hull, remaining, distance_matrix, trace=True)


def _insertion_delta(prev: int, node: int, nxt: int, dist) -> float:
    return dist[prev][node] + dist[node][nxt] - dist[prev][nxt]


def _cheapest_insertion(
    hull: list[int],
    remaining: list[int],
    dist,
    trace: bool,
) -> tuple[list[int], list[TraceStep]]:
    tour = list(hull)
    remaining = list(remaining)
    steps: list[TraceStep] = []

    while remaining:
        best_delta = None
        best_prev = best_node = best_next = None

        n_tour = len(tour)
        for i in range(n_tour):
            prev = tour[i]
            nxt = tour[(i + 1) % n_tour]
            for node in remaining:
                delta = _insertion_delta(prev, node, nxt, dist)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_prev, best_node, best_next = prev, node, nxt

        remaining.remove(best_node)
        idx = tour.index(best_prev)
        tour.insert(idx + 1, best_node)

        if trace:
            steps.append(TraceStep(
                node=best_node,
                inserted_after=best_prev,
                removed_edge=(best_prev, best_next),
                new_edges=[(best_prev, best_node), (best_node, best_next)],
                description=(
                    f"Insert node {best_node} between {best_prev} and {best_next} "
                    f"(Δ={best_delta:.2f})"
                ),
            ))

    return tour, steps
