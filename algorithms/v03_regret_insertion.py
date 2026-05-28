"""
v03_regret_insertion — insert the node we will regret most if we wait.

Cheapest insertion is myopic: it picks the lowest-Δ move available *right
now*. The problem is that a node which currently has one very-cheap
insertion edge and many expensive alternatives is "fragile" — if the
cheap edge later gets consumed by another insertion, we will be forced
to pay one of the expensive ones.

The classic remedy is *regret*-based selection:

    regret(n) = Δ_second_best(n) − Δ_best(n)

A large regret says "if I don't insert ``n`` at its best edge now, the
penalty for waiting will be big". So at each step we pick the node with
the **maximum regret** and place it at its best edge. Nodes whose best
and second-best are similar can be deferred safely.

For Euclidean instances this typically beats cheapest insertion by a
couple of percent in optimality gap.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms._helpers import (
    best_two_insertion_positions,
    best_insertion_position,
    insert_node_at,
)


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour, _ = _regret_insertion(hull, remaining, distance_matrix, trace=False)
    return tour


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    return _regret_insertion(hull, remaining, distance_matrix, trace=True)


def _regret_insertion(
    hull: list[int],
    remaining: list[int],
    dist,
    trace: bool,
) -> tuple[list[int], list[TraceStep]]:
    tour = list(hull)
    remaining = list(remaining)
    steps: list[TraceStep] = []

    while remaining:
        best_node = None
        best_regret = -float("inf")
        best_node_delta = float("inf")
        best_node_idx = 0

        for node in remaining:
            best_d, second_d = best_two_insertion_positions(tour, node, dist)
            # regret = second_best - best (large = urgent)
            regret = second_d - best_d if second_d != float("inf") else 0.0
            if regret > best_regret or (
                regret == best_regret and best_d < best_node_delta
            ):
                best_regret = regret
                best_node = node
                best_node_delta = best_d

        # Re-fetch the actual best position index for the chosen node.
        best_node_idx, _ = best_insertion_position(tour, best_node, dist)
        step = insert_node_at(tour, best_node, best_node_idx)
        if trace:
            step.description = (
                f"Regret-insertion: node {best_node} "
                f"(regret={best_regret:.2f}, Δ={best_node_delta:.2f})"
            )
            steps.append(step)
        remaining.remove(best_node)

    return tour, steps
