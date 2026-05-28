"""
v17_lookahead_k_cheapest — beam-search cheapest insertion.

Cheapest insertion (v01) commits to the cheapest single Δ at every
step. Sometimes the *real* cheapest two-step move is a slightly more
expensive Δ-now that opens a much cheaper Δ-next. v17 looks ahead.

Algorithm (one commit per loop iteration):

  1. Compute the best single insertion (Δ_best, node*, edge*) for every
     remaining node — this is one ``best_per_node`` pass.
  2. Take the top ``K1`` candidate (node, edge) pairs with the lowest
     Δ_best. For each one ``(n_a, e_a)``:
        a. Hypothetically apply the insertion to a *copy* of the tour.
        b. Compute the lowest Δ for any *other* remaining node in the
           new tour — call this ``Δ_next``.
     The total score for ``(n_a, e_a)`` is ``Δ_best(n_a) + Δ_next``.
  3. Commit only the candidate with the lowest score (``n_a``, ``e_a``).

This is essentially **beam-search width K1 with depth 2** instead of
greedy depth 1. ``K1`` is chosen to keep the per-step cost reasonable
on the project's larger instances (a280 has up to 270 interior nodes).
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms._helpers import insert_node_at


_BEAM_WIDTH = 12  # number of (node, edge) candidates to roll out


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
    tour = list(hull)
    remaining = list(remaining)
    steps: list[TraceStep] = []

    while remaining:
        candidates = _top_candidates(tour, remaining, dist, k=_BEAM_WIDTH)
        if len(remaining) == 1:
            # No depth-2 to evaluate — just commit the best move.
            node, edge_idx, delta_now = candidates[0]
            step = insert_node_at(tour, node, edge_idx)
            if trace:
                step.description = (
                    f"Lookahead final: insert {node} Δ={delta_now:.2f}"
                )
                steps.append(step)
            remaining.remove(node)
            continue

        best_score = float("inf")
        best_node = None
        best_edge = 0
        best_delta_now = 0.0
        for node_a, edge_a, delta_a in candidates:
            # Hypothetical tour after committing (node_a, edge_a).
            hyp_tour = list(tour)
            hyp_tour.insert(edge_a + 1, node_a)
            # Best Δ for any other remaining node in this hyp_tour.
            best_next = float("inf")
            for node_b in remaining:
                if node_b == node_a:
                    continue
                d_b = _best_delta(hyp_tour, node_b, dist)
                if d_b < best_next:
                    best_next = d_b
            score = delta_a + best_next
            if score < best_score:
                best_score = score
                best_node = node_a
                best_edge = edge_a
                best_delta_now = delta_a

        step = insert_node_at(tour, best_node, best_edge)
        if trace:
            step.description = (
                f"Lookahead-2: insert {best_node} "
                f"Δ_now={best_delta_now:.2f} score={best_score:.2f}"
            )
            steps.append(step)
        remaining.remove(best_node)

    return tour, steps


def _top_candidates(
    tour: list[int],
    remaining: list[int],
    dist,
    k: int,
) -> list[tuple[int, int, float]]:
    """Return up to k (node, best_edge_idx, Δ) tuples with lowest Δ."""
    n = len(tour)
    items: list[tuple[int, int, float]] = []
    for node in remaining:
        best_delta = float("inf")
        best_edge = 0
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]
            d = dist[a][node] + dist[node][b] - dist[a][b]
            if d < best_delta:
                best_delta = d
                best_edge = i
        items.append((node, best_edge, best_delta))
    items.sort(key=lambda x: x[2])
    return items[:k]


def _best_delta(tour: list[int], node: int, dist) -> float:
    n = len(tour)
    best = float("inf")
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        d = dist[a][node] + dist[node][b] - dist[a][b]
        if d < best:
            best = d
    return best
