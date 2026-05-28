"""
v18_hungarian_assignment — globally optimal one-shot batch insertion.

Cheapest insertion is *locally* optimal per step but each commit is made
ignoring the choices the next nodes will face. v18 takes a different
stance: at every batch we view the assignment of remaining nodes to
tour-edges as a **minimum-cost bipartite matching** problem and solve
it exactly via the Hungarian algorithm.

Round structure:
  * rows = a subset of remaining nodes
  * cols = current tour edges (one slot per edge)
  * cost[i, j] = Δ for inserting row-node ``i`` into col-edge ``j``
  * solve the assignment with `scipy.optimize.linear_sum_assignment`
    → distinct (node, edge) pairs.

If ``|remaining| > |tour_edges|``, only the lowest-Δ ``|tour_edges|``
candidates are entered into the matrix (and the slot count grows after
this round). The matched insertions are applied in descending edge
index so earlier shifts don't disturb later ones, then we iterate.

This is *globally* better than greedy per step in expectation: greedy
can let a low-cost edge be consumed by a node that has many cheap
alternatives, leaving a high-cost slot for a node with no other option;
Hungarian doesn't make that mistake.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from algorithms.protocol import TraceStep


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
        n_edges = len(tour)
        n_rem = len(remaining)

        # Build the full insertion cost matrix once per round.
        # cost[i, j] = Δ of inserting remaining[i] between tour[j] and tour[j+1].
        cost = np.empty((n_rem, n_edges), dtype=float)
        for j in range(n_edges):
            a = tour[j]
            b = tour[(j + 1) % n_edges]
            ab = dist[a][b]
            for i, node in enumerate(remaining):
                cost[i, j] = dist[a][node] + dist[node][b] - ab

        if n_rem <= n_edges:
            # All remaining can be assigned in one round.
            row_ind, col_ind = linear_sum_assignment(cost)
            assignments = list(zip(row_ind.tolist(), col_ind.tolist()))
        else:
            # Pick the n_edges most "urgent" remaining nodes (lowest best Δ).
            # The Hungarian then optimally assigns them to distinct edges.
            best_per_row = cost.min(axis=1)
            top_rows = np.argsort(best_per_row)[: n_edges]
            sub = cost[top_rows, :]
            row_ind_sub, col_ind = linear_sum_assignment(sub)
            assignments = list(zip(top_rows[row_ind_sub].tolist(), col_ind.tolist()))

        # Apply assignments in descending edge index so insertion shifts
        # don't invalidate the other selected edges.
        assignments.sort(key=lambda x: x[1], reverse=True)
        committed_rows = set()
        for row_i, col_j in assignments:
            node = remaining[row_i]
            a = tour[col_j]
            # Remember the original neighbour of ``a`` before insertion, for
            # accurate trace recording.
            b_before = tour[(col_j + 1) % len(tour)]
            tour.insert(col_j + 1, node)
            if trace:
                steps.append(TraceStep(
                    node=node,
                    inserted_after=a,
                    removed_edge=(a, b_before),
                    new_edges=[(a, node), (node, b_before)],
                    description=(
                        f"Hungarian round: assign node {node} to edge "
                        f"({a},{b_before}) Δ={cost[row_i, col_j]:.2f}"
                    ),
                ))
            committed_rows.add(row_i)

        # Remove committed rows from remaining (preserve order otherwise).
        remaining = [n for i, n in enumerate(remaining) if i not in committed_rows]

    return tour, steps
