"""
v26_v01_plus_long_or_opt — Or-opt with longer chain lengths.

v23's Or-opt only relocates chains of length 1, 2, and 3 — enough to fix
isolated nodes and tight pairs, but blind to **misplaced clusters of 4 to
7 nodes**. On larger instances (tsp225, a280) the insertion construction
often dumps a cluster of similar-direction nodes into one arc that
ideally belongs in another arc; only a longer Or-opt move can heal that.

v26 simply extends the chain-length set to ``(1, 2, 3, 4, 5)`` (we cap
at 5 to keep the per-pass O(n²·L) cost manageable on a280). The
hypothesis is that longer chain moves close the remaining 2% gap on
the larger instances where v23 stalls.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms import v01_baseline
from algorithms._helpers import or_opt


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v01_baseline.solve_from_hull(hull, remaining, distance_matrix, coords)
    tour, _ = or_opt(tour, distance_matrix, chain_lengths=(1, 2, 3, 4, 5))
    return tour


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v01_baseline.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    tour, opt_steps = or_opt(tour, distance_matrix, chain_lengths=(1, 2, 3, 4, 5))
    return tour, insert_steps + opt_steps
