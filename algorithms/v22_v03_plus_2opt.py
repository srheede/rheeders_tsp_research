"""
v22_v03_plus_2opt — champion construction + classical 2-opt.

Per the project plan, the construction-only winner across Families A–E
(measured by average optimality gap on the fixed benchmark suite) gets
paired with 2-opt post-processing as the "best of both worlds" variant.

Family A–E results (after running the full suite):

  v03 regret_insertion          → 4.31%   ← winner (construction-only)
  v17 lookahead_k_cheapest      → 5.80%
  v02 furthest_insertion        → 6.33%
  v04 hull_distance_descending  → 7.71%
  v13 centroid_spiral_inward    → 8.29%
  v05 hull_distance_ascending   → 10.64%
  v19 delaunay_bfs              → 10.75%
  (others ≥ 30%)

v22 therefore runs v03's regret-insertion construction and follows up
with 2-opt to convergence. Regret insertion is the most cost-aware of
our greedy variants, so 2-opt is starting from an already-good tour and
mostly has small crossings to fix.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep
from algorithms import v03_regret_insertion
from algorithms._helpers import two_opt


def solve_from_hull(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    tour = v03_regret_insertion.solve_from_hull(
        hull, remaining, distance_matrix, coords
    )
    tour, _ = two_opt(tour, distance_matrix)
    return tour


def solve_from_hull_traced(
    hull: list[int],
    remaining: list[int],
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list[TraceStep]]:
    tour, insert_steps = v03_regret_insertion.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    tour, opt_steps = two_opt(tour, distance_matrix)
    return tour, insert_steps + opt_steps
