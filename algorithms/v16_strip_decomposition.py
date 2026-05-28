"""
v16_strip_decomposition — boustrophedon "lawn-mower" through interior.

Slice the bounding box of the interior nodes into ``K`` vertical strips
(``K = ⌈√n_interior⌉``). Within each strip, sort nodes by ``y`` in an
alternating order — even strips: bottom-to-top, odd strips: top-to-
bottom. This yields a single global "lawn-mower" chain visiting every
interior node with minimal vertical doubling-back.

The chain is then spliced into the hull via the optimal two-bridge cut
(``best_chain_insertion`` enumerates all hull edges and both
orientations).

The classical heuristic-analysis result by Beardwood-Halton-Hammersley
shows that a strip / sweep ordering gives a constant-factor approximation
of the optimal TSP tour for uniformly random points; combined with the
hull this should be especially friendly to large, well-distributed
instances like ``a280`` and ``tsp225``.
"""

from __future__ import annotations

import math

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
        raise ValueError("v16 requires coordinates.")
    if not remaining:
        return list(hull), []
    if len(remaining) == 1:
        idx, _ = best_insertion_position(hull, remaining[0], dist)
        tour = list(hull)
        step = insert_node_at(tour, remaining[0], idx)
        return tour, [step] if trace else []

    pts = coords[remaining]
    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    n_int = len(remaining)
    k = max(1, int(math.ceil(math.sqrt(n_int))))
    # Slightly inflate the bounding box to make the rightmost strip inclusive.
    span = max(x_max - x_min, 1e-9)
    width = span / k

    strips: list[list[int]] = [[] for _ in range(k)]
    for node in remaining:
        x = float(coords[node][0])
        idx = min(k - 1, int((x - x_min) // width))
        strips[idx].append(node)

    # Build per-strip ordered chain (boustrophedon by strip index).
    chain: list[int] = []
    for s_idx, strip in enumerate(strips):
        if not strip:
            continue
        # Even strip → ascending y, odd strip → descending y.
        sorted_strip = sorted(
            strip,
            key=lambda n: coords[n][1],
            reverse=(s_idx % 2 == 1),
        )
        chain.extend(sorted_strip)

    new_tour, steps = splice_chain_into_tour(hull, chain, dist)
    return new_tour, steps if trace else []
