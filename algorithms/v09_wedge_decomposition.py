"""
v09_wedge_decomposition — centroid wedges decide which hull edge owns a node.

Compute the centroid ``C`` of the hull vertices. The ray from ``C``
through each hull vertex partitions the interior into ``|hull|`` angular
wedges. The wedge between two consecutive rays (``C→h_i``, ``C→h_{i+1}``)
is "owned" by the hull edge ``(h_i, h_{i+1})``.

Each interior node lies in exactly one wedge and is therefore assigned
uniquely to one hull edge. Within a wedge we then build the **optimal**
Hamiltonian sub-path with fixed endpoints ``h_i`` and ``h_{i+1}``:

  * wedges with ≤ 11 interior nodes → exact Held-Karp DP
  * larger wedges                  → nearest-neighbour path + 2-opt restricted
                                     to the wedge

Each hull edge is then *replaced* by its computed sub-path and the tour
is the concatenation of all sub-paths.

Compared to v06/v07 this variant respects the *true tour cost* inside
each wedge instead of just sorting nodes by projection.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep


_DP_LIMIT = 11  # max wedge size for exact DP (2^11 * 11^2 ≈ 248k states)


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
        raise ValueError("v09 requires coordinates.")
    if not remaining:
        return list(hull), []

    n_hull = len(hull)
    centroid = coords[hull].mean(axis=0)
    # Angle of each hull vertex around the centroid (in standard CCW order).
    angles = np.array([
        np.arctan2(coords[h][1] - centroid[1], coords[h][0] - centroid[0])
        for h in hull
    ])

    # Assign each interior node to the wedge between two consecutive hull
    # vertices in *hull-list order* — i.e. the wedge "owned" by edge i is
    # the one bounded by the rays through hull[i] and hull[i+1].
    assignments: list[list[int]] = [[] for _ in range(n_hull)]
    for node in remaining:
        p = coords[node]
        node_angle = np.arctan2(p[1] - centroid[1], p[0] - centroid[0])
        # For each edge i, test if node_angle lies in the CCW arc
        # from angles[i] to angles[i+1].
        chosen = -1
        for i in range(n_hull):
            a1 = angles[i]
            a2 = angles[(i + 1) % n_hull]
            if _in_ccw_arc(node_angle, a1, a2):
                chosen = i
                break
        if chosen < 0:
            # Fallback (numerical edge case): assign to nearest-by-edge-midpoint.
            mids = np.array([
                0.5 * (coords[hull[i]] + coords[hull[(i + 1) % n_hull]])
                for i in range(n_hull)
            ])
            chosen = int(np.argmin(np.linalg.norm(mids - p, axis=1)))
        assignments[chosen].append(node)

    tour: list[int] = []
    steps: list[TraceStep] = []
    for i in range(n_hull):
        a = hull[i]
        b = hull[(i + 1) % n_hull]
        tour.append(a)
        bucket = assignments[i]
        if not bucket:
            continue
        sub_path = _shortest_subpath(a, b, bucket, dist)
        # sub_path is [a, n1, n2, ..., nk, b]; we want to append n1..nk
        # (a was already appended, b will be appended at next iteration).
        prev = a
        for j, node in enumerate(sub_path[1:-1]):
            next_after = sub_path[j + 2]  # = sub_path[j+2] which is the next node after `node`
            tour.append(node)
            if trace:
                steps.append(TraceStep(
                    node=node,
                    inserted_after=prev,
                    removed_edge=(prev, next_after) if j > 0 else (a, b),
                    new_edges=[(prev, node), (node, next_after)],
                    description=(
                        f"Wedge ({a},{b}): place node {node} via DP/NN sub-path"
                    ),
                ))
            prev = node
    return tour, steps


def _in_ccw_arc(theta: float, a1: float, a2: float) -> bool:
    """Test whether angle theta lies in the CCW arc from a1 to a2.

    All angles in (-pi, pi]. The arc length can be > pi, going CCW.
    """
    # Normalise to non-negative offsets from a1.
    two_pi = 2.0 * np.pi
    span = (a2 - a1) % two_pi
    offset = (theta - a1) % two_pi
    return offset <= span + 1e-12


def _shortest_subpath(start: int, end: int, mids: list[int], dist) -> list[int]:
    """Optimal Hamiltonian path from start to end through all of ``mids``.

    Exact Held-Karp DP when |mids| ≤ _DP_LIMIT; nearest-neighbour + 2-opt
    fallback for larger sets.
    """
    k = len(mids)
    if k == 0:
        return [start, end]
    if k == 1:
        return [start, mids[0], end]

    if k <= _DP_LIMIT:
        return _held_karp_open(start, end, mids, dist)
    return _nn_then_2opt(start, end, mids, dist)


def _held_karp_open(start: int, end: int, mids: list[int], dist) -> list[int]:
    """Exact shortest s→t Hamiltonian path through ``mids`` via Held-Karp."""
    k = len(mids)
    # dp[(S, j)] = (cost, parent) for visiting set S with last node mids[j]
    # S is bitmask over mids indices.
    INF = float("inf")
    dp = [[INF] * k for _ in range(1 << k)]
    parent = [[-1] * k for _ in range(1 << k)]
    for j in range(k):
        dp[1 << j][j] = dist[start][mids[j]]

    for S in range(1 << k):
        for j in range(k):
            if not (S & (1 << j)) or dp[S][j] == INF:
                continue
            base = dp[S][j]
            for i in range(k):
                if S & (1 << i):
                    continue
                S2 = S | (1 << i)
                new_cost = base + dist[mids[j]][mids[i]]
                if new_cost < dp[S2][i]:
                    dp[S2][i] = new_cost
                    parent[S2][i] = j

    full = (1 << k) - 1
    best_cost = INF
    best_last = -1
    for j in range(k):
        c = dp[full][j] + dist[mids[j]][end]
        if c < best_cost:
            best_cost = c
            best_last = j

    # Reconstruct path.
    order = []
    S = full
    j = best_last
    while j != -1:
        order.append(mids[j])
        prev_j = parent[S][j]
        S ^= 1 << j
        j = prev_j
    order.reverse()
    return [start] + order + [end]


def _nn_then_2opt(start: int, end: int, mids: list[int], dist) -> list[int]:
    """Nearest-neighbour path s→t through ``mids``, refined by 2-opt."""
    # Build NN: from start, repeatedly go to the closest unvisited in mids.
    pool = set(mids)
    path = [start]
    while pool:
        last = path[-1]
        nxt = min(pool, key=lambda n: dist[last][n])
        pool.remove(nxt)
        path.append(nxt)
    path.append(end)

    # 2-opt on the open path (endpoints fixed).
    improved = True
    while improved:
        improved = False
        n = len(path)
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                a, b = path[i - 1], path[i]
                c, d = path[j], path[j + 1]
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                if delta < -1e-12:
                    path[i:j + 1] = list(reversed(path[i:j + 1]))
                    improved = True
    return path
