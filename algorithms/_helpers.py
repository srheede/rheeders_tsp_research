"""
Shared mechanical utilities for hull-anchored TSP variants.

This module intentionally does NOT define any variant: it only holds
generic helpers (local search, two-bridge polygon splice, chain insertion,
geometric primitives) that multiple variants reuse. The runner's variant
discovery skips files that don't start with ``v<digit>`` so this file is
not picked up.

The *algorithmic idea* of each variant should still live inside its own
``algorithms/vNN_*.py`` file; this module only offloads boilerplate.
"""

from __future__ import annotations

import numpy as np

from algorithms.protocol import TraceStep, compute_tour_cost


# ---------------------------------------------------------------------------
# Cheapest insertion (used by several variants as a sub-routine)
# ---------------------------------------------------------------------------

def best_insertion_position(
    tour: list[int],
    node: int,
    dist,
) -> tuple[int, float]:
    """Return (index_after, delta) for the cheapest insertion of ``node``.

    The returned index is the position in ``tour`` AFTER which ``node`` should
    be inserted (so the new tour is ``tour[:i+1] + [node] + tour[i+1:]``).
    """
    n = len(tour)
    best_i = 0
    best_delta = float("inf")
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        delta = dist[a][node] + dist[node][b] - dist[a][b]
        if delta < best_delta:
            best_delta = delta
            best_i = i
    return best_i, best_delta


def best_two_insertion_positions(
    tour: list[int],
    node: int,
    dist,
) -> tuple[float, float]:
    """Return (best_delta, second_best_delta) for inserting ``node``."""
    n = len(tour)
    best = float("inf")
    second = float("inf")
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        delta = dist[a][node] + dist[node][b] - dist[a][b]
        if delta < best:
            second = best
            best = delta
        elif delta < second:
            second = delta
    return best, second


# ---------------------------------------------------------------------------
# Chain insertion: splice a path (chain of nodes) into the tour
# ---------------------------------------------------------------------------

def best_chain_insertion(
    tour: list[int],
    chain: list[int],
    dist,
) -> tuple[int, bool, float]:
    """Find the cheapest edge of ``tour`` to splice ``chain`` into.

    Returns
    -------
    (index_after, reversed, delta)
        index_after: position in ``tour`` after which the chain is inserted.
        reversed:    True iff the chain should be reversed before splicing.
        delta:       total tour-length change from this splice.
    """
    n = len(tour)
    start = chain[0]
    end = chain[-1]
    chain_internal = sum(dist[chain[k]][chain[k + 1]] for k in range(len(chain) - 1))

    best_i = 0
    best_rev = False
    best_delta = float("inf")
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        ab = dist[a][b]
        # Forward orientation: a → start → ... → end → b
        d_fwd = dist[a][start] + chain_internal + dist[end][b] - ab
        if d_fwd < best_delta:
            best_delta = d_fwd
            best_i = i
            best_rev = False
        # Reverse orientation: a → end → ... → start → b
        d_rev = dist[a][end] + chain_internal + dist[start][b] - ab
        if d_rev < best_delta:
            best_delta = d_rev
            best_i = i
            best_rev = True
    return best_i, best_rev, best_delta


def splice_chain_into_tour(
    tour: list[int],
    chain: list[int],
    dist,
) -> tuple[list[int], list[TraceStep]]:
    """Splice the entire ``chain`` into ``tour`` at its cheapest position.

    Generates one TraceStep per chain node in commit order so the visualizer
    can replay the result as a sequence of insertions.
    """
    if not chain:
        return list(tour), []
    idx, reversed_, _ = best_chain_insertion(tour, chain, dist)
    placed_chain = list(reversed(chain)) if reversed_ else list(chain)

    n = len(tour)
    a = tour[idx]
    b = tour[(idx + 1) % n]
    new_tour = list(tour)

    steps: list[TraceStep] = []
    insert_after = a
    next_node = b
    cur_idx = idx
    for k, node in enumerate(placed_chain):
        new_tour.insert(cur_idx + 1, node)
        new_next = placed_chain[k + 1] if k + 1 < len(placed_chain) else b
        steps.append(TraceStep(
            node=node,
            inserted_after=insert_after,
            removed_edge=(insert_after, next_node),
            new_edges=[(insert_after, node), (node, new_next)],
            description=f"Splice chain: insert {node} between {insert_after} and {new_next}",
        ))
        insert_after = node
        next_node = new_next
        cur_idx += 1
    return new_tour, steps


# ---------------------------------------------------------------------------
# Two-bridge polygon splice: merge a closed sub-tour into the outer tour
# ---------------------------------------------------------------------------

def merge_two_polygons(
    outer: list[int],
    inner: list[int],
    dist,
) -> tuple[list[int], list[TraceStep]]:
    """Merge a closed ``inner`` polygon into ``outer`` via two-bridge splice.

    Algorithm: cut one edge in each polygon and reconnect endpoints so the
    union is a single Hamiltonian cycle, minimising added length over all
    (outer-edge, inner-edge, orientation) choices.

    Both ``outer`` and ``inner`` are open-list representations of closed
    cycles (their last node connects back to the first).
    """
    if not inner:
        return list(outer), []
    if len(inner) == 1:
        # Degenerate: single point — fall back to cheapest insertion.
        node = inner[0]
        idx, _ = best_insertion_position(outer, node, dist)
        new_tour = list(outer)
        new_tour.insert(idx + 1, node)
        a = outer[idx]
        b = outer[(idx + 1) % len(outer)]
        return new_tour, [TraceStep(
            node=node,
            inserted_after=a,
            removed_edge=(a, b),
            new_edges=[(a, node), (node, b)],
            description=f"Splice singleton {node} between {a} and {b}",
        )]
    if len(inner) == 2:
        # Two-node "polygon" is a 2-cycle; treat as a 2-node chain instead.
        return splice_chain_into_tour(outer, inner, dist)

    no = len(outer)
    ni = len(inner)
    best_delta = float("inf")
    best_oi = 0
    best_ii = 0
    best_rev = False

    for oi in range(no):
        a = outer[oi]
        b = outer[(oi + 1) % no]
        d_ab = dist[a][b]
        for ii in range(ni):
            c = inner[ii]
            d = inner[(ii + 1) % ni]
            d_cd = dist[c][d]
            # Forward orientation: a → c → ... → d → b (traverse inner from c onward)
            delta_fwd = dist[a][c] + dist[d][b] - d_ab - d_cd
            if delta_fwd < best_delta:
                best_delta = delta_fwd
                best_oi = oi
                best_ii = ii
                best_rev = False
            # Reverse orientation: a → d → ... → c → b
            delta_rev = dist[a][d] + dist[c][b] - d_ab - d_cd
            if delta_rev < best_delta:
                best_delta = delta_rev
                best_oi = oi
                best_ii = ii
                best_rev = True

    # Build the chain to splice in.
    # Edge (c, d) = (inner[best_ii], inner[best_ii + 1]) is cut; the remaining
    # ni - 1 edges of the inner cycle form an open chain that visits every
    # inner node exactly once.
    if best_rev:
        # Reverse orientation: a → d → ... → c → b — walk from d forward
        # along the (unchopped) cycle direction until we reach c.
        chain = [inner[(best_ii + 1 + k) % ni] for k in range(ni)]
    else:
        # Forward orientation: a → c → ... → d → b — walk from c backward
        # through the cycle until we reach d.
        chain = [inner[(best_ii - k) % ni] for k in range(ni)]

    # Insert chain into outer between outer[best_oi] and outer[best_oi+1].
    new_tour = list(outer)
    a = outer[best_oi]
    b = outer[(best_oi + 1) % no]
    steps: list[TraceStep] = []
    cur_idx = best_oi
    insert_after = a
    next_node = b
    for k, node in enumerate(chain):
        new_tour.insert(cur_idx + 1, node)
        new_next = chain[k + 1] if k + 1 < len(chain) else b
        steps.append(TraceStep(
            node=node,
            inserted_after=insert_after,
            removed_edge=(insert_after, next_node),
            new_edges=[(insert_after, node), (node, new_next)],
            description=f"Polygon merge: insert {node} between {insert_after} and {new_next}",
        ))
        insert_after = node
        next_node = new_next
        cur_idx += 1
    return new_tour, steps


# ---------------------------------------------------------------------------
# 2-opt local search (used as post-processing only)
# ---------------------------------------------------------------------------

def two_opt(
    tour: list[int],
    dist,
    max_passes: int = 100,
) -> tuple[list[int], list[TraceStep]]:
    """Improve ``tour`` in place via 2-opt swaps until no improvement.

    Emits a TraceStep for each accepted swap. The "node" field of each
    TraceStep is set to one of the involved nodes for visualizer compatibility.
    """
    tour = list(tour)
    n = len(tour)
    steps: list[TraceStep] = []
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                if delta < -1e-12:
                    tour[i + 1: j + 1] = list(reversed(tour[i + 1: j + 1]))
                    steps.append(TraceStep(
                        node=b,
                        inserted_after=a,
                        removed_edge=(a, b),
                        new_edges=[(a, c), (b, d)],
                        description=f"2-opt: swap edges ({a},{b})-({c},{d}) → ({a},{c})-({b},{d})  Δ={delta:.2f}",
                    ))
                    improved = True
    return tour, steps


# ---------------------------------------------------------------------------
# Or-opt local search (move chains of length 1/2/3 to a better position)
# ---------------------------------------------------------------------------

def or_opt(
    tour: list[int],
    dist,
    chain_lengths: tuple[int, ...] = (1, 2, 3),
    max_passes: int = 100,
) -> tuple[list[int], list[TraceStep]]:
    """Improve ``tour`` via Or-opt segment relocation."""
    tour = list(tour)
    n = len(tour)
    steps: list[TraceStep] = []
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for L in chain_lengths:
            if L >= n - 1:
                continue
            i = 0
            while i < n:
                # Segment [i .. i+L-1] (indices mod n).
                seg_start = i
                seg_end = (i + L - 1) % n
                prev_idx = (seg_start - 1) % n
                next_idx = (seg_end + 1) % n
                a = tour[prev_idx]
                s_first = tour[seg_start]
                s_last = tour[seg_end]
                b = tour[next_idx]
                if a == s_last or b == s_first:
                    i += 1
                    continue
                removed_cost = dist[a][s_first] + dist[s_last][b] - dist[a][b]

                best_delta = 0.0
                best_pos = None
                best_rev = False
                # Try inserting the chain at every other position.
                for j in range(n):
                    # Skip positions overlapping the segment.
                    if _in_segment(j, seg_start, seg_end, n):
                        continue
                    nj = (j + 1) % n
                    if _in_segment(nj, seg_start, seg_end, n):
                        continue
                    p = tour[j]
                    q = tour[nj]
                    if p == a and q == b:
                        continue
                    pq = dist[p][q]
                    insert_fwd = dist[p][s_first] + dist[s_last][q] - pq
                    insert_rev = dist[p][s_last] + dist[s_first][q] - pq
                    delta_fwd = insert_fwd - removed_cost
                    delta_rev = insert_rev - removed_cost
                    if delta_fwd < best_delta - 1e-12:
                        best_delta = delta_fwd
                        best_pos = j
                        best_rev = False
                    if delta_rev < best_delta - 1e-12:
                        best_delta = delta_rev
                        best_pos = j
                        best_rev = True

                if best_pos is not None:
                    segment = [tour[(seg_start + k) % n] for k in range(L)]
                    if best_rev:
                        segment = list(reversed(segment))
                    # Remove segment from tour, then insert.
                    if seg_start <= seg_end:
                        del tour[seg_start: seg_end + 1]
                    else:
                        # Segment wraps around — delete two slices.
                        del tour[seg_start:]
                        del tour[: seg_end + 1]
                    # After deletion the absolute index of best_pos may shift.
                    new_n = len(tour)
                    # Find p in the (possibly shifted) tour, place segment after it.
                    p_node = tour[(best_pos % new_n)] if best_pos < n else None
                    # Safer: recompute via the saved node id `p`.
                    if seg_start <= seg_end:
                        shift = L if best_pos > seg_end else 0
                    else:
                        shift = 0
                    new_pos = best_pos - shift
                    if new_pos < 0:
                        new_pos += new_n
                    insert_at = (new_pos + 1) % (new_n + L)
                    for off, node in enumerate(segment):
                        tour.insert(new_pos + 1 + off, node)
                    n = len(tour)
                    steps.append(TraceStep(
                        node=segment[0],
                        inserted_after=p_node if p_node is not None else -1,
                        removed_edge=None,
                        new_edges=[],
                        description=(
                            f"Or-opt: move chain len={L} "
                            f"(starting {s_first}, ending {s_last}) Δ={best_delta:.2f}"
                        ),
                    ))
                    improved = True
                    # Restart scan because indices shifted.
                    i = 0
                    continue
                i += 1
    return tour, steps


def _in_segment(idx: int, seg_start: int, seg_end: int, n: int) -> bool:
    if seg_start <= seg_end:
        return seg_start <= idx <= seg_end
    return idx >= seg_start or idx <= seg_end


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------

def perpendicular_distance_to_segment(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[float, float]:
    """Return (distance, t) where ``t`` is the projection parameter on ``ab``.

    ``t ∈ [0, 1]`` means the perpendicular foot lies inside the segment.
    Outside that range, the foot is the extrapolation beyond an endpoint,
    but the returned distance is still perpendicular-to-line distance.
    """
    ab = b - a
    ab_sq = float(ab @ ab)
    if ab_sq == 0.0:
        return float(np.linalg.norm(p - a)), 0.0
    t = float((p - a) @ ab / ab_sq)
    foot = a + t * ab
    return float(np.linalg.norm(p - foot)), t


def distance_to_segment(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[float, float]:
    """Return (distance, t_clamped) where distance is to the segment itself.

    If the perpendicular foot lies outside ``ab``, ``t_clamped`` is clamped
    to [0, 1] and the distance is to the nearer endpoint.
    """
    ab = b - a
    ab_sq = float(ab @ ab)
    if ab_sq == 0.0:
        return float(np.linalg.norm(p - a)), 0.0
    t = float((p - a) @ ab / ab_sq)
    t_clamped = max(0.0, min(1.0, t))
    foot = a + t_clamped * ab
    return float(np.linalg.norm(p - foot)), t_clamped


def min_distance_to_hull(
    coords: np.ndarray,
    hull: list[int],
    node: int,
) -> float:
    """Minimum perpendicular distance from ``node`` to any hull edge."""
    p = coords[node]
    n = len(hull)
    best = float("inf")
    for i in range(n):
        a = coords[hull[i]]
        b = coords[hull[(i + 1) % n]]
        d, _ = distance_to_segment(p, a, b)
        if d < best:
            best = d
    return best


# ---------------------------------------------------------------------------
# Trace-step builder for a single cheapest insertion (avoids duplication)
# ---------------------------------------------------------------------------

def insert_node_at(
    tour: list[int],
    node: int,
    index_after: int,
) -> TraceStep:
    """Insert ``node`` after position ``index_after`` and return a TraceStep.

    Mutates ``tour`` in place.
    """
    n = len(tour)
    a = tour[index_after]
    b = tour[(index_after + 1) % n]
    tour.insert(index_after + 1, node)
    return TraceStep(
        node=node,
        inserted_after=a,
        removed_edge=(a, b),
        new_edges=[(a, node), (node, b)],
        description=f"Insert {node} between {a} and {b}",
    )


# Re-export for variant convenience.
__all__ = [
    "best_insertion_position",
    "best_two_insertion_positions",
    "best_chain_insertion",
    "splice_chain_into_tour",
    "merge_two_polygons",
    "two_opt",
    "or_opt",
    "perpendicular_distance_to_segment",
    "distance_to_segment",
    "min_distance_to_hull",
    "insert_node_at",
    "compute_tour_cost",
]
