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
# Compound local search: alternating 2-opt and Or-opt to joint convergence
# ---------------------------------------------------------------------------

def compound_local_search(
    tour: list[int],
    dist,
    or_opt_chain_lengths: tuple[int, ...] = (1, 2, 3),
    max_outer: int = 50,
) -> tuple[list[int], list[TraceStep]]:
    """Alternate 2-opt and Or-opt until neither produces any improvement.

    Either local search may free up the other (Or-opt relocations create
    new crossings 2-opt can fix, and 2-opt rearrangements expose new
    relocation gains for Or-opt). Running them to mutual convergence is
    a classical "VND" (Variable Neighbourhood Descent) strategy.
    """
    tour = list(tour)
    all_steps: list[TraceStep] = []
    last_cost = compute_tour_cost(tour, dist)
    for _ in range(max_outer):
        tour, two_steps = two_opt(tour, dist)
        tour, or_steps = or_opt(tour, dist, or_opt_chain_lengths)
        all_steps.extend(two_steps)
        all_steps.extend(or_steps)
        new_cost = compute_tour_cost(tour, dist)
        if new_cost >= last_cost - 1e-9:
            break
        last_cost = new_cost
    return tour, all_steps


# ---------------------------------------------------------------------------
# 3-opt local search (sequential first-improvement on edge triples)
# ---------------------------------------------------------------------------

def three_opt(
    tour: list[int],
    dist,
    max_passes: int = 30,
) -> tuple[list[int], list[TraceStep]]:
    """Sequential 3-opt: try every triple of edges, accept any improving move.

    For each triple of *non-adjacent* edges (a-b), (c-d), (e-f) (in tour
    order) we consider the 7 possible reconnections (the trivial one is
    the identity). The single one that reduces the tour length the most
    is applied. Run repeatedly until no improving triple exists.

    NOTE: this is O(n³) per pass, so we cap ``max_passes`` and rely on
    early-termination when no improvement appears.
    """
    tour = list(tour)
    n = len(tour)
    steps: list[TraceStep] = []
    for _pass in range(max_passes):
        improved = False
        for i in range(n - 2):
            a_idx = i
            b_idx = i + 1
            for j in range(i + 1, n - 1):
                c_idx = j
                d_idx = j + 1
                for k in range(j + 1, n + (0 if i > 0 else -1)):
                    e_idx = k
                    f_idx = (k + 1) % n
                    if f_idx == a_idx:
                        continue
                    a, b = tour[a_idx], tour[b_idx]
                    c, d = tour[c_idx], tour[d_idx]
                    e, f = tour[e_idx], tour[f_idx]
                    d_old = dist[a][b] + dist[c][d] + dist[e][f]
                    # 7 non-trivial reconnections; pick best.
                    cases = [
                        # Each tuple: (new tour middle segments order, cost)
                        # cost = ab + cd + ef → new edges
                        (1, dist[a][c] + dist[b][d] + dist[e][f]),       # reverse seg1
                        (2, dist[a][b] + dist[c][e] + dist[d][f]),       # reverse seg2
                        (3, dist[a][c] + dist[b][e] + dist[d][f]),       # reverse seg1+seg2
                        (4, dist[a][d] + dist[e][b] + dist[c][f]),       # 3-opt move
                        (5, dist[a][e] + dist[d][b] + dist[c][f]),       # 3-opt move
                        (6, dist[a][d] + dist[e][c] + dist[b][f]),       # 3-opt move
                        (7, dist[a][e] + dist[d][c] + dist[b][f]),       # 3-opt move
                    ]
                    best_case, best_cost = min(cases, key=lambda x: x[1])
                    delta = best_cost - d_old
                    if delta < -1e-9:
                        seg1 = tour[b_idx: d_idx]
                        seg2 = tour[d_idx: f_idx if f_idx > d_idx else n]
                        # Apply chosen reconnection.
                        new_middle = _apply_3opt_case(
                            best_case, seg1, seg2
                        )
                        # Replace tour[b_idx : f_idx] with new_middle (handling wrap).
                        if f_idx > b_idx:
                            tour[b_idx: f_idx] = new_middle
                        else:
                            # f_idx wraps to 0 — replace tour[b_idx:] entirely
                            tour[b_idx:] = new_middle
                        steps.append(TraceStep(
                            node=b,
                            inserted_after=a,
                            removed_edge=(a, b),
                            new_edges=[(a, tour[b_idx])],
                            description=(
                                f"3-opt case {best_case}: Δ={delta:.2f}"
                            ),
                        ))
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return tour, steps


def _apply_3opt_case(
    case: int,
    seg1: list[int],
    seg2: list[int],
) -> list[int]:
    """Return the middle of the tour after applying 3-opt ``case``.

    Cases 1..7 follow the standard enumeration:
      1: reverse seg1
      2: reverse seg2
      3: reverse seg1 and seg2 individually
      4: swap segments (seg2 then seg1)
      5: seg2 then reversed seg1
      6: reversed seg2 then seg1
      7: reversed seg2 then reversed seg1
    """
    rs1 = list(reversed(seg1))
    rs2 = list(reversed(seg2))
    if case == 1:
        return rs1 + seg2
    if case == 2:
        return seg1 + rs2
    if case == 3:
        return rs1 + rs2
    if case == 4:
        return seg2 + seg1
    if case == 5:
        return seg2 + rs1
    if case == 6:
        return rs2 + seg1
    if case == 7:
        return rs2 + rs1
    return seg1 + seg2


# ---------------------------------------------------------------------------
# Double-bridge perturbation (classic ILS kick move)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Neighbor-list-based fast local search (much faster on large instances)
# ---------------------------------------------------------------------------

def build_neighbor_lists(dist, k: int = 20) -> list[list[int]]:
    """For each node, return the k nearest nodes (ascending distance)."""
    n = len(dist)
    arr = np.asarray(dist)
    # argpartition is O(n) per row; we then sort only the top k.
    nb = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        idx = np.argpartition(arr[i], k + 1)[: k + 1]
        idx = idx[idx != i][:k]
        order = np.argsort(arr[i, idx])
        nb[i] = idx[order]
    return nb.tolist()


def two_opt_neighbors(
    tour: list[int],
    dist,
    neighbors: list[list[int]],
    max_passes: int = 200,
) -> list[int]:
    """First-improvement 2-opt restricted to k-nearest-neighbour edges.

    Classical 2-opt examines O(n²) pairs per pass. Restricting the
    "incoming" endpoint to a node's nearest-k neighbour list keeps the
    asymptotic per-pass cost at O(n·k), which is up to 20× faster on
    n=280 with k=20 while retaining most of the improvement opportunities
    (an improving 2-opt move almost always involves a near-neighbour edge).
    """
    tour = list(tour)
    n = len(tour)
    pos = [0] * n
    for i, node in enumerate(tour):
        pos[node] = i
    dont_look = [False] * n

    for _ in range(max_passes):
        improved = False
        for a in range(n):
            if dont_look[a]:
                continue
            local_improved = False
            i = pos[a]
            b = tour[(i + 1) % n]
            d_ab = dist[a][b]
            for c in neighbors[a]:
                if c == a or c == b:
                    continue
                d_ac = dist[a][c]
                if d_ac >= d_ab:
                    # Sorted neighbours: further ones can only be worse.
                    break
                j = pos[c]
                # 2-opt: remove (a,b) and (c,d), add (a,c) and (b,d).
                d_node = tour[(j + 1) % n]
                if d_node == a:
                    continue
                d_cd = dist[c][d_node]
                delta = d_ac + dist[b][d_node] - d_ab - d_cd
                if delta < -1e-12:
                    # Reverse tour[i+1 .. j] (handling wrap correctly).
                    _reverse_segment(tour, (i + 1) % n, j, pos)
                    dont_look[a] = False
                    dont_look[b] = False
                    dont_look[c] = False
                    dont_look[d_node] = False
                    improved = True
                    local_improved = True
                    break
            if not local_improved:
                dont_look[a] = True
        if not improved:
            break
    return tour


def or_opt_neighbors(
    tour: list[int],
    dist,
    neighbors: list[list[int]],
    chain_lengths: tuple[int, ...] = (1, 2, 3),
    max_passes: int = 200,
) -> list[int]:
    """Or-opt restricted to nearest-neighbour insertion targets.

    For each segment ``[s_first .. s_last]`` of length ``L``, the
    re-insertion target ``(p, q)`` is constrained so ``p`` is one of
    ``s_first``'s nearest neighbours, drastically reducing the
    O(n²·L) classical scan to O(n·k·L).
    """
    tour = list(tour)
    n = len(tour)
    pos = [0] * n
    for i, node in enumerate(tour):
        pos[node] = i

    for _ in range(max_passes):
        improved = False
        for L in chain_lengths:
            if L >= n - 1:
                continue
            for s_first in range(n):
                # Compute segment indices via pos[].
                seg_start_idx = pos[s_first]
                seg_end_idx = (seg_start_idx + L - 1) % n
                prev_idx = (seg_start_idx - 1) % n
                next_idx = (seg_end_idx + 1) % n
                a = tour[prev_idx]
                s_last = tour[seg_end_idx]
                b = tour[next_idx]
                if a == s_last or b == s_first:
                    continue
                removed_cost = (
                    dist[a][s_first] + dist[s_last][b] - dist[a][b]
                )
                if removed_cost <= 1e-12:
                    continue

                best_delta = -1e-12
                best_p_idx = -1
                best_rev = False

                # Candidate insertion targets near s_first or s_last.
                candidates = set(neighbors[s_first]) | set(neighbors[s_last])
                for p in candidates:
                    if p == s_first or p == s_last or p == a:
                        continue
                    j = pos[p]
                    if _in_segment(j, seg_start_idx, seg_end_idx, n):
                        continue
                    q = tour[(j + 1) % n]
                    if q == s_first or _in_segment(
                        (j + 1) % n, seg_start_idx, seg_end_idx, n
                    ):
                        continue
                    pq = dist[p][q]
                    ins_fwd = dist[p][s_first] + dist[s_last][q] - pq
                    ins_rev = dist[p][s_last] + dist[s_first][q] - pq
                    delta_fwd = ins_fwd - removed_cost
                    delta_rev = ins_rev - removed_cost
                    if delta_fwd < best_delta:
                        best_delta = delta_fwd
                        best_p_idx = j
                        best_rev = False
                    if delta_rev < best_delta:
                        best_delta = delta_rev
                        best_p_idx = j
                        best_rev = True

                if best_p_idx >= 0:
                    segment = [
                        tour[(seg_start_idx + k) % n] for k in range(L)
                    ]
                    if best_rev:
                        segment = list(reversed(segment))
                    # Remove segment from tour.
                    if seg_start_idx <= seg_end_idx:
                        del tour[seg_start_idx: seg_end_idx + 1]
                    else:
                        del tour[seg_start_idx:]
                        del tour[: seg_end_idx + 1]
                    new_n = len(tour)
                    # Refresh insertion index (positions shifted).
                    if seg_start_idx <= seg_end_idx and best_p_idx > seg_end_idx:
                        new_p = best_p_idx - L
                    else:
                        new_p = best_p_idx
                    if new_p < 0:
                        new_p += new_n
                    for off, node in enumerate(segment):
                        tour.insert(new_p + 1 + off, node)
                    n = len(tour)
                    pos = [0] * n
                    for i, node in enumerate(tour):
                        pos[node] = i
                    improved = True
                    break  # restart outer scan
            if improved:
                break
        if not improved:
            break
    return tour


def _reverse_segment(tour: list[int], start: int, end: int, pos: list[int]) -> None:
    """In-place reverse of ``tour[start..end]`` (inclusive), updating pos[].

    Handles wrap-around (start > end means the segment wraps past index 0).
    """
    n = len(tour)
    if start <= end:
        length = end - start + 1
    else:
        length = (n - start) + end + 1
    for k in range(length // 2):
        i = (start + k) % n
        j = (end - k) % n
        tour[i], tour[j] = tour[j], tour[i]
        pos[tour[i]] = i
        pos[tour[j]] = j


def fast_local_search(
    tour: list[int],
    dist,
    neighbors: list[list[int]],
    or_opt_chain_lengths: tuple[int, ...] = (1, 2, 3, 4, 5),
    max_outer: int = 30,
) -> list[int]:
    """Alternate neighbor-restricted 2-opt and Or-opt to joint convergence.

    Much faster than the full O(n²) loops in compound_local_search, and
    on Euclidean instances produces tours of essentially the same quality
    (the missed moves are dominated by long-edge swaps that rarely improve).
    """
    tour = list(tour)
    last_cost = compute_tour_cost(tour, dist)
    for _ in range(max_outer):
        tour = two_opt_neighbors(tour, dist, neighbors)
        tour = or_opt_neighbors(tour, dist, neighbors, or_opt_chain_lengths)
        new_cost = compute_tour_cost(tour, dist)
        if new_cost >= last_cost - 1e-9:
            break
        last_cost = new_cost
    return tour


def double_bridge(tour: list[int], rng: np.random.Generator) -> list[int]:
    """Apply a double-bridge 4-opt perturbation that no 2-opt or 3-opt can
    undo in a single move. Splits the tour at four random points and
    reconnects the four resulting segments in the swapped order
    ``A + C + B + D``.

    Standard kick move in Iterated Local Search literature; it diversifies
    enough to escape the current LS basin while preserving most of the
    tour structure.
    """
    n = len(tour)
    if n < 8:
        return list(tour)
    p = sorted(rng.choice(range(1, n), size=3, replace=False).tolist())
    p1, p2, p3 = p
    A = tour[:p1]
    B = tour[p1:p2]
    C = tour[p2:p3]
    D = tour[p3:]
    return A + C + B + D


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
    "compound_local_search",
    "three_opt",
    "double_bridge",
    "build_neighbor_lists",
    "two_opt_neighbors",
    "or_opt_neighbors",
    "fast_local_search",
    "perpendicular_distance_to_segment",
    "distance_to_segment",
    "min_distance_to_hull",
    "insert_node_at",
    "compute_tour_cost",
]
