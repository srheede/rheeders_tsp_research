/*
 * _ls_inner.c — tight C kernel for the local-search inner loops.
 *
 * Implements two functions:
 *
 *   fls_two_opt     — k-NN 2-opt with don't-look bits.
 *   fls_or_opt      — k-NN Or-opt (chain lengths 1..L_max), DL bits.
 *
 * The Python wrapper (`_ls_native.py`) flattens the distance matrix
 * and neighbour lists into 1-D arrays and passes pointers via ctypes.
 * Tour is a 1-D int32 array (we mutate in place).
 *
 * Build (executed at import time by the wrapper):
 *
 *   cc -O3 -ffast-math -shared -fPIC _ls_inner.c -o _ls_inner.so
 *
 * No external dependencies.
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

typedef int32_t i32;
typedef double f64;

/* Helper: in-place reverse tour[a..b] inclusive (a, b cyclic). */
static inline void reverse_seg(i32 *tour, i32 *pos, i32 a, i32 b, i32 n) {
    if (a <= b) {
        i32 i = a, j = b;
        while (i < j) {
            i32 tmp = tour[i]; tour[i] = tour[j]; tour[j] = tmp;
            pos[tour[i]] = i;
            pos[tour[j]] = j;
            i++; j--;
        }
        if (i == j) pos[tour[i]] = i;
    } else {
        /* Wrap-around: rotate first then reverse. We materialise the
         * segment, reverse it, write back. */
        i32 len = (n - a) + (b + 1);
        i32 *buf = (i32 *) malloc(sizeof(i32) * len);
        i32 idx = 0;
        for (i32 i = a; i < n; i++) buf[idx++] = tour[i];
        for (i32 i = 0; i <= b; i++) buf[idx++] = tour[i];
        /* Reverse in buf. */
        for (i32 i = 0, j = len - 1; i < j; i++, j--) {
            i32 t = buf[i]; buf[i] = buf[j]; buf[j] = t;
        }
        /* Write back. */
        idx = 0;
        for (i32 i = a; i < n; i++) { tour[i] = buf[idx]; pos[tour[i]] = i; idx++; }
        for (i32 i = 0; i <= b; i++) { tour[i] = buf[idx]; pos[tour[i]] = i; idx++; }
        free(buf);
    }
}

/*
 * fls_two_opt — k-NN 2-opt with don't-look bits.
 *
 *   tour        — int32[n], modified in place
 *   pos         — int32[n], modified in place (pos[node] = its index)
 *   dist        — double[n*n], dist[a*n + b]
 *   neighbours  — int32[n*k]  (sorted ascending by distance from each row's node)
 *   n, k        — sizes
 *   max_passes  — outer loop cap
 */
void fls_two_opt(
    i32 *tour,
    i32 *pos,
    const f64 *dist,
    const i32 *neighbours,
    i32 n,
    i32 k,
    i32 max_passes
) {
    char *dont_look = (char *) calloc(n, 1);
    for (i32 pass = 0; pass < max_passes; pass++) {
        bool improved_pass = false;
        for (i32 a = 0; a < n; a++) {
            if (dont_look[a]) continue;
            i32 i = pos[a];
            i32 b = tour[(i + 1) % n];
            f64 d_ab = dist[a * n + b];
            bool local = false;
            const i32 *nb_a = &neighbours[a * k];
            for (i32 t = 0; t < k; t++) {
                i32 c = nb_a[t];
                if (c == a || c == b) continue;
                f64 d_ac = dist[a * n + c];
                if (d_ac >= d_ab) break;  /* sorted — no further improvement */
                i32 j = pos[c];
                i32 d_node = tour[(j + 1) % n];
                if (d_node == a) continue;
                f64 d_cd = dist[c * n + d_node];
                f64 d_bd = dist[b * n + d_node];
                f64 delta = d_ac + d_bd - d_ab - d_cd;
                if (delta < -1e-12) {
                    /* Apply: reverse segment between b and c (inclusive). */
                    i32 a_next = (i + 1) % n;
                    reverse_seg(tour, pos, a_next, j, n);
                    dont_look[a] = 0;
                    dont_look[b] = 0;
                    dont_look[c] = 0;
                    dont_look[d_node] = 0;
                    improved_pass = true;
                    local = true;
                    break;
                }
            }
            if (!local) dont_look[a] = 1;
        }
        if (!improved_pass) break;
    }
    free(dont_look);
}

/* Cyclic in-segment check: is index `idx` within [start..end] cyclically? */
static inline bool in_seg(i32 idx, i32 start, i32 end, i32 n) {
    if (start <= end) return idx >= start && idx <= end;
    return idx >= start || idx <= end;
}

/*
 * fls_or_opt — k-NN Or-opt with don't-look bits.
 *
 * Same calling convention as fls_two_opt, plus chain_lengths.
 */
void fls_or_opt(
    i32 *tour,
    i32 *pos,
    const f64 *dist,
    const i32 *neighbours,
    i32 n_in,
    i32 k,
    const i32 *chain_lengths,
    i32 n_chains,
    i32 max_passes
) {
    i32 n = n_in;
    char *dont_look = (char *) calloc(n, 1);

    for (i32 pass = 0; pass < max_passes; pass++) {
        bool improved_pass = false;
        for (i32 s_first = 0; s_first < n; s_first++) {
            if (dont_look[s_first]) continue;
            bool local = false;
            for (i32 li = 0; li < n_chains; li++) {
                i32 L = chain_lengths[li];
                if (L >= n - 1) continue;
                i32 seg_start = pos[s_first];
                i32 seg_end = (seg_start + L - 1) % n;
                i32 prev_idx = (seg_start - 1 + n) % n;
                i32 next_idx = (seg_end + 1) % n;
                i32 a = tour[prev_idx];
                i32 s_last = tour[seg_end];
                i32 b = tour[next_idx];
                if (a == s_last || b == s_first) continue;
                f64 d_a_first = dist[a * n + s_first];
                f64 d_last_b = dist[s_last * n + b];
                f64 d_ab = dist[a * n + b];
                f64 removed_cost = d_a_first + d_last_b - d_ab;
                if (removed_cost <= 1e-12) continue;

                f64 best_delta = -1e-12;
                i32 best_p_idx = -1;
                bool best_rev = false;
                /* Candidates: union of neighbours[s_first] and neighbours[s_last].
                 * We just iterate each list and let duplicates fall through —
                 * the comparison work is small. */
                const i32 *nb_first = &neighbours[s_first * k];
                const i32 *nb_last = &neighbours[s_last * k];
                for (i32 src = 0; src < 2; src++) {
                    const i32 *nb = (src == 0) ? nb_first : nb_last;
                    for (i32 t = 0; t < k; t++) {
                        i32 p = nb[t];
                        if (p == s_first || p == s_last || p == a) continue;
                        i32 j = pos[p];
                        if (in_seg(j, seg_start, seg_end, n)) continue;
                        i32 q_idx = (j + 1) % n;
                        if (in_seg(q_idx, seg_start, seg_end, n)) continue;
                        i32 q = tour[q_idx];
                        if (q == s_first) continue;
                        f64 pq = dist[p * n + q];
                        f64 ins_fwd = dist[p * n + s_first] + dist[s_last * n + q] - pq;
                        f64 ins_rev = dist[p * n + s_last] + dist[s_first * n + q] - pq;
                        f64 delta_fwd = ins_fwd - removed_cost;
                        f64 delta_rev = ins_rev - removed_cost;
                        if (delta_fwd < best_delta) {
                            best_delta = delta_fwd;
                            best_p_idx = j;
                            best_rev = false;
                        }
                        if (delta_rev < best_delta) {
                            best_delta = delta_rev;
                            best_p_idx = j;
                            best_rev = true;
                        }
                    }
                }
                if (best_p_idx >= 0) {
                    /* Materialise segment (length L). */
                    i32 segment[16];  /* L bounded by max chain length, < 16 */
                    for (i32 m = 0; m < L; m++)
                        segment[m] = tour[(seg_start + m) % n];
                    if (best_rev) {
                        for (i32 i_ = 0, j_ = L - 1; i_ < j_; i_++, j_--) {
                            i32 t = segment[i_]; segment[i_] = segment[j_]; segment[j_] = t;
                        }
                    }
                    /* Build new tour into a temporary. We do an O(n) shift —
                     * sufficient for our needs (still 50× faster than the
                     * Python equivalent since the body is in C). */
                    i32 *new_tour = (i32 *) malloc(sizeof(i32) * n);
                    i32 nidx = 0;
                    for (i32 i_ = 0; i_ < n; i_++) {
                        if (in_seg(i_, seg_start, seg_end, n)) continue;
                        new_tour[nidx++] = tour[i_];
                    }
                    /* Find new index of the node at best_p_idx in `new_tour`. */
                    i32 p_node = tour[best_p_idx];
                    i32 new_p = -1;
                    for (i32 i_ = 0; i_ < nidx; i_++) {
                        if (new_tour[i_] == p_node) { new_p = i_; break; }
                    }
                    /* Insert `segment` right after new_p. */
                    /* Shift right by L from new_p+1. */
                    for (i32 i_ = nidx - 1; i_ > new_p; i_--) {
                        new_tour[i_ + L] = new_tour[i_];
                    }
                    for (i32 m = 0; m < L; m++) new_tour[new_p + 1 + m] = segment[m];
                    /* Copy back to tour and refresh pos. */
                    memcpy(tour, new_tour, sizeof(i32) * n);
                    free(new_tour);
                    for (i32 i_ = 0; i_ < n; i_++) pos[tour[i_]] = i_;
                    /* Reset DL bits for affected nodes. */
                    for (i32 m = 0; m < L; m++) dont_look[segment[m]] = 0;
                    dont_look[a] = 0;
                    dont_look[b] = 0;
                    dont_look[p_node] = 0;
                    dont_look[tour[(new_p + L + 1) % n]] = 0;
                    improved_pass = true;
                    local = true;
                    break;
                }
            }
            if (!local) dont_look[s_first] = 1;
        }
        if (!improved_pass) break;
    }
    free(dont_look);
}
