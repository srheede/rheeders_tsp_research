/*
 * _ls_inner.c — tight C kernel for the local-search inner loops.
 *
 * Exported functions:
 *
 *   fls_two_opt     — k-NN 2-opt with don't-look bits.
 *   fls_or_opt      — k-NN Or-opt (chain lengths 1..L_max), DL bits.
 *   fls_three_opt   — k-NN 3-opt (true variants 4/5/6), DL bits.
 *   ils_run         — full Iterated Local Search loop, end to end in C
 *                     (perturbation + LS + cost + acceptance). This is
 *                     the high-leverage routine: keeping the whole ILS
 *                     loop in C removes ALL per-iteration Python/numpy
 *                     marshalling and O(n) Python cost loops, which were
 *                     the dominant cost once the inner LS itself was in C.
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

/* ----------------------------------------------------------------------
 * 3-opt — neighbour-restricted, true variants 4/5/6 (segment swaps that
 * are NOT reducible to a sequence of 2-opt moves).
 * -------------------------------------------------------------------- */

/* Apply 3-opt segment-swap variant at cyclic positions p<q<r.
 *   B = tour[p+1 .. q]   C = tour[q+1 .. r]   tail = tour[r+1 .. p]
 *   v4: C + B + tail   v5: C + rev(B) + tail   v6: rev(C) + B + tail
 * `buf` is scratch of length n. */
static void apply_3opt(i32 *tour, i32 *buf, i32 p, i32 q, i32 r, i32 n, int variant) {
    /* Rotate so element at index (p+1) comes first. */
    i32 idx = 0;
    for (i32 i = p + 1; i < n; i++) buf[idx++] = tour[i];
    for (i32 i = 0; i <= p; i++) buf[idx++] = tour[i];
    i32 new_q = (q - p - 1 + n) % n;
    i32 new_r = (r - p - 1 + n) % n;
    i32 o = 0;
    if (variant == 4) {
        for (i32 i = new_q + 1; i <= new_r; i++) tour[o++] = buf[i];
        for (i32 i = 0; i <= new_q; i++)        tour[o++] = buf[i];
    } else if (variant == 5) {
        for (i32 i = new_q + 1; i <= new_r; i++) tour[o++] = buf[i];
        for (i32 i = new_q; i >= 0; i--)         tour[o++] = buf[i];
    } else { /* variant 6 */
        for (i32 i = new_r; i >= new_q + 1; i--) tour[o++] = buf[i];
        for (i32 i = 0; i <= new_q; i++)         tour[o++] = buf[i];
    }
    for (i32 i = new_r + 1; i < n; i++) tour[o++] = buf[i];
}

void fls_three_opt(
    i32 *tour,
    i32 *pos,
    const f64 *dist,
    const i32 *neighbours,
    i32 n,
    i32 k,
    i32 max_passes
) {
    char *dont_look = (char *) calloc(n, 1);
    i32 *buf = (i32 *) malloc(sizeof(i32) * n);
    for (i32 pass = 0; pass < max_passes; pass++) {
        bool improved_pass = false;
        for (i32 a = 0; a < n; a++) {
            if (dont_look[a]) continue;
            i32 p = pos[a];
            i32 b = tour[(p + 1) % n];
            f64 d_ab = dist[a * n + b];
            bool local = false;
            const i32 *nb_a = &neighbours[a * k];
            for (i32 ti = 0; ti < k && !local; ti++) {
                i32 d = nb_a[ti];
                if (d == a || d == b) continue;
                f64 d_ad = dist[a * n + d];
                i32 q = (pos[d] - 1 + n) % n;
                if (q == p) continue;
                i32 c = tour[q];
                if (c == a || c == b) continue;
                f64 d_cd = dist[c * n + d];
                const i32 *nb_b = &neighbours[b * k];
                for (i32 tj = 0; tj < k; tj++) {
                    i32 e = nb_b[tj];
                    if (e == a || e == b || e == c || e == d) continue;
                    i32 r = pos[e];
                    i32 f = tour[(r + 1) % n];
                    if (f == a || f == b || f == c || f == d) continue;
                    /* Cyclic strict order p < q < r. */
                    i32 qp = (q - p + n) % n;
                    i32 rp = (r - p + n) % n;
                    if (!(0 < qp && qp < rp && rp < n)) continue;
                    f64 d_ef = dist[e * n + f];
                    f64 base = d_ab + d_cd + d_ef;
                    f64 d4 = dist[e * n + b] + dist[c * n + f] + d_ad - base;
                    f64 d5 = dist[e * n + c] + dist[b * n + f] + d_ad - base;
                    f64 d6 = dist[a * n + e] + dist[d * n + b] + dist[c * n + f] - base;
                    f64 bd = -1e-9;
                    int bv = 0;
                    if (d4 < bd) { bd = d4; bv = 4; }
                    if (d5 < bd) { bd = d5; bv = 5; }
                    if (d6 < bd) { bd = d6; bv = 6; }
                    if (bv > 0) {
                        apply_3opt(tour, buf, p, q, r, n, bv);
                        for (i32 ii = 0; ii < n; ii++) pos[tour[ii]] = ii;
                        dont_look[a] = 0; dont_look[b] = 0; dont_look[c] = 0;
                        dont_look[d] = 0; dont_look[e] = 0; dont_look[f] = 0;
                        improved_pass = true;
                        local = true;
                        break;
                    }
                }
            }
            if (!local) dont_look[a] = 1;
        }
        if (!improved_pass) break;
    }
    free(dont_look);
    free(buf);
}

/* ----------------------------------------------------------------------
 * Full ILS loop in C.
 * -------------------------------------------------------------------- */

/* xoshiro256** PRNG — fast, high quality, deterministic per-seed. */
typedef struct { uint64_t s[4]; } rng_state;

static inline uint64_t rotl(uint64_t x, int kk) { return (x << kk) | (x >> (64 - kk)); }

static uint64_t rng_next(rng_state *st) {
    uint64_t *s = st->s;
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

static void rng_seed(rng_state *st, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        st->s[i] = z;
    }
}

static inline i32 rng_below(rng_state *st, i32 m) {
    return (i32)(rng_next(st) % (uint64_t) m);
}

static f64 tour_cost(const i32 *tour, const f64 *dist, i32 n) {
    f64 s = 0.0;
    for (i32 i = 0; i < n - 1; i++) s += dist[tour[i] * n + tour[i + 1]];
    s += dist[tour[n - 1] * n + tour[0]];
    return s;
}

/* Double-bridge 4-opt perturbation: A + C + B + D. dst must differ from src. */
static void double_bridge(const i32 *src, i32 *dst, i32 n, rng_state *rng) {
    i32 p1, p2, p3;
    p1 = 1 + rng_below(rng, n - 1);
    do { p2 = 1 + rng_below(rng, n - 1); } while (p2 == p1);
    do { p3 = 1 + rng_below(rng, n - 1); } while (p3 == p1 || p3 == p2);
    /* sort p1 < p2 < p3 */
    i32 t;
    if (p1 > p2) { t = p1; p1 = p2; p2 = t; }
    if (p2 > p3) { t = p2; p2 = p3; p3 = t; }
    if (p1 > p2) { t = p1; p1 = p2; p2 = t; }
    i32 o = 0;
    for (i32 i = 0;  i < p1; i++) dst[o++] = src[i];  /* A */
    for (i32 i = p2; i < p3; i++) dst[o++] = src[i];  /* C */
    for (i32 i = p1; i < p2; i++) dst[o++] = src[i];  /* B */
    for (i32 i = p3; i < n;  i++) dst[o++] = src[i];  /* D */
}

/* Combined local search to joint convergence: 2-opt + Or-opt (+ optional
 * 3-opt). Operates in place; pos must be valid on entry. */
static void local_search(
    i32 *tour, i32 *pos, const f64 *dist, const i32 *neighbours,
    i32 n, i32 k, const i32 *chain_lengths, i32 n_chains,
    i32 max_outer, int use_3opt
) {
    f64 last = tour_cost(tour, dist, n);
    for (i32 it = 0; it < max_outer; it++) {
        fls_two_opt(tour, pos, dist, neighbours, n, k, 200);
        fls_or_opt(tour, pos, dist, neighbours, n, k, chain_lengths, n_chains, 200);
        if (use_3opt)
            fls_three_opt(tour, pos, dist, neighbours, n, k, 200);
        f64 c = tour_cost(tour, dist, n);
        if (c >= last - 1e-9) break;
        last = c;
    }
}

/*
 * ils_run — run a full Iterated Local Search chain in C.
 *
 *   tour          int32[n]  in/out; on return holds the best tour found.
 *   dist          double[n*n]
 *   neighbours    int32[n*k] (ascending by distance per row)
 *   n, k          sizes
 *   chain_lengths int32[n_chains]  Or-opt segment lengths
 *   iterations    number of double-bridge ILS kicks
 *   seed          PRNG seed (per chain)
 *   init_kick     if non-zero, diversify the starting tour with 2 kicks
 *   use_3opt      if non-zero, include 3-opt in the LS neighbourhood
 */
void ils_run(
    i32 *tour,
    const f64 *dist,
    const i32 *neighbours,
    i32 n,
    i32 k,
    const i32 *chain_lengths,
    i32 n_chains,
    i32 iterations,
    uint64_t seed,
    i32 init_kick,
    i32 use_3opt
) {
    rng_state rng;
    rng_seed(&rng, seed);

    i32 *pos  = (i32 *) malloc(sizeof(i32) * n);
    i32 *cur  = (i32 *) malloc(sizeof(i32) * n);
    i32 *best = (i32 *) malloc(sizeof(i32) * n);
    i32 *tmp  = (i32 *) malloc(sizeof(i32) * n);

    if (init_kick && n >= 8) {
        double_bridge(tour, tmp, n, &rng);
        double_bridge(tmp, cur, n, &rng);
    } else {
        memcpy(cur, tour, sizeof(i32) * n);
    }
    for (i32 i = 0; i < n; i++) pos[cur[i]] = i;
    local_search(cur, pos, dist, neighbours, n, k, chain_lengths, n_chains, 30, use_3opt);

    memcpy(best, cur, sizeof(i32) * n);
    f64 best_cost = tour_cost(best, dist, n);

    for (i32 it = 0; it < iterations; it++) {
        double_bridge(best, cur, n, &rng);
        for (i32 i = 0; i < n; i++) pos[cur[i]] = i;
        local_search(cur, pos, dist, neighbours, n, k, chain_lengths, n_chains, 30, use_3opt);
        f64 c = tour_cost(cur, dist, n);
        if (c < best_cost - 1e-9) {
            memcpy(best, cur, sizeof(i32) * n);
            best_cost = c;
        }
    }

    memcpy(tour, best, sizeof(i32) * n);
    free(pos); free(cur); free(best); free(tmp);
}
