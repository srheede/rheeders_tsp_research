"""
_ls_native — ctypes wrapper around the C kernel ``_ls_inner.c``.

On import the module compiles ``_ls_inner.c`` once (caches the .so /
.dylib next to it). Provides:

    fast_local_search_c(tour, dist, neighbours,
                        or_opt_chain_lengths=(1,2,3,4,5),
                        max_outer=30) -> list[int]

with the same semantics as the pure-Python ``fast_local_search`` but
roughly 30-100× faster on n ≥ 250.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


_HERE = Path(__file__).parent
_SRC = _HERE / "_ls_inner.c"

if sys.platform == "darwin":
    _LIB_NAME = "_ls_inner.dylib"
elif sys.platform.startswith("linux"):
    _LIB_NAME = "_ls_inner.so"
elif sys.platform.startswith("win"):
    _LIB_NAME = "_ls_inner.dll"
else:
    _LIB_NAME = "_ls_inner.so"

_LIB_PATH = _HERE / _LIB_NAME


def _compile() -> None:
    """Compile the C kernel into a shared library."""
    cc = os.environ.get("CC", "cc")
    cmd = [
        cc, "-O3", "-ffast-math", "-shared", "-fPIC",
        str(_SRC), "-o", str(_LIB_PATH),
    ]
    subprocess.check_call(cmd)


def _ensure_lib() -> None:
    if not _LIB_PATH.exists() or _SRC.stat().st_mtime > _LIB_PATH.stat().st_mtime:
        _compile()


_ensure_lib()
_lib = ctypes.CDLL(str(_LIB_PATH))

# void fls_two_opt(int32 *tour, int32 *pos, double *dist,
#                  int32 *neighbours, int32 n, int32 k, int32 max_passes)
_lib.fls_two_opt.argtypes = [
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
]
_lib.fls_two_opt.restype = None

# void fls_or_opt(int32 *tour, int32 *pos, double *dist,
#                 int32 *neighbours, int32 n, int32 k,
#                 int32 *chain_lengths, int32 n_chains, int32 max_passes)
_lib.fls_or_opt.argtypes = [
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.c_int32,
]
_lib.fls_or_opt.restype = None

# void fls_three_opt(int32 *tour, int32 *pos, double *dist,
#                    int32 *neighbours, int32 n, int32 k, int32 max_passes)
_lib.fls_three_opt.argtypes = [
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
]
_lib.fls_three_opt.restype = None

# void ils_run(int32 *tour, double *dist, int32 *neighbours, int32 n, int32 k,
#              int32 *chain_lengths, int32 n_chains, int32 iterations,
#              uint64 seed, int32 init_kick, int32 use_3opt)
_lib.ils_run.argtypes = [
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_uint64,
    ctypes.c_int32,
    ctypes.c_int32,
]
_lib.ils_run.restype = None


def _prepare_inputs(tour: list[int], dist, neighbours):
    """Convert Python objects to contiguous numpy arrays for the C call."""
    n = len(tour)
    tour_arr = np.asarray(tour, dtype=np.int32).copy()
    pos_arr = np.empty(n, dtype=np.int32)
    for i, node in enumerate(tour):
        pos_arr[node] = i
    dist_arr = np.ascontiguousarray(dist, dtype=np.float64)
    nb_arr = np.asarray(neighbours, dtype=np.int32)
    if nb_arr.ndim == 1:
        # Already flat
        k = nb_arr.size // n
    else:
        k = nb_arr.shape[1]
        nb_arr = np.ascontiguousarray(nb_arr.reshape(-1), dtype=np.int32)
    return tour_arr, pos_arr, dist_arr, nb_arr, n, k


def fast_local_search_c(
    tour: list[int],
    dist,
    neighbours,
    or_opt_chain_lengths: tuple[int, ...] = (1, 2, 3, 4, 5),
    max_outer: int = 30,
) -> list[int]:
    """Same semantics as fast_local_search; uses the C kernel."""
    tour_arr, pos_arr, dist_arr, nb_arr, n, k = _prepare_inputs(
        tour, dist, neighbours
    )
    chain_arr = np.asarray(list(or_opt_chain_lengths), dtype=np.int32)

    last_cost = _tour_cost(tour_arr, dist_arr, n)
    for _ in range(max_outer):
        _lib.fls_two_opt(
            tour_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            pos_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            dist_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nb_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n, k, 200,
        )
        _lib.fls_or_opt(
            tour_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            pos_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            dist_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nb_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n, k,
            chain_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            len(chain_arr), 200,
        )
        new_cost = _tour_cost(tour_arr, dist_arr, n)
        if new_cost >= last_cost - 1e-9:
            break
        last_cost = new_cost
    return tour_arr.tolist()


def _tour_cost(tour_arr, dist_arr, n):
    s = 0.0
    for i in range(n):
        s += dist_arr[tour_arr[i], tour_arr[(i + 1) % n]]
    return s


# Persistent caches so repeated calls within one process don't re-flatten
# the (potentially large) distance matrix / neighbour lists.
_DIST_CACHE: dict[int, np.ndarray] = {}
_NB_CACHE: dict[int, tuple[np.ndarray, int]] = {}


def _get_dist_arr(dist):
    key = id(dist)
    cached = _DIST_CACHE.get(key)
    if cached is None:
        cached = np.ascontiguousarray(dist, dtype=np.float64)
        _DIST_CACHE.clear()  # keep at most one — matrices are big
        _DIST_CACHE[key] = cached
    return cached


def _get_nb_arr(neighbours, n):
    key = id(neighbours)
    cached = _NB_CACHE.get(key)
    if cached is None:
        nb = np.asarray(neighbours, dtype=np.int32)
        if nb.ndim == 1:
            k = nb.size // n
            flat = np.ascontiguousarray(nb)
        else:
            k = nb.shape[1]
            flat = np.ascontiguousarray(nb.reshape(-1))
        cached = (flat, k)
        _NB_CACHE.clear()
        _NB_CACHE[key] = cached
    return cached


def ils_run_c(
    tour: list[int],
    dist,
    neighbours,
    iterations: int,
    seed: int,
    init_kick: bool = False,
    or_opt_chain_lengths: tuple[int, ...] = (1, 2, 3, 4, 5),
    use_3opt: bool = False,
) -> list[int]:
    """Run a full Iterated Local Search chain entirely in C.

    The double-bridge perturbation, the 2-opt / Or-opt (optionally 3-opt)
    local search, the tour-cost evaluation and the accept-if-better logic
    all execute natively — Python is only touched once at the start (to
    marshal the arrays) and once at the end (to read the result back).
    This removes the O(n) Python cost loop and list↔array conversions that
    previously ran on every one of the millions of ILS iterations.
    """
    n = len(tour)
    tour_arr = np.asarray(tour, dtype=np.int32).copy()
    dist_arr = _get_dist_arr(dist)
    nb_flat, k = _get_nb_arr(neighbours, n)
    chain_arr = np.asarray(list(or_opt_chain_lengths), dtype=np.int32)

    _lib.ils_run(
        tour_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        dist_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        nb_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        n, k,
        chain_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        len(chain_arr),
        int(iterations),
        ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF),
        1 if init_kick else 0,
        1 if use_3opt else 0,
    )
    return tour_arr.tolist()
