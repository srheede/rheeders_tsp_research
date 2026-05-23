"""
Utilities for loading TSPLIB instances and their known-optimal tours.

Returns plain numpy arrays so algorithm modules have no dependency on tsplib95.
"""

from __future__ import annotations
import os
import numpy as np
import tsplib95


def load_instance(
    name: str,
    tsp_dir: str,
) -> tuple[np.ndarray, np.ndarray, float | None, list[int] | None]:
    """Load a TSPLIB instance by stem name (e.g. ``"berlin52"``).

    Parameters
    ----------
    name:
        Filename stem without extension.
    tsp_dir:
        Directory that contains ``<name>.tsp`` and optionally ``<name>.opt.tour``.

    Returns
    -------
    coords:
        N×2 float array of (x, y) positions, 0-based row = node index.
    distance_matrix:
        N×N float64 array, 0-based.
    optimal_cost:
        Total length of the known-optimal tour, or ``None`` if no
        ``.opt.tour`` file is present.
    optimal_tour:
        0-based node order of the known-optimal tour, or ``None`` if no
        ``.opt.tour`` file is present.
    """
    tsp_path = os.path.join(tsp_dir, f"{name}.tsp")
    problem = tsplib95.load(tsp_path)
    n = problem.dimension

    # Build 0-based coordinate array.
    # Prefer node_coords; fall back to display_data (used by some EXPLICIT instances).
    coord_source = problem.node_coords if problem.node_coords else problem.display_data
    coords: np.ndarray | None = None
    if coord_source:
        coords = np.array([coord_source[i + 1] for i in range(n)], dtype=float)

    # Build distance matrix via tsplib95 so edge-weight type is respected.
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = problem.get_weight(i + 1, j + 1)

    # Load optimal tour if the .opt.tour file exists.
    optimal_cost: float | None = None
    optimal_tour: list[int] | None = None
    opt_path = os.path.join(tsp_dir, f"{name}.opt.tour")
    if os.path.exists(opt_path):
        opt_problem = tsplib95.load(opt_path)
        opt_tour_1based = opt_problem.tours[0]           # 1-based node IDs
        optimal_tour = [node - 1 for node in opt_tour_1based]  # → 0-based
        optimal_cost = float(
            sum(
                dist[optimal_tour[i]][optimal_tour[(i + 1) % n]]
                for i in range(n)
            )
        )

    return coords, dist, optimal_cost, optimal_tour
