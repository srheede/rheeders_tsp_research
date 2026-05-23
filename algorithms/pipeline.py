"""
Orchestration layer: shared hull construction + variant-specific insertion.

Benchmarks and the visualizer call into this module rather than invoking
variant modules directly, so the hull step is always applied consistently.
"""

from __future__ import annotations

import importlib

import numpy as np

from algorithms.convex_hull import build_hull, build_hull_traced
from algorithms.protocol import TraceStep


def solve_variant(
    variant_name: str,
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> list[int]:
    """Run hull construction then the named variant's insertion logic."""
    if coords is None:
        raise ValueError("Coordinates are required for hull construction.")

    hull, remaining = build_hull(coords)
    module = importlib.import_module(f"algorithms.{variant_name}")
    return module.solve_from_hull(hull, remaining, distance_matrix, coords)


def solve_variant_traced(
    variant_name: str,
    distance_matrix: np.ndarray,
    coords: np.ndarray | None = None,
) -> tuple[list[int], list]:
    """Run traced solve; returns (tour, all_steps).

    Steps are a mixed list of HullStep (phase='hull') and TraceStep
    (phase='insertion') objects suitable for the visualizer.
    """
    if coords is None:
        raise ValueError("Coordinates are required for hull construction.")

    hull, remaining, hull_steps = build_hull_traced(coords)
    module = importlib.import_module(f"algorithms.{variant_name}")
    tour, insert_steps = module.solve_from_hull_traced(
        hull, remaining, distance_matrix, coords
    )
    return tour, hull_steps + insert_steps
