"""
Algorithm protocol for TSP R&D variants.

Shared hull construction lives in ``algorithms/convex_hull.py`` and is applied
before every variant.  Variants only implement the insertion phase:

    solve_from_hull(hull, remaining, distance_matrix, coords) -> list[int]
    solve_from_hull_traced(...) -> tuple[list[int], list[TraceStep]]

Benchmarks and the visualizer call ``algorithms.pipeline`` which orchestrates
hull + variant.  Node indices are always 0-based.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class HullStep:
    """One node added to the shared convex hull chain."""

    node: int
    action: str
    removed_edge: tuple[int, int] | None = None
    new_edges: list[tuple[int, int]] = field(default_factory=list)
    description: str = ""

    @property
    def phase(self) -> str:
        return "hull"


@dataclass
class TraceStep:
    """One insertion decision made by a variant after hull construction."""

    node: int
    inserted_after: int
    removed_edge: tuple[int, int] | None = None
    new_edges: list[tuple[int, int]] = field(default_factory=list)
    description: str = ""

    @property
    def phase(self) -> str:
        return "insertion"

    @property
    def action(self) -> str:
        return "insert_node"


def compute_tour_cost(tour: list[int], distance_matrix) -> float:
    """Return the total cost of a closed tour."""
    n = len(tour)
    return sum(distance_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))
