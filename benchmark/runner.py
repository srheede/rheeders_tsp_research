"""
Benchmark runner.

Usage (from the project root):

    python -m benchmark.runner v01_baseline          # run one variant
    python -m benchmark.runner v01_baseline v02_foo  # run two variants
    python -m benchmark.runner --all                 # run every variant found
    python -m benchmark.runner --compare             # print table, no re-runs

Results are written to results/<variant_name>.json.  Re-running a variant
overwrites its file; other variants' files are never touched.
"""

from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

# Ensure the project root is on the path when invoked as a module.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from algorithms.pipeline import solve_variant
from benchmark.loader import load_instance
from benchmark.report import print_comparison_table
from config.test_suite import TEST_INSTANCES, TSP_DIR


RESULTS_DIR = os.path.join(_ROOT, "results")
ALGORITHMS_DIR = os.path.join(_ROOT, "algorithms")


# ---------------------------------------------------------------------------
# Variant discovery
# ---------------------------------------------------------------------------

def discover_variants() -> list[str]:
    """Return sorted list of variant stems found in algorithms/."""
    stems = []
    for fname in os.listdir(ALGORITHMS_DIR):
        if fname.startswith("v") and fname.endswith(".py") and fname[1].isdigit():
            stems.append(fname[:-3])
    return sorted(stems)


def load_variant(name: str):
    """Import and return the variant module by stem name."""
    try:
        return importlib.import_module(f"algorithms.{name}")
    except ModuleNotFoundError as exc:
        sys.exit(f"Error: could not import algorithms/{name}.py — {exc}")


# ---------------------------------------------------------------------------
# Single variant benchmark
# ---------------------------------------------------------------------------

def run_variant(variant_name: str, tsp_dir: str) -> dict:
    """Run variant against all test-suite instances, return results dict."""
    module = load_variant(variant_name)

    if not hasattr(module, "solve_from_hull"):
        sys.exit(
            f"Error: algorithms/{variant_name}.py has no solve_from_hull() function."
        )

    print(f"\nBenchmarking {variant_name} ...")
    print(f"  {'Instance':<16} {'Cost':>12} {'Optimal':>12} {'Gap %':>8} {'Time (s)':>10}")
    print("  " + "-" * 62)

    instance_results: dict[str, dict] = {}

    for instance_name in TEST_INSTANCES:
        coords, dist, optimal_cost = load_instance(instance_name, tsp_dir)

        t0 = time.perf_counter()
        tour = solve_variant(variant_name, dist, coords)
        elapsed = time.perf_counter() - t0

        cost = float(sum(dist[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour))))

        gap_pct: float | None = None
        if optimal_cost is not None:
            gap_pct = round((cost - optimal_cost) / optimal_cost * 100, 4)

        instance_results[instance_name] = {
            "cost": round(cost, 4),
            "optimal_cost": round(optimal_cost, 4) if optimal_cost is not None else None,
            "gap_pct": gap_pct,
            "time_sec": round(elapsed, 6),
            "n_nodes": len(tour),
        }

        gap_str = f"{gap_pct:.2f}%" if gap_pct is not None else "  n/a"
        opt_str = f"{optimal_cost:.1f}" if optimal_cost is not None else "   n/a"
        print(
            f"  {instance_name:<16} {cost:>12.1f} {opt_str:>12} "
            f"{gap_str:>8} {elapsed:>10.4f}"
        )

    # Aggregate summary across instances that have known optima.
    gaps = [v["gap_pct"] for v in instance_results.values() if v["gap_pct"] is not None]
    avg_gap = round(sum(gaps) / len(gaps), 4) if gaps else None
    total_time = round(sum(v["time_sec"] for v in instance_results.values()), 6)

    print(f"\n  Average gap: {avg_gap:.2f}%  |  Total time: {total_time:.3f}s")

    result = {
        "variant": variant_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instances": instance_results,
        "summary": {
            "avg_gap_pct": avg_gap,
            "total_time_sec": total_time,
            "instances_with_optimal": len(gaps),
        },
    }

    _save_result(variant_name, result)
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_result(variant_name: str, result: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{variant_name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → results/{variant_name}.json")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TSP R&D benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "variants",
        nargs="*",
        metavar="VARIANT",
        help="One or more variant names (e.g. v01_baseline v02_2opt). "
             "Omit when using --all or --compare.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every variant found in algorithms/.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print a comparison table from stored results (no re-running).",
    )

    args = parser.parse_args()

    tsp_dir = os.path.join(_ROOT, TSP_DIR)

    if args.compare:
        print_comparison_table(RESULTS_DIR, TEST_INSTANCES)
        return

    targets: list[str] = []
    if args.all:
        targets = discover_variants()
        if not targets:
            sys.exit("No variant modules found in algorithms/.")
        print(f"Found variants: {', '.join(targets)}")
    elif args.variants:
        targets = args.variants
    else:
        parser.print_help()
        return

    for name in targets:
        run_variant(name, tsp_dir)

    if len(targets) > 1:
        print("\n" + "=" * 70)
        print_comparison_table(RESULTS_DIR, TEST_INSTANCES)


if __name__ == "__main__":
    main()
