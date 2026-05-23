"""
Load per-variant JSON result files and print a comparison table.

Can be called directly:
    python -m benchmark.report

Or via the runner:
    python -m benchmark.runner --compare
"""

from __future__ import annotations
import json
import os
import sys


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "results")


def load_all_results(results_dir: str) -> dict[str, dict]:
    """Return a dict of {variant_name: result_dict} for all stored JSON files."""
    results = {}
    if not os.path.isdir(results_dir):
        return results
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                data = json.load(f)
            results[data["variant"]] = data
    return results


def print_comparison_table(results_dir: str, instance_order: list[str]) -> None:
    """Print a formatted comparison table across all stored variants."""
    results = load_all_results(results_dir)
    if not results:
        print("No results found in results/. Run a variant first.")
        return

    variants = sorted(results.keys())
    col_w = max(14, max(len(v) for v in variants) + 2)

    # Header
    header = f"{'Instance':<18}" + "".join(f"{v:>{col_w}}" for v in variants)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    # Per-instance rows
    for name in instance_order:
        row = f"{name:<18}"
        for v in variants:
            inst = results[v]["instances"].get(name)
            if inst is None:
                row += f"{'—':>{col_w}}"
            elif inst["gap_pct"] is None:
                row += f"{'(no opt)':>{col_w}}"
            else:
                row += f"{inst['gap_pct']:>{col_w-1}.2f}%"
        print(row)

    print("-" * len(header))

    # Summary row: average gap
    avg_row = f"{'AVERAGE gap%':<18}"
    for v in variants:
        s = results[v].get("summary", {})
        avg = s.get("avg_gap_pct")
        avg_row += f"{avg:>{col_w-1}.2f}%" if avg is not None else f"{'—':>{col_w}}"
    print(avg_row)

    # Summary row: total time
    time_row = f"{'Total time (s)':<18}"
    for v in variants:
        s = results[v].get("summary", {})
        t = s.get("total_time_sec")
        time_row += f"{t:>{col_w-1}.3f}s" if t is not None else f"{'—':>{col_w}}"
    print(time_row)

    print("=" * len(header))

    # Timestamps
    print("\nResult timestamps:")
    for v in variants:
        ts = results[v].get("timestamp", "unknown")
        print(f"  {v}: {ts}")
    print()


def main() -> None:
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from config.test_suite import TEST_INSTANCES
    print_comparison_table(RESULTS_DIR, TEST_INSTANCES)


if __name__ == "__main__":
    main()
