# TSP Algorithm R&D

Research and development workspace for iteratively improving the Rheeders TSP heuristic. The goal is **tour quality** (shortest route), not execution speed.

## Structure

```
tsp_rd/
├── algorithms/
│   ├── protocol.py       # HullStep + TraceStep dataclasses
│   ├── convex_hull.py    # Shared hull construction (fixed for all variants)
│   ├── pipeline.py       # Orchestrates hull + variant insertion
│   ├── v01_baseline.py   # Baseline cheapest-insertion variant
│   └── vNN_name.py       # New variants (insertion logic only)
├── benchmark/
│   ├── runner.py         # Run variants against the fixed test suite
│   └── report.py         # Compare stored results across variants
├── config/
│   └── test_suite.py     # Fixed list of benchmark instances
├── datasets/             # TSPLIB instances + known-optimal tours
├── results/              # Per-variant JSON result files
├── visualizer/
│   └── app.py            # Step-through hull + insertion viewer
└── analysis/
    └── compare.ipynb     # Notebook for plotting result comparisons
```

## Architecture

Every solve runs in two fixed stages:

1. **Hull construction** (`convex_hull.py`) — identical for all variants. Builds the outer boundary chain from node coordinates.
2. **Variant insertion** (`vNN_*.py`) — your experimental logic. Inserts remaining nodes into the hull tour.

Variants only implement `solve_from_hull()` and `solve_from_hull_traced()`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run benchmark for a single variant

```bash
python -m benchmark.runner v01_baseline
```

### Run all variants

```bash
python -m benchmark.runner --all
```

### Compare stored results (no re-running)

```bash
python -m benchmark.runner --compare
```

### Visualize hull construction + variant insertion

```bash
python -m visualizer.app v01_baseline berlin52
```

The visualizer shows hull steps first (shared), then variant insertion steps. Each step highlights the added node, the removed edge (red dashed), and the two new edges (orange).

## Adding a New Variant

1. Copy `algorithms/v01_baseline.py` to `algorithms/vNN_yourname.py`
2. Modify `solve_from_hull()` and `solve_from_hull_traced()` with your insertion logic
3. Run `python -m benchmark.runner vNN_yourname`
4. Compare: `python -m benchmark.runner --compare`

Do **not** reimplement hull construction in variants — it is shared and applied automatically.

## Variant Naming

Use the pattern `vNN_description`, e.g.:
- `v02_2opt_postprocess`
- `v03_nearest_neighbor_init`
- `v04_triangle_improvement`

## Benchmarking

All variants are measured against the same fixed set of TSPLIB instances defined in `config/test_suite.py`. The primary metric is **optimality gap**:

```
gap% = (cost - optimal_cost) / optimal_cost * 100
```

Results are stored in `results/vNN_name.json`. Old variants are never re-run unless explicitly requested.
