"""
Fixed benchmark test suite.

These instances are always used for benchmarking — never changed — so that
results from different algorithm variants are directly comparable over time.

All selected instances have a known optimal tour (.opt.tour) in datasets/tsp/,
enabling optimality gap measurement. Sizes range from 22 to 280 nodes to give
coverage across problem scales.
"""

# Instances from datasets/tsp/ to run every variant against.
# Format: filename stem (no extension).
TEST_INSTANCES = [
    "ulysses22",   # 22 nodes  — GEO coords
    "bayg29",      # 29 nodes  — EXPLICIT, display_data coords
    "att48",       # 48 nodes  — ATT coords
    "eil51",       # 51 nodes  — EUC_2D
    "berlin52",    # 52 nodes  — EUC_2D
    "st70",        # 70 nodes  — EUC_2D
    "eil76",       # 76 nodes  — EUC_2D
    "gr96",        # 96 nodes  — GEO coords
    "kroA100",     # 100 nodes — EUC_2D
    "lin105",      # 105 nodes — EUC_2D
    "ch130",       # 130 nodes — EUC_2D
    "tsp225",      # 225 nodes — EUC_2D
    "a280",        # 280 nodes — EUC_2D
]

# Directory (relative to project root) where .tsp and .opt.tour files live.
TSP_DIR = "datasets/tsp"
