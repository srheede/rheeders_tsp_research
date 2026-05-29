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
#
# This suite focuses on LARGER instances (150–666 nodes) — the suite of
# v01..v32 measurements lives in the json result files for the prior
# (22–280) suite. All instances below have known-optimal tours.
TEST_INSTANCES = [
    "a280",        #  280 nodes — EUC_2D    (sanity check, was 0% on v37)
    "pcb442",      #  442 nodes — EUC_2D    (was 0% on v37)
    "gr666",       #  666 nodes — GEO       (was 0.05% on v37)
    "pr1002",      # 1002 nodes — EUC_2D    (HUGE)
    "pr2392",      # 2392 nodes — EUC_2D    (HUGE)
]

# Directory (relative to project root) where .tsp and .opt.tour files live.
TSP_DIR = "datasets/tsp"
