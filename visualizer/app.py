"""
Step-through tour construction visualizer.

Usage:
    python -m visualizer.app v01_baseline berlin52
    python -m visualizer.app v01_baseline lu980
"""

from __future__ import annotations
import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmark.loader import load_instance
from algorithms.pipeline import solve_variant_traced
from algorithms.protocol import HullStep, TraceStep
from config.test_suite import TSP_DIR


Step = HullStep | TraceStep


class _EdgeBuffer:
    def __init__(self, coords: np.ndarray, capacity: int) -> None:
        self._coords = coords
        self._a = np.empty(capacity, dtype=np.int32)
        self._b = np.empty(capacity, dtype=np.int32)
        self._segments = np.empty((capacity, 2, 2), dtype=np.float32)
        self._index: dict[tuple[int, int], int] = {}
        self._size = 0

    def clear(self) -> None:
        self._index.clear()
        self._size = 0

    def add(self, edge: tuple[int, int]) -> None:
        if edge in self._index:
            return
        i = self._size
        a, b = edge
        self._a[i] = a
        self._b[i] = b
        self._segments[i, 0] = self._coords[a]
        self._segments[i, 1] = self._coords[b]
        self._index[edge] = i
        self._size += 1

    def remove(self, edge: tuple[int, int]) -> None:
        if edge not in self._index:
            return
        i = self._index.pop(edge)
        last = self._size - 1
        if i != last:
            self._a[i] = self._a[last]
            self._b[i] = self._b[last]
            self._segments[i] = self._segments[last]
            moved = (int(self._a[i]), int(self._b[i]))
            self._index[moved] = i
        self._size -= 1

    def segments(self) -> np.ndarray:
        return self._segments[: self._size]


class TourVisualizer:
    _AUTOPLAY_INTERVAL = 0.15
    _TICK_MS = 50

    def __init__(
        self,
        coords: np.ndarray,
        steps: list[Step],
        instance_name: str,
        variant_name: str,
    ) -> None:
        self.coords = coords.astype(np.float32, copy=False)
        self.steps = steps
        self.instance_name = instance_name
        self.variant_name = variant_name
        self.n_nodes = len(coords)
        self.current = 0
        self._autoplay = False
        self._autoplay_speed = self._AUTOPLAY_INTERVAL
        self._last_advance = 0.0
        self._timer = None

        self._blue_edges = _EdgeBuffer(self.coords, max(len(steps) * 2, 16))
        self._highlight_removed: tuple[int, int] | None = None
        self._highlight_new: set[tuple[int, int]] = set()

        pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.05 or 1.0
        pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.05 or 1.0
        self._xlim = (coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
        self._ylim = (coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

        self._large_map = self.n_nodes > 200
        if self.n_nodes > 1000:
            self._node_size = 2.0
            self._path_width = 0.5
        elif self._large_map:
            self._node_size = 5.0
            self._path_width = 0.8
        else:
            self._node_size = 30.0
            self._path_width = 1.2

        self._build_figure()
        self._set_step(0, jump=True)
        self._start_timer()

    def _build_figure(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.subplots_adjust(bottom=0.18)

        title = f"{self.variant_name}  ·  {self.instance_name}  ({self.n_nodes} nodes)"
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(title)

        self.ax.set_xlim(self._xlim)
        self.ax.set_ylim(self._ylim)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.axis("off")

        # Static node layer
        self._nodes_artist = self.ax.scatter(
            self.coords[:, 0], self.coords[:, 1],
            s=self._node_size, color="#aaaaaa", linewidths=0, zorder=2,
        )

        # Node labels — static on small maps, updated highlight on large maps
        self._label_artists: dict[int, plt.Text] = {}
        if not self._large_map:
            for i, (x, y) in enumerate(self.coords):
                self._label_artists[i] = self.ax.text(
                    x, y, str(i), fontsize=6, color="#888888",
                    ha="left", va="bottom", zorder=3,
                )

        empty = np.empty((0, 2, 2), dtype=np.float32)
        self._blue_lc = LineCollection(
            empty, colors="#4477cc", linewidths=self._path_width, zorder=3,
        )
        self._removed_lc = LineCollection(
            empty, colors="#cc2222", linewidths=2.0, linestyles="--", zorder=4,
        )
        self._new_lc = LineCollection(
            empty, colors="#ff8800", linewidths=2.4, zorder=5,
        )
        self._active_artist = self.ax.scatter(
            [], [], s=max(self._node_size * 4, 80), color="#dd2222", zorder=6, linewidths=0,
        )

        self.ax.add_collection(self._blue_lc)
        self.ax.add_collection(self._removed_lc)
        self.ax.add_collection(self._new_lc)

        # Phase badge — colour updated each step
        self._phase_badge = mpatches.FancyBboxPatch(
            (0.01, 0.91), 0.18, 0.07,
            transform=self.ax.transAxes,
            boxstyle="round,pad=0.01",
            facecolor="#2266aa", edgecolor="none", alpha=0.85, zorder=7,
        )
        self.ax.add_patch(self._phase_badge)
        self._phase_text = self.ax.text(
            0.10, 0.945, "HULL", transform=self.ax.transAxes,
            fontsize=8, color="white", fontweight="bold",
            ha="center", va="center", zorder=8,
        )

        self._title_artist = self.ax.set_title("", fontsize=10, loc="left", pad=10)

        handles = [
            mpatches.Patch(color="#4477cc", label="Existing edge"),
            mpatches.Patch(color="#cc2222", label="Removed edge"),
            mpatches.Patch(color="#ff8800", label="New edge(s)"),
            mpatches.Patch(color="#dd2222", label="Added node"),
        ]
        self.ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.7)

        btn_y, btn_h = 0.04, 0.06
        self.btn_first = Button(self.fig.add_axes([0.05, btn_y, 0.08, btn_h]), "|◀")
        self.btn_prev = Button(self.fig.add_axes([0.14, btn_y, 0.08, btn_h]), "◀")
        self.btn_play = Button(self.fig.add_axes([0.23, btn_y, 0.10, btn_h]), "▶ Play")
        self.btn_next = Button(self.fig.add_axes([0.34, btn_y, 0.08, btn_h]), "▶|")
        self.btn_last = Button(self.fig.add_axes([0.43, btn_y, 0.08, btn_h]), "▶▶|")

        self.btn_first.on_clicked(lambda _e: self._go(0))
        self.btn_prev.on_clicked(lambda _e: self._step(-1))
        self.btn_play.on_clicked(lambda _e: self._toggle_autoplay())
        self.btn_next.on_clicked(lambda _e: self._step(1))
        self.btn_last.on_clicked(lambda _e: self._go(len(self.steps) - 1))

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _edges_to_segments(self, edges: set[tuple[int, int]]) -> np.ndarray:
        if not edges:
            return np.empty((0, 2, 2), dtype=np.float32)
        arr = np.array(list(edges), dtype=np.intp)
        a, b = arr[:, 0], arr[:, 1]
        return np.stack([self.coords[a], self.coords[b]], axis=1)

    def _start_timer(self) -> None:
        self._stop_timer()
        self._timer = self.fig.canvas.new_timer(interval=self._TICK_MS)
        self._timer.add_callback(self._on_timer)
        self._timer.start()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _on_timer(self) -> None:
        if not self._autoplay:
            return
        if time.monotonic() - self._last_advance < self._autoplay_speed:
            return
        if self.current >= len(self.steps) - 1:
            self._autoplay = False
            self._update_play_button()
            self.fig.canvas.draw_idle()
            return
        self._last_advance = time.monotonic()
        self._set_step(self.current + 1, from_autoplay=True)

    def _commit_previous_highlights(self) -> None:
        for edge in self._highlight_new:
            self._blue_edges.add(edge)
        self._highlight_new.clear()
        self._highlight_removed = None

    def _apply_forward(self, step: Step) -> None:
        self._commit_previous_highlights()
        self._highlight_removed = (
            tuple(step.removed_edge) if step.removed_edge else None
        )
        self._highlight_new = {tuple(e) for e in step.new_edges}
        if step.removed_edge:
            self._blue_edges.remove(tuple(step.removed_edge))

    def _apply_backward(self, step: Step) -> None:
        for edge in step.new_edges:
            self._blue_edges.remove(tuple(edge))
        if step.removed_edge:
            self._blue_edges.add(tuple(step.removed_edge))
        self._highlight_removed = (
            tuple(step.removed_edge) if step.removed_edge else None
        )
        self._highlight_new = {tuple(e) for e in step.new_edges}

    def _rebuild_state_to(self, target: int) -> None:
        self._blue_edges.clear()
        self._highlight_removed = None
        self._highlight_new.clear()
        for i in range(1, target + 1):
            self._apply_forward(self.steps[i])

    def _set_step(self, index: int, *, jump: bool = False, from_autoplay: bool = False) -> None:
        index = max(0, min(index, len(self.steps) - 1))

        if jump or abs(index - self.current) > 1:
            self._rebuild_state_to(index)
        elif index > self.current:
            self._apply_forward(self.steps[index])
        elif index < self.current:
            self._apply_backward(self.steps[self.current])

        self.current = index
        if not from_autoplay:
            self._autoplay = False
            self._last_advance = time.monotonic()
        self._refresh()

    def _refresh(self) -> None:
        step = self.steps[self.current]

        self._blue_lc.set_segments(self._blue_edges.segments())
        self._removed_lc.set_segments(
            self._edges_to_segments({self._highlight_removed})
            if self._highlight_removed else np.empty((0, 2, 2), dtype=np.float32)
        )
        self._new_lc.set_segments(self._edges_to_segments(self._highlight_new))

        if step.node >= 0:
            self._active_artist.set_offsets(self.coords[step.node : step.node + 1])
            self._active_artist.set_visible(True)
        else:
            self._active_artist.set_visible(False)

        # Update label colours on small maps
        if not self._large_map:
            for i, artist in self._label_artists.items():
                if i == step.node:
                    artist.set_color("#dd2222")
                    artist.set_fontweight("bold")
                    artist.set_fontsize(8)
                else:
                    artist.set_color("#888888")
                    artist.set_fontweight("normal")
                    artist.set_fontsize(6)

        phase_colors = {"hull": "#2266aa", "insertion": "#228822"}
        self._phase_badge.set_facecolor(phase_colors.get(step.phase, "#666666"))
        self._phase_text.set_text(step.phase.upper())

        total = len(self.steps)
        self._title_artist.set_text(
            f"Step {self.current + 1} / {total}  ·  {step.action}\n{step.description}"
        )

        self._update_play_button()
        self.fig.canvas.draw_idle()

    def _update_play_button(self) -> None:
        self.btn_play.label.set_text("⏸ Pause" if self._autoplay else "▶ Play")

    def _go(self, index: int) -> None:
        self._set_step(index, jump=True)

    def _step(self, delta: int) -> None:
        self._set_step(self.current + delta)

    def _toggle_autoplay(self) -> None:
        self._autoplay = not self._autoplay
        self._last_advance = time.monotonic()
        self._update_play_button()
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        key = event.key
        if key in ("right", "l"):
            self._step(1)
        elif key in ("left", "h"):
            self._step(-1)
        elif key == " ":
            self._toggle_autoplay()
        elif key == "home":
            self._go(0)
        elif key == "end":
            self._go(len(self.steps) - 1)
        elif key in ("+", "="):
            self._autoplay_speed = max(0.05, self._autoplay_speed - 0.05)
        elif key == "-":
            self._autoplay_speed = min(1.5, self._autoplay_speed + 0.05)
        elif key in ("q", "escape"):
            plt.close(self.fig)

    def _on_close(self, _event) -> None:
        self._stop_timer()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step-through visualizer for TSP algorithm variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("variant", help="Variant module stem, e.g. v01_baseline")
    parser.add_argument("instance", help="Instance name, e.g. berlin52")
    args = parser.parse_args()

    tsp_dir = os.path.join(_ROOT, TSP_DIR)
    map_dir = os.path.join(_ROOT, "datasets/map")
    print(f"Loading {args.instance} ...")
    if os.path.exists(os.path.join(map_dir, f"{args.instance}.tsp")):
        coords, dist, optimal_cost = load_instance(args.instance, map_dir)
    else:
        coords, dist, optimal_cost = load_instance(args.instance, tsp_dir)

    print(f"Running hull + {args.variant} on {args.instance} ({len(coords)} nodes) ...")
    if len(coords) > 200:
        print("  (trace generation may take a minute on large instances)")
    t0 = time.perf_counter()
    tour, steps = solve_variant_traced(args.variant, dist, coords)
    solve_time = time.perf_counter() - t0
    cost = sum(dist[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))

    hull_steps = sum(1 for s in steps if isinstance(s, HullStep))
    insert_steps = sum(1 for s in steps if isinstance(s, TraceStep))

    gap_str = ""
    if optimal_cost is not None:
        gap = (cost - optimal_cost) / optimal_cost * 100
        gap_str = f"  |  optimal: {optimal_cost:.1f}  |  gap: {gap:.2f}%"

    print(f"Tour cost: {cost:.1f}{gap_str}")
    print(f"Steps: {hull_steps} hull + {insert_steps} insertion = {len(steps)} total")
    print(f"Solve+trace: {solve_time:.1f}s")
    print("\nControls:  ← / → step  |  Space play  |  +/- speed  |  Q quit")

    TourVisualizer(coords, steps, args.instance, args.variant)
    plt.show()


if __name__ == "__main__":
    main()
