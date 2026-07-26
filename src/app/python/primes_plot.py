"""Prediction heatmap for the primality experiment.

One row per integer, one column per logged step, colour = P(prime). Two narrow gutters
on the left carry ground truth and split membership, so memorisation can be read
straight off the picture: train rows converge to the correct colour while test and
unseen rows stay muddled around the base rate.

Reads runs/<run_id>/predictions.jsonl, which the Trainer writes as
    {"step":N,"split":"test","id":97,"target":1,"pred":0.83}

Live, while training runs in another terminal:
  .\\.venv\\Scripts\\python.exe src/app/python/primes_plot.py --live

Static PNG of whatever has been logged so far:
  .\\.venv\\Scripts\\python.exe src/app/python/primes_plot.py --snapshot runs/primes.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, to_rgb

from live_plot import RUNS_DIR, JsonlTailer, read_status, resolve_run

# Ground-truth gutter: light = composite, dark = prime.
TRUTH_CMAP = ListedColormap(["#f0f0f0", "#1a1a1a"])
# Split gutter. Unseen ranges get their own colours in first-seen order.
SPLIT_COLOURS = {"train": "#1f77b4", "test": "#ff7f0e"}
FALLBACK_COLOURS = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
MAX_COLUMNS = 800  # drawn columns; longer runs are thinned by striding


@dataclass
class PredictionGrid:
    """Accumulates predictions.jsonl into a (integer x step) matrix.

    Rows are integers in ascending order, columns are logged steps in ascending order,
    cells are P(prime). Cells with no record yet stay NaN and draw as blank.
    """

    # id -> step -> prediction
    by_id: dict[int, dict[int, float]] = field(default_factory=dict)
    target: dict[int, float] = field(default_factory=dict)
    split_of: dict[int, str] = field(default_factory=dict)
    steps: set[int] = field(default_factory=set)
    split_order: list[str] = field(default_factory=list)

    def clear(self) -> None:
        self.by_id.clear()
        self.target.clear()
        self.split_of.clear()
        self.steps.clear()
        self.split_order.clear()

    def add(self, record: dict) -> None:
        try:
            n = int(round(float(record["id"])))
            step = int(record["step"])
            pred = float(record["pred"])
        except (KeyError, TypeError, ValueError):
            return
        split = str(record.get("split", "?"))
        self.by_id.setdefault(n, {})[step] = pred
        self.target[n] = float(record.get("target", 0.0))
        self.split_of[n] = split
        self.steps.add(step)
        if split not in self.split_order:
            self.split_order.append(split)

    @property
    def empty(self) -> bool:
        return not self.steps or not self.by_id

    def ids(self) -> list[int]:
        return sorted(self.by_id)

    def columns(self) -> list[int]:
        cols = sorted(self.steps)
        if len(cols) > MAX_COLUMNS:
            stride = len(cols) // MAX_COLUMNS + 1
            # Always keep the newest column: it is the one being watched.
            cols = cols[::stride] + ([cols[-1]] if cols[-1] not in cols[::stride] else [])
        return cols

    def matrix(self, ids: list[int], cols: list[int]) -> np.ndarray:
        grid = np.full((len(ids), len(cols)), np.nan)
        col_index = {step: j for j, step in enumerate(cols)}
        for i, n in enumerate(ids):
            for step, pred in self.by_id[n].items():
                j = col_index.get(step)
                if j is not None:
                    grid[i, j] = pred
        return grid

    def accuracy_at_last_step(self) -> dict[str, tuple[int, int]]:
        """Per split, (correct, total) at each integer's most recent prediction."""
        tally: dict[str, list[int]] = {}
        for n, history in self.by_id.items():
            if not history:
                continue
            last_step = max(history)
            predicted_prime = history[last_step] >= 0.5
            is_prime = self.target.get(n, 0.0) >= 0.5
            slot = tally.setdefault(self.split_of.get(n, "?"), [0, 0])
            slot[1] += 1
            if predicted_prime == is_prime:
                slot[0] += 1
        return {k: (v[0], v[1]) for k, v in tally.items()}


def split_colour(name: str, order: int) -> str:
    if name in SPLIT_COLOURS:
        return SPLIT_COLOURS[name]
    return FALLBACK_COLOURS[order % len(FALLBACK_COLOURS)]


def build_plot(run_dir: Path):
    """Create the figure and return (fig, update), shared by the live and snapshot
    paths so both render identically."""
    grid = PredictionGrid()
    tailer = JsonlTailer(run_dir / "predictions.jsonl")

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1, 1, 40, 1.5], wspace=0.06, left=0.09, right=0.95
    )
    ax_truth = fig.add_subplot(gs[0, 0])
    ax_split = fig.add_subplot(gs[0, 1], sharey=ax_truth)
    ax_heat = fig.add_subplot(gs[0, 2], sharey=ax_truth)
    ax_cbar = fig.add_subplot(gs[0, 3])

    for ax, label in ((ax_truth, "truth"), (ax_split, "split")):
        ax.set_xticks([])
        ax.set_title(label, fontsize=8, rotation=90, loc="left")
    ax_split.tick_params(labelleft=False)
    ax_heat.tick_params(labelleft=False)
    ax_truth.set_ylabel("n")
    ax_heat.set_xlabel("training step")

    heat_im = ax_heat.imshow(
        np.zeros((1, 1)),
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="coolwarm",
        interpolation="nearest",
    )
    fig.colorbar(heat_im, cax=ax_cbar, label="P(prime)")

    truth_im = ax_truth.imshow(
        np.zeros((1, 1)),
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap=TRUTH_CMAP,
        interpolation="nearest",
    )
    split_im = ax_split.imshow(
        np.zeros((1, 1, 3)), aspect="auto", origin="lower", interpolation="nearest"
    )

    def update(_frame=None):
        records, reset = tailer.poll()
        if reset:
            grid.clear()
        for record in records:
            grid.add(record)

        status = read_status(run_dir)
        if grid.empty:
            fig.suptitle(f"{run_dir.name}  [{status}]  waiting for predictions")
            return heat_im, truth_im, split_im

        ids = grid.ids()
        cols = grid.columns()
        data = grid.matrix(ids, cols)

        # Columns are drawn at equal width (one per logged step) and the x axis is
        # labelled with the real step numbers. Scaling the axis by step value instead
        # would silently distort widths whenever the logging cadence changes.
        heat_im.set_data(data)
        heat_im.set_extent((-0.5, len(cols) - 0.5, ids[0], ids[-1]))
        tick_stride = max(1, len(cols) // 8)
        tick_idx = list(range(0, len(cols), tick_stride))
        if tick_idx[-1] != len(cols) - 1:
            tick_idx.append(len(cols) - 1)
        ax_heat.set_xticks(tick_idx)
        ax_heat.set_xticklabels([str(cols[i]) for i in tick_idx])
        ax_heat.set_xlim(-0.5, len(cols) - 0.5)

        truth_col = np.array([[grid.target.get(n, 0.0)] for n in ids])
        truth_im.set_data(truth_col)
        truth_im.set_extent((0, 1, ids[0], ids[-1]))

        rgb = np.zeros((len(ids), 1, 3))
        for i, n in enumerate(ids):
            name = grid.split_of.get(n, "?")
            order = grid.split_order.index(name) if name in grid.split_order else 0
            rgb[i, 0, :] = to_rgb(split_colour(name, order))
        split_im.set_data(rgb)
        split_im.set_extent((0, 1, ids[0], ids[-1]))

        ax_truth.set_ylim(ids[0], ids[-1])

        acc = grid.accuracy_at_last_step()
        summary = "  ".join(
            f"{name} {c}/{t} ({c / t:.0%})"
            for name, (c, t) in sorted(
                acc.items(),
                key=lambda kv: grid.split_order.index(kv[0])
                if kv[0] in grid.split_order
                else 99,
            )
            if t
        )
        legend_handles = [
            plt.Line2D(
                [], [], marker="s", linestyle="", markersize=8,
                color=split_colour(name, i), label=name
            )
            for i, name in enumerate(grid.split_order)
        ]
        ax_heat.legend(
            handles=legend_handles, loc="lower right", fontsize="small", framealpha=0.9
        )
        fig.suptitle(
            f"{run_dir.name}  [{status}]  step {cols[-1]}\n"
            f"accuracy at latest logged step:  {summary}",
            fontsize=10,
        )
        return heat_im, truth_im, split_im

    return fig, update


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Primality prediction heatmap from predictions.jsonl."
    )
    parser.add_argument("--run", default=None, help="run_id or path (default: newest run)")
    parser.add_argument("--live", action="store_true", help="poll and redraw while training")
    parser.add_argument("--interval", type=float, default=1.0, help="redraw seconds")
    parser.add_argument(
        "--snapshot", default=None, help="render current state to this PNG and exit"
    )
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    if run_dir is None:
        raise SystemExit(
            f"No run found under {RUNS_DIR}.\n"
            "Start a run first, e.g.:\n"
            "  .\\build\\src\\app\\train_primes\\train_primes.exe --name primes_bits"
        )
    if not (run_dir / "predictions.jsonl").is_file():
        raise SystemExit(f"{run_dir} has no predictions.jsonl")
    print(f"Reading {run_dir}")

    fig, update = build_plot(run_dir)

    if args.snapshot:
        update()
        fig.savefig(args.snapshot, dpi=120)
        print(f"Wrote {args.snapshot}")
        return

    if args.live:
        # Held in a local so the animation is not garbage collected before plt.show().
        anim = FuncAnimation(
            fig, update, interval=int(args.interval * 1000), cache_frame_data=False
        )
        plt.show()
        del anim
    else:
        update()
        plt.show()


if __name__ == "__main__":
    main()
