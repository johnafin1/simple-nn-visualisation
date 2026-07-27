"""Live loss plot that tails a run's metrics.jsonl while training is in progress.

Reads runs/<run_id>/metrics.jsonl incrementally (byte offset, partial-line safe) and
redraws: loss per split, weight_norm / grad_norm, and - for a classification run -
accuracy per split. Weight norm is the diagnostic that matters for grokking, so it gets
its own panel; accuracy is where generalisation (or its absence) shows up.

Every split the log mentions gets its own line, so the primes run's two unseen ranges
appear automatically alongside train and test.

Run training in one terminal and this in another:
  .\\build\\src\\app\\train_x2\\train_x2.exe --steps 300000 --name live
  .\\.venv\\Scripts\\python.exe src/app/python/live_plot.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "runs"
MAX_POINTS = 5000  # plotted points per series; keeps long runs responsive

# Split order for legends and colours. Anything unrecognised is appended in the order
# it first appears in the log.
SPLIT_ORDER = ["train", "test"]


def find_latest_run(runs_dir: Path) -> Path | None:
    """Newest run directory (by modification time), ignoring the demo snapshot dir."""
    if not runs_dir.is_dir():
        return None
    candidates = [
        d for d in runs_dir.iterdir() if d.is_dir() and (d / "metrics.jsonl").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def resolve_run(arg: str | None) -> Path | None:
    """Resolve --run (a run_id or a path) or fall back to the newest run."""
    if arg is None:
        return find_latest_run(RUNS_DIR)
    as_path = Path(arg)
    if as_path.is_dir():
        return as_path
    candidate = RUNS_DIR / arg
    return candidate if candidate.is_dir() else None


def read_config(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


@dataclass
class SplitSeries:
    """One split's metric history."""

    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    accuracy_steps: list[int] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.steps.clear()
        self.loss.clear()
        self.accuracy_steps.clear()
        self.accuracy.clear()


@dataclass
class Series:
    """Metric series accumulated from the log, keyed by split name."""

    splits: dict[str, SplitSeries] = field(default_factory=dict)
    weight_norm: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)

    def clear(self) -> None:
        for s in self.splits.values():
            s.clear()
        self.weight_norm.clear()
        self.grad_norm.clear()

    def add(self, record: dict) -> None:
        split = record.get("split")
        step = record.get("step")
        loss = record.get("loss")
        if split is None or step is None or loss is None:
            return
        series = self.splits.setdefault(split, SplitSeries())
        series.steps.append(int(step))
        series.loss.append(float(loss))
        if "accuracy" in record:
            series.accuracy_steps.append(int(step))
            series.accuracy.append(float(record["accuracy"]))
        # Norms are only present on train rows.
        if "weight_norm" in record:
            self.weight_norm.append(float(record["weight_norm"]))
        if "grad_norm" in record:
            self.grad_norm.append(float(record["grad_norm"]))

    def ordered_names(self) -> list[str]:
        known = [n for n in SPLIT_ORDER if n in self.splits]
        rest = sorted(n for n in self.splits if n not in SPLIT_ORDER)
        return known + rest


class JsonlTailer:
    """Incremental JSONL reader.

    Tracks a byte offset and buffers any trailing fragment, so a line that is only
    half-written when we poll is never parsed. Detects truncation (the Trainer opens
    log files with trunc) and resets.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._offset = 0
        self._buffer = ""

    def poll(self) -> tuple[list[dict], bool]:
        """Return (new records, reset_happened)."""
        if not self.path.is_file():
            return [], False

        size = self.path.stat().st_size
        reset = False
        if size < self._offset:  # file was replaced/truncated
            self._offset = 0
            self._buffer = ""
            reset = True
        if size == self._offset:
            return [], reset

        with self.path.open("r", encoding="utf-8") as fh:
            fh.seek(self._offset)
            chunk = fh.read()
            self._offset = fh.tell()

        self._buffer += chunk
        *complete, self._buffer = self._buffer.split("\n")

        records = []
        for line in complete:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn line we could not reassemble; skip rather than crash.
                continue
        return records, reset


def read_status(run_dir: Path) -> str:
    meta = run_dir / "meta.json"
    if not meta.is_file():
        return "unknown"
    try:
        with meta.open("r", encoding="utf-8") as fh:
            return json.load(fh).get("status", "unknown")
    except (json.JSONDecodeError, OSError):
        return "unknown"


def decimate(xs: list, ys: list) -> tuple[list, list]:
    """Thin a series down to about MAX_POINTS for drawing."""
    n = len(xs)
    if n <= MAX_POINTS:
        return xs, ys
    stride = n // MAX_POINTS + 1
    return xs[::stride], ys[::stride]


def apply_scale(ax, values: list[float], logy: bool) -> None:
    """Log scale is only valid for strictly positive data."""
    if logy and values and min(values) > 0:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")


def build_plot(run_dir: Path, logx: bool):
    """Create the figure and return (fig, update) so the animation and the static
    snapshot path share exactly the same drawing code."""
    series = Series()
    tailer = JsonlTailer(run_dir / "metrics.jsonl")
    classification = bool(read_config(run_dir).get("classification", False))

    n_panels = 3 if classification else 2
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(9, 3.2 * n_panels), sharex=True, squeeze=False
    )
    ax_loss = axes[0][0]
    ax_norm = axes[1][0]
    ax_acc = axes[2][0] if classification else None

    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)

    (wnorm_line,) = ax_norm.plot([], [], label="weight_norm", linewidth=1.2)
    ax_norm.set_ylabel("weight_norm")
    ax_norm.grid(True, alpha=0.3)
    ax_grad = ax_norm.twinx()
    (gnorm_line,) = ax_grad.plot(
        [], [], label="grad_norm", linewidth=1.0, color="tab:green"
    )
    ax_grad.set_ylabel("grad_norm")
    ax_norm.legend(handles=[wnorm_line, gnorm_line], loc="upper right")

    if ax_acc is not None:
        ax_acc.set_ylabel("accuracy")
        ax_acc.set_ylim(0.0, 1.02)
        ax_acc.grid(True, alpha=0.3)
    axes[-1][0].set_xlabel("step")

    if logx:
        for row in axes:
            row[0].set_xscale("symlog")

    # Lines are created lazily: we do not know the split names until the log arrives.
    loss_lines: dict[str, object] = {}
    acc_lines: dict[str, object] = {}

    def line_for(ax, cache: dict, name: str, order: int):
        if name not in cache:
            (cache[name],) = ax.plot(
                [], [], label=name, linewidth=1.2, color=f"C{order % 10}"
            )
            ax.legend(loc="upper right", fontsize="small")
        return cache[name]

    def update(_frame=None):
        records, reset = tailer.poll()
        if reset:
            series.clear()
        for record in records:
            series.add(record)

        all_losses: list[float] = []
        for order, name in enumerate(series.ordered_names()):
            s = series.splits[name]
            xs, ys = decimate(s.steps, s.loss)
            line_for(ax_loss, loss_lines, name, order).set_data(xs, ys)
            all_losses.extend(s.loss)
            if ax_acc is not None and s.accuracy:
                axs, ays = decimate(s.accuracy_steps, s.accuracy)
                line_for(ax_acc, acc_lines, name, order).set_data(axs, ays)

        apply_scale(ax_loss, all_losses, logy=True)
        ax_loss.relim()
        ax_loss.autoscale_view()

        train = series.splits.get("train", SplitSeries())
        wx, wy = decimate(train.steps, series.weight_norm)
        wnorm_line.set_data(wx[: len(wy)], wy)
        ax_norm.relim()
        ax_norm.autoscale_view()

        gx, gy = decimate(train.steps, series.grad_norm)
        gnorm_line.set_data(gx[: len(gy)], gy)
        apply_scale(ax_grad, series.grad_norm, logy=True)
        ax_grad.relim()
        ax_grad.autoscale_view()

        if ax_acc is not None:
            ax_acc.relim()
            ax_acc.autoscale_view(scaley=False)

        status = read_status(run_dir)
        latest = f"step {train.steps[-1]}" if train.steps else "waiting for data"
        loss_txt = f", loss {train.loss[-1]:.6g}" if train.loss else ""
        acc_txt = f", acc {train.accuracy[-1]:.3f}" if train.accuracy else ""
        fig.suptitle(f"{run_dir.name}  [{status}]  {latest}{loss_txt}{acc_txt}")
        return tuple(loss_lines.values()) + (wnorm_line, gnorm_line)

    return fig, update


def main() -> None:
    parser = argparse.ArgumentParser(description="Live training plot from JSONL logs.")
    parser.add_argument("--run", default=None, help="run_id or path (default: newest run)")
    parser.add_argument("--interval", type=float, default=0.5, help="redraw seconds")
    parser.add_argument("--logx", action="store_true", help="log-scale the step axis")
    parser.add_argument(
        "--snapshot", default=None, help="render current state to this PNG and exit"
    )
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    if run_dir is None:
        raise SystemExit(
            f"No run found under {RUNS_DIR}.\n"
            "Start a run first, e.g.:\n"
            "  .\\build\\src\\app\\train_x2\\train_x2.exe --steps 300000 --name live"
        )
    print(f"Tailing {run_dir}")

    fig, update = build_plot(run_dir, logx=args.logx)

    if args.snapshot:
        update()
        fig.tight_layout()
        fig.savefig(args.snapshot, dpi=120)
        print(f"Wrote {args.snapshot}")
        return

    # Held in a local so the animation is not garbage collected before plt.show().
    anim = FuncAnimation(
        fig,
        update,
        interval=int(args.interval * 1000),
        cache_frame_data=False,
    )
    fig.tight_layout()
    plt.show()
    del anim


if __name__ == "__main__":
    main()
