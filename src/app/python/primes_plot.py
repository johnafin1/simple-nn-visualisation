"""Live generalisation dashboard for primality experiments.

The former per-integer red/blue prediction heatmap did not scale beyond a few
hundred values. This dashboard tails metrics.jsonl instead and shows:

* whole-experiment progress across modulo pretraining and prime-head training;
* loss per phase/split;
* balanced accuracy plus prime/composite recall;
* the latest held-out confusion matrix;
* a conservative generalisation/grokking status updated at the C++ eval cadence.

Run while training:
  .\\.venv\\Scripts\\python.exe src/app/python/primes_plot.py --live
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from live_plot import RUNS_DIR, JsonlTailer, read_config, read_status, resolve_run

MAX_POINTS = 4000
SPLIT_ORDER = ["train", "validation", "test"]
SPLIT_COLOURS = {
    "train": "#355070",
    "validation": "#6d597a",
    "test": "#4f772d",
}


@dataclass
class MetricHistory:
    values: dict[tuple[str, str, str], list[tuple[int, float]]] = field(
        default_factory=dict
    )
    binary_by_split: dict[str, list[dict]] = field(default_factory=dict)
    phase_order: list[str] = field(default_factory=list)
    _latest_step: int | None = None
    _latest_phase: str = "waiting"

    def clear(self) -> None:
        self.values.clear()
        self.binary_by_split.clear()
        self.phase_order.clear()
        self._latest_step = None
        self._latest_phase = "waiting"

    def add(self, record: dict) -> None:
        if "step" not in record or "split" not in record:
            return
        try:
            step = int(record["step"])
        except (TypeError, ValueError):
            return
        phase = str(record.get("phase", "train"))
        split = str(record["split"])
        if phase not in self.phase_order:
            self.phase_order.append(phase)
        self._latest_step = (
            step if self._latest_step is None else max(self._latest_step, step)
        )
        self._latest_phase = phase
        for metric in (
            "loss",
            "accuracy",
            "balanced_accuracy",
            "prime_recall",
            "composite_recall",
            "precision",
        ):
            if metric not in record:
                continue
            try:
                value = float(record[metric])
            except (TypeError, ValueError):
                continue
            self.values.setdefault((phase, split, metric), []).append((step, value))
        if "balanced_accuracy" in record:
            self.binary_by_split.setdefault(split, []).append(record)

    def latest_step(self) -> int | None:
        return self._latest_step

    def latest_phase(self) -> str:
        return self._latest_phase

    def series(
        self, metric: str, *, phase: str | None = None, split: str | None = None
    ) -> tuple[list[int], list[float]]:
        points: list[tuple[int, float]] = []
        for (phase_name, split_name, metric_name), metric_points in self.values.items():
            if metric_name != metric:
                continue
            if phase is not None and phase_name != phase:
                continue
            if split is not None and split_name != split:
                continue
            points.extend(metric_points)
        points.sort(key=lambda point: point[0])
        if len(points) > MAX_POINTS:
            stride = len(points) // MAX_POINTS + 1
            points = points[::stride] + (
                [points[-1]] if points[-1] not in points[::stride] else []
            )
        return [p[0] for p in points], [p[1] for p in points]

    def phases(self) -> list[str]:
        return list(self.phase_order)

    def latest_binary(self, split: str) -> dict | None:
        records = self.binary_by_split.get(split, [])
        return records[-1] if records else None

    def binary_records(self, split: str) -> list[dict]:
        return self.binary_by_split.get(split, [])


def heldout_name(history: MetricHistory) -> str | None:
    for name in ("validation", "test"):
        if history.latest_binary(name) is not None:
            return name
    return None


def threshold_pass(record: dict) -> bool:
    return (
        float(record.get("balanced_accuracy", 0.0)) >= 0.90
        and float(record.get("prime_recall", 0.0)) >= 0.80
        and float(record.get("composite_recall", 0.0)) >= 0.80
    )


def generalisation_status(
    history: MetricHistory, eval_interval: int
) -> tuple[str, str, str | None]:
    """Return (headline, explanation, held-out split)."""
    train = history.latest_binary("train")
    heldout = heldout_name(history)
    validation = history.latest_binary(heldout) if heldout else None

    if train is None:
        return (
            "MODULO PRETRAINING",
            "Learning reusable cyclic residue features; prime generalisation has not been tested yet.",
            heldout,
        )
    if validation is None:
        return (
            "PRIME HEAD LEARNING",
            "The frozen encoder is being probed, but no held-out classification result exists yet.",
            heldout,
        )

    train_balanced = float(train.get("balanced_accuracy", 0.0))
    heldout_balanced = float(validation.get("balanced_accuracy", 0.0))
    heldout_history = history.binary_records(heldout)
    sustained = len(heldout_history) >= 3 and all(
        threshold_pass(record) for record in heldout_history[-3:]
    )

    if sustained:
        train_fit_step = next(
            (
                int(record["step"])
                for record in history.binary_records("train")
                if float(record.get("balanced_accuracy", 0.0)) >= 0.98
            ),
            None,
        )
        general_step = int(heldout_history[-3]["step"])
        if (
            train_fit_step is not None
            and general_step - train_fit_step >= max(3 * eval_interval, 3000)
        ):
            return (
                "CANDIDATE GROKKING",
                "Held-out performance crossed the generalisation threshold well after the training fit.",
                heldout,
            )
        return (
            "GENERALISING",
            "Balanced accuracy and both class recalls passed their thresholds for three evaluations.",
            heldout,
        )

    if train_balanced >= 0.98 and heldout_balanced <= 0.55:
        return (
            "MEMORISED, NOT GENERALISING",
            "Training is almost perfect while held-out balanced accuracy remains near chance.",
            heldout,
        )
    if heldout_balanced > 0.55:
        return (
            "PARTIALLY GENERALISING",
            "Held-out balanced accuracy is above chance but has not met the sustained threshold.",
            heldout,
        )
    return (
        "PRIME HEAD LEARNING",
        "The classifier has not yet demonstrated held-out performance above chance.",
        heldout,
    )


def build_plot(run_dir: Path):
    history = MetricHistory()
    tailer = JsonlTailer(run_dir / "metrics.jsonl")
    config = read_config(run_dir)

    fig = plt.figure(figsize=(12, 8.5))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.28, 1.0, 1.0],
        hspace=0.42,
        wspace=0.28,
        left=0.08,
        right=0.96,
        top=0.92,
        bottom=0.08,
    )
    ax_progress = fig.add_subplot(grid[0, :])
    ax_loss = fig.add_subplot(grid[1, 0])
    ax_class = fig.add_subplot(grid[1, 1])
    ax_confusion = fig.add_subplot(grid[2, 0])
    ax_status = fig.add_subplot(grid[2, 1])

    total_steps = int(config.get("total_steps", config.get("steps", 0)) or 0)
    eval_interval = int(config.get("eval_interval", 1000) or 1000)

    def update(_frame=None):
        records, reset = tailer.poll()
        if reset:
            history.clear()
        for record in records:
            history.add(record)

        latest_step = history.latest_step()
        phase = history.latest_phase()
        status = read_status(run_dir)

        ax_progress.clear()
        progress = (
            min(1.0, max(0.0, (latest_step + 1) / total_steps))
            if latest_step is not None and total_steps > 0
            else 0.0
        )
        ax_progress.barh([0], [1.0], color="#e9ecef", height=0.55)
        ax_progress.barh([0], [progress], color="#52796f", height=0.55)
        ax_progress.set_xlim(0.0, 1.0)
        ax_progress.set_yticks([])
        ax_progress.set_xticks([])
        step_text = (
            f"{latest_step + 1:,} / {total_steps:,} steps"
            if latest_step is not None and total_steps > 0
            else "waiting for metrics"
        )
        ax_progress.text(
            0.5,
            0,
            f"{progress:.1%}   {step_text}   phase: {phase}",
            ha="center",
            va="center",
            color="white" if progress > 0.55 else "#212529",
            fontweight="bold",
        )
        ax_progress.set_title("Training progress", loc="left", fontsize=10)

        ax_loss.clear()
        loss_values: list[float] = []
        for phase_name in history.phases():
            for split in SPLIT_ORDER:
                xs, ys = history.series("loss", phase=phase_name, split=split)
                if not xs:
                    continue
                loss_values.extend(ys)
                label = f"{phase_name}: {split}"
                ax_loss.plot(
                    xs,
                    ys,
                    label=label,
                    linewidth=1.25 if split == "train" else 1.8,
                    linestyle="-" if split == "train" else "--",
                    color=SPLIT_COLOURS.get(split),
                )
        if loss_values and min(loss_values) > 0:
            ax_loss.set_yscale("log")
        ax_loss.set_title("Loss by phase and split")
        ax_loss.set_xlabel("global step")
        ax_loss.set_ylabel("loss")
        ax_loss.grid(True, alpha=0.25)
        if ax_loss.lines:
            ax_loss.legend(fontsize=7, loc="best")

        ax_class.clear()
        for split in SPLIT_ORDER:
            colour = SPLIT_COLOURS.get(split)
            for metric, label, linestyle in (
                ("balanced_accuracy", "balanced", "-"),
                ("prime_recall", "prime recall", "--"),
                ("composite_recall", "composite recall", ":"),
            ):
                xs, ys = history.series(metric, split=split)
                if xs:
                    ax_class.plot(
                        xs,
                        ys,
                        color=colour,
                        linestyle=linestyle,
                        linewidth=1.6,
                        label=f"{split}: {label}",
                    )
        ax_class.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--")
        ax_class.axhline(0.9, color="#52796f", linewidth=0.8, linestyle="--")
        ax_class.set_ylim(0.0, 1.02)
        ax_class.set_title("Class-balanced generalisation")
        ax_class.set_xlabel("global step")
        ax_class.set_ylabel("score")
        ax_class.grid(True, alpha=0.25)
        if ax_class.lines:
            ax_class.legend(fontsize=7, loc="lower right", ncol=2)

        headline, explanation, heldout = generalisation_status(history, eval_interval)
        latest_heldout = history.latest_binary(heldout) if heldout else None

        ax_confusion.clear()
        if latest_heldout is None:
            ax_confusion.axis("off")
            ax_confusion.text(
                0.5,
                0.5,
                "Confusion matrix appears when\nprime-head evaluation begins.",
                ha="center",
                va="center",
                fontsize=11,
            )
        else:
            matrix = np.array(
                [
                    [
                        int(latest_heldout.get("true_negative", 0)),
                        int(latest_heldout.get("false_positive", 0)),
                    ],
                    [
                        int(latest_heldout.get("false_negative", 0)),
                        int(latest_heldout.get("true_positive", 0)),
                    ],
                ]
            )
            ax_confusion.imshow(matrix, cmap="Greys", vmin=0)
            for row in range(2):
                for col in range(2):
                    ax_confusion.text(
                        col,
                        row,
                        str(matrix[row, col]),
                        ha="center",
                        va="center",
                        fontsize=18,
                        color="white"
                        if matrix[row, col] > max(1, matrix.max() / 2)
                        else "black",
                    )
            ax_confusion.set_xticks([0, 1], ["pred composite", "pred prime"])
            ax_confusion.set_yticks([0, 1], ["actual composite", "actual prime"])
            ax_confusion.set_title(f"Latest {heldout} confusion matrix")

        ax_status.clear()
        ax_status.axis("off")
        ax_status.text(
            0.0,
            0.88,
            headline,
            fontsize=17,
            fontweight="bold",
            color="#2f3e46",
            transform=ax_status.transAxes,
        )
        ax_status.text(
            0.0,
            0.68,
            explanation,
            fontsize=10,
            wrap=True,
            transform=ax_status.transAxes,
        )
        if latest_heldout is not None:
            summary = (
                f"Held-out split: {heldout}\n"
                f"Balanced accuracy: {float(latest_heldout['balanced_accuracy']):.1%}\n"
                f"Prime recall: {float(latest_heldout['prime_recall']):.1%}\n"
                f"Composite recall: {float(latest_heldout['composite_recall']):.1%}\n"
                f"Raw accuracy: {float(latest_heldout['accuracy']):.1%}"
            )
            ax_status.text(
                0.0, 0.18, summary, fontsize=11, linespacing=1.45, transform=ax_status.transAxes
            )

        fig.suptitle(f"{run_dir.name}  [{status}]", fontsize=12, fontweight="bold")
        return ()

    return fig, update


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live class-balanced primality generalisation dashboard."
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
            "Start train_prime_transfer first, then launch this dashboard."
        )
    if not (run_dir / "metrics.jsonl").is_file():
        raise SystemExit(f"{run_dir} has no metrics.jsonl")
    print(f"Reading {run_dir}")

    fig, update = build_plot(run_dir)
    if args.snapshot:
        update()
        fig.savefig(args.snapshot, dpi=130)
        print(f"Wrote {args.snapshot}")
        return

    if args.live:
        animation = FuncAnimation(
            fig,
            update,
            interval=int(args.interval * 1000),
            cache_frame_data=False,
        )
        plt.show()
        del animation
    else:
        update()
        plt.show()


if __name__ == "__main__":
    main()
