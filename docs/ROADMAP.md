# Roadmap

For the canonical current state, read [STATUS.md](STATUS.md). For known code/documentation gaps
and their acceptance criteria, read [RECTIFICATIONS.md](RECTIFICATIONS.md). This file defines
phase order and gates.

Phased, discussion-driven. For every phase we follow the same loop:
**discuss the technique -> confirm structures (mermaid) -> implement together -> visualise.**

Nothing in a later phase starts until the earlier phase's exit criteria are met and you've
signed off.

| Phase | Focus | Key deliverables | Exit criteria |
|-------|-------|------------------|---------------|
| 0 | Docs + toolchain | This doc set; CMake/Python installed; buildable skeleton ([plan](plans/phase-0.md)) | Docs approved; `SETUP.md` verify checklist passes |
| 1 | Math layer | `nn::math` pure functions; unit tests ([plan](plans/phase-1.md)) | matmul/activation/mse tested; numeric checks pass |
| 2 | Object layer + backprop | `Tensor`, `Layer`/`DenseLayer`, `Loss`, `Network`; hand-written backward | Gradient check: analytic vs numeric within tolerance |
| 3 | Logging + train loop | `Loggable`, `JsonlSink`, `Trainer`, `Optimizer` (SGD) | A run produces valid `metrics.jsonl`; loss decreases |
| 4 | Live visualisation | `src/app/python/live_plot.py` tails logs | Loss curve updates in real time during a run |
| 5 | Grokking (`x^2` + primality) | Multi-split `Dataset`, `Sigmoid`, fused BCE, weight decay, class-balanced metrics, live generalisation dashboard | Test loss drops sharply well after train loss (grok), or documented result |
| 6 | Parallel experiments | `ExperimentRunner`; multi-run compare | N configs run concurrently; overlaid comparison plot |
| 7 | Video / analysis pack | `make_video.py`, `query.py`, Parquet path | `.mp4` of loss + prediction curve from logs |
| 8 | Intra-op parallelism | Threaded/OpenMP matmul behind `nn::math` | Speedup on larger nets; identical results to serial |
| 9 | Optional: GUI / GPU | ImGui live view and/or GPU math backend | Only if we want it; core unchanged |

## Progress

- Phases 0-2 complete. **Phase 3 complete**: `Loggable`, hand-rolled `JsonLine` + `JsonlSink`,
  `Optimizer`/`Sgd`, `RunConfig`, `Dataset`, and `Trainer` are implemented and tested; `train_x2`
  produces a valid `runs/<id>/metrics.jsonl` with decreasing loss. Bonus: a Streamlit network
  forward/backward visualiser (`src/app/python/network_demo.py`) reads a C++-emitted snapshot.
- **Phase 4 complete**: `JsonlSink` now flushes on a timer as well as a record count, `train_x2`
  takes CLI args for long runs, and `src/app/python/live_plot.py` tails `metrics.jsonl`
  incrementally (partial-line safe) to plot train/test loss plus `weight_norm`/`grad_norm` live.
  Verified mid-run: rows arriving continuously while `meta.json` status is `running`.
- **Phase 5 complete**: the classification infrastructure is in (`Sigmoid`, fused
  `BceWithLogitsLoss`, `SgdWeightDecay`, per-split accuracy), `Dataset` was generalised into
  named vector-valued `Split`s with per-sample `ids`, `Trainer` now evaluates every non-train
  split and writes a generic per-sample predictions schema. The **primality on [2, 200]**
  experiment was run in both
  `bits` and `onehot` encodings with 201-255 and 256-300 held out entirely.
  **Result: no grok** — train accuracy hit 100% while every held-out split sat at or below its
  base rate, and weight decay shrank the weight norm without any generalisation following. Full
  numbers in [EXPERIMENTS.md](EXPERIMENTS.md#actual-results-phase-5). This is the documented-result
  branch of the exit criterion, not the grok branch.
- **Phase 5 follow-up implemented**: `train_prime_transfer` pretrains a 64-wide encoder on
  cyclic residues for moduli 2, 3, 5, 7, 11, 13, 17, and 19, then replaces the head, freezes the
  encoder, and trains with class-weighted BCE on a stratified 60/20/20 split through 500.
  `primes_plot.py` is now a scalable progress/generalisation dashboard with balanced accuracy,
  per-class recall, confusion counts, and conservative generalisation/grokking labels. The
  implementation is verified; the long-run empirical result is pending.
- Next: resolve the high-priority provenance/metadata work in
  [RECTIFICATIONS.md](RECTIFICATIONS.md), then begin Phase 6 (parallel experiments). A
  weight-decay sweep is the natural first use. Modular arithmetic is the stronger candidate for
  observing a textbook grok, but remains an experiment rather than a guaranteed result.

## Notes

- **Parallel experiments (Phase 6)** are low-risk and high-value for comparing hyperparameters
  (e.g. weight-decay sweeps for grokking). Runs are isolated by `run_id`, so we can always add
  more.
- **Phase 5 honesty, in hindsight:** sharp textbook grokking is most reliable on algorithmic
  tasks, and the primality run bore that out — it overfit cleanly rather than grokking. The one
  transferable rule it found ("even numbers are composite") appeared in the first ~50 steps and
  was then trained away. See [EXPERIMENTS.md](EXPERIMENTS.md) and the modular-arithmetic fallback.
- Phases 8-9 are explicitly optional and gated on real need.
