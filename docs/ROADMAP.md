# Roadmap

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
| 5 | Grokking `x^2` | `Dataset`, weight decay, long-run config | Test loss drops sharply well after train loss (grok), or documented result |
| 6 | Parallel experiments | `ExperimentRunner`; multi-run compare | N configs run concurrently; overlaid comparison plot |
| 7 | Video / analysis pack | `make_video.py`, `query.py`, Parquet path | `.mp4` of loss + prediction curve from logs |
| 8 | Intra-op parallelism | Threaded/OpenMP matmul behind `nn::math` | Speedup on larger nets; identical results to serial |
| 9 | Optional: GUI / GPU | ImGui live view and/or GPU math backend | Only if we want it; core unchanged |

## Notes

- **Parallel experiments (Phase 6)** are low-risk and high-value for comparing hyperparameters
  (e.g. weight-decay sweeps for grokking). Runs are isolated by `run_id`, so we can always add
  more.
- **Phase 5 honesty:** sharp textbook grokking is most reliable on algorithmic tasks. `x^2`
  regression may show gradual generalisation instead of a hard phase transition. See
  [EXPERIMENTS.md](EXPERIMENTS.md) for the plan and the modular-arithmetic fallback.
- Phases 8-9 are explicitly optional and gated on real need.
