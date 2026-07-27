# Project Status

This is the canonical record of what exists **now**. `ROADMAP.md` describes phase order and
future work; `RECTIFICATIONS.md` tracks known gaps between the intended design and the current
implementation.

Last updated: **2026-07-26**

## At a glance

- **Completed through:** Phase 5.
- **Active focus:** running grokking-oriented experiments and learning from the results.
- **Observed result:** no successful grok has been observed yet.
- **Next planned phase:** Phase 6, parallel experiments with an `ExperimentRunner`.
- **C++ dependency policy:** implement the neural-network and logging mechanics by hand. Keep the
  runtime library free of third-party C++ dependencies; build/test dependency management is
  allowed when it supports the learning work without hiding NN behaviour.

## What is implemented

### C++ library

- `nn::math`: linear algebra, activations, losses, gradients, and seeded Xavier/He
  initialisation.
- `nn::core`: `Tensor`, polymorphic layers/losses/optimisers, `Network`, backpropagation,
  weighted BCE, frozen dense parameters, replaceable final heads, `Loggable`, `JsonLine`, and
  append-capable `JsonlSink`.
- `nn::api`: named/vector-valued datasets, deterministic stratified train/validation/test
  splits, residue targets, staged run configuration, and the single-run `Trainer`.
- Training logs: `config.json`, `meta.json`, `metrics.jsonl`, `params.jsonl`, and
  `predictions.jsonl` under an isolated `runs/<run_id>/` directory. Multi-phase runs also write
  `config_<phase>.json` and use continuous global steps.

### C++ applications

- `src/app/smoke/`: toolchain proof.
- `src/app/demo/`: small network/logging demonstration.
- `src/app/train_x2/`: `x^2` regression experiment.
- `src/app/train_primes/`: primality classification experiment.
- `src/app/train_prime_transfer/`: modulo-residue pretraining followed by a frozen,
  class-balanced prime head on integers 2-500.

### Python tools

- `test_dashboard.py`: rebuild/test dashboard.
- `network_demo.py`: network forward/backward visualiser.
- `live_plot.py`: live metrics plotter.
- `primes_plot.py`: live/snapshot progress and class-balanced generalisation dashboard.

`make_video.py` and `query.py` are Phase 7 work and do not exist yet.

## Experiment status

- **`x^2`:** runnable and under active experimentation. No grok is currently documented.
- **Primality:** completed for bit and one-hot encodings. Both memorised their training data and
  failed to generalise; see [EXPERIMENTS.md](EXPERIMENTS.md#actual-results-phase-5).
- **Modulo-to-prime transfer:** implemented and smoke-verified, but no material long-run result
  is documented yet. It uses residue sin/cos pretraining, a frozen 64-wide encoder, weighted
  BCE, and held-out balanced metrics on a 60/20/20 split through 500.
- **Two-input modular addition:** still not implemented; it remains the textbook-grok comparison.
- **Parallel sweeps:** not implemented yet. Runs are individually isolated, but there is no
  `ExperimentRunner`.

An unsuccessful grokking run is still a valid experiment result. Do not describe a run as
grokking unless its logged train/test behaviour supports that conclusion.

## Known gaps

The current logging implementation works and now records phase, model/loss/optimiser labels,
cadences, class weights, and global progress. Its provenance and metadata still need
strengthening before parallel experiment work:

- model topology is still a human-readable description rather than a complete structured
  manifest, and the git revision is not captured;
- `meta.json` does not preserve complete start/end/host lifecycle information.
- `params.jsonl` and `predictions.jsonl` rely on their parent run directory for identity instead
  of carrying `run_id` in every row.

The detailed tasks and acceptance criteria live in [RECTIFICATIONS.md](RECTIFICATIONS.md).

## What comes next

1. Run and record the modulo-to-prime transfer experiment; compare held-out balanced accuracy
   with the original scratch baseline.
2. Resolve the high-priority experiment provenance and metadata rectifications.
3. Phase 6: design and implement `ExperimentRunner`, then run controlled parallel sweeps.
4. Continue grokking experiments; implement two-input modular addition when agreed.
5. Phase 7: add reusable post-hoc querying and video generation.
6. Phases 8-9 remain optional performance/UI/backend work.

## Source-of-truth order

When documents appear to disagree, use this order:

1. `STATUS.md` for current phase, implemented capabilities, and immediate next work.
2. `RECTIFICATIONS.md` for known defects and documentation/code mismatches.
3. `ROADMAP.md` for planned phase order and exit criteria.
4. `TECHNIQUES.md` for individual algorithm implementation status.
5. `EXPERIMENTS.md` for experiment designs, commands, and observed results.
6. Historical phase plans for the decisions made during those completed phases.

## Updating this status

Update this document whenever a capability is implemented, an experiment produces a material
result, a phase gate changes, or the immediate next work changes. Base updates on repository
evidence and verification, not on plans alone.

When updating:

1. Change the date and the smallest relevant current-state sections.
2. Keep `ROADMAP.md` progress consistent with phase changes.
3. Update `TECHNIQUES.md` when an algorithm status changes.
4. Record experiment evidence in `EXPERIMENTS.md`.
5. Close or add work in `RECTIFICATIONS.md`.
6. Update `.cursor` rules/skills if the agent workflow or architectural contract changes.
