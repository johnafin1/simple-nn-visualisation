---
name: run-experiment
description: Set up, run, and compare training experiments in this repo, including launching multiple runs in parallel (e.g. weight-decay sweeps for grokking). Use when configuring an experiment, adding a run config, running a hyperparameter sweep, or comparing runs.
---

# Run an Experiment

Experiments are defined by a `RunConfig` and executed via `nn::api::Trainer` /
`ExperimentRunner`. Runs are isolated by `run_id` so many can run in parallel and be compared.
See `docs/EXPERIMENTS.md`.

## Single run

```
- [ ] 1. Define RunConfig: task, model dims, optimiser, lr, weight_decay, steps, seed, log intervals
- [ ] 2. Assign a unique run_id -> runs/<run_id>/
- [ ] 3. Train; emit metrics.jsonl (+ params.jsonl / predictions.jsonl per config)
- [ ] 4. Verify logs are valid JSONL and loss behaves as expected
```

- Persist the full config + git hash + seed to `runs/<run_id>/config.json` for reproducibility.
- Set `param_log_interval` appropriately: large for long runs; `1` only when a per-round
  parameter movie is wanted (see logging rule / `docs/LOGGING.md`).

## Parallel sweep (compare configs)

Use `ExperimentRunner` to launch several configs concurrently. Runs share nothing but the
`runs/` root; each has its own directory.

- Natural first sweep: **weight decay** for grokking (`0`, `1e-3`, `1e-2`, `1e-1`).
- Other sweeps: learning rate, hidden width, train-set size.
- Adding more runs later is free — just launch more.

## Compare

Query all runs at once with DuckDB, then plot (see visualise-training skill):

```sql
SELECT run_id, step, loss
FROM 'runs/*/metrics.jsonl'
WHERE split = 'test'
ORDER BY run_id, step;
```

## Grokking notes

- Small train set + weight decay + long training are the key levers.
- `x^2` may show gradual (not sharp) generalisation; if a textbook-sharp grok is needed, use the
  modular-arithmetic fallback task documented in `docs/EXPERIMENTS.md`.
