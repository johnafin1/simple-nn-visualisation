---
name: run-experiment
description: Configure, run, inspect, and compare the currently implemented single-run x-squared, primality, and modulo-to-prime transfer experiments. Use when changing RunConfig values, launching train_x2, train_primes, or train_prime_transfer, checking run JSON/JSONL artefacts, comparing completed runs, or recording whether an experiment grokked. Do not assume the future ExperimentRunner exists.
---

# Run an Experiment

Read `docs/STATUS.md`, then `docs/EXPERIMENTS.md`. The current repository has a single-run
`nn::api::Trainer`; `ExperimentRunner` is Phase 6 and is not implemented.

## Single run

```text
- [ ] 1. Choose the implemented executable: train_x2, train_primes, or train_prime_transfer
- [ ] 2. Set model and RunConfig values through its supported CLI
- [ ] 3. Train; Trainer generates run_id and runs/<run_id>/
- [ ] 4. Verify config.json, meta.json, and emitted JSONL are valid
- [ ] 5. Inspect train and held-out behaviour
- [ ] 6. Record material configuration and results in docs/EXPERIMENTS.md
```

- Seed all randomness and keep the seed in the run configuration.
- Use a large `param_log_interval` for long runs; use `1` only for a per-step parameter trace.
- `train_prime_transfer` appends modulo pretraining and frozen prime-head phases to one run
  directory. Use balanced accuracy and both class recalls for its generalisation verdict.
- `config.json` does not yet contain full model/source provenance. Do not claim a run is fully
  reproducible until RECT-001 in `docs/RECTIFICATIONS.md` is complete.

## Compare completed runs

Metrics rows carry `run_id`, so DuckDB can compare run directories:

```sql
SELECT run_id, step, loss
FROM 'runs/*/metrics.jsonl'
WHERE split = 'test'
ORDER BY run_id, step;
```

`params.jsonl` and `predictions.jsonl` do not currently carry `run_id`; derive identity from the
file path or query one run directory at a time until RECT-003 is resolved.

## Grokking claims

- No experiment in the project has produced a documented grok yet.
- The original scratch primality experiments memorised and failed to generalise.
- The modulo-to-prime transfer application is implemented, but no material long-run result is
  documented yet.
- Treat `x^2` as active empirical work and record negative results as well as promising ones.
- Call a result a grok only when logs show delayed held-out generalisation after training fit.
- Two-input modular addition is proposed but not implemented; do not confuse it with unary
  modulo-residue pretraining.

## Parallel work (future Phase 6)

Do not call or create an `ExperimentRunner` as though its API were settled. Before implementing
Phase 6, discuss concurrency, failure handling, provenance, and exact object/file structure with
the user. Track the work in `docs/RECTIFICATIONS.md`.
