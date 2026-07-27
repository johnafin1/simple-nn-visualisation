---
name: visualise-training
description: Use the repository's implemented Python tools to visualise tests, network behaviour, live training metrics, or primality predictions, and query current JSONL logs. Use for test_dashboard.py, network_demo.py, live_plot.py, primes_plot.py, plots, run comparisons, or planning the future post-hoc video pipeline.
---

# Visualise Training

The C++ core emits JSON/JSONL and Python handles visualisation. Read `docs/STATUS.md` before
choosing a tool so planned Phase 7 utilities are not mistaken for existing scripts.

## Implemented tools

- `src/app/python/test_dashboard.py`: rebuilds and displays the test suite.
- `src/app/python/network_demo.py`: visualises a C++-emitted forward/backward snapshot.
- `src/app/python/live_plot.py`: tails `metrics.jsonl` and plots loss/norms live.
- `src/app/python/primes_plot.py`: renders the scalable primality progress/generalisation
  dashboard from `metrics.jsonl`, including balanced accuracy, class recalls, confusion matrix,
  and conservative generalisation/grokking status.

Use the script's `--help` or source as the authority for current CLI flags.

## Live metrics

`live_plot.py` incrementally reads a run's `metrics.jsonl`. It must tolerate a partial final line
because `JsonlSink` may be mid-write. The sink flushes by record count and elapsed time.

`primes_plot.py` uses the same partial-line-safe tailing path. For `train_prime_transfer`, its
progress bar uses continuous staged steps and `total_steps` from `config.json`; held-out status
changes only when a new evaluation row arrives.

## Current run comparison

DuckDB can query metrics directly:

```python
import duckdb

df = duckdb.sql("""
  SELECT run_id, step, loss
  FROM 'runs/*/metrics.jsonl'
  WHERE split = 'test'
  ORDER BY run_id, step
""").df()
```

Parameter and prediction rows currently rely on the parent directory for run identity. Do not
blindly merge those streams across runs without adding the filename/run identity.

## Post-hoc video (future Phase 7)

`make_video.py` and `query.py` do not exist yet. The planned pipeline is:

1. Query frame data from run logs.
2. Render loss/prediction frames with matplotlib.
3. Stitch frames into `.mp4` with imageio/ffmpeg.

Do not report this pipeline as implemented. Discuss its interface and layout with the user before
adding it.

For repeated heavy parameter analysis, a future query tool may promote JSONL to Parquet; Parquet
outputs remain generated artefacts.
