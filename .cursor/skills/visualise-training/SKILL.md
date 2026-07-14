---
name: visualise-training
description: Visualise training from JSONL run logs - live matplotlib plots while training, and post-hoc video/graph generation using DuckDB queries. Use when building or running the Python analysis track, plotting loss curves, comparing runs, or generating a training video.
---

# Visualise Training

The C++ core emits JSONL logs; Python does all visualisation. Everything is reconstructed from
`runs/<run_id>/*.jsonl`, so plots and video are reproducible without retraining. See
`docs/LOGGING.md`.

## Tooling

- **DuckDB** to query logs (SQL directly over `.jsonl`, no import).
- **matplotlib** for figures/frames; **imageio** + ffmpeg to stitch `.mp4`.
- Python scripts live in `src/app/python/` (`live_plot.py`, `make_video.py`, `query.py`).

## Live plot (during training)

```
- [ ] Poll/tail runs/<run_id>/metrics.jsonl on an interval
- [ ] Plot train/test loss vs step (log x-axis), redraw
```

The C++ `JsonlSink` flushes on an interval, so tailing sees fresh rows. Handle partial last
lines (a line may be mid-write).

## Post-hoc video

```
- [ ] 1. Query per-step frame data with DuckDB (loss-so-far + current prediction curve)
- [ ] 2. Render each frame with matplotlib
- [ ] 3. Stitch frames -> mp4 with imageio/ffmpeg
```

Prediction-curve frames come from `predictions.jsonl`; overlay `y_hat` on the true `x^2`.
Per-parameter evolution videos come from `params.jsonl`.

## Comparing parallel runs

```python
import duckdb
df = duckdb.sql("""
  SELECT run_id, step, loss
  FROM 'runs/*/metrics.jsonl'
  WHERE split = 'test'
  ORDER BY run_id, step
""").df()
# overlay one line per run_id, log x-axis
```

## Performance

`params.jsonl` can be millions of rows. For repeated heavy queries, convert once to Parquet:

```sql
COPY (SELECT * FROM 'runs/<id>/params.jsonl')
TO 'runs/<id>/params.parquet' (FORMAT parquet);
```
