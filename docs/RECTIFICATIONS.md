# Rectifications

This document breaks known mismatches and engineering debt into actionable work. It is not the
future phase plan; see [ROADMAP.md](ROADMAP.md) for that. Current project state lives in
[STATUS.md](STATUS.md).

Last reviewed: **2026-07-26**

## Status legend

- **Open:** confirmed work remains.
- **Blocked:** cannot proceed without a decision or prerequisite.
- **Done:** implemented and verified; retained briefly for traceability.

## Active rectifications

### RECT-001 — Complete experiment provenance

- **Priority:** High
- **Status:** Open
- **Area:** `RunConfig`, `Trainer`, run artefacts
- **Current condition:** staged runs now record cadence, global progress, class weights, and
  human-readable model/loss/optimiser labels in `config.json` plus `config_<phase>.json`.
  The manifest is still not a complete structured topology and does not record git revision.
- **Why it matters:** an experiment cannot be reproduced confidently from its run directory
  alone, and parallel comparisons need trustworthy configuration labels.

Required work:

- [ ] Agree on a serialisable run-manifest shape before changing the API.
- [x] Persist training/evaluation/parameter/prediction and flush cadences.
- [ ] Persist selected prediction splits.
- [ ] Persist model topology and initialisation as structured data; keep loss and optimiser
  settings consistent across every application.
- [ ] Record the source git revision, including whether the worktree was dirty.
- [ ] Add tests that parse the generated config and assert required fields.
- [ ] Update `LOGGING.md` with the final schema.

Acceptance: a run directory contains enough configuration and source provenance to reconstruct
the same model, data split, optimiser, and logging behaviour.

### RECT-002 — Preserve the full run lifecycle

- **Priority:** High
- **Status:** Open
- **Area:** `meta.json`, `Trainer`
- **Current condition:** the start record is overwritten at completion; the final file contains
  the end time and status but loses the start time. Host information and failure state are not
  recorded.
- **Why it matters:** long and parallel runs need duration, host, and failure diagnostics.

Required work:

- [ ] Choose a single-object metadata schema or an append-only lifecycle stream.
- [ ] Preserve start time, end time, host, and final status together.
- [ ] Record failed/interrupted runs where practical.
- [ ] Write atomically enough that a live reader does not see malformed JSON.
- [ ] Add lifecycle tests for running and completed states.

Acceptance: the final metadata represents the complete lifecycle without discarding earlier
information.

### RECT-003 — Standardise stream identity

- **Priority:** Medium
- **Status:** Open
- **Area:** JSONL schemas and multi-run analysis
- **Current condition:** `metrics.jsonl` includes `run_id`; `params.jsonl` and
  `predictions.jsonl` rely on the parent directory for run identity.
- **Why it matters:** direct glob queries across runs are simpler and safer when every row is
  self-identifying.

Required work:

- [ ] Decide whether all streams must carry `run_id` or whether readers must derive it from the
  filename.
- [ ] Apply the decision consistently to the writer, tests, DuckDB examples, and Python readers.
- [ ] Document the stable schema and compatibility expectations.

Acceptance: multi-run queries cannot silently mix parameter or prediction rows without a run
identity.

### RECT-004 — Keep experiment conclusions evidence-based

- **Priority:** Medium
- **Status:** Open, ongoing
- **Area:** experiment records
- **Current condition:** grokking-oriented runs are active, but no successful grok is documented.
- **Why it matters:** the project should distinguish a planned grokking experiment from an
  observed grok.

Required work:

- [ ] Record material `x^2` configurations and results in `EXPERIMENTS.md`.
- [ ] Preserve negative results and the criteria used to classify them.
- [ ] Do not mark grokking successful without a delayed train/test generalisation transition.
- [x] Add the approved unary modulo-residue-to-prime transfer experiment.
- [ ] Record its first material long-run result without conflating transfer generalisation with
  delayed grokking.

Acceptance: every headline experiment claim points to reproducible configuration and logged
evidence.

## Planned phase work with dependencies

### RECT-005 — Phase 6 parallel experiment runner

- **Priority:** Next phase
- **Status:** Blocked by design confirmation and preferably RECT-001/002
- **Current condition:** no `ExperimentRunner` exists.

Required work:

- [ ] Discuss process/thread model, failure handling, and concurrency limits.
- [ ] Confirm the API and exact source layout with the user.
- [ ] Implement isolated concurrent runs without shared mutable logging state.
- [ ] Compare a controlled sweep with a multi-run plot/query.

Acceptance: N configurations run concurrently, retain complete provenance, and produce a correct
overlaid comparison.

### RECT-006 — Phase 7 post-hoc analysis pack

- **Priority:** Future
- **Status:** Blocked by phase order
- **Current condition:** `make_video.py` and `query.py` do not exist.

Required work:

- [ ] Add reusable DuckDB queries.
- [ ] Build the log-to-frames-to-video pipeline.
- [ ] Document and test the Parquet promotion path for heavy parameter data.

Acceptance: an `.mp4` can be reconstructed from run logs without retraining.

## Completed documentation rectifications

### 2026-07-26

- **Done:** aligned the logging rule/docs with `Loggable`, `ParamView`, `JsonLine`, and
  `JsonlSink` as implemented.
- **Done:** replaced stale phase handoff guidance with `STATUS.md` and a project-status skill.
- **Done:** made current versus future experiment/visualisation capabilities explicit.
- **Done:** updated setup and dependency guidance to reflect the zero-runtime-dependency C++
  preference.
- **Done:** standardised `Tensor` on the Rule of Zero.
