---
name: project-status-check
description: Check, report, reconcile, and update this repository's canonical project status. Use when asked where the project is up to, what is implemented, what comes next, whether docs match the repository, how to resume work, whether a phase is complete, or when implementation/experiment results require status, roadmap, technique, experiment, or rectification updates.
---

# Project Status Check

Use `docs/STATUS.md` as the current-state authority. Keep it concise and evidence-based.

## Check status

1. Read `.gitignore` first. Exclude ignored generated files and directories from the inventory
   unless the user explicitly puts them in scope.
2. Read, in order:
   - `docs/STATUS.md` for current facts and immediate next work;
   - `docs/RECTIFICATIONS.md` for known gaps;
   - `docs/ROADMAP.md` for phase order and exit criteria.
3. Read `docs/TECHNIQUES.md` or `docs/EXPERIMENTS.md` only when the question involves algorithm
   state or experiment evidence.
4. Verify material claims against non-ignored repository files. Check source/header presence,
   CMake registration, tests, and relevant implementation details.
5. Report:
   - completed phase and implemented capabilities;
   - active work and observed experiment result;
   - open high-priority rectifications;
   - next phase and future work.
6. Distinguish **documented**, **verified**, **planned**, and **unknown**. Never infer
   "implemented" from a plan or filename alone.

For a read-only status request, do not edit files.

## Update status

Update status after verified implementation, a material experiment result, a phase-gate change,
or a change to immediate priorities:

1. Update the date and smallest relevant sections of `docs/STATUS.md`.
2. Update `docs/ROADMAP.md` only when phase progress or exit criteria change.
3. Update `docs/TECHNIQUES.md` when an algorithm moves between planned, discussed, and
   implemented.
4. Add experiment configuration, evidence, and conclusions to `docs/EXPERIMENTS.md`.
5. Add, revise, or close items in `docs/RECTIFICATIONS.md`.
6. Update `.cursor` rules/skills when the workflow or architectural contract changes.
7. Validate local links and search for contradictory phase/status claims before finishing.

## Guardrails

- `STATUS.md` describes now; `ROADMAP.md` describes phase order; `RECTIFICATIONS.md` describes
  known gaps. Do not duplicate full plans across them.
- Do not mark a phase complete until its exit criteria are verified and the user has signed off
  where required.
- Do not call an experiment a grok without logged evidence of delayed generalisation.
- Do not claim `ExperimentRunner`, `make_video.py`, `query.py`, or two-input modular addition
  exist until they are present, registered where applicable, and verified. Unary modulo-residue
  pretraining in `train_prime_transfer` is implemented but is not the same experiment.
- Preserve the project's learning goal: hand-write NN/runtime mechanics and keep the C++ runtime
  free of third-party dependencies. Build/test library management is allowed.
