---
name: implement-technique
description: Implement a neural-network technique (layer, activation, loss, optimiser, init, backprop step) from scratch in the C++ core following the project's discuss-confirm-implement-visualise loop. Use when adding or changing any NN algorithm, math op, or training component in this repo.
---

# Implement a Technique

Follow this loop for any new/changed NN technique. Do not skip steps or jump phases
(`docs/ROADMAP.md`).

## Workflow

```
- [ ] 1. Discuss: math + placement
- [ ] 2. Confirm: structures via mermaid, get sign-off
- [ ] 3. Implement: functions/classes as agreed
- [ ] 4. Verify: gradient check / unit test
- [ ] 5. Visualise + update docs
```

### 1. Discuss

- State the math (formula + forward role + backward/gradient) plainly.
- Decide the layer: `nn::math` (pure fn) vs `nn::core` (object) — see `docs/CPP_CONVENTIONS.md`.
- Add or update the entry in `docs/TECHNIQUES.md` with status `discussed`.

### 2. Confirm structures (before coding)

- Present the object structure / signatures with a mermaid `classDiagram` or `flowchart`, plus
  the exact files to touch (`src/includes/<bucket>/...`, `src/<bucket>/...`).
- Wait for the user's explicit sign-off. Keep them heavily involved.

### 3. Implement

- Math goes in `nn::math` as pure free functions over `std::span`.
- Objects go in `nn::core`, calling `nn::math`. Respect the big 6/7 rules and polymorphism
  guidance in `docs/CPP_CONVENTIONS.md`.
- If the component is observable during training, implement `nn::log::Loggable`.

### 4. Verify

- For anything with a gradient, add a **numeric gradient check** (analytic vs finite-difference
  within tolerance) in `tests/`.
- For pure ops, add a closed-form unit test on small inputs.

### 5. Visualise + document

- Ensure the technique's effect is observable via logs (see the visualise-training skill).
- Flip `docs/TECHNIQUES.md` status to `implemented`.
- Update `docs/STATUS.md` if the capability, active phase, or immediate next work changed.
- Close or add `docs/RECTIFICATIONS.md` items as needed.

## Guardrails

- No black-box ML frameworks in the core.
- Deterministic: seed any randomness (init) and record the seed in `RunConfig`.
