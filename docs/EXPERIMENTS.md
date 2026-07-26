# Experiments

## Flagship: grokking `f(x) = x^2`

### Goal

Train a tiny MLP to regress `x^2` on a **small** sample and keep training long after it has
fit the training data, watching for **grokking**: test loss stays high while train loss is
already low, then drops sharply — delayed generalisation.

### Setup (numbers finalised at the start of Phase 5)

- **Target:** `y = x^2` on `x` in `[-1, 1]`.
- **Data:** small train set (order ~10-20 points), larger held-out test/grid for evaluation.
  Small data is what makes memorise-then-generalise observable.
- **Model:** tiny MLP, starting point `1 -> 8 -> 8 -> 1` with `tanh` hidden activations
  (smooth target favours tanh over ReLU). Exact width decided when we implement.
- **Optimiser:** SGD **with weight decay** — weight decay is the primary grokking lever.
- **Training length:** long (well past the point train loss plateaus near zero).
- **Seed:** fixed and logged for reproducibility.

### What we log (see [LOGGING.md](LOGGING.md))

- `metrics.jsonl`: train/test loss, `weight_norm`, `grad_norm`, `lr` per step.
- `params.jsonl`: per-parameter value + grad (interval-gated; interval = 1 when we want a
  per-round parameter movie).
- `predictions.jsonl`: `y_hat` across a dense `x` grid at snapshot steps, for the prediction-
  curve video.

### Success / result criteria

- **Ideal:** a visible gap where test loss lags, then a sharp drop = grok.
- **Acceptable/documented:** if `x^2` only shows gradual generalisation (likely, since smooth
  regression lacks the sharp phase transition of algorithmic tasks), we document that and pivot
  the "sharp grok" demo to the fallback below. This is a learning outcome, not a failure.

### Fallback for a textbook-sharp grok: modular arithmetic

Grokking was originally and most reliably shown on algorithmic tasks (Power et al., 2022), e.g.
`(a + b) mod p` as classification. If we want the dramatic curve, we add a second experiment:

- **Task:** `(a + b) mod p` for small prime `p`, framed as classification over `p` classes.
- **Model:** small MLP/embedding; heavy weight decay; train on a fraction of all `(a,b)` pairs.
- Reuses the same `Trainer`, logging, and viz — only `Dataset` and the output head change.

We lead with `x^2` (your stated interest, and a clean regression/visualisation story) and keep
modular arithmetic ready as the guaranteed-grok comparison.

## Grokking experiment 2: primality on [2, 200]

- Status: **run** (Phase 5) — see [Actual results](#actual-results-phase-5) below.

### Goal

Classify "is `n` prime" for `n` in `[2, 200]`, training long past the point train loss
bottoms out, and watch whether test accuracy ever lifts off the base rate.

### Data

- 199 samples (`n = 2..200`), of which **46 are prime** (`π(200) = 46`), so the base rate is
  **23.1%**. A model that always predicts "not prime" scores 76.9% accuracy — which is why we
  log **accuracy alongside loss**, and why the comparison against base rate matters more than
  raw accuracy.
- Stratified ~50/50 train/test split, so both splits contain primes (23 each).
- **Unseen evaluation ranges** beyond the training pool, as separate splits:

| Split | Range | Samples | Primes | Base rate (always "composite") |
|---|---|---|---|---|
| `train` | subset of 2-200 | 100 | 23 | 77.0% |
| `test` | rest of 2-200 | 99 | 23 | 76.8% |
| `unseen_201_255` | 201-255 | 55 | 8 | 85.5% |
| `unseen_256_300` | 256-300 | 45 | 8 | 82.2% |

### The two encodings (this is the experiment)

| Encoding | Inputs | Why |
|---|---|---|
| `bits` | 9 (binary digits of `n`) | Bits are **shared structure** across inputs, so a rule learned on training integers can transfer to held-out ones. Generalisation is at least *possible*. |
| `onehot` | 199 (exactly one hot) | Deliberate **memorisation control**. Each integer is an independent symbol, so a held-out `n` has an input weight that was never trained. Test accuracy *cannot* beat the base rate. |

**Why 9 bits, not 8.** The input width is fixed when the network is constructed, so
evaluating on integers up to 300 requires 9 bits (which covers 2-511). Bit 8 is therefore
**always 0 throughout training** and switches on for the first time at `n = 256`. That makes
256-300 a strictly harder test than 201-255 — the network is being asked about a feature it
has literally never seen active — so the two ranges are logged as separate splits rather than
averaged together.

**One-hot has no unseen splits.** There is no input slot for 201-300 under one-hot, so
`Dataset::primes` produces only `train` and `test` for that encoding. That absence *is* the
control: the encoding makes generalisation beyond the training pool not merely hard but
structurally impossible.

Running both and overlaying them is the teaching result: it shows concretely that grokking
requires a generalisable structure to discover, not just weight decay and patience.

### Setup

- **Model:** `9 -> 32 -> 32 -> 1` (bits) / `199 -> 32 -> 32 -> 1` (one-hot), `tanh` hidden
  activations, **linear output** emitting a logit.
- **Loss:** `BceWithLogitsLoss` — binary cross-entropy with the sigmoid **fused in**. The naive
  `-[y log p + (1-y) log(1-p)]` overflows once `p` saturates; the fused form
  `max(z,0) - z·y + log(1 + e^-|z|)` is exact and never evaluates `exp` on a positive argument.
  Its gradient collapses to the very stable `(p - y)/N`.
- **Optimiser:** `SgdWeightDecay` — the grokking lever. Biases are not decayed.
- **Metrics:** loss + **accuracy** for every split, plus `weight_norm` and `grad_norm`.
- **Seed:** fixed and logged.

### Honest expectation

Primality is not compactly expressible by a small MLP over bit inputs — there is no neat
circuit analogous to the Fourier features that make modular arithmetic grok. Realistic
outcome: the `bits` run generalises **weakly above base rate** and may never show a sharp
phase transition; the `onehot` run flatlines at base rate by construction. Both are useful
results and we document whatever happens. Modular arithmetic (below) remains the
guaranteed-grok comparison.

Outcome: the `onehot` prediction was exactly right, and the `bits` prediction was *optimistic* —
it did not generalise weakly, it failed to generalise at all. Numbers below.

### Infrastructure Phase 5 added

All of it is now implemented:

1. **Vector-valued inputs** — `Dataset` became a set of named `Split`s with flat row-major
   inputs, targets, and a per-sample `id` ([../src/includes/api/dataset.hpp](../src/includes/api/dataset.hpp)).
2. **`Sigmoid`** — `nn::math::sigmoid` / `sigmoid_grad` (sign-branched, overflow-safe) plus
   `nn::core::SigmoidLayer`.
3. **`BceWithLogitsLoss`** — `nn::math::bce_with_logits` / `_grad` plus the `nn::core` wrapper.
   Two virtuals were added to the `Loss` base so the `Trainer` stays task-agnostic:
   `activate()` (identity for regression, sigmoid for BCE) and `is_classification()`.
4. **`SgdWeightDecay`** — `w -= lr * (g + λw)`, skipping `bias` parameters.
5. **Accuracy** — the `Trainer` now evaluates *every* non-train split at `eval_interval` and
   emits `accuracy` alongside `loss` when the loss is a classification loss.

### How to reproduce

```powershell
# bits (structure available, generalisation possible in principle)
.\build\src\app\train_primes\train_primes.exe --encoding bits `
  --steps 6000 --lr 0.5 --weight-decay 0.001 --hidden 32 --seed 42 `
  --name primes_bits --predict-interval 50

# onehot (memorisation control)
.\build\src\app\train_primes\train_primes.exe --encoding onehot `
  --steps 6000 --lr 0.5 --weight-decay 0.001 --hidden 32 --seed 42 `
  --name primes_onehot --predict-interval 50

# watch either one live, or snapshot afterwards
.\.venv\Scripts\python.exe src/app/python/primes_plot.py --live
.\.venv\Scripts\python.exe src/app/python/live_plot.py --live
```

### Actual results (Phase 5)

6000 steps, `lr = 0.5`, `weight_decay = 0.001`, `hidden = 32`, `seed = 42`. Accuracy is at the
final step; **base rate** is the score for always answering "composite".

| Run | Split | Final loss | Final accuracy | Base rate | Verdict |
|---|---|---|---|---|---|
| `bits` | train | 0.0057 | **100%** | 77.0% | memorised completely (100% by step 411) |
| `bits` | test | 1.356 | 75.8% | 76.8% | **at/below base rate — no generalisation** |
| `bits` | unseen_201_255 | 1.363 | 76.4% | 85.5% | **worse than always saying "composite"** |
| `bits` | unseen_256_300 | 1.283 | 71.1% | 82.2% | worse still, as predicted for the new bit |
| `onehot` | train | 0.0025 | **100%** | 77.0% | memorised by step 48 |
| `onehot` | test | 0.914 | 76.8% | 76.8% | **exactly the base rate**, by construction |

**No grok.** Train loss fell ~240x while every held-out loss *rose* monotonically to ~1.3 — the
textbook overfitting signature, not a delayed-generalisation one. Weight norm peaked at 11.47
(step 802) and weight decay then pulled it down to 9.38 by step 6000, with no accompanying
improvement on any held-out split. Weight decay was doing its job; there was simply no
generalising solution for it to find.

The `onehot` heatmap is the cleanest teaching artefact: exactly the *training* primes turn red
and every other row stays blue, so the model answers "composite" for all 99 test integers and
lands precisely on the 76.8% base rate. The `bits` run looks superficially busier — it scatters
confident red rows across the held-out ranges — but those are false positives, which is why it
scores *below* base rate on both unseen ranges.

The one genuine signal: `unseen_201_255` peaked at 87.3% accuracy at step 50 and
`unseen_256_300` at 82.2% at step 0, i.e. **early, before memorisation set in**. At that point
the network had learned little more than "even numbers are composite" (bit 0), which is a real
if trivial rule that transfers. Training then destroyed it in favour of memorising the training
set. Generalisation here does not arrive late — it departs early.

This is the expectation in the honesty note above, confirmed. Primality has no compact circuit
over bit inputs analogous to the Fourier features that make modular arithmetic grok, so the
`bits` encoding gives the optimiser *possible* structure but not *learnable* structure. If we
want to see a real grok, the modular-arithmetic experiment below is the one to run.

## Parallel comparison runs

Because runs are isolated by `run_id`, we launch several configs at once via
`ExperimentRunner` and overlay them. Natural first sweep:

```mermaid
flowchart LR
  Runner["ExperimentRunner"] --> R1["run: wd=0.0"]
  Runner --> R2["run: wd=1e-3"]
  Runner --> R3["run: wd=1e-2"]
  Runner --> R4["run: wd=1e-1"]
  R1 --> Cmp["overlay test-loss curves"]
  R2 --> Cmp
  R3 --> Cmp
  R4 --> Cmp
```

- **Weight-decay sweep:** the clearest illustration of what triggers grokking.
- Other easy sweeps: learning rate, hidden width, train-set size.
- Comparison plot: overlay `test loss` vs `step` (log x-axis) for all runs, pulled with one
  DuckDB query over `runs/*/metrics.jsonl`.

Adding more runs later is free — just launch more; nothing needs restructuring.

## Outputs

- **Prediction heatmap** (`src/app/python/primes_plot.py`, Phase 5): one row per integer, one
  column per logged step, colour = `P(prime)`, with ground-truth and split-membership gutters
  down the left. This is the view that makes memorisation legible at a glance — train rows
  converge to the correct colour while held-out rows do not. Supports `--live` and `--snapshot`.
- Live loss curve during training (Phase 4).
- Post-hoc `.mp4`: loss curves + evolving prediction curve overlaid on the true `x^2`.
- Optional per-parameter evolution video from `params.jsonl`.
- Comparison figures across parallel runs.
