# Techniques Catalogue

A living list of every algorithm/technique we implement by hand. We **discuss each entry before
coding it** — this file is where that discussion is captured. Nothing here is implemented until
its status says so.

## Status legend

- `planned` — identified, not yet discussed in depth
- `discussed` — math + placement agreed; ready to implement
- `implemented` — done and tested

## Template (copy for each new technique)

```
### <name>
- Status: planned | discussed | implemented
- Layer: nn::math | nn::core | nn::api
- Math: <short formula / derivation sketch>
- Forward role: <what it computes>
- Backward role: <gradient it contributes>
- Notes / gotchas:
- Tests: <how we verify, e.g. numeric gradient check>
```

---

## Seed entries

### Dense (fully-connected) layer
- Status: implemented
- Layer: nn::core `DenseLayer` (uses nn::math `matvec`, `matvec_t`, `outer`, `axpy`)
- Math: `y = W x + b`
- Backward: `dW = dy x^T` (`outer`), `db = dy`, `dx = W^T dy` (`matvec_t`); grads accumulate.
- Notes: main carrier of parameters; implements `nn::log::Loggable` and exposes per-param
  value+grad views (`ParamView` with shape) — where per-param logging matters most.
- Files: `src/includes/classes/dense_layer.hpp`, `src/classes/dense_layer.cpp`
- Tests: forward value check, param shapes, numeric gradient check on W and b, zero_grad
  (`tests/test_dense_layer.cpp`)

### ReLU activation
- Status: implemented (nn::math free functions + nn::core `ReluLayer`)
- Layer: nn::math (fn) + thin nn::core layer wrapper
- Math: `relu(z) = max(0, z)`
- Backward: `1 if z > 0 else 0` (subgradient 0 at z == 0); layer caches z, gates upstream grad.
- Notes: dead-unit risk on tiny nets; watch during grokking runs.
- Files: `src/includes/helpers/activations.hpp`, `src/helpers/activations.cpp`,
  `src/includes/classes/activation_layers.hpp`, `src/classes/activation_layers.cpp`
- Tests: fn value + numeric gradient check (`tests/test_activations.cpp`); layer
  forward/backward (`tests/test_activation_layers.cpp`)

### Tanh activation
- Status: implemented (nn::math free functions + nn::core `TanhLayer`)
- Layer: nn::math + nn::core wrapper
- Math: `tanh(z)`
- Backward: `1 - tanh(z)^2`; layer caches z, multiplies upstream grad by local derivative.
- Notes: smooth; often a better fit for a smooth target like `x^2`.
- Files: `src/includes/helpers/activations.hpp`, `src/helpers/activations.cpp`,
  `src/includes/classes/activation_layers.hpp`, `src/classes/activation_layers.cpp`
- Tests: fn matches `std::tanh` + numeric gradient check (`tests/test_activations.cpp`);
  layer forward/backward (`tests/test_activation_layers.cpp`)

### MSE loss
- Status: implemented (nn::math free functions + nn::core `MseLoss`)
- Layer: nn::math + nn::core `MseLoss`
- Math: `L = mean((y_hat - y)^2)`
- Backward: `dL/dy_hat = 2 (y_hat - y) / N` (seed gradient for backprop)
- Files: `src/includes/helpers/loss.hpp`, `src/helpers/loss.cpp`,
  `src/includes/classes/loss.hpp`, `src/classes/loss.cpp`
- Tests: fn closed-form + numeric gradient check (`tests/test_loss.cpp`); wrapper value+grad
  (`tests/test_core_loss.cpp`)

### SGD optimiser
- Status: implemented
- Layer: nn::core `Sgd` (abstract `Optimizer` base)
- Math: `w <- w - lr * grad`
- Notes: baseline; steps over the flattened `ParamView`s from `Network::parameters()`. Trainer
  scales the seed gradient by 1/batch so the objective is the mean over the batch.
- Files: `src/includes/classes/optimizer.hpp`, `src/classes/optimizer.cpp`
- Tests: element-wise update + network step (`tests/test_optimizer.cpp`)

### SGD + weight decay
- Status: implemented (Phase 5)
- Layer: nn::core `SgdWeightDecay`
- Math: `w <- w - lr * (grad + lambda * w)`
- Notes: **key knob for grokking** — weight decay drives the transition to generalisation.
  Subclasses `Optimizer` alongside the existing `Sgd`. Biases are **excluded** from decay by
  default (`decay_bias = false`): decaying a bias shifts the decision boundary without reducing
  model complexity. Settled question; pass `decay_bias = true` to opt in.
- Files: `src/includes/classes/optimizer.hpp`, `src/classes/optimizer.cpp`
- Tests: weights decay, biases do not, decay combines with the gradient, and an unconstrained
  weight shrinks towards zero over many steps (`tests/test_optimizer.cpp`)

### Sigmoid activation
- Status: implemented (Phase 5)
- Layer: nn::math (fn) + nn::core `SigmoidLayer`
- Math: `sigma(z) = 1 / (1 + exp(-z))`
- Backward: `sigma(z) * (1 - sigma(z))`
- Notes: turns a logit into a probability. Overflow is avoided by branching on the sign of `z`:
  `1/(1+e^-z)` for `z >= 0`, `e^z/(1+e^z)` for `z < 0`, so the `exp` argument is never positive.
- Files: `src/includes/helpers/activations.hpp`, `src/helpers/activations.cpp`,
  `src/includes/classes/activation_layers.hpp`, `src/classes/activation_layers.cpp`
- Tests: monotonicity and range, `sigma(0) = 0.5`, `sigma'(0) = 0.25`, finiteness at `z = ±1000`,
  numeric gradient check (`tests/test_activations.cpp`, `tests/test_activation_layers.cpp`)

### Binary cross-entropy loss (fused with sigmoid)
- Status: implemented (Phase 5)
- Layer: nn::math (fn) + nn::core `BceWithLogitsLoss`
- Math: takes **logits**, not probabilities. `L = max(z, 0) - z*y + log(1 + exp(-|z|))`, averaged
  over outputs. This is algebraically identical to `-[y log(p) + (1-y) log(1-p)]` but never
  evaluates `exp` on a positive argument, so it cannot overflow.
- Backward: `dL/dz = (p - y) / N` — the sigmoid derivative cancels the `p(1-p)` denominator of
  plain BCE, which is exactly why fusing is worth it.
- Notes: **fused** form chosen (logit in, loss out). The network's output layer stays linear.
  `SigmoidLayer` still exists separately for inspecting probabilities in the network GUI.
  Two virtuals were added to the `Loss` base to keep the `Trainer` task-agnostic:
  `activate()` (identity by default, sigmoid here) and `is_classification()`.
- Files: `src/includes/helpers/loss.hpp`, `src/helpers/loss.cpp`,
  `src/includes/classes/loss.hpp`, `src/classes/loss.cpp`
- Tests: closed-form value at `z = 0`, confident-correct vs confident-wrong, numeric gradient
  check, and exact finite values at `z = ±800` where the naive form gives `inf`
  (`tests/test_loss.cpp`, `tests/test_core_loss.cpp`)

### Class-weighted binary cross-entropy with logits
- Status: implemented
- Layer: nn::math pure functions + nn::core `WeightedBceWithLogitsLoss`
- Math:
  `L = mean(w(y) * [max(z,0) - z*y + log(1 + exp(-|z|))])`, where `w(1)=w_pos` and
  `w(0)=w_neg`.
- Backward: `dL/dz = w(y) * (sigmoid(z) - y) / N`.
- Forward role: balances the aggregate training influence of rare primes and common composites
  without changing the natural validation/test distribution.
- Notes: `train_prime_transfer` uses normalised inverse-frequency weights
  `N/(2*N_class)`, giving average sample weight approximately one. Both weights must be positive.
- Files: `src/includes/helpers/loss.hpp`, `src/helpers/loss.cpp`,
  `src/includes/classes/loss.hpp`, `src/classes/loss.cpp`
- Tests: closed-form weighted value/gradient, numeric gradient check, core wrapper activation,
  and invalid-weight rejection (`tests/test_loss.cpp`, `tests/test_core_loss.cpp`)

### Classification accuracy metric
- Status: implemented (Phase 5; balanced binary metrics added in the prime-transfer follow-up)
- Layer: nn::api `Trainer`
- Notes: fraction of correct predictions after thresholding `Loss::activate(y_hat)` at 0.5,
  logged per split into `metrics.jsonl` next to `loss`. Emitted only when
  `Loss::is_classification()`, so regression runs are unchanged. Essential for the primes
  experiment because the 23.1% base rate means loss alone hides whether the model is doing
  better than "always composite". Single-output classifiers additionally log balanced accuracy,
  prime/composite recall, precision, and TP/TN/FP/FN counts.
- Files: `src/includes/api/trainer.hpp`, `src/api/trainer.cpp`
- Tests: every classification row carries in-range accuracy and binary rows carry the balanced
  metrics/counts; regression rows carry none (`tests/test_trainer.cpp`)

### Frozen dense parameters and replaceable transfer head
- Status: implemented
- Layer: nn::core `DenseLayer`, `ParamView`, `Network`, optimisers
- Math: a frozen block still propagates `dL/dx = W^T dL/dy`, but does not accumulate `dL/dW` or
  `dL/db`; optimisers skip its `ParamView` entirely, including weight decay.
- Forward role: preserves a learned encoder while a new output head is trained.
- Backward role: keeps gradient flow structurally valid while guaranteeing frozen values remain
  unchanged.
- Notes: `Network::remove_last()` replaces only the final head and rejects an empty network.
- Files: `src/includes/classes/{layer,dense_layer,network}.hpp`,
  `src/classes/{dense_layer,network,optimizer}.cpp`
- Tests: frozen gradients/optimizer update, input-gradient propagation, head replacement, and
  empty-network rejection (`tests/test_dense_layer.cpp`, `tests/test_optimizer.cpp`,
  `tests/test_network.cpp`)

### Weight initialisation (Xavier / He)
- Status: implemented
- Layer: nn::math
- Math: Xavier `~ U[-sqrt(6/(fan_in+fan_out)), +...]`; He `~ U[-sqrt(6/fan_in), +...]` (ReLU).
- Notes: seedable (`std::mt19937_64`) for reproducibility across parallel runs.
- Files: `src/includes/helpers/init.hpp`, `src/helpers/init.cpp`
- Tests: determinism per seed, range bound, mean ~0, variance ~ a^2/3 (`tests/test_init.cpp`)

### Backpropagation (the orchestration)
- Status: implemented
- Layer: nn::core `Network` (`forward` in order, `backward` in reverse, `parameters`, `zero_grad`)
- Math: reverse-mode chain rule across layers; each `Layer::backward` returns `dL/dx` and
  accumulates its own parameter grads.
- Notes: `Network` owns layers via `std::unique_ptr` and implements `nn::log::Loggable`;
  `parameters()` flattens every layer's `ParamView`s in forward order for logging.
- Files: `src/includes/classes/network.hpp`, `src/classes/network.cpp`,
  `src/includes/classes/layer.hpp`, `src/includes/classes/loggable.hpp`
- Tests: end-to-end numeric gradient check on the whole 1->4(tanh)->1 network — Phase 2 exit
  criterion (`tests/test_network.cpp`)

### Train/test split & sampling
- Status: implemented (generalised to named multi-split, vector-valued data in Phase 5)
- Layer: nn::api `Dataset` / `Split`
- Notes: a `Dataset` is a set of named `Split`s sharing one input/output dimensionality. Each
  `Split` stores flat row-major `inputs`, `targets`, and an `ids` vector — the per-sample
  identity used as the independent variable when plotting (`n` for primes, `x` for `x^2`). That
  single field is what lets one predictions schema serve both experiments.
  - `Dataset::x_squared(n_train, n_test, seed, n_grid)` → `train` (random), `test` and `grid`
    (even grids over [-1, 1]). Small train set is central to inducing grokking.
  - `Dataset::primes(PrimesConfig)` → `train` / `test` from a **stratified** shuffle (primes and
    composites split separately, so both sides carry primes at the same base rate), plus
    `unseen_201_255` / `unseen_256_300` for the bit encoding. See
    [EXPERIMENTS.md](EXPERIMENTS.md) for the encoding and 9-bit reasoning.
  - A non-zero `validation_fraction` creates a third stratified split before the remainder is
    assigned to test.
  - `Dataset::prime_residues(...)` reuses those exact IDs/inputs and replaces the labels with
    `[sin(2*pi*n/q), cos(2*pi*n/q)]` pairs for each modulus.
- Files: `src/includes/api/dataset.hpp`, `src/api/dataset.cpp`
- Tests: split sizes and disjointness, labels against a known prime list, stratification,
  60/20/20 class counts through 500, residue targets, bit-encoding round-trip, one-hot
  one-hotness, determinism (`tests/test_dataset.cpp`)

### Sieve of Eratosthenes (data generation)
- Status: implemented (Phase 5)
- Layer: nn::api free function `prime_sieve(n)`
- Math: mark multiples of each prime `p` starting at `p*p`; anything unmarked is prime.
- Notes: hand-written like everything else — no library lookup tables. Used for prime labels,
  class-balance counts, and held-out metric verification.
- Files: `src/includes/api/dataset.hpp`, `src/api/dataset.cpp`
- Tests: exact match against the 46 known primes ≤ 200 (`tests/test_dataset.cpp`)

### Training loop + JSONL logging (orchestration)
- Status: implemented
- Layer: nn::api `Trainer`, nn::log `JsonLine` / `JsonlSink`
- Notes: full-batch loop (forward per sample, grad accumulation, one `Sgd` step, `zero_grad`).
  Emits `metrics.jsonl` (per-step scalars), `params.jsonl` (per-param value+grad, interval-gated),
  `predictions.jsonl` (dense x-grid), plus `config.json` / `meta.json`, into `runs/<run_id>/`.
  Sequential phases may append to the same streams with a phase label and continuous global
  step; each phase also writes `config_<phase>.json`.
  JSON is hand-rolled (zero runtime deps) per the Phase 3 decision. `Loggable` supplies stable
  identity only; `Trainer` serialises metrics and `ParamView` data with `JsonLine`. Full config
  provenance, metadata lifecycle, and cross-stream run identity remain tracked in
  [RECTIFICATIONS.md](RECTIFICATIONS.md).
- Files: `src/includes/api/{run_config,trainer}.hpp`, `src/api/trainer.cpp`,
  `src/includes/classes/{json_line,jsonl_sink}.hpp`, `src/classes/{json_line,jsonl_sink}.cpp`
- Tests: JSON format/escaping, sink append round-trip (`tests/test_json.cpp`), and staged
  end-to-end runs with continuous steps (`tests/test_trainer.cpp`).

### (later) Adam / AdamW
- Status: planned
- Layer: nn::core
- Notes: revisit if SGD+decay is too slow to grok.
