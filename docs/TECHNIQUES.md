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
- Status: planned
- Layer: nn::core (uses nn::math)
- Math: `y = W x + b`
- Backward: `dW = dy x^T`, `db = dy`, `dx = W^T dy`
- Notes: main carrier of parameters; where per-param logging matters most.
- Tests: numeric gradient check on W and b.

### ReLU activation
- Status: planned
- Layer: nn::math (fn) + thin nn::core layer wrapper
- Math: `relu(z) = max(0, z)`
- Backward: `1 if z > 0 else 0`
- Notes: dead-unit risk on tiny nets; watch during grokking runs.

### Tanh activation
- Status: planned
- Layer: nn::math + nn::core wrapper
- Math: `tanh(z)`
- Backward: `1 - tanh(z)^2`
- Notes: smooth; often a better fit for a smooth target like `x^2`.

### MSE loss
- Status: planned
- Layer: nn::math + nn::core `MseLoss`
- Math: `L = mean((y_hat - y)^2)`
- Backward: `dL/dy_hat = 2 (y_hat - y) / N`
- Tests: closed-form check on small inputs.

### SGD optimiser
- Status: planned
- Layer: nn::core `Sgd`
- Math: `w <- w - lr * grad`
- Notes: baseline; deterministic with fixed seed.

### SGD + weight decay
- Status: planned
- Layer: nn::core `SgdWeightDecay`
- Math: `w <- w - lr * (grad + lambda * w)`
- Notes: **key knob for grokking** — weight decay drives the transition to generalisation.

### Weight initialisation (Xavier / He)
- Status: planned
- Layer: nn::math
- Math: Xavier `~ U[-sqrt(6/(fan_in+fan_out)), +...]`; He scaled for ReLU.
- Notes: seedable for reproducibility across parallel runs.

### Backpropagation (the orchestration)
- Status: planned
- Layer: nn::core `Network::backward`
- Math: reverse-mode chain rule across layers.
- Tests: end-to-end numeric gradient check on the whole network.

### Train/test split & sampling
- Status: planned
- Layer: nn::api `Dataset`
- Notes: small train set is central to inducing grokking; exact sampling decided in
  [EXPERIMENTS.md](EXPERIMENTS.md).

### (later) Adam / AdamW
- Status: planned
- Layer: nn::core
- Notes: revisit if SGD+decay is too slow to grok.
