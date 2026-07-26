# Phase 1 Handoff

Prescriptive continuation guide for finishing the `nn::math` layer. Follow the
checklist exactly. Do **not** invent new APIs, add C++ dependencies, or start
Phase 2 (object layer) without explicit user sign-off.

## Where we are

Phase 1 math layer is **partly done**:

| Module | Status | Files |
|--------|--------|-------|
| `linalg` | **done** | `src/includes/helpers/linalg.hpp`, `src/helpers/linalg.cpp`, `tests/test_linalg.cpp` |
| `activations` | **done** | `src/includes/helpers/activations.hpp`, `src/helpers/activations.cpp`, `tests/test_activations.cpp` |
| `loss` | **TODO** | — |
| `init` | **TODO** | — |

Authoritative signatures and exit criteria: [docs/plans/phase-1.md](plans/phase-1.md).
Architecture / conventions: [docs/ARCHITECTURE.md](ARCHITECTURE.md),
[docs/CPP_CONVENTIONS.md](CPP_CONVENTIONS.md),
[.cursor/rules/project-workflow.mdc](../.cursor/rules/project-workflow.mdc).

## Guardrails (do not violate)

- `nn::math` = **pure free functions** over `std::span`. No classes, no hidden state,
  no allocation on the caller's behalf.
- Scalar type: **`double`**. Matrices are **row-major**; shapes passed explicitly.
- No new third-party C++ dependencies. doctest is already wired.
- Gradient-check tolerance: **`1e-6`** (central differences via
  [tests/grad_check.hpp](../tests/grad_check.hpp)).
- Do **not** start Phase 2 (`Tensor`, `Layer`, `Network`, backprop objects) until
  Phase 1 exit criteria are met and the user signs off.
- Keep the user involved: present mermaid structures before new class/module work
  (not needed for the remaining free functions — copy the activations pattern).

## The proven recipe (copy activations exactly)

For each remaining module (`loss`, then `init`):

1. Add declarations to `src/includes/helpers/<name>.hpp` using the **exact**
   signatures below (also in [docs/plans/phase-1.md](plans/phase-1.md)).
2. Add implementations to `src/helpers/<name>.cpp`.
3. Add the `.cpp` to the `nn` library sources in [src/CMakeLists.txt](../src/CMakeLists.txt)
   (currently lists `helpers/linalg.cpp` and `helpers/activations.cpp`).
4. Add `tests/test_<name>.cpp`:
   - closed-form / property checks for values;
   - where a gradient exists, numeric check with `nn::test::numeric_gradient`
     (see [tests/test_activations.cpp](../tests/test_activations.cpp) for the pattern).
5. Add the test file to [tests/CMakeLists.txt](../tests/CMakeLists.txt)
   (`nn_tests` executable sources).
6. Build + test:
   ```powershell
   cmake --build build
   ctest --test-dir build --output-on-failure
   ```
   Or use the Streamlit dashboard (below).
7. Flip status to `implemented` in [docs/TECHNIQUES.md](TECHNIQUES.md) with file
   + test references (match the ReLU/Tanh entries).

## Exact next task 1: `loss`

### Header — `src/includes/helpers/loss.hpp`

```cpp
#pragma once

#include <span>

namespace nn::math {

[[nodiscard]] double mse(std::span<const double> y_hat, std::span<const double> y);
// dL/dy_hat = 2*(y_hat - y)/N
void mse_grad(std::span<const double> y_hat, std::span<const double> y,
              std::span<double> grad_out);

}  // namespace nn::math
```

### Implementation notes — `src/helpers/loss.cpp`

- Assert equal lengths for `y_hat` and `y` (and `grad_out` for `mse_grad`).
- `mse`: `mean((y_hat - y)^2)` = `sum((y_hat[i]-y[i])^2) / N`. Empty span → treat
  carefully (assert non-empty or return 0.0; prefer assert non-empty like other ops).
- `mse_grad`: for each `i`, `grad_out[i] = 2.0 * (y_hat[i] - y[i]) / N`.

### Tests — `tests/test_loss.cpp`

- Closed-form: e.g. `y_hat={1,2,3}`, `y={1,1,1}` → MSE = `((0)^2+(1)^2+(2)^2)/3 = 5/3`.
- Gradient check: analytic `mse_grad` vs `numeric_gradient` of `mse` w.r.t. `y_hat`
  (fix `y`, vary `y_hat`), tolerance `1e-6`.

### TECHNIQUES.md

Update **MSE loss** from `planned` → `implemented (nn::math free functions; MseLoss
wrapper pending in nn::core)` with files + tests.

## Exact next task 2: `init`

### Header — `src/includes/helpers/init.hpp`

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace nn::math {

// Fill weights in-place using a seeded RNG.
void xavier_uniform(std::span<double> w, std::size_t fan_in, std::size_t fan_out,
                    std::uint64_t seed);
void he_uniform(std::span<double> w, std::size_t fan_in, std::uint64_t seed);

}  // namespace nn::math
```

### Implementation notes — `src/helpers/init.cpp`

- Use `std::mt19937_64` constructed from `seed` (deterministic).
- **Xavier uniform:** sample each weight from
  `U[-a, a]` where `a = sqrt(6.0 / (fan_in + fan_out))`.
- **He uniform:** sample each weight from
  `U[-a, a]` where `a = sqrt(6.0 / fan_in)` (common He/Kaiming uniform scale for ReLU).
- Fill `w` in-place; do not allocate a new buffer.

### Tests — `tests/test_init.cpp`

- **Determinism:** same args + seed → identical fills (element-wise equal).
- **Different seeds:** fills differ.
- **Mean ~ 0:** large `w` (e.g. 10_000 elements), `|mean|` small (e.g. `< 0.05`).
- **Variance / range:** samples stay within `[-a, a]`; optional loose variance check
  against `a^2 / 3` (uniform variance) with a generous tolerance.

### TECHNIQUES.md

Update **Weight initialisation (Xavier / He)** from `planned` → `implemented` with
files + tests.

## How to see results

### Streamlit test dashboard (preferred while iterating)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\streamlit.exe run src/app/python/test_dashboard.py
```

Auto-rebuilds and re-runs tests when you save `.cpp`/`.hpp` under `src/` or `tests/`.
Shows per-case green/red and compiler errors on build failure.

### CLI

```powershell
cmake --build build
ctest --test-dir build --output-on-failure
# or:
.\build\tests\nn_tests.exe
```

If `build/` is missing:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=g++
```

## Definition of done (Phase 1)

- [ ] `loss` implemented + tested (closed-form + numeric grad).
- [ ] `init` implemented + tested (determinism + stats).
- [ ] Both `.cpp` files registered in `src/CMakeLists.txt`.
- [ ] Both test files registered in `tests/CMakeLists.txt`.
- [ ] `ctest` green (all cases pass).
- [ ] [docs/TECHNIQUES.md](TECHNIQUES.md) updated for MSE and Xavier/He.
- [ ] No Phase 2 code started.

Then stop and wait for the user to sign off before Phase 2.
