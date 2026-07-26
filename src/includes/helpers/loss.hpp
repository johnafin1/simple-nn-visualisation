#pragma once

#include <span>

namespace nn::math {

// Mean squared error: mean((y_hat - y)^2) over all N elements.
[[nodiscard]] double mse(std::span<const double> y_hat, std::span<const double> y);

// Gradient of MSE w.r.t. predictions: dL/dy_hat[i] = 2 * (y_hat[i] - y[i]) / N.
void mse_grad(std::span<const double> y_hat, std::span<const double> y,
              std::span<double> grad_out);

// Binary cross-entropy computed directly from logits z (NOT probabilities), averaged
// over the N elements. Targets y must be 0 or 1.
//
// Fusing the sigmoid into the loss is what makes this numerically safe: the naive
// -[y log(p) + (1-y) log(1-p)] blows up as p saturates to 0 or 1. The identity
//     L = max(z, 0) - z*y + log(1 + exp(-|z|))
// is exact and never evaluates exp() on a positive argument.
[[nodiscard]] double bce_with_logits(std::span<const double> z, std::span<const double> y);

// Gradient w.r.t. the logits, which collapses to the very stable
//     dL/dz[i] = (sigmoid(z[i]) - y[i]) / N
// because the sigmoid derivative cancels the p(1-p) denominator of plain BCE.
void bce_with_logits_grad(std::span<const double> z, std::span<const double> y,
                          std::span<double> grad_out);

}  // namespace nn::math
