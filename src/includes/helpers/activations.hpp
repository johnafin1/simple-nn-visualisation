#pragma once

#include <span>

namespace nn::math {

// Element-wise activations. Each writes into `out`, which must match `z` in length.
// Gradients are element-wise derivatives d(act)/dz evaluated at z (the local
// derivative used during backprop), NOT multiplied by any upstream gradient.

// ReLU: out[i] = max(0, z[i]).
void relu(std::span<const double> z, std::span<double> out);
// d/dz ReLU: 1 if z[i] > 0 else 0 (subgradient 0 at z == 0).
void relu_grad(std::span<const double> z, std::span<double> out);

// tanh (trailing underscore to avoid colliding with std::tanh).
void tanh_(std::span<const double> z, std::span<double> out);
// d/dz tanh = 1 - tanh(z)^2.
void tanh_grad(std::span<const double> z, std::span<double> out);

// Logistic sigmoid: out[i] = 1 / (1 + exp(-z[i])).
// Branches on the sign of z so exp() never overflows for large |z|.
void sigmoid(std::span<const double> z, std::span<double> out);
// d/dz sigmoid = sigmoid(z) * (1 - sigmoid(z)).
void sigmoid_grad(std::span<const double> z, std::span<double> out);

}  // namespace nn::math
