#include "helpers/activations.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

namespace nn::math {

void relu(std::span<const double> z, std::span<double> out) {
    assert(z.size() == out.size() && "relu: size mismatch");
    for (std::size_t i = 0; i < z.size(); ++i) {
        out[i] = z[i] > 0.0 ? z[i] : 0.0;
    }
}

void relu_grad(std::span<const double> z, std::span<double> out) {
    assert(z.size() == out.size() && "relu_grad: size mismatch");
    for (std::size_t i = 0; i < z.size(); ++i) {
        out[i] = z[i] > 0.0 ? 1.0 : 0.0;
    }
}

void tanh_(std::span<const double> z, std::span<double> out) {
    assert(z.size() == out.size() && "tanh_: size mismatch");
    for (std::size_t i = 0; i < z.size(); ++i) {
        out[i] = std::tanh(z[i]);
    }
}

void tanh_grad(std::span<const double> z, std::span<double> out) {
    assert(z.size() == out.size() && "tanh_grad: size mismatch");
    for (std::size_t i = 0; i < z.size(); ++i) {
        const double t = std::tanh(z[i]);
        out[i] = 1.0 - t * t;
    }
}

namespace {

// Overflow-safe logistic sigmoid. For z >= 0 use 1/(1+exp(-z)); for z < 0 use
// exp(z)/(1+exp(z)). Either way the exp() argument is <= 0, so it cannot overflow.
double sigmoid_one(double z) {
    if (z >= 0.0) {
        return 1.0 / (1.0 + std::exp(-z));
    }
    const double e = std::exp(z);
    return e / (1.0 + e);
}

}  // namespace

void sigmoid(std::span<const double> z, std::span<double> out) {
    assert(z.size() == out.size() && "sigmoid: size mismatch");
    for (std::size_t i = 0; i < z.size(); ++i) {
        out[i] = sigmoid_one(z[i]);
    }
}

void sigmoid_grad(std::span<const double> z, std::span<double> out) {
    assert(z.size() == out.size() && "sigmoid_grad: size mismatch");
    for (std::size_t i = 0; i < z.size(); ++i) {
        const double s = sigmoid_one(z[i]);
        out[i] = s * (1.0 - s);
    }
}

}  // namespace nn::math
