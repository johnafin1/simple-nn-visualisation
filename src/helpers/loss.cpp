#include "helpers/loss.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

#include "helpers/activations.hpp"

namespace nn::math {

double mse(std::span<const double> y_hat, std::span<const double> y) {
    assert(y_hat.size() == y.size() && "mse: size mismatch");
    assert(!y_hat.empty() && "mse: empty input");
    double sum = 0.0;
    for (std::size_t i = 0; i < y_hat.size(); ++i) {
        const double diff = y_hat[i] - y[i];
        sum += diff * diff;
    }
    return sum / static_cast<double>(y_hat.size());
}

void mse_grad(std::span<const double> y_hat, std::span<const double> y,
              std::span<double> grad_out) {
    assert(y_hat.size() == y.size() && "mse_grad: size mismatch");
    assert(y_hat.size() == grad_out.size() && "mse_grad: grad_out size mismatch");
    assert(!y_hat.empty() && "mse_grad: empty input");
    const double scale = 2.0 / static_cast<double>(y_hat.size());
    for (std::size_t i = 0; i < y_hat.size(); ++i) {
        grad_out[i] = scale * (y_hat[i] - y[i]);
    }
}

double bce_with_logits(std::span<const double> z, std::span<const double> y) {
    assert(z.size() == y.size() && "bce_with_logits: size mismatch");
    assert(!z.empty() && "bce_with_logits: empty input");
    double sum = 0.0;
    for (std::size_t i = 0; i < z.size(); ++i) {
        const double zi = z[i];
        sum += std::max(zi, 0.0) - zi * y[i] + std::log1p(std::exp(-std::abs(zi)));
    }
    return sum / static_cast<double>(z.size());
}

void bce_with_logits_grad(std::span<const double> z, std::span<const double> y,
                          std::span<double> grad_out) {
    assert(z.size() == y.size() && "bce_with_logits_grad: size mismatch");
    assert(z.size() == grad_out.size() && "bce_with_logits_grad: grad_out size mismatch");
    assert(!z.empty() && "bce_with_logits_grad: empty input");
    sigmoid(z, grad_out);  // grad_out = p
    const double inv_n = 1.0 / static_cast<double>(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) {
        grad_out[i] = (grad_out[i] - y[i]) * inv_n;
    }
}

}  // namespace nn::math
