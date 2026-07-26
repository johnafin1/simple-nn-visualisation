#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace nn::test {

// Central-difference numeric gradient of a scalar function f: R^n -> R.
// f takes the current point as a std::span<const double> and returns a scalar.
// Returns df/dx_i for every i, accurate to O(h^2).
template <class F>
std::vector<double> numeric_gradient(F&& f, std::vector<double> x, double h = 1e-6) {
    std::vector<double> grad(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double original = x[i];

        x[i] = original + h;
        const double f_plus = f(std::span<const double>(x));

        x[i] = original - h;
        const double f_minus = f(std::span<const double>(x));

        x[i] = original;
        grad[i] = (f_plus - f_minus) / (2.0 * h);
    }
    return grad;
}

}  // namespace nn::test
