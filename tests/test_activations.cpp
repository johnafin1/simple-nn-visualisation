#include <array>
#include <cmath>
#include <numeric>
#include <span>
#include <vector>

#include <doctest/doctest.h>

#include "grad_check.hpp"
#include "helpers/activations.hpp"

using nn::math::relu;
using nn::math::relu_grad;
using nn::math::sigmoid;
using nn::math::sigmoid_grad;
using nn::math::tanh_;
using nn::math::tanh_grad;

TEST_CASE("sigmoid maps 0 to 0.5 and saturates monotonically") {
    const std::array<double, 5> z{-4.0, -1.0, 0.0, 1.0, 4.0};
    std::array<double, 5> out{};
    sigmoid(z, out);
    CHECK(out[2] == doctest::Approx(0.5));
    CHECK(out[1] == doctest::Approx(1.0 / (1.0 + std::exp(1.0))));
    CHECK(out[3] == doctest::Approx(1.0 / (1.0 + std::exp(-1.0))));
    for (std::size_t i = 1; i < out.size(); ++i) {
        CHECK(out[i] > out[i - 1]);
    }
    for (const double p : out) {
        CHECK(p > 0.0);
        CHECK(p < 1.0);
    }
}

TEST_CASE("sigmoid does not overflow at extreme inputs") {
    // The naive 1/(1+exp(-z)) overflows exp() for very negative z; the sign-branching
    // implementation must stay finite and in range.
    const std::array<double, 4> z{-1000.0, -750.0, 750.0, 1000.0};
    std::array<double, 4> out{};
    sigmoid(z, out);
    for (const double p : out) {
        CHECK(std::isfinite(p));
        CHECK(p >= 0.0);
        CHECK(p <= 1.0);
    }
    CHECK(out[0] == doctest::Approx(0.0));
    CHECK(out[3] == doctest::Approx(1.0));

    std::array<double, 4> grad{};
    sigmoid_grad(z, grad);
    for (const double g : grad) {
        CHECK(std::isfinite(g));
    }
}

TEST_CASE("sigmoid_grad peaks at 0.25 when z = 0") {
    const std::array<double, 1> z{0.0};
    std::array<double, 1> out{};
    sigmoid_grad(z, out);
    CHECK(out[0] == doctest::Approx(0.25));
}

TEST_CASE("sigmoid_grad matches the numeric derivative") {
    const std::vector<double> z{-2.5, -0.4, 0.0, 0.9, 3.1};
    std::vector<double> analytic(z.size());
    sigmoid_grad(std::span<const double>(z), std::span<double>(analytic));

    for (std::size_t i = 0; i < z.size(); ++i) {
        const auto numeric = nn::test::numeric_gradient(
            [](std::span<const double> v) {
                std::array<double, 1> out{};
                sigmoid(v, out);
                return out[0];
            },
            std::vector<double>{z[i]});
        CHECK(analytic[i] == doctest::Approx(numeric[0]).epsilon(1e-6));
    }
}

TEST_CASE("relu clamps negatives to zero") {
    const std::array<double, 4> z{-2.0, -0.5, 0.5, 3.0};
    std::array<double, 4> out{};
    relu(z, out);
    CHECK(out[0] == doctest::Approx(0.0));
    CHECK(out[1] == doctest::Approx(0.0));
    CHECK(out[2] == doctest::Approx(0.5));
    CHECK(out[3] == doctest::Approx(3.0));
}

TEST_CASE("tanh_ matches std::tanh") {
    const std::array<double, 3> z{-1.0, 0.0, 1.0};
    std::array<double, 3> out{};
    tanh_(z, out);
    CHECK(out[0] == doctest::Approx(std::tanh(-1.0)));
    CHECK(out[1] == doctest::Approx(0.0));
    CHECK(out[2] == doctest::Approx(std::tanh(1.0)));
}

// Gradient checks: for an element-wise activation, d/dz_i of sum(act(z)) equals
// act_grad(z)_i. Compare the hand-written grad to a central-difference estimate.
TEST_CASE("relu_grad matches the numeric gradient (away from 0)") {
    const std::vector<double> z{-2.0, -0.3, 0.7, 2.5};
    std::array<double, 4> analytic{};
    relu_grad(std::span<const double>(z), analytic);

    const auto numeric = nn::test::numeric_gradient(
        [](std::span<const double> v) {
            std::vector<double> o(v.size());
            relu(v, o);
            return std::accumulate(o.begin(), o.end(), 0.0);
        },
        z);

    for (std::size_t i = 0; i < z.size(); ++i) {
        CHECK(analytic[i] == doctest::Approx(numeric[i]).epsilon(1e-6));
    }
}

TEST_CASE("tanh_grad matches the numeric gradient") {
    const std::vector<double> z{-1.5, -0.2, 0.4, 1.1};
    std::array<double, 4> analytic{};
    tanh_grad(std::span<const double>(z), analytic);

    const auto numeric = nn::test::numeric_gradient(
        [](std::span<const double> v) {
            std::vector<double> o(v.size());
            tanh_(v, o);
            return std::accumulate(o.begin(), o.end(), 0.0);
        },
        z);

    for (std::size_t i = 0; i < z.size(); ++i) {
        CHECK(analytic[i] == doctest::Approx(numeric[i]).epsilon(1e-6));
    }
}
