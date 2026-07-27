#include <array>
#include <cmath>
#include <span>
#include <vector>

#include <doctest/doctest.h>

#include "grad_check.hpp"
#include "helpers/loss.hpp"

using nn::math::bce_with_logits;
using nn::math::bce_with_logits_grad;
using nn::math::mse;
using nn::math::mse_grad;
using nn::math::weighted_bce_with_logits;
using nn::math::weighted_bce_with_logits_grad;

TEST_CASE("mse computes mean squared error") {
    // y_hat = {1,2,3}, y = {1,1,1} -> (0 + 1 + 4) / 3 = 5/3
    const std::array<double, 3> y_hat{1.0, 2.0, 3.0};
    const std::array<double, 3> y{1.0, 1.0, 1.0};
    CHECK(mse(y_hat, y) == doctest::Approx(5.0 / 3.0));
}

TEST_CASE("mse is zero for a perfect prediction") {
    const std::array<double, 4> y_hat{-2.0, 0.5, 3.0, 7.0};
    CHECK(mse(y_hat, y_hat) == doctest::Approx(0.0));
}

TEST_CASE("mse_grad matches the closed form 2*(y_hat - y)/N") {
    const std::array<double, 3> y_hat{1.0, 2.0, 3.0};
    const std::array<double, 3> y{1.0, 1.0, 1.0};
    std::array<double, 3> grad{};
    mse_grad(y_hat, y, grad);
    // 2/3 * {0, 1, 2}
    CHECK(grad[0] == doctest::Approx(0.0));
    CHECK(grad[1] == doctest::Approx(2.0 / 3.0));
    CHECK(grad[2] == doctest::Approx(4.0 / 3.0));
}

TEST_CASE("mse_grad matches the numeric gradient w.r.t. y_hat") {
    const std::vector<double> y_hat{0.3, -1.2, 2.5, 0.8};
    const std::vector<double> y{0.0, 1.0, 2.0, -0.5};

    std::array<double, 4> analytic{};
    mse_grad(std::span<const double>(y_hat), std::span<const double>(y), analytic);

    const auto numeric = nn::test::numeric_gradient(
        [&y](std::span<const double> v) { return mse(v, std::span<const double>(y)); },
        y_hat);

    for (std::size_t i = 0; i < y_hat.size(); ++i) {
        CHECK(analytic[i] == doctest::Approx(numeric[i]).epsilon(1e-6));
    }
}

TEST_CASE("bce_with_logits matches the closed form at z = 0") {
    // p = 0.5 for either target, so L = -log(0.5) = log(2) per element.
    const std::array<double, 2> z{0.0, 0.0};
    const std::array<double, 2> y{0.0, 1.0};
    CHECK(bce_with_logits(z, y) == doctest::Approx(std::log(2.0)));
}

TEST_CASE("bce_with_logits rewards confident correct predictions") {
    const std::array<double, 2> z{8.0, -8.0};
    const std::array<double, 2> correct{1.0, 0.0};
    const std::array<double, 2> wrong{0.0, 1.0};
    CHECK(bce_with_logits(z, correct) < 0.001);
    CHECK(bce_with_logits(z, wrong) > 7.0);
}

TEST_CASE("bce_with_logits stays finite at extreme logits") {
    // The naive -[y log p + (1-y) log(1-p)] form would produce inf/NaN here because
    // sigmoid(+-800) saturates exactly to 1.0 / 0.0 in double precision.
    const std::array<double, 4> z{800.0, -800.0, 750.0, -750.0};
    const std::array<double, 4> y{1.0, 0.0, 0.0, 1.0};
    const double L = bce_with_logits(z, y);
    CHECK(std::isfinite(L));
    // Two confident-correct (contributing ~0) and two confident-wrong (contributing |z|).
    CHECK(L == doctest::Approx((750.0 + 750.0) / 4.0));

    std::array<double, 4> grad{};
    bce_with_logits_grad(z, y, grad);
    for (const double g : grad) {
        CHECK(std::isfinite(g));
    }
}

TEST_CASE("bce_with_logits_grad is (sigmoid(z) - y)/N") {
    const std::array<double, 2> z{0.0, 0.0};
    const std::array<double, 2> y{1.0, 0.0};
    std::array<double, 2> grad{};
    bce_with_logits_grad(z, y, grad);
    CHECK(grad[0] == doctest::Approx((0.5 - 1.0) / 2.0));
    CHECK(grad[1] == doctest::Approx((0.5 - 0.0) / 2.0));
}

TEST_CASE("bce_with_logits_grad matches the numeric gradient w.r.t. the logits") {
    const std::vector<double> z{0.4, -1.7, 3.2, -0.2};
    const std::vector<double> y{1.0, 0.0, 1.0, 0.0};

    std::array<double, 4> analytic{};
    bce_with_logits_grad(std::span<const double>(z), std::span<const double>(y), analytic);

    const auto numeric = nn::test::numeric_gradient(
        [&y](std::span<const double> v) {
            return bce_with_logits(v, std::span<const double>(y));
        },
        z);

    for (std::size_t i = 0; i < z.size(); ++i) {
        CHECK(analytic[i] == doctest::Approx(numeric[i]).epsilon(1e-6));
    }
}

TEST_CASE("weighted_bce_with_logits applies separate class weights") {
    const std::array<double, 2> z{0.0, 0.0};
    const std::array<double, 2> y{1.0, 0.0};
    CHECK(weighted_bce_with_logits(z, y, 4.0, 1.0) ==
          doctest::Approx(2.5 * std::log(2.0)));

    std::array<double, 2> grad{};
    weighted_bce_with_logits_grad(z, y, 4.0, 1.0, grad);
    CHECK(grad[0] == doctest::Approx(-1.0));
    CHECK(grad[1] == doctest::Approx(0.25));
}

TEST_CASE("weighted_bce_with_logits_grad matches its numeric gradient") {
    const std::vector<double> z{0.4, -1.7, 3.2, -0.2};
    const std::vector<double> y{1.0, 0.0, 1.0, 0.0};
    std::array<double, 4> analytic{};
    weighted_bce_with_logits_grad(z, y, 2.5, 0.6, analytic);

    const auto numeric = nn::test::numeric_gradient(
        [&y](std::span<const double> v) {
            return weighted_bce_with_logits(v, std::span<const double>(y), 2.5, 0.6);
        },
        z);
    for (std::size_t i = 0; i < z.size(); ++i) {
        CHECK(analytic[i] == doctest::Approx(numeric[i]).epsilon(1e-6));
    }
}
