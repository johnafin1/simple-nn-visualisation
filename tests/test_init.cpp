#include <cmath>
#include <cstddef>
#include <vector>

#include <doctest/doctest.h>

#include "helpers/init.hpp"

using nn::math::he_uniform;
using nn::math::xavier_uniform;

namespace {

double mean(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double max_abs(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

}  // namespace

TEST_CASE("xavier_uniform is deterministic for a fixed seed") {
    std::vector<double> a(64), b(64);
    xavier_uniform(a, 8, 8, 42);
    xavier_uniform(b, 8, 8, 42);
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i] == doctest::Approx(b[i]));
    }
}

TEST_CASE("different seeds produce different fills") {
    std::vector<double> a(64), b(64);
    xavier_uniform(a, 8, 8, 1);
    xavier_uniform(b, 8, 8, 2);
    bool any_different = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            any_different = true;
            break;
        }
    }
    CHECK(any_different);
}

TEST_CASE("xavier_uniform respects its range and is roughly zero-mean") {
    const std::size_t fan_in = 30, fan_out = 20;
    std::vector<double> w(10000);
    xavier_uniform(w, fan_in, fan_out, 123);

    const double limit = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
    CHECK(max_abs(w) <= limit);          // never exceeds the uniform bound
    CHECK(std::abs(mean(w)) < 0.02);     // symmetric distribution -> mean ~ 0
}

TEST_CASE("he_uniform respects its range and is roughly zero-mean") {
    const std::size_t fan_in = 50;
    std::vector<double> w(10000);
    he_uniform(w, fan_in, 7);

    const double limit = std::sqrt(6.0 / static_cast<double>(fan_in));
    CHECK(max_abs(w) <= limit);
    CHECK(std::abs(mean(w)) < 0.02);
}

TEST_CASE("he_uniform variance is close to the uniform expectation a^2/3") {
    const std::size_t fan_in = 40;
    std::vector<double> w(20000);
    he_uniform(w, fan_in, 99);

    const double limit = std::sqrt(6.0 / static_cast<double>(fan_in));
    const double expected_var = (limit * limit) / 3.0;

    const double m = mean(w);
    double var = 0.0;
    for (double x : w) var += (x - m) * (x - m);
    var /= static_cast<double>(w.size());

    // Generous tolerance: sampling noise on a finite draw.
    CHECK(var == doctest::Approx(expected_var).epsilon(0.1));
}
