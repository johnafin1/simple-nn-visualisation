#include <span>
#include <vector>

#include <doctest/doctest.h>

#include "classes/dense_layer.hpp"
#include "classes/tensor.hpp"
#include "grad_check.hpp"

using nn::core::DenseLayer;
using nn::core::InitKind;
using nn::core::ParamView;
using nn::core::Tensor;
using nn::test::numeric_gradient;

namespace {

// Scalar objective L = dot(y, c) for a fixed vector c, so dL/dy = c.
double objective(const Tensor& y, std::span<const double> c) {
    double sum = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        sum += y.cspan()[i] * c[i];
    }
    return sum;
}

}  // namespace

TEST_CASE("DenseLayer forward computes Wx + b") {
    DenseLayer dense(2, 3, InitKind::Xavier, 42, "dense.0");
    // Overwrite weights/bias to known values via parameters().
    auto params = dense.parameters();
    // params[0] = weight (3x2), params[1] = bias (3x1)
    std::vector<double> w{1, 2, 3, 4, 5, 6};
    for (std::size_t i = 0; i < w.size(); ++i) params[0].values[i] = w[i];
    std::vector<double> b{0.5, -0.5, 1.0};
    for (std::size_t i = 0; i < b.size(); ++i) params[1].values[i] = b[i];

    Tensor y = dense.forward(Tensor::column({1.0, 1.0}));
    // row0: 1+2+0.5=3.5, row1: 3+4-0.5=6.5, row2: 5+6+1=12
    CHECK(y.at(0, 0) == doctest::Approx(3.5));
    CHECK(y.at(1, 0) == doctest::Approx(6.5));
    CHECK(y.at(2, 0) == doctest::Approx(12.0));
}

TEST_CASE("DenseLayer parameter shapes and log name") {
    DenseLayer dense(2, 3, InitKind::He, 7, "dense.0");
    CHECK(dense.log_name() == "dense.0");
    auto params = dense.parameters();
    REQUIRE(params.size() == 2);
    CHECK(params[0].name == "weight");
    CHECK(params[0].rows == 3);
    CHECK(params[0].cols == 2);
    CHECK(params[1].name == "bias");
    CHECK(params[1].rows == 3);
    CHECK(params[1].cols == 1);
}

TEST_CASE("DenseLayer analytic gradients match numeric") {
    const std::size_t in = 3, out = 2;
    DenseLayer dense(in, out, InitKind::Xavier, 123, "dense.0");
    const Tensor x = Tensor::column({0.5, -1.5, 2.0});
    const std::vector<double> c{0.7, -1.3};  // dL/dy

    dense.zero_grad();
    dense.forward(x);
    dense.backward(Tensor::column(c));

    auto params = dense.parameters();

    // --- weights ---
    std::vector<double> w0(params[0].values.begin(), params[0].values.end());
    auto num_w = numeric_gradient(
        [&](std::span<const double> wv) {
            for (std::size_t i = 0; i < wv.size(); ++i) params[0].values[i] = wv[i];
            Tensor y = dense.forward(x);
            return objective(y, c);
        },
        w0);
    for (std::size_t i = 0; i < w0.size(); ++i) params[0].values[i] = w0[i];  // restore
    for (std::size_t i = 0; i < num_w.size(); ++i) {
        CHECK(params[0].grads[i] == doctest::Approx(num_w[i]).epsilon(1e-6));
    }

    // --- bias ---
    std::vector<double> b0(params[1].values.begin(), params[1].values.end());
    auto num_b = numeric_gradient(
        [&](std::span<const double> bv) {
            for (std::size_t i = 0; i < bv.size(); ++i) params[1].values[i] = bv[i];
            Tensor y = dense.forward(x);
            return objective(y, c);
        },
        b0);
    for (std::size_t i = 0; i < b0.size(); ++i) params[1].values[i] = b0[i];
    for (std::size_t i = 0; i < num_b.size(); ++i) {
        CHECK(params[1].grads[i] == doctest::Approx(num_b[i]).epsilon(1e-6));
    }
}

TEST_CASE("DenseLayer zero_grad clears accumulation") {
    DenseLayer dense(2, 2, InitKind::He, 1, "dense.0");
    dense.forward(Tensor::column({1.0, 1.0}));
    dense.backward(Tensor::column({1.0, 1.0}));
    dense.zero_grad();
    auto params = dense.parameters();
    for (double g : params[0].grads) CHECK(g == doctest::Approx(0.0));
    for (double g : params[1].grads) CHECK(g == doctest::Approx(0.0));
}
