#include <vector>

#include <doctest/doctest.h>

#include "classes/activation_layers.hpp"
#include "classes/tensor.hpp"

using nn::core::ReluLayer;
using nn::core::SigmoidLayer;
using nn::core::TanhLayer;
using nn::core::Tensor;

TEST_CASE("ReluLayer forward clamps negatives") {
    ReluLayer relu;
    Tensor y = relu.forward(Tensor::column({-2.0, 0.0, 3.0}));
    CHECK(y.at(0, 0) == doctest::Approx(0.0));
    CHECK(y.at(1, 0) == doctest::Approx(0.0));
    CHECK(y.at(2, 0) == doctest::Approx(3.0));
}

TEST_CASE("ReluLayer backward gates by the cached input sign") {
    ReluLayer relu;
    relu.forward(Tensor::column({-2.0, 0.0, 3.0}));
    Tensor g = relu.backward(Tensor::column({1.0, 1.0, 1.0}));
    CHECK(g.at(0, 0) == doctest::Approx(0.0));
    CHECK(g.at(1, 0) == doctest::Approx(0.0));  // subgradient 0 at z == 0
    CHECK(g.at(2, 0) == doctest::Approx(1.0));
}

TEST_CASE("TanhLayer forward and backward") {
    TanhLayer tanh;
    Tensor y = tanh.forward(Tensor::column({0.0}));
    CHECK(y.at(0, 0) == doctest::Approx(0.0));
    // grad at z = 0 is 1 - tanh(0)^2 = 1
    Tensor g = tanh.backward(Tensor::column({2.0}));
    CHECK(g.at(0, 0) == doctest::Approx(2.0));
}

TEST_CASE("SigmoidLayer forward and backward") {
    SigmoidLayer sig;
    Tensor y = sig.forward(Tensor::column({0.0}));
    CHECK(y.at(0, 0) == doctest::Approx(0.5));
    // local grad at z = 0 is 0.25, so upstream 2.0 -> 0.5
    Tensor g = sig.backward(Tensor::column({2.0}));
    CHECK(g.at(0, 0) == doctest::Approx(0.5));
}

TEST_CASE("activation layers have no parameters") {
    ReluLayer relu;
    TanhLayer tanh;
    SigmoidLayer sig;
    CHECK(relu.parameters().empty());
    CHECK(tanh.parameters().empty());
    CHECK(sig.parameters().empty());
}
