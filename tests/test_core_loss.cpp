#include <cmath>
#include <stdexcept>

#include <doctest/doctest.h>

#include "classes/loss.hpp"
#include "classes/tensor.hpp"

using nn::core::BceWithLogitsLoss;
using nn::core::MseLoss;
using nn::core::Tensor;
using nn::core::WeightedBceWithLogitsLoss;

TEST_CASE("MseLoss value matches mean squared error") {
    MseLoss loss;
    Tensor y_hat = Tensor::column({1.0, 2.0, 3.0});
    Tensor y = Tensor::column({1.0, 0.0, 3.0});
    // errors: 0, 2, 0 -> mean(0, 4, 0) = 4/3
    CHECK(loss.value(y_hat, y) == doctest::Approx(4.0 / 3.0));
}

TEST_CASE("MseLoss grad is 2*(y_hat - y)/N") {
    MseLoss loss;
    Tensor y_hat = Tensor::column({1.0, 2.0, 3.0});
    Tensor y = Tensor::column({1.0, 0.0, 3.0});
    Tensor g = loss.grad(y_hat, y);
    CHECK(g.at(0, 0) == doctest::Approx(0.0));
    CHECK(g.at(1, 0) == doctest::Approx(2.0 * 2.0 / 3.0));
    CHECK(g.at(2, 0) == doctest::Approx(0.0));
}

TEST_CASE("MseLoss is a regression loss and reports predictions unchanged") {
    MseLoss loss;
    CHECK_FALSE(loss.is_classification());
    Tensor y_hat = Tensor::column({0.3, -1.0});
    Tensor out = loss.activate(y_hat);
    CHECK(out.at(0, 0) == doctest::Approx(0.3));
    CHECK(out.at(1, 0) == doctest::Approx(-1.0));
}

TEST_CASE("BceWithLogitsLoss activates logits into probabilities") {
    BceWithLogitsLoss loss;
    CHECK(loss.is_classification());
    Tensor p = loss.activate(Tensor::column({0.0, 2.0, -2.0}));
    CHECK(p.at(0, 0) == doctest::Approx(0.5));
    CHECK(p.at(1, 0) > 0.85);
    CHECK(p.at(2, 0) < 0.15);
}

TEST_CASE("BceWithLogitsLoss value and grad on logits") {
    BceWithLogitsLoss loss;
    Tensor logits = Tensor::column({0.0});
    Tensor y = Tensor::column({1.0});
    CHECK(loss.value(logits, y) == doctest::Approx(std::log(2.0)));
    Tensor g = loss.grad(logits, y);
    CHECK(g.at(0, 0) == doctest::Approx(0.5 - 1.0));
}

TEST_CASE("WeightedBceWithLogitsLoss exposes weights and scales positive errors") {
    WeightedBceWithLogitsLoss loss(4.0, 1.0);
    CHECK(loss.is_classification());
    CHECK(loss.positive_weight() == doctest::Approx(4.0));
    CHECK(loss.negative_weight() == doctest::Approx(1.0));
    const Tensor logits = Tensor::column({0.0});
    const Tensor positive = Tensor::column({1.0});
    CHECK(loss.value(logits, positive) == doctest::Approx(4.0 * std::log(2.0)));
    CHECK(loss.grad(logits, positive).at(0, 0) == doctest::Approx(-2.0));
    CHECK(loss.activate(logits).at(0, 0) == doctest::Approx(0.5));
}

TEST_CASE("WeightedBceWithLogitsLoss rejects non-positive weights") {
    CHECK_THROWS_AS(WeightedBceWithLogitsLoss(0.0, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(WeightedBceWithLogitsLoss(1.0, -1.0), std::invalid_argument);
}
