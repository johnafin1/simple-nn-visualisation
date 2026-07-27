#include <memory>
#include <vector>

#include <doctest/doctest.h>

#include "classes/dense_layer.hpp"
#include "classes/network.hpp"
#include "classes/optimizer.hpp"
#include "classes/tensor.hpp"

using nn::core::DenseLayer;
using nn::core::InitKind;
using nn::core::Network;
using nn::core::Sgd;
using nn::core::SgdWeightDecay;
using nn::core::Tensor;

TEST_CASE("Sgd applies w -= lr * grad to every parameter") {
    DenseLayer dense(2, 2, InitKind::Xavier, 5, "dense.0");
    auto params = dense.parameters();
    // Set known values and grads directly through the views.
    for (std::size_t i = 0; i < params[0].values.size(); ++i) {
        params[0].values[i] = 1.0;
        params[0].grads[i] = 0.5;  // -> 1.0 - 0.1*0.5 = 0.95
    }
    for (std::size_t i = 0; i < params[1].values.size(); ++i) {
        params[1].values[i] = 2.0;
        params[1].grads[i] = -1.0;  // -> 2.0 - 0.1*(-1.0) = 2.1
    }

    Sgd opt(0.1);
    CHECK(opt.learning_rate() == doctest::Approx(0.1));
    opt.step(params);

    for (double v : params[0].values) CHECK(v == doctest::Approx(0.95));
    for (double v : params[1].values) CHECK(v == doctest::Approx(2.1));
}

TEST_CASE("SgdWeightDecay decays weights but leaves biases alone by default") {
    DenseLayer dense(2, 2, InitKind::Xavier, 5, "dense.0");
    auto params = dense.parameters();
    for (std::size_t i = 0; i < params[0].values.size(); ++i) {
        params[0].values[i] = 2.0;
        params[0].grads[i] = 0.0;  // isolate the decay term
    }
    for (std::size_t i = 0; i < params[1].values.size(); ++i) {
        params[1].values[i] = 2.0;
        params[1].grads[i] = 0.0;
    }

    SgdWeightDecay opt(/*lr=*/0.1, /*lambda=*/0.5);
    CHECK(opt.weight_decay() == doctest::Approx(0.5));
    opt.step(params);

    // weight: 2 - 0.1 * (0 + 0.5*2) = 1.9
    for (double v : params[0].values) CHECK(v == doctest::Approx(1.9));
    // bias: untouched because grad is 0 and decay is skipped
    for (double v : params[1].values) CHECK(v == doctest::Approx(2.0));
}

TEST_CASE("SgdWeightDecay can decay biases when asked") {
    DenseLayer dense(2, 2, InitKind::Xavier, 5, "dense.0");
    auto params = dense.parameters();
    for (std::size_t i = 0; i < params[1].values.size(); ++i) {
        params[1].values[i] = 2.0;
        params[1].grads[i] = 0.0;
    }

    SgdWeightDecay opt(/*lr=*/0.1, /*lambda=*/0.5, /*decay_bias=*/true);
    opt.step(params);
    for (double v : params[1].values) CHECK(v == doctest::Approx(1.9));
}

TEST_CASE("SgdWeightDecay combines the gradient and the decay term") {
    DenseLayer dense(1, 1, InitKind::Xavier, 3, "dense.0");
    auto params = dense.parameters();
    params[0].values[0] = 4.0;
    params[0].grads[0] = 1.0;

    SgdWeightDecay opt(/*lr=*/0.5, /*lambda=*/0.25);
    opt.step(params);
    // 4 - 0.5 * (1 + 0.25*4) = 4 - 0.5*2 = 3.0
    CHECK(params[0].values[0] == doctest::Approx(3.0));
}

TEST_CASE("weight decay shrinks an unconstrained weight towards zero over many steps") {
    DenseLayer dense(1, 1, InitKind::Xavier, 3, "dense.0");
    auto params = dense.parameters();
    params[0].values[0] = 1.0;

    SgdWeightDecay opt(/*lr=*/0.1, /*lambda=*/0.5);
    for (int i = 0; i < 100; ++i) {
        auto ps = dense.parameters();
        ps[0].grads[0] = 0.0;  // no data pressure, only decay
        opt.step(ps);
    }
    CHECK(dense.parameters()[0].values[0] < 0.05);
}

TEST_CASE("optimisers skip frozen parameter blocks") {
    DenseLayer dense(1, 1, InitKind::Xavier, 3, "dense.0");
    auto before = dense.parameters();
    before[0].values[0] = 2.0;
    before[0].grads[0] = 3.0;
    dense.set_trainable(false);
    auto frozen = dense.parameters();
    REQUIRE_FALSE(frozen[0].trainable);

    SgdWeightDecay opt(/*lr=*/0.5, /*lambda=*/0.25);
    opt.step(frozen);
    CHECK(frozen[0].values[0] == doctest::Approx(2.0));
}

TEST_CASE("one SGD step reduces a simple network's loss direction") {
    // Sanity: after computing grads and stepping, the same params moved opposite grad.
    Network net("network");
    net.add(std::make_unique<DenseLayer>(1, 1, InitKind::Xavier, 9, "dense.0"));
    auto before = net.parameters();
    std::vector<double> w_before(before[0].values.begin(), before[0].values.end());
    before[0].grads[0] = 1.0;  // pretend gradient

    Sgd opt(0.5);
    auto params = net.parameters();
    params[0].grads[0] = 1.0;
    opt.step(params);

    auto after = net.parameters();
    CHECK(after[0].values[0] == doctest::Approx(w_before[0] - 0.5 * 1.0));
}
