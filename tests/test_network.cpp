#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <doctest/doctest.h>

#include "classes/activation_layers.hpp"
#include "classes/dense_layer.hpp"
#include "classes/loss.hpp"
#include "classes/network.hpp"
#include "classes/tensor.hpp"
#include "grad_check.hpp"

using nn::core::DenseLayer;
using nn::core::InitKind;
using nn::core::MseLoss;
using nn::core::Network;
using nn::core::ParamView;
using nn::core::TanhLayer;
using nn::core::Tensor;
using nn::test::numeric_gradient;

namespace {

Network make_net() {
    Network net("network");
    net.add(std::make_unique<DenseLayer>(1, 4, InitKind::Xavier, 11, "dense.0"));
    net.add(std::make_unique<TanhLayer>());
    net.add(std::make_unique<DenseLayer>(4, 1, InitKind::Xavier, 22, "dense.1"));
    return net;
}

}  // namespace

TEST_CASE("Network forward chains layers and preserves output shape") {
    Network net = make_net();
    Tensor y = net.forward(Tensor::column({0.5}));
    CHECK(y.rows() == 1);
    CHECK(y.cols() == 1);
}

TEST_CASE("Network flattened parameters expose every layer's params in order") {
    Network net = make_net();
    auto params = net.parameters();
    // dense0: weight+bias, dense1: weight+bias -> 4 blocks
    REQUIRE(params.size() == 4);
    CHECK(params[0].rows == 4);  // dense0 weight 4x1
    CHECK(params[0].cols == 1);
    CHECK(params[2].rows == 1);  // dense1 weight 1x4
    CHECK(params[2].cols == 4);
    CHECK(net.log_name() == "network");
}

TEST_CASE("Network end-to-end analytic gradients match numeric (Phase 2 exit)") {
    Network net = make_net();
    MseLoss loss;
    const Tensor x = Tensor::column({0.7});
    const Tensor target = Tensor::column({0.25});

    net.zero_grad();
    Tensor y_hat = net.forward(x);
    Tensor grad_loss = loss.grad(y_hat, target);
    net.backward(grad_loss);

    auto params = net.parameters();

    // Check every parameter block against a central-difference estimate.
    for (auto& p : params) {
        std::vector<double> p0(p.values.begin(), p.values.end());
        auto numeric = numeric_gradient(
            [&](std::span<const double> pv) {
                for (std::size_t i = 0; i < pv.size(); ++i) p.values[i] = pv[i];
                Tensor out = net.forward(x);
                return loss.value(out, target);
            },
            p0);
        for (std::size_t i = 0; i < p0.size(); ++i) p.values[i] = p0[i];  // restore
        for (std::size_t i = 0; i < numeric.size(); ++i) {
            CHECK(p.grads[i] == doctest::Approx(numeric[i]).epsilon(1e-6));
        }
    }
}

TEST_CASE("Network zero_grad clears all layer gradients") {
    Network net = make_net();
    MseLoss loss;
    const Tensor x = Tensor::column({0.7});
    const Tensor target = Tensor::column({0.25});
    Tensor y_hat = net.forward(x);
    net.backward(loss.grad(y_hat, target));
    net.zero_grad();
    for (auto& p : net.parameters()) {
        for (double g : p.grads) CHECK(g == doctest::Approx(0.0));
    }
}

TEST_CASE("Network can replace its final transfer-learning head") {
    Network net = make_net();
    CHECK(net.num_layers() == 3);
    net.remove_last();
    CHECK(net.num_layers() == 2);
    net.add(std::make_unique<DenseLayer>(4, 2, InitKind::Xavier, 33, "replacement_head"));
    CHECK(net.forward(Tensor::column({0.5})).size() == 2);
}

TEST_CASE("Network remove_last rejects an empty network") {
    Network net;
    CHECK_THROWS_AS(net.remove_last(), std::out_of_range);
}
