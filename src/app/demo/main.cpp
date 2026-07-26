#include <iostream>
#include <string_view>

#include "classes/activation_layers.hpp"
#include "classes/dense_layer.hpp"
#include "classes/json_line.hpp"
#include "classes/jsonl_sink.hpp"
#include "classes/loss.hpp"
#include "classes/tensor.hpp"

using nn::core::DenseLayer;
using nn::core::InitKind;
using nn::core::MseLoss;
using nn::core::ParamView;
using nn::core::TanhLayer;
using nn::core::Tensor;
using nn::log::JsonLine;
using nn::log::JsonlSink;

// Runs ONE forward+backward pass through a 1 -> 4 (tanh) -> 1 network and emits a
// snapshot of every node value and gradient plus every weight/bias and its gradient.
// The layers are driven by hand (not via Network) so we can capture the intermediate
// linear-algebra results the GUI wants to show: z = Wx + b, a = tanh(z), and on the
// way back dL/da, dL/dz, dL/dx.
//
// Output: runs/demo/snapshot.jsonl  (record types: node | edge | bias | loss)
int main() {
    constexpr std::size_t kIn = 1;
    constexpr std::size_t kHidden = 4;
    constexpr std::size_t kOut = 1;

    DenseLayer dense0(kIn, kHidden, InitKind::Xavier, 11, "dense.0");
    TanhLayer tanh0;
    DenseLayer dense1(kHidden, kOut, InitKind::Xavier, 22, "dense.1");
    MseLoss loss;

    const Tensor x = Tensor::column({0.7});
    const Tensor target = Tensor::column({0.25});

    // ---- forward, capturing intermediates ----
    dense0.zero_grad();
    dense1.zero_grad();
    const Tensor z0 = dense0.forward(x);   // pre-activation (Wx + b)
    const Tensor a0 = tanh0.forward(z0);   // hidden activation
    const Tensor y_hat = dense1.forward(a0);
    const double L = loss.value(y_hat, target);

    // ---- backward, capturing gradients that flow between layers ----
    const Tensor grad_y = loss.grad(y_hat, target);  // dL/dy_hat
    const Tensor grad_a0 = dense1.backward(grad_y);   // dL/da0
    const Tensor grad_z0 = tanh0.backward(grad_a0);   // dL/dz0
    const Tensor grad_x = dense0.backward(grad_z0);    // dL/dx

    JsonlSink snap("runs/demo/snapshot.jsonl", /*flush_every=*/64);

    auto node = [&](std::string_view layer, std::size_t idx, double value, double grad) {
        snap.write(JsonLine{}
                       .add("type", std::string_view{"node"})
                       .add("layer", layer)
                       .add("idx", idx)
                       .add("value", value)
                       .add("grad", grad)
                       .str());
    };

    // input layer
    node("input", 0, x.at(0, 0), grad_x.at(0, 0));
    // hidden pre-activations and activations
    for (std::size_t j = 0; j < kHidden; ++j) {
        node("dense.0.pre", j, z0.at(j, 0), grad_z0.at(j, 0));
        node("dense.0.act", j, a0.at(j, 0), grad_a0.at(j, 0));
    }
    // output
    node("output", 0, y_hat.at(0, 0), grad_y.at(0, 0));

    // edges + biases from both dense layers
    auto emit_params = [&](DenseLayer& layer) {
        for (const ParamView& p : layer.parameters()) {
            if (p.name == "weight") {
                for (std::size_t r = 0; r < p.rows; ++r) {
                    for (std::size_t c = 0; c < p.cols; ++c) {
                        const std::size_t i = r * p.cols + c;
                        snap.write(JsonLine{}
                                       .add("type", std::string_view{"edge"})
                                       .add("layer", std::string_view{p.layer})
                                       .add("row", r)
                                       .add("col", c)
                                       .add("weight", p.values[i])
                                       .add("grad", p.grads[i])
                                       .str());
                    }
                }
            } else {  // bias
                for (std::size_t r = 0; r < p.rows; ++r) {
                    snap.write(JsonLine{}
                                   .add("type", std::string_view{"bias"})
                                   .add("layer", std::string_view{p.layer})
                                   .add("idx", r)
                                   .add("value", p.values[r])
                                   .add("grad", p.grads[r])
                                   .str());
                }
            }
        }
    };
    emit_params(dense0);
    emit_params(dense1);

    snap.write(JsonLine{}
                   .add("type", std::string_view{"loss"})
                   .add("value", L)
                   .add("target", target.at(0, 0))
                   .str());
    snap.flush();

    std::cout << "Snapshot written to runs/demo/snapshot.jsonl\n";
    std::cout << "  y_hat = " << y_hat.at(0, 0) << ", target = " << target.at(0, 0)
              << ", MSE loss = " << L << "\n";
    std::cout << "  View it with: streamlit run src/app/python/network_demo.py\n";
    return 0;
}
