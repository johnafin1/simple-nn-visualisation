#pragma once

#include <memory>
#include <string>
#include <vector>

#include "classes/layer.hpp"
#include "classes/loggable.hpp"
#include "classes/tensor.hpp"

namespace nn::core {

// Owns an ordered stack of layers and runs the forward/backward passes.
// forward() calls each layer in order; backward() walks them in reverse, threading
// the gradient. Implements Loggable so the whole network's parameters can be
// enumerated and streamed for visualisation.
class Network final : public nn::log::Loggable {
public:
    explicit Network(std::string log_id = "network") : log_id_(std::move(log_id)) {}

    void add(std::unique_ptr<Layer> layer);

    Tensor forward(const Tensor& x);
    // Seeds backprop with grad_loss = dL/dy_hat and propagates to the input.
    Tensor backward(const Tensor& grad_loss);

    // Flattened parameter views across all layers, in forward order.
    [[nodiscard]] std::vector<ParamView> parameters();
    void zero_grad();

    [[nodiscard]] std::size_t num_layers() const { return layers_.size(); }
    [[nodiscard]] Layer& layer(std::size_t i) { return *layers_[i]; }

    [[nodiscard]] std::string log_name() const override { return log_id_; }

private:
    std::string log_id_;
    std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace nn::core
