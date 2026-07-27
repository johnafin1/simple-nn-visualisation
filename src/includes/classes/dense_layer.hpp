#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "classes/layer.hpp"
#include "classes/loggable.hpp"
#include "classes/tensor.hpp"

namespace nn::core {

// How to initialise a dense layer's weights (bias is always zero).
enum class InitKind { Xavier, He };

// Fully-connected layer: y = W x + b, with W of shape (out_features x in_features).
// Implements Loggable so its per-parameter value+grad can be streamed for later
// visualisation.
class DenseLayer final : public Layer, public nn::log::Loggable {
public:
    DenseLayer(std::size_t in_features, std::size_t out_features, InitKind init,
               std::uint64_t seed, std::string log_id);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<ParamView> parameters() override;
    void zero_grad() override;
    [[nodiscard]] std::string name() const override { return "dense"; }

    [[nodiscard]] std::string log_name() const override { return log_id_; }

    [[nodiscard]] std::size_t in_features() const { return in_features_; }
    [[nodiscard]] std::size_t out_features() const { return out_features_; }
    void set_trainable(bool trainable);
    [[nodiscard]] bool is_trainable() const { return trainable_; }

private:
    std::size_t in_features_;
    std::size_t out_features_;
    std::string log_id_;

    Tensor weights_;   // out x in
    Tensor bias_;      // out x 1
    Tensor grad_w_;    // out x in
    Tensor grad_b_;    // out x 1
    Tensor cached_x_;  // in x 1, saved during forward for use in backward
    bool trainable_ = true;
};

}  // namespace nn::core
