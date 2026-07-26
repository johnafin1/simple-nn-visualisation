#pragma once

#include <string>
#include <vector>

#include "classes/layer.hpp"
#include "classes/tensor.hpp"

namespace nn::core {

// Parameter-free element-wise activation layers. forward() caches the pre-activation
// input z; backward() multiplies the upstream gradient by the local derivative at z.

class ReluLayer final : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<ParamView> parameters() override { return {}; }
    void zero_grad() override {}
    [[nodiscard]] std::string name() const override { return "relu"; }

private:
    Tensor cached_z_;
};

class TanhLayer final : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<ParamView> parameters() override { return {}; }
    void zero_grad() override {}
    [[nodiscard]] std::string name() const override { return "tanh"; }

private:
    Tensor cached_z_;
};

// Squashes to (0, 1). For training a classifier prefer leaving the network output as a
// raw logit and using BceWithLogitsLoss, which fuses this in more stably; this layer is
// for inspecting probabilities directly (e.g. in the network GUI).
class SigmoidLayer final : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<ParamView> parameters() override { return {}; }
    void zero_grad() override {}
    [[nodiscard]] std::string name() const override { return "sigmoid"; }

private:
    Tensor cached_z_;
};

}  // namespace nn::core
