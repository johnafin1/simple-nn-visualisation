#pragma once

#include <vector>

#include "classes/layer.hpp"

namespace nn::core {

// Abstract parameter update rule. Given the network's flattened parameters (value +
// grad views), apply one update in place. Does not clear grads (the Trainer calls
// Network::zero_grad after stepping).
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(std::vector<ParamView>& params) = 0;
    [[nodiscard]] virtual double learning_rate() const = 0;
};

// Plain stochastic gradient descent: w <- w - lr * grad.
class Sgd final : public Optimizer {
public:
    explicit Sgd(double lr) : lr_(lr) {}
    void step(std::vector<ParamView>& params) override;
    [[nodiscard]] double learning_rate() const override { return lr_; }

private:
    double lr_;
};

// SGD with L2 weight decay: w <- w - lr * (grad + lambda * w).
// This is the primary lever for grokking: it penalises large weights, so once the
// training data is fit the optimiser keeps searching for a smaller-norm solution.
// Biases are excluded by default - decaying them shifts the decision boundary without
// reducing model complexity, and rarely helps.
class SgdWeightDecay final : public Optimizer {
public:
    SgdWeightDecay(double lr, double lambda, bool decay_bias = false)
        : lr_(lr), lambda_(lambda), decay_bias_(decay_bias) {}
    void step(std::vector<ParamView>& params) override;
    [[nodiscard]] double learning_rate() const override { return lr_; }
    [[nodiscard]] double weight_decay() const { return lambda_; }

private:
    double lr_;
    double lambda_;
    bool decay_bias_;
};

}  // namespace nn::core
