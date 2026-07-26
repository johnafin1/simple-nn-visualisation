#pragma once

#include "classes/tensor.hpp"

namespace nn::core {

// Abstract scalar loss over a prediction/target pair.
class Loss {
public:
    virtual ~Loss() = default;

    // Scalar loss value for (y_hat, y).
    [[nodiscard]] virtual double value(const Tensor& y_hat, const Tensor& y) const = 0;

    // dL/dy_hat, same shape as y_hat. This is the seed gradient for backprop.
    [[nodiscard]] virtual Tensor grad(const Tensor& y_hat, const Tensor& y) const = 0;

    // Maps the raw network output to the reportable prediction. Identity for
    // regression; for a logit-based classifier this applies the sigmoid. The Trainer
    // uses it when logging predictions so it stays task-agnostic.
    [[nodiscard]] virtual Tensor activate(const Tensor& y_hat) const { return y_hat; }

    // When true the Trainer also reports accuracy, thresholding activate(y_hat) at 0.5.
    [[nodiscard]] virtual bool is_classification() const { return false; }
};

// Mean squared error wrapper around nn::math::mse / mse_grad.
class MseLoss final : public Loss {
public:
    [[nodiscard]] double value(const Tensor& y_hat, const Tensor& y) const override;
    [[nodiscard]] Tensor grad(const Tensor& y_hat, const Tensor& y) const override;
};

// Binary cross-entropy on logits, with the sigmoid fused in for numerical stability.
// The network's final layer should be linear (emit a logit, not a probability).
class BceWithLogitsLoss final : public Loss {
public:
    [[nodiscard]] double value(const Tensor& logits, const Tensor& y) const override;
    [[nodiscard]] Tensor grad(const Tensor& logits, const Tensor& y) const override;
    [[nodiscard]] Tensor activate(const Tensor& logits) const override;
    [[nodiscard]] bool is_classification() const override { return true; }
};

}  // namespace nn::core
