#include "classes/loss.hpp"

#include <stdexcept>

#include "helpers/activations.hpp"
#include "helpers/loss.hpp"

namespace nn::core {

double MseLoss::value(const Tensor& y_hat, const Tensor& y) const {
    return nn::math::mse(y_hat.cspan(), y.cspan());
}

Tensor MseLoss::grad(const Tensor& y_hat, const Tensor& y) const {
    Tensor g(y_hat.rows(), y_hat.cols());
    nn::math::mse_grad(y_hat.cspan(), y.cspan(), g.span());
    return g;
}

double BceWithLogitsLoss::value(const Tensor& logits, const Tensor& y) const {
    return nn::math::bce_with_logits(logits.cspan(), y.cspan());
}

Tensor BceWithLogitsLoss::grad(const Tensor& logits, const Tensor& y) const {
    Tensor g(logits.rows(), logits.cols());
    nn::math::bce_with_logits_grad(logits.cspan(), y.cspan(), g.span());
    return g;
}

Tensor BceWithLogitsLoss::activate(const Tensor& logits) const {
    Tensor p(logits.rows(), logits.cols());
    nn::math::sigmoid(logits.cspan(), p.span());
    return p;
}

WeightedBceWithLogitsLoss::WeightedBceWithLogitsLoss(double positive_weight,
                                                     double negative_weight)
    : positive_weight_(positive_weight), negative_weight_(negative_weight) {
    if (positive_weight <= 0.0 || negative_weight <= 0.0) {
        throw std::invalid_argument("WeightedBceWithLogitsLoss: weights must be positive");
    }
}

double WeightedBceWithLogitsLoss::value(const Tensor& logits, const Tensor& y) const {
    return nn::math::weighted_bce_with_logits(logits.cspan(), y.cspan(), positive_weight_,
                                              negative_weight_);
}

Tensor WeightedBceWithLogitsLoss::grad(const Tensor& logits, const Tensor& y) const {
    Tensor g(logits.rows(), logits.cols());
    nn::math::weighted_bce_with_logits_grad(logits.cspan(), y.cspan(), positive_weight_,
                                            negative_weight_, g.span());
    return g;
}

Tensor WeightedBceWithLogitsLoss::activate(const Tensor& logits) const {
    Tensor p(logits.rows(), logits.cols());
    nn::math::sigmoid(logits.cspan(), p.span());
    return p;
}

}  // namespace nn::core
