#include "classes/activation_layers.hpp"

#include "helpers/activations.hpp"
#include "helpers/linalg.hpp"

namespace nn::core {

Tensor ReluLayer::forward(const Tensor& x) {
    cached_z_ = x;
    Tensor y(x.rows(), x.cols());
    nn::math::relu(x.cspan(), y.span());
    return y;
}

Tensor ReluLayer::backward(const Tensor& grad_out) {
    // dL/dz = grad_out * relu'(z)
    Tensor local(cached_z_.rows(), cached_z_.cols());
    nn::math::relu_grad(cached_z_.cspan(), local.span());
    Tensor grad_in(grad_out.rows(), grad_out.cols());
    nn::math::mul(grad_out.cspan(), local.cspan(), grad_in.span());
    return grad_in;
}

Tensor TanhLayer::forward(const Tensor& x) {
    cached_z_ = x;
    Tensor y(x.rows(), x.cols());
    nn::math::tanh_(x.cspan(), y.span());
    return y;
}

Tensor TanhLayer::backward(const Tensor& grad_out) {
    // dL/dz = grad_out * tanh'(z)
    Tensor local(cached_z_.rows(), cached_z_.cols());
    nn::math::tanh_grad(cached_z_.cspan(), local.span());
    Tensor grad_in(grad_out.rows(), grad_out.cols());
    nn::math::mul(grad_out.cspan(), local.cspan(), grad_in.span());
    return grad_in;
}

Tensor SigmoidLayer::forward(const Tensor& x) {
    cached_z_ = x;
    Tensor y(x.rows(), x.cols());
    nn::math::sigmoid(x.cspan(), y.span());
    return y;
}

Tensor SigmoidLayer::backward(const Tensor& grad_out) {
    // dL/dz = grad_out * sigmoid'(z)
    Tensor local(cached_z_.rows(), cached_z_.cols());
    nn::math::sigmoid_grad(cached_z_.cspan(), local.span());
    Tensor grad_in(grad_out.rows(), grad_out.cols());
    nn::math::mul(grad_out.cspan(), local.cspan(), grad_in.span());
    return grad_in;
}

}  // namespace nn::core
