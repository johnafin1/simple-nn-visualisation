#include "classes/dense_layer.hpp"

#include <cassert>
#include <utility>

#include "helpers/init.hpp"
#include "helpers/linalg.hpp"

namespace nn::core {

DenseLayer::DenseLayer(std::size_t in_features, std::size_t out_features, InitKind init,
                       std::uint64_t seed, std::string log_id)
    : in_features_(in_features),
      out_features_(out_features),
      log_id_(std::move(log_id)),
      weights_(out_features, in_features),
      bias_(out_features, 1),
      grad_w_(out_features, in_features),
      grad_b_(out_features, 1),
      cached_x_(in_features, 1) {
    if (init == InitKind::Xavier) {
        nn::math::xavier_uniform(weights_.span(), in_features, out_features, seed);
    } else {
        nn::math::he_uniform(weights_.span(), in_features, seed);
    }
    // bias initialised to zero by Tensor's constructor.
}

Tensor DenseLayer::forward(const Tensor& x) {
    assert(x.size() == in_features_ && "DenseLayer::forward: x has wrong size");
    cached_x_ = x;  // save for backward (dL/dW = dL/dz * x^T)

    Tensor y(out_features_, 1);
    nn::math::matvec(weights_.cspan(), x.cspan(), y.span(), out_features_, in_features_);
    nn::math::axpy(1.0, bias_.cspan(), y.span());  // y += b
    return y;
}

Tensor DenseLayer::backward(const Tensor& grad_out) {
    assert(grad_out.size() == out_features_ && "DenseLayer::backward: grad_out has wrong size");

    // dL/db = grad_out  (accumulate)
    nn::math::axpy(1.0, grad_out.cspan(), grad_b_.span());

    // dL/dW += grad_out * x^T  (outer product, out x in), accumulated
    Tensor dw(out_features_, in_features_);
    nn::math::outer(grad_out.cspan(), cached_x_.cspan(), dw.span(), out_features_, in_features_);
    nn::math::axpy(1.0, dw.cspan(), grad_w_.span());

    // dL/dx = W^T * grad_out
    Tensor grad_in(in_features_, 1);
    nn::math::matvec_t(weights_.cspan(), grad_out.cspan(), grad_in.span(), out_features_,
                       in_features_);
    return grad_in;
}

std::vector<ParamView> DenseLayer::parameters() {
    return {
        ParamView{log_id_, "weight", weights_.span(), grad_w_.span(), out_features_,
                  in_features_},
        ParamView{log_id_, "bias", bias_.span(), grad_b_.span(), out_features_, 1},
    };
}

void DenseLayer::zero_grad() {
    grad_w_.fill(0.0);
    grad_b_.fill(0.0);
}

}  // namespace nn::core
