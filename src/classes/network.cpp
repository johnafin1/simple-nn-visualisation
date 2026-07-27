#include "classes/network.hpp"

#include <stdexcept>
#include <utility>

namespace nn::core {

void Network::add(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

void Network::remove_last() {
    if (layers_.empty()) {
        throw std::out_of_range("Network::remove_last: network is empty");
    }
    layers_.pop_back();
}

Tensor Network::forward(const Tensor& x) {
    Tensor current = x;
    for (auto& layer : layers_) {
        current = layer->forward(current);
    }
    return current;
}

Tensor Network::backward(const Tensor& grad_loss) {
    Tensor grad = grad_loss;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
    return grad;
}

std::vector<ParamView> Network::parameters() {
    std::vector<ParamView> out;
    for (auto& layer : layers_) {
        for (auto& p : layer->parameters()) {
            out.push_back(std::move(p));
        }
    }
    return out;
}

void Network::zero_grad() {
    for (auto& layer : layers_) {
        layer->zero_grad();
    }
}

}  // namespace nn::core
