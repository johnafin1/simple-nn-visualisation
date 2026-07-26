#include "classes/optimizer.hpp"

#include <cstddef>

namespace nn::core {

void Sgd::step(std::vector<ParamView>& params) {
    for (ParamView& p : params) {
        for (std::size_t i = 0; i < p.values.size(); ++i) {
            p.values[i] -= lr_ * p.grads[i];
        }
    }
}

void SgdWeightDecay::step(std::vector<ParamView>& params) {
    for (ParamView& p : params) {
        const bool decay = decay_bias_ || p.name != "bias";
        const double lambda = decay ? lambda_ : 0.0;
        for (std::size_t i = 0; i < p.values.size(); ++i) {
            p.values[i] -= lr_ * (p.grads[i] + lambda * p.values[i]);
        }
    }
}

}  // namespace nn::core
