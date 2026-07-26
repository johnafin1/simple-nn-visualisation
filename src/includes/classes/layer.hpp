#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "classes/tensor.hpp"

namespace nn::core {

// A non-owning view of one named parameter block (e.g. a weight matrix or a bias
// vector) together with its gradient buffer and shape. The shape lets a logger map
// a flat index i -> (i / cols, i % cols) for long-format per-parameter logging.
struct ParamView {
    std::string layer;          // owning layer's log id, e.g. "dense.0"
    std::string name;           // e.g. "weight", "bias"
    std::span<double> values;   // live parameter buffer
    std::span<double> grads;    // matching gradient buffer (dL/dparam per element)
    std::size_t rows = 0;
    std::size_t cols = 0;
};

// Abstract single-sample layer. forward() caches whatever backward() needs.
// Gradients accumulate into internal buffers so full-batch == sum over samples;
// call zero_grad() before starting a batch.
class Layer {
public:
    virtual ~Layer() = default;

    // x -> y for one sample. Caches state needed by backward().
    virtual Tensor forward(const Tensor& x) = 0;

    // grad_out = dL/dy -> returns dL/dx. Accumulates parameter gradients internally.
    virtual Tensor backward(const Tensor& grad_out) = 0;

    // Trainable parameters (value + grad views). Empty for parameter-free layers.
    virtual std::vector<ParamView> parameters() = 0;

    // Reset accumulated gradients to zero.
    virtual void zero_grad() = 0;

    [[nodiscard]] virtual std::string name() const = 0;
};

}  // namespace nn::core
