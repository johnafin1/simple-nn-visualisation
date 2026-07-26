#include "classes/tensor.hpp"

#include <cassert>
#include <utility>

namespace nn::core {

Tensor::Tensor(std::size_t rows, std::size_t cols)
    : data_(rows * cols, 0.0), rows_(rows), cols_(cols) {}

Tensor::Tensor(std::size_t rows, std::size_t cols, std::vector<double> data)
    : data_(std::move(data)), rows_(rows), cols_(cols) {
    assert(data_.size() == rows_ * cols_ && "Tensor: data size must equal rows*cols");
}

Tensor Tensor::column(std::vector<double> data) {
    const std::size_t n = data.size();
    return Tensor(n, 1, std::move(data));
}

void Tensor::fill(double value) {
    for (double& x : data_) {
        x = value;
    }
}

}  // namespace nn::core
