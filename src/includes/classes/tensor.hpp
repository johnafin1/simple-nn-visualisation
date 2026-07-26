#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace nn::core {

// A 2D, row-major, owning matrix of doubles. A vector is just a Tensor with one
// column (rows x 1) or one row (1 x cols). Rule of Zero: the std::vector member
// gives correct copy/move/destruction for free.
class Tensor {
public:
    Tensor() = default;
    Tensor(std::size_t rows, std::size_t cols);
    Tensor(std::size_t rows, std::size_t cols, std::vector<double> data);

    [[nodiscard]] static Tensor column(std::vector<double> data);  // rows x 1

    [[nodiscard]] std::size_t rows() const { return rows_; }
    [[nodiscard]] std::size_t cols() const { return cols_; }
    [[nodiscard]] std::size_t size() const { return data_.size(); }

    [[nodiscard]] double& at(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
    [[nodiscard]] double at(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

    [[nodiscard]] std::span<double> span() { return {data_.data(), data_.size()}; }
    [[nodiscard]] std::span<const double> cspan() const { return {data_.data(), data_.size()}; }

    void fill(double value);

private:
    std::vector<double> data_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
};

}  // namespace nn::core
