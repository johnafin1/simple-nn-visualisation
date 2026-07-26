#pragma once

#include <cstddef>
#include <span>

namespace nn::math {

// Dot product of two equal-length vectors. Caller guarantees a.size() == b.size().
[[nodiscard]] double dot(std::span<const double> a, std::span<const double> b);

// In-place scaled accumulation: y += alpha * x. x and y must be equal length.
void axpy(double alpha, std::span<const double> x, std::span<double> y);

// Euclidean (L2) norm: sqrt(sum(v_i^2)).
[[nodiscard]] double l2_norm(std::span<const double> v);

// Matrix-vector product y = A * x.
//   A is row-major with shape M x N, x has length N, y has length M.
void matvec(std::span<const double> A, std::span<const double> x,
            std::span<double> y, std::size_t M, std::size_t N);

// Matrix-matrix product C = A * B (all row-major).
//   A: M x K, B: K x N, C: M x N.
void matmul(std::span<const double> A, std::span<const double> B,
            std::span<double> C, std::size_t M, std::size_t K, std::size_t N);

// Transposed matrix-vector product out = A^T * y.
//   A is row-major M x N, y has length M, out has length N.
//   Used in backprop: dL/dx = W^T * dL/dz.
void matvec_t(std::span<const double> A, std::span<const double> y,
              std::span<double> out, std::size_t M, std::size_t N);

// Outer product out = a * b^T.
//   a has length M, b has length N, out is row-major M x N.
//   Used in backprop: dL/dW = dL/dz * x^T.
void outer(std::span<const double> a, std::span<const double> b,
           std::span<double> out, std::size_t M, std::size_t N);

// Element-wise (Hadamard) product out[i] = a[i] * b[i]. All equal length.
void mul(std::span<const double> a, std::span<const double> b, std::span<double> out);

}  // namespace nn::math
