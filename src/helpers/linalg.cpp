#include "helpers/linalg.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

namespace nn::math {

double dot(std::span<const double> a, std::span<const double> b) {
    assert(a.size() == b.size() && "dot: vectors must have equal length");
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void axpy(double alpha, std::span<const double> x, std::span<double> y) {
    assert(x.size() == y.size() && "axpy: vectors must have equal length");
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] += alpha * x[i];
    }
}

double l2_norm(std::span<const double> v) {
    return std::sqrt(dot(v, v));
}

void matvec(std::span<const double> A, std::span<const double> x,
            std::span<double> y, std::size_t M, std::size_t N) {
    assert(A.size() == M * N && "matvec: A must have M*N elements");
    assert(x.size() == N && "matvec: x must have N elements");
    assert(y.size() == M && "matvec: y must have M elements");
    for (std::size_t i = 0; i < M; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < N; ++j) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

void matmul(std::span<const double> A, std::span<const double> B,
            std::span<double> C, std::size_t M, std::size_t K, std::size_t N) {
    assert(A.size() == M * K && "matmul: A must have M*K elements");
    assert(B.size() == K * N && "matmul: B must have K*N elements");
    assert(C.size() == M * N && "matmul: C must have M*N elements");
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matvec_t(std::span<const double> A, std::span<const double> y,
              std::span<double> out, std::size_t M, std::size_t N) {
    assert(A.size() == M * N && "matvec_t: A must have M*N elements");
    assert(y.size() == M && "matvec_t: y must have M elements");
    assert(out.size() == N && "matvec_t: out must have N elements");
    for (std::size_t j = 0; j < N; ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < M; ++i) {
            sum += A[i * N + j] * y[i];
        }
        out[j] = sum;
    }
}

void outer(std::span<const double> a, std::span<const double> b,
           std::span<double> out, std::size_t M, std::size_t N) {
    assert(a.size() == M && "outer: a must have M elements");
    assert(b.size() == N && "outer: b must have N elements");
    assert(out.size() == M * N && "outer: out must have M*N elements");
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            out[i * N + j] = a[i] * b[j];
        }
    }
}

void mul(std::span<const double> a, std::span<const double> b, std::span<double> out) {
    assert(a.size() == b.size() && "mul: size mismatch");
    assert(a.size() == out.size() && "mul: out size mismatch");
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * b[i];
    }
}

}  // namespace nn::math
