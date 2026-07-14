#include <array>

#include <doctest/doctest.h>

#include "helpers/linalg.hpp"

using nn::math::axpy;
using nn::math::dot;
using nn::math::l2_norm;
using nn::math::matmul;
using nn::math::matvec;

TEST_CASE("dot computes the inner product") {
    const std::array<double, 3> a{1.0, 2.0, 3.0};
    const std::array<double, 3> b{4.0, 5.0, 6.0};
    CHECK(dot(a, b) == doctest::Approx(32.0));
}

TEST_CASE("dot of empty spans is zero") {
    const std::array<double, 0> e{};
    CHECK(dot(e, e) == doctest::Approx(0.0));
}

TEST_CASE("axpy accumulates y += alpha * x") {
    const std::array<double, 3> x{1.0, 2.0, 3.0};
    std::array<double, 3> y{10.0, 10.0, 10.0};
    axpy(2.0, x, y);
    CHECK(y[0] == doctest::Approx(12.0));
    CHECK(y[1] == doctest::Approx(14.0));
    CHECK(y[2] == doctest::Approx(16.0));
}

TEST_CASE("l2_norm is the Euclidean length") {
    const std::array<double, 2> v{3.0, 4.0};
    CHECK(l2_norm(v) == doctest::Approx(5.0));
}

TEST_CASE("matvec multiplies a 2x3 matrix by a length-3 vector") {
    // A = [[1, 2, 3], [4, 5, 6]], x = [1, 0, -1]  ->  y = [-2, -2]
    const std::array<double, 6> A{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::array<double, 3> x{1.0, 0.0, -1.0};
    std::array<double, 2> y{};
    matvec(A, x, y, 2, 3);
    CHECK(y[0] == doctest::Approx(-2.0));
    CHECK(y[1] == doctest::Approx(-2.0));
}

TEST_CASE("matmul multiplies 2x3 by 3x2") {
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // C = [[58, 64], [139, 154]]
    const std::array<double, 6> A{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::array<double, 6> B{7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::array<double, 4> C{};
    matmul(A, B, C, 2, 3, 2);
    CHECK(C[0] == doctest::Approx(58.0));
    CHECK(C[1] == doctest::Approx(64.0));
    CHECK(C[2] == doctest::Approx(139.0));
    CHECK(C[3] == doctest::Approx(154.0));
}
