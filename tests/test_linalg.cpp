#include <array>

#include <doctest/doctest.h>

#include "helpers/linalg.hpp"

using nn::math::axpy;
using nn::math::dot;
using nn::math::l2_norm;
using nn::math::matmul;
using nn::math::matvec;
using nn::math::matvec_t;
using nn::math::mul;
using nn::math::outer;

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

TEST_CASE("matvec_t multiplies A^T by a length-M vector") {
    // A = [[1,2,3],[4,5,6]] (2x3), y = [1, 2]
    // A^T y = [1*1+4*2, 2*1+5*2, 3*1+6*2] = [9, 12, 15]
    const std::array<double, 6> A{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::array<double, 2> y{1.0, 2.0};
    std::array<double, 3> out{};
    matvec_t(A, y, out, 2, 3);
    CHECK(out[0] == doctest::Approx(9.0));
    CHECK(out[1] == doctest::Approx(12.0));
    CHECK(out[2] == doctest::Approx(15.0));
}

TEST_CASE("outer product a b^T") {
    // a = [1, 2], b = [3, 4, 5] -> [[3,4,5],[6,8,10]]
    const std::array<double, 2> a{1.0, 2.0};
    const std::array<double, 3> b{3.0, 4.0, 5.0};
    std::array<double, 6> out{};
    outer(a, b, out, 2, 3);
    CHECK(out[0] == doctest::Approx(3.0));
    CHECK(out[1] == doctest::Approx(4.0));
    CHECK(out[2] == doctest::Approx(5.0));
    CHECK(out[3] == doctest::Approx(6.0));
    CHECK(out[4] == doctest::Approx(8.0));
    CHECK(out[5] == doctest::Approx(10.0));
}

TEST_CASE("mul is the element-wise product") {
    const std::array<double, 3> a{1.0, 2.0, 3.0};
    const std::array<double, 3> b{4.0, -5.0, 6.0};
    std::array<double, 3> out{};
    mul(a, b, out);
    CHECK(out[0] == doctest::Approx(4.0));
    CHECK(out[1] == doctest::Approx(-10.0));
    CHECK(out[2] == doctest::Approx(18.0));
}
