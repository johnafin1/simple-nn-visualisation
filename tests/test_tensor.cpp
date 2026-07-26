#include <vector>

#include <doctest/doctest.h>

#include "classes/tensor.hpp"

using nn::core::Tensor;

TEST_CASE("default Tensor is empty") {
    Tensor t;
    CHECK(t.rows() == 0);
    CHECK(t.cols() == 0);
    CHECK(t.size() == 0);
}

TEST_CASE("sized Tensor is zero-initialised") {
    Tensor t(2, 3);
    CHECK(t.rows() == 2);
    CHECK(t.cols() == 3);
    CHECK(t.size() == 6);
    for (std::size_t r = 0; r < 2; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            CHECK(t.at(r, c) == doctest::Approx(0.0));
        }
    }
}

TEST_CASE("row-major indexing matches data order") {
    Tensor t(2, 3, std::vector<double>{1, 2, 3, 4, 5, 6});
    CHECK(t.at(0, 0) == doctest::Approx(1.0));
    CHECK(t.at(0, 2) == doctest::Approx(3.0));
    CHECK(t.at(1, 0) == doctest::Approx(4.0));
    CHECK(t.at(1, 2) == doctest::Approx(6.0));
}

TEST_CASE("column factory makes an N x 1 tensor") {
    Tensor c = Tensor::column({7, 8, 9});
    CHECK(c.rows() == 3);
    CHECK(c.cols() == 1);
    CHECK(c.at(1, 0) == doctest::Approx(8.0));
}

TEST_CASE("mutation through at and fill") {
    Tensor t(2, 2);
    t.at(0, 1) = 5.0;
    CHECK(t.at(0, 1) == doctest::Approx(5.0));
    t.fill(-1.0);
    CHECK(t.at(0, 1) == doctest::Approx(-1.0));
    CHECK(t.at(1, 1) == doctest::Approx(-1.0));
}
