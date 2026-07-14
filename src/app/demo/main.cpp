#include <array>
#include <iostream>

#include "helpers/linalg.hpp"

// First runnable slice of the real math layer: call nn::math::dot end to end.
int main() {
    const std::array<double, 3> a{1.0, 2.0, 3.0};
    const std::array<double, 3> b{4.0, 5.0, 6.0};

    const double result = nn::math::dot(a, b);

    std::cout << "nn::math::dot([1,2,3], [4,5,6]) = " << result
              << "  (expected 32)\n";
    return 0;
}
