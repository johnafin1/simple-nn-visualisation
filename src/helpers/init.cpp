#include "helpers/init.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>

namespace nn::math {

namespace {

// Fill w in-place with samples from U[-limit, limit] using a seeded mt19937_64.
void fill_uniform(std::span<double> w, double limit, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (double& value : w) {
        value = dist(rng);
    }
}

}  // namespace

void xavier_uniform(std::span<double> w, std::size_t fan_in, std::size_t fan_out,
                    std::uint64_t seed) {
    assert((fan_in + fan_out) > 0 && "xavier_uniform: fan_in + fan_out must be > 0");
    const double limit =
        std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
    fill_uniform(w, limit, seed);
}

void he_uniform(std::span<double> w, std::size_t fan_in, std::uint64_t seed) {
    assert(fan_in > 0 && "he_uniform: fan_in must be > 0");
    const double limit = std::sqrt(6.0 / static_cast<double>(fan_in));
    fill_uniform(w, limit, seed);
}

}  // namespace nn::math
