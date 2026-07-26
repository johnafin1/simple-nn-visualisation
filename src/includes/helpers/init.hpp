#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace nn::math {

// Weight initialisation. Each fills `w` in-place using a deterministic seeded RNG,
// so the same (size, fan values, seed) always produce identical weights.

// Xavier/Glorot uniform: w ~ U[-a, a] with a = sqrt(6 / (fan_in + fan_out)).
void xavier_uniform(std::span<double> w, std::size_t fan_in, std::size_t fan_out,
                    std::uint64_t seed);

// He/Kaiming uniform (for ReLU): w ~ U[-a, a] with a = sqrt(6 / fan_in).
void he_uniform(std::span<double> w, std::size_t fan_in, std::uint64_t seed);

}  // namespace nn::math
