#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nn::api {

// One named group of samples (train / test / an unseen evaluation range / a plotting
// grid). Inputs and targets are stored flat and row-major so a sample is a contiguous
// span, matching the Tensor and nn::math conventions.
struct Split {
    std::string name;
    std::vector<double> inputs;   // count * input_dim
    std::vector<double> targets;  // count * output_dim
    // Per-sample identity, used as the independent variable when plotting: the integer
    // n for the primes task, the x value for x^2.
    std::vector<double> ids;
    std::size_t count = 0;
    std::size_t input_dim = 0;
    std::size_t output_dim = 0;

    [[nodiscard]] std::span<const double> input(std::size_t i) const {
        return {inputs.data() + i * input_dim, input_dim};
    }
    [[nodiscard]] std::span<const double> target(std::size_t i) const {
        return {targets.data() + i * output_dim, output_dim};
    }
};

// How an integer is presented to the network for the primality task.
enum class PrimeEncoding {
    // Binary digits of n, `bits` inputs wide. Bits are shared structure across inputs,
    // so a rule learned on training integers can in principle transfer to unseen ones.
    Bits,
    // One input per integer in the train/test range, exactly one hot. A held-out
    // integer's weight is never trained, so this cannot generalise - it is the
    // memorisation control.
    OneHot,
};

struct PrimesConfig {
    int lo = 2;              // first integer of the train/test pool
    int hi = 200;            // last integer of the train/test pool
    int unseen_lo = 201;     // first integer of the unseen evaluation range
    int unseen_hi = 300;     // last integer of the unseen evaluation range
    double train_fraction = 0.5;
    std::uint64_t seed = 0;
    PrimeEncoding encoding = PrimeEncoding::Bits;
    // Input width for Bits. Must cover unseen_hi (9 bits reaches 511) so the same
    // network can be fed integers beyond the training range.
    int bits = 9;
};

// A collection of named splits sharing one input/output dimensionality.
class Dataset {
public:
    Dataset() = default;

    // Target f(x) = x^2 on [-1, 1]. Train points are randomly sampled; `test` is an
    // evenly spaced grid, and `grid` is a denser grid kept for prediction-curve plots.
    [[nodiscard]] static Dataset x_squared(int n_train, int n_test, std::uint64_t seed,
                                           int n_grid = 41);

    // "Is n prime" as binary classification. Produces `train` and `test` from
    // [lo, hi], and for the Bits encoding also `unseen_<lo>_<hi>` splits covering
    // [unseen_lo, unseen_hi]. OneHot produces no unseen splits: those integers have no
    // input slot, which is precisely why that encoding cannot generalise.
    [[nodiscard]] static Dataset primes(const PrimesConfig& cfg);

    [[nodiscard]] const std::vector<Split>& splits() const { return splits_; }
    // Throws std::out_of_range if absent.
    [[nodiscard]] const Split& split(std::string_view name) const;
    [[nodiscard]] bool has_split(std::string_view name) const;

    [[nodiscard]] std::size_t input_dim() const { return input_dim_; }
    [[nodiscard]] std::size_t output_dim() const { return output_dim_; }

private:
    void add(Split split);

    std::vector<Split> splits_;
    std::size_t input_dim_ = 0;
    std::size_t output_dim_ = 0;
};

// Sieve of Eratosthenes: is_prime[i] for i in [0, n]. Exposed for tests and for
// labelling plots.
[[nodiscard]] std::vector<bool> prime_sieve(int n);

}  // namespace nn::api
