#include "api/dataset.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace nn::api {

namespace {

// Builds one sample's input row for integer n and appends it to `out`.
void encode(int n, const PrimesConfig& cfg, std::vector<double>& out) {
    if (cfg.encoding == PrimeEncoding::Bits) {
        for (int b = 0; b < cfg.bits; ++b) {
            out.push_back(((n >> b) & 1) ? 1.0 : 0.0);
        }
    } else {
        const int width = cfg.hi - cfg.lo + 1;
        for (int i = 0; i < width; ++i) {
            out.push_back((cfg.lo + i == n) ? 1.0 : 0.0);
        }
    }
}

std::size_t encoded_width(const PrimesConfig& cfg) {
    return cfg.encoding == PrimeEncoding::Bits
               ? static_cast<std::size_t>(cfg.bits)
               : static_cast<std::size_t>(cfg.hi - cfg.lo + 1);
}

// Assembles a split from a list of integers.
Split make_prime_split(std::string name, const std::vector<int>& numbers,
                       const std::vector<bool>& is_prime, const PrimesConfig& cfg) {
    Split s;
    s.name = std::move(name);
    s.input_dim = encoded_width(cfg);
    s.output_dim = 1;
    s.count = numbers.size();
    s.inputs.reserve(numbers.size() * s.input_dim);
    s.targets.reserve(numbers.size());
    s.ids.reserve(numbers.size());
    for (const int n : numbers) {
        encode(n, cfg, s.inputs);
        s.targets.push_back(is_prime[static_cast<std::size_t>(n)] ? 1.0 : 0.0);
        s.ids.push_back(static_cast<double>(n));
    }
    return s;
}

}  // namespace

std::vector<bool> prime_sieve(int n) {
    if (n < 0) n = 0;
    std::vector<bool> is_prime(static_cast<std::size_t>(n) + 1, true);
    if (!is_prime.empty()) is_prime[0] = false;
    if (is_prime.size() > 1) is_prime[1] = false;
    for (int p = 2; static_cast<long long>(p) * p <= n; ++p) {
        if (!is_prime[static_cast<std::size_t>(p)]) continue;
        for (int m = p * p; m <= n; m += p) {
            is_prime[static_cast<std::size_t>(m)] = false;
        }
    }
    return is_prime;
}

void Dataset::add(Split split) {
    if (splits_.empty()) {
        input_dim_ = split.input_dim;
        output_dim_ = split.output_dim;
    } else {
        assert(split.input_dim == input_dim_ && "Dataset: split input_dim mismatch");
        assert(split.output_dim == output_dim_ && "Dataset: split output_dim mismatch");
    }
    splits_.push_back(std::move(split));
}

const Split& Dataset::split(std::string_view name) const {
    for (const Split& s : splits_) {
        if (s.name == name) return s;
    }
    throw std::out_of_range("Dataset: no split named " + std::string(name));
}

bool Dataset::has_split(std::string_view name) const {
    return std::any_of(splits_.begin(), splits_.end(),
                       [name](const Split& s) { return s.name == name; });
}

Dataset Dataset::x_squared(int n_train, int n_test, std::uint64_t seed, int n_grid) {
    Dataset d;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unif(-1.0, 1.0);

    Split train;
    train.name = "train";
    train.input_dim = 1;
    train.output_dim = 1;
    train.count = static_cast<std::size_t>(std::max(n_train, 0));
    for (int i = 0; i < n_train; ++i) {
        const double x = unif(rng);
        train.inputs.push_back(x);
        train.targets.push_back(x * x);
        train.ids.push_back(x);
    }
    d.add(std::move(train));

    // Evenly spaced grids for evaluation and for prediction-curve plots.
    const auto grid_split = [](const std::string& name, int n) {
        Split s;
        s.name = name;
        s.input_dim = 1;
        s.output_dim = 1;
        s.count = static_cast<std::size_t>(std::max(n, 0));
        for (int i = 0; i < n; ++i) {
            const double t = (n > 1) ? static_cast<double>(i) / (n - 1) : 0.5;
            const double x = -1.0 + 2.0 * t;
            s.inputs.push_back(x);
            s.targets.push_back(x * x);
            s.ids.push_back(x);
        }
        return s;
    };
    d.add(grid_split("test", n_test));
    d.add(grid_split("grid", n_grid));
    return d;
}

Dataset Dataset::primes(const PrimesConfig& cfg) {
    if (cfg.train_fraction < 0.0 || cfg.validation_fraction < 0.0 ||
        cfg.train_fraction + cfg.validation_fraction > 1.0) {
        throw std::invalid_argument(
            "Dataset::primes: train and validation fractions must fit in [0, 1]");
    }

    Dataset d;
    const int sieve_max = std::max(cfg.hi, cfg.unseen_hi);
    const std::vector<bool> is_prime = prime_sieve(sieve_max);

    // Stratified split: shuffle primes and composites separately, then take
    // train_fraction of each, so both splits contain primes at the same base rate.
    std::vector<int> primes_pool;
    std::vector<int> composites_pool;
    for (int n = cfg.lo; n <= cfg.hi; ++n) {
        if (is_prime[static_cast<std::size_t>(n)]) {
            primes_pool.push_back(n);
        } else {
            composites_pool.push_back(n);
        }
    }

    std::mt19937_64 rng(cfg.seed);
    std::shuffle(primes_pool.begin(), primes_pool.end(), rng);
    std::shuffle(composites_pool.begin(), composites_pool.end(), rng);

    std::vector<int> train_numbers;
    std::vector<int> validation_numbers;
    std::vector<int> test_numbers;
    const auto take = [&](const std::vector<int>& pool) {
        const auto rounded_count = [&pool](double fraction) {
            return static_cast<std::size_t>(
                static_cast<double>(pool.size()) * fraction + 0.5);
        };
        const std::size_t n_train = std::min(rounded_count(cfg.train_fraction), pool.size());
        const std::size_t available = pool.size() - n_train;
        const std::size_t n_validation =
            std::min(rounded_count(cfg.validation_fraction), available);
        for (std::size_t i = 0; i < pool.size(); ++i) {
            if (i < n_train) {
                train_numbers.push_back(pool[i]);
            } else if (i < n_train + n_validation) {
                validation_numbers.push_back(pool[i]);
            } else {
                test_numbers.push_back(pool[i]);
            }
        }
    };
    take(primes_pool);
    take(composites_pool);

    // Sort so logs and diagnostic rows come out in integer order.
    std::sort(train_numbers.begin(), train_numbers.end());
    std::sort(validation_numbers.begin(), validation_numbers.end());
    std::sort(test_numbers.begin(), test_numbers.end());

    d.add(make_prime_split("train", train_numbers, is_prime, cfg));
    if (!validation_numbers.empty()) {
        d.add(make_prime_split("validation", validation_numbers, is_prime, cfg));
    }
    d.add(make_prime_split("test", test_numbers, is_prime, cfg));

    // Unseen ranges only exist for Bits: one-hot has no input slot for integers
    // outside [lo, hi], so it cannot even be evaluated there.
    if (cfg.encoding == PrimeEncoding::Bits && cfg.unseen_hi >= cfg.unseen_lo) {
        // Split at the point where a new high bit switches on, because integers that
        // activate a bit never seen during training are a much harder test.
        const int bit_boundary = 1 << (cfg.bits - 1);  // e.g. 256 for 9 bits
        std::vector<int> in_range;
        std::vector<int> beyond;
        for (int n = cfg.unseen_lo; n <= cfg.unseen_hi; ++n) {
            (n < bit_boundary ? in_range : beyond).push_back(n);
        }
        if (!in_range.empty()) {
            d.add(make_prime_split(
                "unseen_" + std::to_string(in_range.front()) + "_" +
                    std::to_string(in_range.back()),
                in_range, is_prime, cfg));
        }
        if (!beyond.empty()) {
            d.add(make_prime_split(
                "unseen_" + std::to_string(beyond.front()) + "_" +
                    std::to_string(beyond.back()),
                beyond, is_prime, cfg));
        }
    }
    return d;
}

Dataset Dataset::prime_residues(const PrimesConfig& cfg, std::span<const int> moduli) {
    if (moduli.empty()) {
        throw std::invalid_argument("Dataset::prime_residues: moduli must not be empty");
    }
    for (const int q : moduli) {
        if (q <= 1) {
            throw std::invalid_argument("Dataset::prime_residues: every modulus must exceed 1");
        }
    }

    Dataset d = Dataset::primes(cfg);
    d.output_dim_ = moduli.size() * 2;
    for (Split& split : d.splits_) {
        split.output_dim = d.output_dim_;
        split.targets.clear();
        split.targets.reserve(split.count * split.output_dim);
        for (const double id : split.ids) {
            const double n = id;
            for (const int q : moduli) {
                const double angle =
                    2.0 * std::numbers::pi_v<double> * n / static_cast<double>(q);
                split.targets.push_back(std::sin(angle));
                split.targets.push_back(std::cos(angle));
            }
        }
    }
    return d;
}

}  // namespace nn::api
