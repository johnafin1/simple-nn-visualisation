#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "api/dataset.hpp"

using nn::api::Dataset;
using nn::api::PrimeEncoding;
using nn::api::PrimesConfig;
using nn::api::prime_sieve;

namespace {

const std::vector<int> kPrimesTo200{2,   3,   5,   7,   11,  13,  17,  19,  23,  29,
                                    31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
                                    73,  79,  83,  89,  97,  101, 103, 107, 109, 113,
                                    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
                                    179, 181, 191, 193, 197, 199};

std::vector<int> ids_of(const nn::api::Split& s) {
    std::vector<int> out;
    for (const double id : s.ids) out.push_back(static_cast<int>(id));
    return out;
}

}  // namespace

TEST_CASE("prime_sieve matches the known primes up to 200") {
    const auto is_prime = prime_sieve(200);
    REQUIRE(is_prime.size() == 201);
    CHECK_FALSE(is_prime[0]);
    CHECK_FALSE(is_prime[1]);

    const std::set<int> expected(kPrimesTo200.begin(), kPrimesTo200.end());
    for (int n = 0; n <= 200; ++n) {
        CHECK(is_prime[static_cast<std::size_t>(n)] == (expected.count(n) == 1));
    }
    CHECK(expected.size() == 46);
}

TEST_CASE("x_squared produces train, test and grid splits with matching targets") {
    const Dataset d = Dataset::x_squared(20, 11, 42, 41);
    CHECK(d.input_dim() == 1);
    CHECK(d.output_dim() == 1);
    CHECK(d.has_split("train"));
    CHECK(d.has_split("test"));
    CHECK(d.has_split("grid"));

    CHECK(d.split("train").count == 20);
    CHECK(d.split("test").count == 11);
    CHECK(d.split("grid").count == 41);

    for (const auto& s : d.splits()) {
        for (std::size_t i = 0; i < s.count; ++i) {
            const double x = s.input(i)[0];
            CHECK(x >= -1.0);
            CHECK(x <= 1.0);
            CHECK(s.target(i)[0] == doctest::Approx(x * x));
            CHECK(s.ids[i] == doctest::Approx(x));
        }
    }

    // Grid endpoints span the domain.
    const auto& g = d.split("grid");
    CHECK(g.ids.front() == doctest::Approx(-1.0));
    CHECK(g.ids.back() == doctest::Approx(1.0));
}

TEST_CASE("x_squared is deterministic for a given seed") {
    const Dataset a = Dataset::x_squared(15, 5, 7);
    const Dataset b = Dataset::x_squared(15, 5, 7);
    const Dataset c = Dataset::x_squared(15, 5, 8);
    CHECK(a.split("train").inputs == b.split("train").inputs);
    CHECK(a.split("train").inputs != c.split("train").inputs);
}

TEST_CASE("primes bits encoding covers the pool exactly once across train and test") {
    PrimesConfig cfg;
    const Dataset d = Dataset::primes(cfg);

    const auto train = ids_of(d.split("train"));
    const auto test = ids_of(d.split("test"));
    CHECK(train.size() + test.size() == 199);  // 2..200

    std::set<int> all(train.begin(), train.end());
    for (const int n : test) {
        CHECK(all.insert(n).second);  // no overlap
    }
    CHECK(all.size() == 199);
    CHECK(*all.begin() == 2);
    CHECK(*all.rbegin() == 200);
}

TEST_CASE("primes labels are correct and the split is stratified") {
    PrimesConfig cfg;
    const Dataset d = Dataset::primes(cfg);
    const std::set<int> expected(kPrimesTo200.begin(), kPrimesTo200.end());

    std::size_t total_primes = 0;
    for (const auto& s : d.splits()) {
        if (s.name != "train" && s.name != "test") continue;
        for (std::size_t i = 0; i < s.count; ++i) {
            const int n = static_cast<int>(s.ids[i]);
            const bool label = s.target(i)[0] > 0.5;
            CHECK(label == (expected.count(n) == 1));
            if (label) ++total_primes;
        }
    }
    CHECK(total_primes == 46);

    // Stratification: 23 primes each side at train_fraction = 0.5.
    const auto count_primes = [](const nn::api::Split& s) {
        return std::count_if(s.targets.begin(), s.targets.end(),
                             [](double t) { return t > 0.5; });
    };
    CHECK(count_primes(d.split("train")) == 23);
    CHECK(count_primes(d.split("test")) == 23);
}

TEST_CASE("primes bit encoding round-trips the integer") {
    PrimesConfig cfg;
    const Dataset d = Dataset::primes(cfg);
    CHECK(d.input_dim() == 9);

    for (const auto& s : d.splits()) {
        for (std::size_t i = 0; i < s.count; ++i) {
            const auto row = s.input(i);
            int decoded = 0;
            for (std::size_t b = 0; b < row.size(); ++b) {
                CHECK((row[b] == 0.0 || row[b] == 1.0));
                if (row[b] > 0.5) decoded |= (1 << b);
            }
            CHECK(decoded == static_cast<int>(s.ids[i]));
        }
    }
}

TEST_CASE("primes bits encoding adds two unseen splits at the bit boundary") {
    PrimesConfig cfg;
    const Dataset d = Dataset::primes(cfg);
    REQUIRE(d.has_split("unseen_201_255"));
    REQUIRE(d.has_split("unseen_256_300"));
    CHECK(d.split("unseen_201_255").count == 55);
    CHECK(d.split("unseen_256_300").count == 45);
    CHECK(d.splits().size() == 4);

    // Bit 8 is off for everything the network trains on, and on for 256+.
    for (const auto& row_owner : {std::string("train"), std::string("test")}) {
        const auto& s = d.split(row_owner);
        for (std::size_t i = 0; i < s.count; ++i) {
            CHECK(s.input(i)[8] == 0.0);
        }
    }
    const auto& beyond = d.split("unseen_256_300");
    for (std::size_t i = 0; i < beyond.count; ++i) {
        CHECK(beyond.input(i)[8] == 1.0);
    }
}

TEST_CASE("primes one-hot encoding produces no unseen splits") {
    PrimesConfig cfg;
    cfg.encoding = PrimeEncoding::OneHot;
    const Dataset d = Dataset::primes(cfg);

    CHECK(d.splits().size() == 2);
    CHECK(d.has_split("train"));
    CHECK(d.has_split("test"));
    CHECK_FALSE(d.has_split("unseen_201_255"));
    CHECK(d.input_dim() == 199);

    // Exactly one hot, in the slot for n - lo.
    const auto& s = d.split("train");
    for (std::size_t i = 0; i < s.count; ++i) {
        const auto row = s.input(i);
        const auto hot = std::count(row.begin(), row.end(), 1.0);
        CHECK(hot == 1);
        CHECK(row[static_cast<std::size_t>(s.ids[i]) - 2] == 1.0);
    }
}

TEST_CASE("primes split is deterministic for a seed and varies across seeds") {
    PrimesConfig a;
    PrimesConfig b;
    PrimesConfig c;
    c.seed = 99;
    CHECK(ids_of(Dataset::primes(a).split("train")) ==
          ids_of(Dataset::primes(b).split("train")));
    CHECK(ids_of(Dataset::primes(a).split("train")) !=
          ids_of(Dataset::primes(c).split("train")));
}

TEST_CASE("primes supports a stratified 60/20/20 split through 500") {
    PrimesConfig cfg;
    cfg.hi = 500;
    cfg.unseen_lo = 1;
    cfg.unseen_hi = 0;
    cfg.train_fraction = 0.6;
    cfg.validation_fraction = 0.2;
    const Dataset d = Dataset::primes(cfg);

    CHECK(d.split("train").count == 299);
    CHECK(d.split("validation").count == 100);
    CHECK(d.split("test").count == 100);

    const auto count_primes = [](const nn::api::Split& split) {
        return std::count_if(split.targets.begin(), split.targets.end(),
                             [](double target) { return target > 0.5; });
    };
    CHECK(count_primes(d.split("train")) == 57);
    CHECK(count_primes(d.split("validation")) == 19);
    CHECK(count_primes(d.split("test")) == 19);
}

TEST_CASE("prime_residues uses the prime split and cyclic sin/cos targets") {
    PrimesConfig cfg;
    cfg.hi = 30;
    cfg.unseen_lo = 1;
    cfg.unseen_hi = 0;
    cfg.train_fraction = 0.6;
    cfg.validation_fraction = 0.2;
    const std::vector<int> moduli{2, 3};
    const Dataset d = Dataset::prime_residues(cfg, moduli);
    CHECK(d.input_dim() == 9);
    CHECK(d.output_dim() == 4);

    bool checked_ten = false;
    for (const auto& split : d.splits()) {
        for (std::size_t i = 0; i < split.count; ++i) {
            if (static_cast<int>(split.ids[i]) != 10) continue;
            const auto target = split.target(i);
            CHECK(target[0] == doctest::Approx(0.0).epsilon(1e-10));
            CHECK(target[1] == doctest::Approx(1.0).epsilon(1e-10));
            CHECK(target[2] == doctest::Approx(std::sin(20.0 * std::acos(-1.0) / 3.0)));
            CHECK(target[3] == doctest::Approx(std::cos(20.0 * std::acos(-1.0) / 3.0)));
            checked_ten = true;
        }
    }
    CHECK(checked_ten);
}

TEST_CASE("prime split fractions must be valid") {
    PrimesConfig cfg;
    cfg.train_fraction = 0.9;
    cfg.validation_fraction = 0.2;
    CHECK_THROWS_AS(static_cast<void>(Dataset::primes(cfg)), std::invalid_argument);
}

TEST_CASE("split() throws for an unknown name") {
    const Dataset d = Dataset::x_squared(4, 4, 1);
    CHECK_THROWS_AS([[maybe_unused]] const auto& s = d.split("nope"), std::out_of_range);
}
