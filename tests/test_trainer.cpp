#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "api/dataset.hpp"
#include "api/run_config.hpp"
#include "api/trainer.hpp"
#include "classes/activation_layers.hpp"
#include "classes/dense_layer.hpp"
#include "classes/loss.hpp"
#include "classes/network.hpp"
#include "classes/optimizer.hpp"

using nn::api::Dataset;
using nn::api::PrimesConfig;
using nn::api::RunConfig;
using nn::api::Trainer;
using nn::core::BceWithLogitsLoss;
using nn::core::DenseLayer;
using nn::core::InitKind;
using nn::core::MseLoss;
using nn::core::Network;
using nn::core::Sgd;
using nn::core::SgdWeightDecay;
using nn::core::TanhLayer;

namespace {

std::vector<std::string> read_lines(const std::filesystem::path& file) {
    std::vector<std::string> out;
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

// Extracts a numeric field from a flat JSON line; returns nullopt if absent.
std::optional<double> field(const std::string& line, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    const auto at = line.find(needle);
    if (at == std::string::npos) return std::nullopt;
    const std::size_t start = at + needle.size();
    const std::size_t stop = line.find_first_of(",}", start);
    return std::stod(line.substr(start, stop - start));
}

bool has_split(const std::string& line, const std::string& name) {
    return line.find("\"split\":\"" + name + "\"") != std::string::npos;
}

// Losses for one split, in file order.
std::vector<double> losses_for(const std::filesystem::path& file, const std::string& split) {
    std::vector<double> out;
    for (const std::string& line : read_lines(file)) {
        if (!has_split(line, split)) continue;
        if (const auto v = field(line, "loss")) out.push_back(*v);
    }
    return out;
}

}  // namespace

TEST_CASE("Trainer produces valid streams and decreasing loss (Phase 3 exit)") {
    Network net("network");
    net.add(std::make_unique<DenseLayer>(1, 8, InitKind::Xavier, 1, "dense.0"));
    net.add(std::make_unique<TanhLayer>());
    net.add(std::make_unique<DenseLayer>(8, 1, InitKind::Xavier, 2, "dense.1"));

    MseLoss loss;
    Sgd opt(0.1);

    RunConfig cfg;
    cfg.name = "unit";
    cfg.seed = 123;
    cfg.lr = 0.1;
    cfg.steps = 300;
    cfg.eval_interval = 50;
    cfg.param_log_interval = 100;
    cfg.predict_interval = 100;
    cfg.predict_splits = {"grid"};
    cfg.runs_dir = std::filesystem::temp_directory_path() / "nn_trainer_test_runs";
    std::error_code ec;
    std::filesystem::remove_all(cfg.runs_dir, ec);

    Dataset data = Dataset::x_squared(16, 21, 7, 11);

    Trainer trainer(net, loss, opt, cfg);
    const std::filesystem::path run_dir = trainer.train(data);

    CHECK(std::filesystem::exists(run_dir / "metrics.jsonl"));
    CHECK(std::filesystem::exists(run_dir / "params.jsonl"));
    CHECK(std::filesystem::exists(run_dir / "predictions.jsonl"));
    CHECK(std::filesystem::exists(run_dir / "config.json"));
    CHECK(std::filesystem::exists(run_dir / "meta.json"));

    const auto losses = losses_for(run_dir / "metrics.jsonl", "train");
    REQUIRE(losses.size() >= 2);
    CHECK(losses.back() < losses.front());

    // Every non-train split gets evaluated, and regression logs no accuracy.
    CHECK_FALSE(losses_for(run_dir / "metrics.jsonl", "test").empty());
    CHECK_FALSE(losses_for(run_dir / "metrics.jsonl", "grid").empty());
    for (const std::string& line : read_lines(run_dir / "metrics.jsonl")) {
        CHECK(line.find("\"accuracy\"") == std::string::npos);
    }

    // Predictions are restricted to the requested split and carry the generic schema.
    const auto pred_lines = read_lines(run_dir / "predictions.jsonl");
    REQUIRE_FALSE(pred_lines.empty());
    for (const std::string& line : pred_lines) {
        CHECK(has_split(line, "grid"));
        CHECK(field(line, "id").has_value());
        CHECK(field(line, "target").has_value());
        CHECK(field(line, "pred").has_value());
    }

    std::filesystem::remove_all(cfg.runs_dir, ec);
}

TEST_CASE("Trainer logs accuracy per split for a classification loss") {
    // 9 -> 16 -> 1 logit on a small primes dataset; enough steps to beat the base rate
    // on train, which is all this test asserts.
    Network net("network");
    net.add(std::make_unique<DenseLayer>(9, 16, InitKind::Xavier, 1, "dense.0"));
    net.add(std::make_unique<TanhLayer>());
    net.add(std::make_unique<DenseLayer>(16, 1, InitKind::Xavier, 2, "dense.1"));

    BceWithLogitsLoss loss;
    SgdWeightDecay opt(0.3, 0.001);

    RunConfig cfg;
    cfg.name = "unit_cls";
    cfg.seed = 5;
    cfg.steps = 200;
    cfg.eval_interval = 50;
    cfg.param_log_interval = 0;
    cfg.predict_interval = 100;
    cfg.runs_dir = std::filesystem::temp_directory_path() / "nn_trainer_cls_test_runs";
    std::error_code ec;
    std::filesystem::remove_all(cfg.runs_dir, ec);

    PrimesConfig pc;
    pc.hi = 60;
    pc.unseen_lo = 61;
    pc.unseen_hi = 80;
    Dataset data = Dataset::primes(pc);

    Trainer trainer(net, loss, opt, cfg);
    const std::filesystem::path run_dir = trainer.train(data);

    const auto lines = read_lines(run_dir / "metrics.jsonl");
    std::size_t with_accuracy = 0;
    std::size_t with_balanced_accuracy = 0;
    for (const std::string& line : lines) {
        if (const auto a = field(line, "accuracy")) {
            ++with_accuracy;
            CHECK(*a >= 0.0);
            CHECK(*a <= 1.0);
        }
        if (const auto balanced = field(line, "balanced_accuracy")) {
            ++with_balanced_accuracy;
            CHECK(*balanced >= 0.0);
            CHECK(*balanced <= 1.0);
            CHECK(field(line, "prime_recall").has_value());
            CHECK(field(line, "composite_recall").has_value());
            CHECK(field(line, "true_positive").has_value());
            CHECK(field(line, "true_negative").has_value());
            CHECK(field(line, "false_positive").has_value());
            CHECK(field(line, "false_negative").has_value());
        }
    }
    CHECK(with_accuracy == lines.size());
    CHECK(with_balanced_accuracy == lines.size());

    // Unseen splits are evaluated too.
    CHECK_FALSE(losses_for(run_dir / "metrics.jsonl", "unseen_61_80").empty());

    // Predictions default to every split and are probabilities, not logits.
    const auto preds = read_lines(run_dir / "predictions.jsonl");
    REQUIRE_FALSE(preds.empty());
    bool saw_unseen = false;
    for (const std::string& line : preds) {
        const auto p = field(line, "pred");
        REQUIRE(p.has_value());
        CHECK(*p >= 0.0);
        CHECK(*p <= 1.0);
        if (has_split(line, "unseen_61_80")) saw_unseen = true;
    }
    CHECK(saw_unseen);

    std::filesystem::remove_all(cfg.runs_dir, ec);
}

TEST_CASE("Trainer appends staged phases with global steps into one run") {
    Network net("network");
    net.add(std::make_unique<DenseLayer>(1, 1, InitKind::Xavier, 1, "dense.0"));
    MseLoss loss;
    Sgd opt(0.01);
    Dataset data = Dataset::x_squared(4, 3, 7, 3);

    RunConfig first;
    first.name = "unit_staged";
    first.phase = "first";
    first.steps = 1;
    first.total_experiment_steps = 2;
    first.eval_interval = 1;
    first.param_log_interval = 0;
    first.predict_interval = 0;
    first.finalize_run = false;
    first.runs_dir = std::filesystem::temp_directory_path() / "nn_trainer_staged_runs";
    std::error_code ec;
    std::filesystem::remove_all(first.runs_dir, ec);

    Trainer first_trainer(net, loss, opt, first);
    const auto run_dir = first_trainer.train(data);

    RunConfig second = first;
    second.run_id = first_trainer.run_id();
    second.phase = "second";
    second.step_offset = 1;
    second.append_logs = true;
    second.finalize_run = true;
    Trainer second_trainer(net, loss, opt, second);
    second_trainer.train(data);

    const auto lines = read_lines(run_dir / "metrics.jsonl");
    std::vector<std::string> train_lines;
    for (const auto& line : lines) {
        if (has_split(line, "train")) train_lines.push_back(line);
    }
    REQUIRE(train_lines.size() == 2);
    CHECK(train_lines[0].find("\"phase\":\"first\"") != std::string::npos);
    CHECK(train_lines[1].find("\"phase\":\"second\"") != std::string::npos);
    const auto first_step = field(train_lines[0], "step");
    const auto second_step = field(train_lines[1], "step");
    REQUIRE(first_step.has_value());
    REQUIRE(second_step.has_value());
    CHECK(*first_step == doctest::Approx(0.0));
    CHECK(*second_step == doctest::Approx(1.0));
    CHECK(std::filesystem::exists(run_dir / "config_first.json"));
    CHECK(std::filesystem::exists(run_dir / "config_second.json"));

    std::filesystem::remove_all(first.runs_dir, ec);
}

TEST_CASE("one-hot primes config trains without unseen splits") {
    Network net("network");
    net.add(std::make_unique<DenseLayer>(59, 8, InitKind::Xavier, 1, "dense.0"));
    net.add(std::make_unique<TanhLayer>());
    net.add(std::make_unique<DenseLayer>(8, 1, InitKind::Xavier, 2, "dense.1"));

    BceWithLogitsLoss loss;
    Sgd opt(0.2);

    RunConfig cfg;
    cfg.name = "unit_onehot";
    cfg.steps = 20;
    cfg.eval_interval = 10;
    cfg.param_log_interval = 0;
    cfg.predict_interval = 10;
    cfg.runs_dir = std::filesystem::temp_directory_path() / "nn_trainer_onehot_test_runs";
    std::error_code ec;
    std::filesystem::remove_all(cfg.runs_dir, ec);

    PrimesConfig pc;
    pc.hi = 60;
    pc.encoding = nn::api::PrimeEncoding::OneHot;
    Dataset data = Dataset::primes(pc);
    REQUIRE(data.input_dim() == 59);

    Trainer trainer(net, loss, opt, cfg);
    const std::filesystem::path run_dir = trainer.train(data);

    for (const std::string& line : read_lines(run_dir / "metrics.jsonl")) {
        CHECK(line.find("unseen") == std::string::npos);
    }

    std::filesystem::remove_all(cfg.runs_dir, ec);
}
