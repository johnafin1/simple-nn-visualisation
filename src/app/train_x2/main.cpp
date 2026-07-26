#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "api/dataset.hpp"
#include "api/run_config.hpp"
#include "api/trainer.hpp"
#include "classes/activation_layers.hpp"
#include "classes/dense_layer.hpp"
#include "classes/loss.hpp"
#include "classes/network.hpp"
#include "classes/optimizer.hpp"

using nn::api::Dataset;
using nn::api::RunConfig;
using nn::api::Trainer;
using nn::core::DenseLayer;
using nn::core::InitKind;
using nn::core::MseLoss;
using nn::core::Network;
using nn::core::Sgd;
using nn::core::TanhLayer;

namespace {

void print_usage() {
    std::cout << "Usage: train_x2 [options]\n"
              << "  --steps N                 training steps (default 3000)\n"
              << "  --lr F                    learning rate (default 0.1)\n"
              << "  --seed N                  RNG seed (default 42)\n"
              << "  --name S                  run name, used in run_id (default x2)\n"
              << "  --param-log-interval N    steps between params.jsonl rows, 0 disables\n"
              << "  --eval-interval N         steps between test-split rows\n"
              << "  --help\n";
}

// Returns false if the flag is missing its value.
bool take_value(const std::vector<std::string_view>& args, std::size_t& i,
                std::string_view& out) {
    if (i + 1 >= args.size()) {
        std::cerr << "Missing value for " << args[i] << "\n";
        return false;
    }
    out = args[++i];
    return true;
}

}  // namespace

// Flagship-in-miniature: train a tiny MLP to regress f(x) = x^2 and log everything
// to runs/<run_id>/. Phase 5 will extend this into the full grokking experiment.
//
// Watch it live in another terminal:
//   python src/app/python/live_plot.py
int main(int argc, char** argv) {
    RunConfig cfg;
    cfg.name = "x2";
    cfg.seed = 42;
    cfg.lr = 0.1;
    cfg.steps = 3000;
    cfg.eval_interval = 25;
    cfg.param_log_interval = 100;
    cfg.predict_interval = 100;
    // Only the dense grid needs prediction rows: it is what the curve plot draws.
    cfg.predict_splits = {"grid"};

    const std::vector<std::string_view> args(argv + 1, argv + argc);
    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string_view value;
        if (args[i] == "--help" || args[i] == "-h") {
            print_usage();
            return 0;
        }
        if (args[i] == "--steps") {
            if (!take_value(args, i, value)) return 1;
            cfg.steps = std::atoi(std::string(value).c_str());
        } else if (args[i] == "--lr") {
            if (!take_value(args, i, value)) return 1;
            cfg.lr = std::atof(std::string(value).c_str());
        } else if (args[i] == "--seed") {
            if (!take_value(args, i, value)) return 1;
            cfg.seed = std::strtoull(std::string(value).c_str(), nullptr, 10);
        } else if (args[i] == "--name") {
            if (!take_value(args, i, value)) return 1;
            cfg.name = std::string(value);
        } else if (args[i] == "--param-log-interval") {
            if (!take_value(args, i, value)) return 1;
            cfg.param_log_interval = std::atoi(std::string(value).c_str());
        } else if (args[i] == "--eval-interval") {
            if (!take_value(args, i, value)) return 1;
            cfg.eval_interval = std::atoi(std::string(value).c_str());
        } else {
            std::cerr << "Unknown option: " << args[i] << "\n";
            print_usage();
            return 1;
        }
    }

    Network net("network");
    net.add(std::make_unique<DenseLayer>(1, 8, InitKind::Xavier, cfg.seed + 1, "dense.0"));
    net.add(std::make_unique<TanhLayer>());
    net.add(std::make_unique<DenseLayer>(8, 8, InitKind::Xavier, cfg.seed + 2, "dense.1"));
    net.add(std::make_unique<TanhLayer>());
    net.add(std::make_unique<DenseLayer>(8, 1, InitKind::Xavier, cfg.seed + 3, "dense.2"));

    MseLoss loss;
    Sgd opt(cfg.lr);

    Dataset data =
        Dataset::x_squared(/*n_train=*/20, /*n_test=*/41, /*seed=*/7, /*n_grid=*/41);

    Trainer trainer(net, loss, opt, cfg);
    std::cout << "Starting run " << trainer.run_id() << " (" << cfg.steps << " steps, lr "
              << cfg.lr << ")\n";
    std::cout << "  Watch live: python src/app/python/live_plot.py\n";

    const auto run_dir = trainer.train(data);

    std::cout << "Run complete: " << run_dir.string() << "\n";
    std::cout << "  metrics.jsonl / params.jsonl / predictions.jsonl written.\n";
    return 0;
}
