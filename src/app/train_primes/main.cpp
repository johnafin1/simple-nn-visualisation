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
using nn::api::PrimeEncoding;
using nn::api::PrimesConfig;
using nn::api::RunConfig;
using nn::api::Trainer;
using nn::core::BceWithLogitsLoss;
using nn::core::DenseLayer;
using nn::core::InitKind;
using nn::core::Network;
using nn::core::SgdWeightDecay;
using nn::core::TanhLayer;

namespace {

void print_usage() {
    std::cout
        << "Usage: train_primes [options]\n"
        << "  --encoding bits|onehot    input encoding (default bits)\n"
        << "  --steps N                 training steps (default 4000)\n"
        << "  --lr F                    learning rate (default 0.5)\n"
        << "  --weight-decay F          L2 penalty on weights (default 0.001)\n"
        << "  --hidden N                width of both hidden layers (default 32)\n"
        << "  --seed N                  RNG seed (default 42)\n"
        << "  --name S                  run name, used in run_id (default primes)\n"
        << "  --predict-interval N      steps between predictions.jsonl snapshots\n"
        << "  --param-log-interval N    steps between params.jsonl rows, 0 disables\n"
        << "  --eval-interval N         steps between non-train split rows\n"
        << "  --help\n";
}

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

// Primality as binary classification on [2, 200], evaluated on the unseen ranges
// [201, 255] and [256, 300].
//
// The two encodings answer different questions. `bits` gives the network shared
// structure across integers, so a genuine rule could transfer to numbers it never saw.
// `onehot` gives every integer its own private input weight, so it can only memorise -
// it is the control that shows what "not generalising" looks like.
//
// Watch it live in another terminal:
//   python src/app/python/primes_plot.py --live
int main(int argc, char** argv) {
    RunConfig cfg;
    cfg.name = "primes";
    cfg.seed = 42;
    cfg.lr = 0.5;
    cfg.weight_decay = 0.001;
    cfg.steps = 4000;
    cfg.eval_interval = 25;
    cfg.param_log_interval = 250;
    cfg.predict_interval = 0;
    // The current prime dashboard reads aggregate metrics, so per-sample predictions
    // are disabled unless explicitly requested through the CLI.

    PrimesConfig data_cfg;
    int hidden = 32;

    const std::vector<std::string_view> args(argv + 1, argv + argc);
    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string_view value;
        if (args[i] == "--help" || args[i] == "-h") {
            print_usage();
            return 0;
        }
        if (args[i] == "--encoding") {
            if (!take_value(args, i, value)) return 1;
            if (value == "bits") {
                data_cfg.encoding = PrimeEncoding::Bits;
            } else if (value == "onehot") {
                data_cfg.encoding = PrimeEncoding::OneHot;
            } else {
                std::cerr << "Unknown encoding: " << value << " (expected bits|onehot)\n";
                return 1;
            }
        } else if (args[i] == "--steps") {
            if (!take_value(args, i, value)) return 1;
            cfg.steps = std::atoi(std::string(value).c_str());
        } else if (args[i] == "--lr") {
            if (!take_value(args, i, value)) return 1;
            cfg.lr = std::atof(std::string(value).c_str());
        } else if (args[i] == "--weight-decay") {
            if (!take_value(args, i, value)) return 1;
            cfg.weight_decay = std::atof(std::string(value).c_str());
        } else if (args[i] == "--hidden") {
            if (!take_value(args, i, value)) return 1;
            hidden = std::atoi(std::string(value).c_str());
        } else if (args[i] == "--seed") {
            if (!take_value(args, i, value)) return 1;
            cfg.seed = std::strtoull(std::string(value).c_str(), nullptr, 10);
            data_cfg.seed = cfg.seed;
        } else if (args[i] == "--name") {
            if (!take_value(args, i, value)) return 1;
            cfg.name = std::string(value);
        } else if (args[i] == "--predict-interval") {
            if (!take_value(args, i, value)) return 1;
            cfg.predict_interval = std::atoi(std::string(value).c_str());
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

    if (hidden < 1) {
        std::cerr << "--hidden must be at least 1\n";
        return 1;
    }

    Dataset data = Dataset::primes(data_cfg);
    const auto in_dim = static_cast<std::size_t>(data.input_dim());

    // Final layer is linear: it emits a logit, and BceWithLogitsLoss applies the
    // sigmoid internally where it can be done stably.
    Network net("network");
    net.add(std::make_unique<DenseLayer>(in_dim, hidden, InitKind::Xavier, cfg.seed + 1,
                                        "dense.0"));
    net.add(std::make_unique<TanhLayer>());
    net.add(std::make_unique<DenseLayer>(hidden, hidden, InitKind::Xavier, cfg.seed + 2,
                                        "dense.1"));
    net.add(std::make_unique<TanhLayer>());
    net.add(std::make_unique<DenseLayer>(hidden, 1, InitKind::Xavier, cfg.seed + 3,
                                        "dense.2"));

    BceWithLogitsLoss loss;
    SgdWeightDecay opt(cfg.lr, cfg.weight_decay);

    Trainer trainer(net, loss, opt, cfg);
    std::cout << "Starting run " << trainer.run_id() << "\n"
              << "  encoding    "
              << (data_cfg.encoding == PrimeEncoding::Bits ? "bits" : "onehot") << " ("
              << in_dim << " inputs)\n"
              << "  net         " << in_dim << " -> " << hidden << " -> " << hidden
              << " -> 1\n"
              << "  steps       " << cfg.steps << ", lr " << cfg.lr << ", weight decay "
              << cfg.weight_decay << "\n"
              << "  splits      ";
    for (const auto& s : data.splits()) {
        std::cout << s.name << "(" << s.count << ") ";
    }
    std::cout << "\n  Watch live: python src/app/python/primes_plot.py --live\n";

    const auto run_dir = trainer.train(data);

    std::cout << "Run complete: " << run_dir.string() << "\n";
    return 0;
}
