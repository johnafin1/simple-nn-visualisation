#include <algorithm>
#include <array>
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
using nn::core::DenseLayer;
using nn::core::InitKind;
using nn::core::MseLoss;
using nn::core::Network;
using nn::core::SgdWeightDecay;
using nn::core::TanhLayer;
using nn::core::WeightedBceWithLogitsLoss;

namespace {

constexpr std::array<int, 8> kPrimeModuli{2, 3, 5, 7, 11, 13, 17, 19};

void print_usage() {
    std::cout
        << "Usage: train_prime_transfer [options]\n"
        << "  --pretrain-steps N        modulo-residue pretraining steps (default 20000)\n"
        << "  --prime-steps N           frozen-encoder prime steps (default 200000)\n"
        << "  --pretrain-lr F           static modulo-phase learning rate (default 0.1)\n"
        << "  --prime-lr F              static prime-head learning rate (default 0.5)\n"
        << "  --weight-decay F          L2 penalty on trainable weights (default 0.0003)\n"
        << "  --hidden N                shared encoder width (default 64)\n"
        << "  --seed N                  RNG/data-split seed (default 42)\n"
        << "  --name S                  run name (default prime_transfer)\n"
        << "  --eval-interval N         held-out evaluation cadence (default 1000)\n"
        << "  --param-log-interval N    parameter snapshot cadence, 0 disables (default 10000)\n"
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

RunConfig phase_config(const RunConfig& base, std::string phase, int steps, int offset,
                       int total_steps) {
    RunConfig cfg = base;
    cfg.phase = std::move(phase);
    cfg.steps = steps;
    cfg.step_offset = offset;
    cfg.total_experiment_steps = total_steps;
    return cfg;
}

}  // namespace

// Two-stage transfer experiment:
//   1. Learn cyclic residue features for every small prime divisor needed up to 500.
//   2. Replace the residue head, freeze the shared encoder, and fit a class-balanced
//      primality head on a stratified 60/20/20 split.
int main(int argc, char** argv) {
    RunConfig base;
    base.name = "prime_transfer";
    base.experiment = "prime_modulo_transfer";
    base.seed = 42;
    base.weight_decay = 0.0003;
    base.eval_interval = 1000;
    base.param_log_interval = 10000;
    base.predict_interval = 0;
    base.experiment_has_classification = true;

    int pretrain_steps = 20000;
    int prime_steps = 200000;
    int hidden = 64;
    double pretrain_lr = 0.1;
    double prime_lr = 0.5;

    const std::vector<std::string_view> args(argv + 1, argv + argc);
    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string_view value;
        if (args[i] == "--help" || args[i] == "-h") {
            print_usage();
            return 0;
        }
        if (args[i] == "--pretrain-steps") {
            if (!take_value(args, i, value)) return 1;
            pretrain_steps = std::atoi(std::string(value).c_str());
        } else if (args[i] == "--prime-steps") {
            if (!take_value(args, i, value)) return 1;
            prime_steps = std::atoi(std::string(value).c_str());
        } else if (args[i] == "--pretrain-lr") {
            if (!take_value(args, i, value)) return 1;
            pretrain_lr = std::atof(std::string(value).c_str());
        } else if (args[i] == "--prime-lr") {
            if (!take_value(args, i, value)) return 1;
            prime_lr = std::atof(std::string(value).c_str());
        } else if (args[i] == "--weight-decay") {
            if (!take_value(args, i, value)) return 1;
            base.weight_decay = std::atof(std::string(value).c_str());
        } else if (args[i] == "--hidden") {
            if (!take_value(args, i, value)) return 1;
            hidden = std::atoi(std::string(value).c_str());
        } else if (args[i] == "--seed") {
            if (!take_value(args, i, value)) return 1;
            base.seed = std::strtoull(std::string(value).c_str(), nullptr, 10);
        } else if (args[i] == "--name") {
            if (!take_value(args, i, value)) return 1;
            base.name = std::string(value);
        } else if (args[i] == "--eval-interval") {
            if (!take_value(args, i, value)) return 1;
            base.eval_interval = std::atoi(std::string(value).c_str());
        } else if (args[i] == "--param-log-interval") {
            if (!take_value(args, i, value)) return 1;
            base.param_log_interval = std::atoi(std::string(value).c_str());
        } else {
            std::cerr << "Unknown option: " << args[i] << "\n";
            print_usage();
            return 1;
        }
    }

    if (pretrain_steps < 1 || prime_steps < 1 || hidden < 1 || pretrain_lr <= 0.0 ||
        prime_lr <= 0.0 || base.weight_decay < 0.0 || base.eval_interval < 1) {
        std::cerr << "Steps, width, learning rates, and eval interval must be positive; "
                     "weight decay must be non-negative.\n";
        return 1;
    }

    PrimesConfig data_cfg;
    data_cfg.lo = 2;
    data_cfg.hi = 500;
    data_cfg.unseen_lo = 1;
    data_cfg.unseen_hi = 0;  // strict held-out evaluation stays inside 2..500
    data_cfg.train_fraction = 0.6;
    data_cfg.validation_fraction = 0.2;
    data_cfg.seed = base.seed;
    data_cfg.encoding = PrimeEncoding::Bits;
    data_cfg.bits = 9;

    Dataset residue_data = Dataset::prime_residues(data_cfg, kPrimeModuli);
    Dataset prime_data = Dataset::primes(data_cfg);
    const auto in_dim = residue_data.input_dim();
    const auto residue_dim = residue_data.output_dim();
    const int total_steps = pretrain_steps + prime_steps;

    Network net("network");
    auto first_dense = std::make_unique<DenseLayer>(
        in_dim, static_cast<std::size_t>(hidden), InitKind::Xavier, base.seed + 1,
        "encoder.0");
    DenseLayer* const encoder0 = first_dense.get();
    net.add(std::move(first_dense));
    net.add(std::make_unique<TanhLayer>());
    auto second_dense = std::make_unique<DenseLayer>(
        static_cast<std::size_t>(hidden), static_cast<std::size_t>(hidden),
        InitKind::Xavier, base.seed + 2, "encoder.1");
    DenseLayer* const encoder1 = second_dense.get();
    net.add(std::move(second_dense));
    net.add(std::make_unique<TanhLayer>());
    net.add(std::make_unique<DenseLayer>(
        static_cast<std::size_t>(hidden), residue_dim, InitKind::Xavier, base.seed + 3,
        "residue_head"));

    RunConfig modulo_cfg =
        phase_config(base, "modulo_pretrain", pretrain_steps, 0, total_steps);
    modulo_cfg.lr = pretrain_lr;
    modulo_cfg.model_description =
        "9 -> " + std::to_string(hidden) + " -> " + std::to_string(hidden) +
        " -> 16 residue sin/cos outputs";
    modulo_cfg.loss_name = "MseLoss";
    modulo_cfg.optimizer_name = "SgdWeightDecay";
    modulo_cfg.finalize_run = false;

    MseLoss residue_loss;
    SgdWeightDecay residue_opt(modulo_cfg.lr, modulo_cfg.weight_decay);
    Trainer modulo_trainer(net, residue_loss, residue_opt, modulo_cfg);

    std::cout << "Starting transfer experiment " << modulo_trainer.run_id() << "\n"
              << "  data        2..500, stratified train/validation/test = 60/20/20\n"
              << "  encoder     9 -> " << hidden << " -> " << hidden << "\n"
              << "  phase 1     " << pretrain_steps
              << " modulo-residue steps, lr " << pretrain_lr << "\n"
              << "  phase 2     " << prime_steps
              << " frozen-encoder prime steps, lr " << prime_lr << "\n"
              << "  weight decay " << base.weight_decay << "\n"
              << "  Dashboard: .\\.venv\\Scripts\\python.exe "
                 "src/app/python/primes_plot.py --live\n"
              << "  Metrics:   .\\.venv\\Scripts\\python.exe "
                 "src/app/python/live_plot.py\n";

    const auto run_dir = modulo_trainer.train(residue_data);

    // Replace the pretraining head but keep the learned encoder.
    net.remove_last();
    encoder0->set_trainable(false);
    encoder1->set_trainable(false);
    net.add(std::make_unique<DenseLayer>(
        static_cast<std::size_t>(hidden), 1, InitKind::Xavier, base.seed + 4,
        "prime_head"));

    const auto& prime_train = prime_data.split("train");
    const auto positives = static_cast<std::size_t>(std::count_if(
        prime_train.targets.begin(), prime_train.targets.end(),
        [](double target) { return target >= 0.5; }));
    const std::size_t negatives = prime_train.count - positives;
    if (positives == 0 || negatives == 0) {
        std::cerr << "Prime training split must contain both classes.\n";
        return 1;
    }
    const double train_count = static_cast<double>(prime_train.count);
    const double positive_weight = train_count / (2.0 * static_cast<double>(positives));
    const double negative_weight = train_count / (2.0 * static_cast<double>(negatives));

    RunConfig prime_cfg =
        phase_config(base, "prime_head", prime_steps, pretrain_steps, total_steps);
    prime_cfg.run_id = modulo_trainer.run_id();
    prime_cfg.lr = prime_lr;
    prime_cfg.append_logs = true;
    prime_cfg.finalize_run = true;
    prime_cfg.positive_class_weight = positive_weight;
    prime_cfg.negative_class_weight = negative_weight;
    prime_cfg.model_description =
        "frozen 9 -> " + std::to_string(hidden) + " -> " +
        std::to_string(hidden) + " encoder -> 1 prime logit";
    prime_cfg.loss_name = "WeightedBceWithLogitsLoss";
    prime_cfg.optimizer_name = "SgdWeightDecay";

    WeightedBceWithLogitsLoss prime_loss(positive_weight, negative_weight);
    SgdWeightDecay prime_opt(prime_cfg.lr, prime_cfg.weight_decay);
    Trainer prime_trainer(net, prime_loss, prime_opt, prime_cfg);

    std::cout << "Modulo pretraining complete; encoder frozen.\n"
              << "  weighted BCE: prime " << positive_weight << ", composite "
              << negative_weight << "\n";
    prime_trainer.train(prime_data);

    std::cout << "Run complete: " << run_dir.string() << "\n";
    return 0;
}
