#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nn::api {

// Hyperparameters + logging cadence for a single training run. Everything the
// Trainer needs to run and log a reproducible experiment.
struct RunConfig {
    std::string name = "run";           // human tag, part of the run_id
    std::uint64_t seed = 0;             // RNG seed for reproducibility
    double lr = 0.05;                   // learning rate
    double weight_decay = 0.0;          // L2 penalty, when using SgdWeightDecay
    int steps = 2000;                   // training steps (full-batch passes)

    int eval_interval = 50;            // steps between non-train split metric rows
    int param_log_interval = 100;      // steps between params.jsonl snapshots (0 disables)
    int predict_interval = 100;        // steps between predictions.jsonl snapshots (0 disables)

    // Splits to write into predictions.jsonl. Empty means every split in the dataset;
    // x^2 narrows this to its dense "grid" split so the curve plot stays cheap.
    std::vector<std::string> predict_splits{};

    std::size_t flush_every = 256;     // JsonlSink flush cadence (records)
    int flush_interval_ms = 250;       // JsonlSink flush cadence (time), for live tailing
    std::filesystem::path runs_dir = "runs";
};

}  // namespace nn::api
