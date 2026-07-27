#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nn::api {

// Hyperparameters + logging cadence for a single training run or one phase of a
// staged run. Full source/topology provenance remains tracked as a rectification.
struct RunConfig {
    std::string name = "run";           // human tag, part of the run_id
    // Empty creates a new id. A later training phase can reuse an existing id and
    // append to the same run directory.
    std::string run_id{};
    std::string experiment{};           // multi-phase experiment name, when applicable
    std::string phase = "train";        // label emitted on config/metric rows
    std::string model_description{};    // human-readable topology/provenance
    std::string loss_name{};
    std::string optimizer_name{};
    std::uint64_t seed = 0;             // RNG seed for reproducibility
    double lr = 0.05;                   // learning rate
    double weight_decay = 0.0;          // L2 penalty, when using SgdWeightDecay
    int steps = 2000;                   // training steps (full-batch passes)
    int step_offset = 0;                // global step offset for an appended phase
    int total_experiment_steps = 0;      // zero means use `steps`
    bool append_logs = false;
    bool finalize_run = true;
    bool experiment_has_classification = false;
    double positive_class_weight = 1.0;
    double negative_class_weight = 1.0;

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
