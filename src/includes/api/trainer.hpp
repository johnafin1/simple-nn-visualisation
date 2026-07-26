#pragma once

#include <filesystem>
#include <string>

#include "api/dataset.hpp"
#include "api/run_config.hpp"
#include "classes/loss.hpp"
#include "classes/network.hpp"
#include "classes/optimizer.hpp"

namespace nn::api {

// Orchestrates one training run: full-batch SGD over the dataset, emitting the three
// JSONL streams (metrics / params / predictions) into runs/<run_id>/. Holds
// references to the model pieces (does not own them).
class Trainer {
public:
    Trainer(nn::core::Network& net, nn::core::Loss& loss, nn::core::Optimizer& opt,
            RunConfig cfg);

    // Runs the loop and returns the created run directory.
    std::filesystem::path train(const Dataset& data);

    [[nodiscard]] const std::string& run_id() const { return run_id_; }

    // Mean loss and (for a classification loss) accuracy over a split.
    struct SplitMetrics {
        double loss = 0.0;
        double accuracy = 0.0;
    };

private:
    SplitMetrics evaluate(const Split& split);

    nn::core::Network& net_;
    nn::core::Loss& loss_;
    nn::core::Optimizer& opt_;
    RunConfig cfg_;
    std::string run_id_;
};

}  // namespace nn::api
