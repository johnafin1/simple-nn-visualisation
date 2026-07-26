#pragma once

#include <string>

namespace nn::log {

// Cross-cutting interface for components whose state we want to observe during
// training. Phase 2 establishes the seam (a stable identifier); Phase 3 adds
// to_json() plus the JsonlSink that writes the per-parameter streams.
class Loggable {
public:
    virtual ~Loggable() = default;

    // Stable identifier for this component, e.g. "dense.0" or "network".
    [[nodiscard]] virtual std::string log_name() const = 0;

    // Phase 3 will add:
    //   virtual nlohmann::json to_json() const = 0;  // built from parameters()
};

}  // namespace nn::log
