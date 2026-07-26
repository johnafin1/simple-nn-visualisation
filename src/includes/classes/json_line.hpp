#pragma once

#include <string>
#include <string_view>

namespace nn::log {

// Minimal hand-rolled builder for a single JSON object emitted as one line.
// Only the value types we actually log are supported (double, integer, string).
// Chainable: JsonLine{}.add("step", 5).add("loss", 0.1).str() -> {"step":5,"loss":0.1}
class JsonLine {
public:
    JsonLine& add(std::string_view key, double value);
    JsonLine& add(std::string_view key, long long value);
    JsonLine& add(std::string_view key, int value);
    JsonLine& add(std::string_view key, std::size_t value);
    JsonLine& add(std::string_view key, std::string_view value);
    // Emits the JSON literals true/false. Needed explicitly, otherwise a bool would
    // ambiguously match all the numeric overloads.
    JsonLine& add(std::string_view key, bool value);

    // The complete JSON object, e.g. {"a":1,"b":"x"}. No trailing newline.
    [[nodiscard]] std::string str() const;

private:
    void begin_key(std::string_view key);

    std::string buf_ = "{";
    bool first_ = true;
};

// Escapes a string for embedding in JSON (quotes, backslashes, control chars).
[[nodiscard]] std::string json_escape(std::string_view s);

}  // namespace nn::log
