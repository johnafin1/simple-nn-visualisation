#include "classes/json_line.hpp"

#include <array>
#include <cmath>
#include <cstdio>

namespace nn::log {

std::string json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    std::array<char, 8> u{};
                    std::snprintf(u.data(), u.size(), "\\u%04x", c);
                    out += u.data();
                } else {
                    out += c;
                }
        }
    }
    return out;
}

namespace {

// Format a double without locale surprises, enough precision to round-trip.
std::string format_double(double value) {
    if (std::isnan(value)) return "\"NaN\"";
    if (std::isinf(value)) return value > 0 ? "\"Infinity\"" : "\"-Infinity\"";
    std::array<char, 32> b{};
    std::snprintf(b.data(), b.size(), "%.17g", value);
    return std::string(b.data());
}

}  // namespace

void JsonLine::begin_key(std::string_view key) {
    if (!first_) {
        buf_ += ',';
    }
    first_ = false;
    buf_ += '"';
    buf_ += json_escape(key);
    buf_ += "\":";
}

JsonLine& JsonLine::add(std::string_view key, double value) {
    begin_key(key);
    buf_ += format_double(value);
    return *this;
}

JsonLine& JsonLine::add(std::string_view key, long long value) {
    begin_key(key);
    buf_ += std::to_string(value);
    return *this;
}

JsonLine& JsonLine::add(std::string_view key, int value) {
    return add(key, static_cast<long long>(value));
}

JsonLine& JsonLine::add(std::string_view key, std::size_t value) {
    begin_key(key);
    buf_ += std::to_string(value);
    return *this;
}

JsonLine& JsonLine::add(std::string_view key, std::string_view value) {
    begin_key(key);
    buf_ += '"';
    buf_ += json_escape(value);
    buf_ += '"';
    return *this;
}

JsonLine& JsonLine::add(std::string_view key, bool value) {
    begin_key(key);
    buf_ += value ? "true" : "false";
    return *this;
}

std::string JsonLine::str() const {
    return buf_ + "}";
}

}  // namespace nn::log
