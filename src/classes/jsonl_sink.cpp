#include "classes/jsonl_sink.hpp"

#include <utility>

namespace nn::log {

JsonlSink::JsonlSink(std::filesystem::path file, std::size_t flush_every,
                     int flush_interval_ms, bool append)
    : path_(std::move(file)),
      flush_every_(flush_every == 0 ? 1 : flush_every),
      flush_interval_(std::chrono::milliseconds(flush_interval_ms < 0 ? 0
                                                                     : flush_interval_ms)) {
    if (path_.has_parent_path()) {
        std::filesystem::create_directories(path_.parent_path());
    }
    const auto mode = append ? (std::ios::out | std::ios::app)
                             : (std::ios::out | std::ios::trunc);
    out_.open(path_, mode);
}

JsonlSink::~JsonlSink() {
    if (out_.is_open()) {
        flush();
    }
}

void JsonlSink::write(std::string_view line) {
    out_ << line << '\n';
    ++since_flush_;
    const bool count_reached = since_flush_ >= flush_every_;
    const bool interval_elapsed =
        std::chrono::steady_clock::now() - last_flush_ >= flush_interval_;
    if (count_reached || interval_elapsed) {
        flush();
    }
}

void JsonlSink::flush() {
    out_.flush();
    since_flush_ = 0;
    last_flush_ = std::chrono::steady_clock::now();
}

}  // namespace nn::log
