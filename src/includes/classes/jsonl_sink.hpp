#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace nn::log {

// Buffered writer for a single JSONL stream (one JSON object per line). Writes are
// accumulated and flushed to disk every `flush_every` records, whenever
// `flush_interval_ms` has elapsed since the last flush, and on destruction. The timed
// flush is what lets a live tailer (src/app/python/live_plot.py) see fresh rows even
// when steps are slow enough that the record count alone would not trigger a flush.
class JsonlSink {
public:
    explicit JsonlSink(std::filesystem::path file, std::size_t flush_every = 256,
                       int flush_interval_ms = 250, bool append = false);
    ~JsonlSink();

    JsonlSink(const JsonlSink&) = delete;
    JsonlSink& operator=(const JsonlSink&) = delete;
    JsonlSink(JsonlSink&&) noexcept = default;
    JsonlSink& operator=(JsonlSink&&) noexcept = default;

    // Append one record (a complete JSON object). A newline is added.
    void write(std::string_view line);
    void flush();

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
    std::ofstream out_;
    std::size_t flush_every_;
    std::chrono::milliseconds flush_interval_;
    std::size_t since_flush_ = 0;
    std::chrono::steady_clock::time_point last_flush_ = std::chrono::steady_clock::now();
};

}  // namespace nn::log
