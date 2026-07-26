#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <doctest/doctest.h>

#include "classes/json_line.hpp"
#include "classes/jsonl_sink.hpp"

using nn::log::JsonLine;
using nn::log::JsonlSink;
using nn::log::json_escape;

TEST_CASE("JsonLine builds an object with mixed types") {
    const std::string s = JsonLine{}
                              .add("step", static_cast<long long>(5))
                              .add("split", std::string_view{"train"})
                              .add("loss", 0.5)
                              .str();
    CHECK(s == "{\"step\":5,\"split\":\"train\",\"loss\":0.5}");
}

TEST_CASE("empty JsonLine is an empty object") {
    CHECK(JsonLine{}.str() == "{}");
}

TEST_CASE("json_escape handles quotes, backslashes and control chars") {
    CHECK(json_escape("a\"b") == "a\\\"b");
    CHECK(json_escape("a\\b") == "a\\\\b");
    CHECK(json_escape("a\nb") == "a\\nb");
    CHECK(json_escape(std::string(1, '\x01')) == "\\u0001");
}

TEST_CASE("JsonLine escapes string values") {
    const std::string s = JsonLine{}.add("name", std::string_view{"dense\"0"}).str();
    CHECK(s == "{\"name\":\"dense\\\"0\"}");
}

TEST_CASE("JsonlSink writes one record per line and flushes on destruction") {
    const auto path = std::filesystem::temp_directory_path() / "nn_test_sink.jsonl";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    {
        JsonlSink sink(path, 256);
        sink.write(JsonLine{}.add("i", static_cast<long long>(0)).str());
        sink.write(JsonLine{}.add("i", static_cast<long long>(1)).str());
    }  // destructor flushes

    std::ifstream in(path);
    REQUIRE(in.is_open());
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    REQUIRE(lines.size() == 2);
    CHECK(lines[0] == "{\"i\":0}");
    CHECK(lines[1] == "{\"i\":1}");
    in.close();
    std::filesystem::remove(path, ec);
}

TEST_CASE("JsonlSink flushes on the timer, not just the record count") {
    // Large record threshold so only the elapsed-time rule can trigger a flush. This is
    // what lets live_plot.py see rows while a run is still in progress.
    const auto path = std::filesystem::temp_directory_path() / "nn_test_sink_timed.jsonl";
    std::error_code ec;
    std::filesystem::remove(path, ec);

    JsonlSink sink(path, /*flush_every=*/1'000'000, /*flush_interval_ms=*/1);
    sink.write(JsonLine{}.add("i", static_cast<long long>(0)).str());
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    // This write sees an elapsed interval, so everything buffered so far hits disk.
    sink.write(JsonLine{}.add("i", static_cast<long long>(1)).str());

    std::ifstream in(path);
    REQUIRE(in.is_open());
    std::string first;
    REQUIRE(std::getline(in, first));
    CHECK(first == "{\"i\":0}");
    in.close();

    std::filesystem::remove(path, ec);
}

TEST_CASE("JsonlSink withholds records when neither flush rule has fired") {
    const auto path = std::filesystem::temp_directory_path() / "nn_test_sink_buffered.jsonl";
    std::error_code ec;
    std::filesystem::remove(path, ec);

    JsonlSink sink(path, /*flush_every=*/1'000'000, /*flush_interval_ms=*/1'000'000);
    sink.write(JsonLine{}.add("i", static_cast<long long>(0)).str());

    std::ifstream in(path);
    REQUIRE(in.is_open());
    std::string line;
    CHECK_FALSE(std::getline(in, line));  // still buffered
    in.close();

    std::filesystem::remove(path, ec);
}

TEST_CASE("JsonlSink creates missing parent directories") {
    const auto dir = std::filesystem::temp_directory_path() / "nn_test_sink_dir";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    const auto path = dir / "nested" / "out.jsonl";
    {
        JsonlSink sink(path, 1);
        sink.write("{}");
    }
    CHECK(std::filesystem::exists(path));
    std::filesystem::remove_all(dir, ec);
}
