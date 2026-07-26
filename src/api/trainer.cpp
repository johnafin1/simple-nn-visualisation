#include "api/trainer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "classes/json_line.hpp"
#include "classes/jsonl_sink.hpp"
#include "classes/tensor.hpp"

namespace nn::api {

using nn::core::ParamView;
using nn::core::Tensor;
using nn::log::JsonLine;
using nn::log::JsonlSink;

namespace {

std::string timestamp_utc() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::array<char, 32> buf{};
    std::strftime(buf.data(), buf.size(), "%Y%m%dT%H%M%S", &tm);
    return std::string(buf.data());
}

std::string short_hex(std::uint64_t seed) {
    std::mt19937_64 rng(seed ^ std::random_device{}());
    std::array<char, 8> buf{};
    std::snprintf(buf.data(), buf.size(), "%06x", static_cast<unsigned>(rng() & 0xffffff));
    return std::string(buf.data());
}

// A sample row becomes a column vector: the network treats one sample at a time.
Tensor column_from(std::span<const double> row) {
    Tensor t(row.size(), 1);
    std::copy(row.begin(), row.end(), t.span().begin());
    return t;
}

}  // namespace

Trainer::Trainer(nn::core::Network& net, nn::core::Loss& loss, nn::core::Optimizer& opt,
                 RunConfig cfg)
    : net_(net), loss_(loss), opt_(opt), cfg_(std::move(cfg)) {
    run_id_ = timestamp_utc() + "_" + cfg_.name + "_" + short_hex(cfg_.seed);
}

Trainer::SplitMetrics Trainer::evaluate(const Split& split) {
    SplitMetrics m;
    if (split.count == 0) return m;

    std::size_t correct = 0;
    for (std::size_t i = 0; i < split.count; ++i) {
        const Tensor x = column_from(split.input(i));
        const Tensor target = column_from(split.target(i));
        const Tensor y_hat = net_.forward(x);
        m.loss += loss_.value(y_hat, target);

        if (loss_.is_classification()) {
            const Tensor p = loss_.activate(y_hat);
            bool all_right = true;
            for (std::size_t k = 0; k < p.size(); ++k) {
                if ((p.cspan()[k] >= 0.5) != (target.cspan()[k] >= 0.5)) {
                    all_right = false;
                    break;
                }
            }
            if (all_right) ++correct;
        }
    }
    const auto n = static_cast<double>(split.count);
    m.loss /= n;
    m.accuracy = static_cast<double>(correct) / n;
    return m;
}

std::filesystem::path Trainer::train(const Dataset& data) {
    const std::filesystem::path run_dir = cfg_.runs_dir / run_id_;
    std::filesystem::create_directories(run_dir);

    // config.json + meta.json (start)
    {
        JsonLine line;
        line.add("run_id", std::string_view{run_id_})
            .add("name", std::string_view{cfg_.name})
            .add("seed", static_cast<long long>(cfg_.seed))
            .add("lr", cfg_.lr)
            .add("weight_decay", cfg_.weight_decay)
            .add("steps", cfg_.steps)
            .add("param_log_interval", cfg_.param_log_interval)
            .add("predict_interval", cfg_.predict_interval)
            .add("input_dim", data.input_dim())
            .add("output_dim", data.output_dim())
            .add("classification", loss_.is_classification());
        // One size_<split> field per split keeps config.json flat and greppable.
        for (const Split& s : data.splits()) {
            line.add("size_" + s.name, s.count);
        }
        std::ofstream cfg_file(run_dir / "config.json");
        cfg_file << line.str() << '\n';
    }
    {
        std::ofstream meta_file(run_dir / "meta.json");
        meta_file << JsonLine{}
                         .add("run_id", std::string_view{run_id_})
                         .add("start", std::string_view{timestamp_utc()})
                         .add("status", std::string_view{"running"})
                         .str()
                  << '\n';
    }

    JsonlSink metrics(run_dir / "metrics.jsonl", cfg_.flush_every, cfg_.flush_interval_ms);
    JsonlSink params_sink(run_dir / "params.jsonl", cfg_.flush_every, cfg_.flush_interval_ms);
    JsonlSink preds(run_dir / "predictions.jsonl", cfg_.flush_every, cfg_.flush_interval_ms);

    const Split& train_split = data.split("train");
    const double inv_n =
        train_split.count == 0 ? 1.0 : 1.0 / static_cast<double>(train_split.count);

    // Which splits get per-sample predictions logged.
    std::vector<const Split*> predict_targets;
    for (const Split& s : data.splits()) {
        const bool wanted =
            cfg_.predict_splits.empty() ||
            std::find(cfg_.predict_splits.begin(), cfg_.predict_splits.end(), s.name) !=
                cfg_.predict_splits.end();
        if (wanted) predict_targets.push_back(&s);
    }

    const auto wall_start = std::chrono::steady_clock::now();

    for (int step = 0; step < cfg_.steps; ++step) {
        net_.zero_grad();

        double train_loss = 0.0;
        std::size_t train_correct = 0;
        for (std::size_t i = 0; i < train_split.count; ++i) {
            const Tensor x = column_from(train_split.input(i));
            const Tensor target = column_from(train_split.target(i));
            const Tensor y_hat = net_.forward(x);
            train_loss += loss_.value(y_hat, target);

            if (loss_.is_classification()) {
                const Tensor p = loss_.activate(y_hat);
                bool all_right = true;
                for (std::size_t k = 0; k < p.size(); ++k) {
                    if ((p.cspan()[k] >= 0.5) != (target.cspan()[k] >= 0.5)) {
                        all_right = false;
                        break;
                    }
                }
                if (all_right) ++train_correct;
            }

            // Seed gradient scaled so the objective is the mean over the batch.
            Tensor g = loss_.grad(y_hat, target);
            for (std::size_t k = 0; k < g.size(); ++k) {
                g.span()[k] *= inv_n;
            }
            net_.backward(g);
        }
        train_loss *= inv_n;

        // Norms over the accumulated batch gradient / current weights.
        auto ps = net_.parameters();
        double w_sq = 0.0;
        double g_sq = 0.0;
        for (const ParamView& p : ps) {
            for (double v : p.values) w_sq += v * v;
            for (double v : p.grads) g_sq += v * v;
        }
        const double weight_norm = std::sqrt(w_sq);
        const double grad_norm = std::sqrt(g_sq);

        opt_.step(ps);

        const auto wall_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - wall_start)
                .count();

        {
            JsonLine line;
            line.add("run_id", std::string_view{run_id_})
                .add("step", step)
                .add("split", std::string_view{"train"})
                .add("loss", train_loss)
                .add("lr", opt_.learning_rate())
                .add("weight_norm", weight_norm)
                .add("grad_norm", grad_norm)
                .add("wall_ms", static_cast<long long>(wall_ms));
            if (loss_.is_classification()) {
                line.add("accuracy",
                         static_cast<double>(train_correct) * inv_n);
            }
            metrics.write(line.str());
        }

        // Every non-train split, evaluated on the same cadence.
        if (cfg_.eval_interval > 0 &&
            (step % cfg_.eval_interval == 0 || step == cfg_.steps - 1)) {
            for (const Split& s : data.splits()) {
                if (s.name == "train") continue;
                const SplitMetrics m = evaluate(s);
                JsonLine line;
                line.add("run_id", std::string_view{run_id_})
                    .add("step", step)
                    .add("split", std::string_view{s.name})
                    .add("loss", m.loss);
                if (loss_.is_classification()) {
                    line.add("accuracy", m.accuracy);
                }
                metrics.write(line.str());
            }
        }

        // Per-parameter snapshot (heavy, interval-gated).
        if (cfg_.param_log_interval > 0 &&
            (step % cfg_.param_log_interval == 0 || step == cfg_.steps - 1)) {
            for (const ParamView& p : ps) {
                for (std::size_t r = 0; r < p.rows; ++r) {
                    for (std::size_t c = 0; c < p.cols; ++c) {
                        const std::size_t idx = r * p.cols + c;
                        params_sink.write(JsonLine{}
                                              .add("step", step)
                                              .add("layer", std::string_view{p.layer})
                                              .add("kind", std::string_view{p.name})
                                              .add("row", r)
                                              .add("col", c)
                                              .add("value", p.values[idx])
                                              .add("grad", p.grads[idx])
                                              .str());
                    }
                }
            }
        }

        // Per-sample predictions: one row per sample per logged step, keyed by the
        // split's `id`. Same schema for every task, which is what lets one plotting
        // script serve both the x^2 curve and the primes heatmap.
        if (cfg_.predict_interval > 0 &&
            (step % cfg_.predict_interval == 0 || step == cfg_.steps - 1)) {
            for (const Split* s : predict_targets) {
                for (std::size_t i = 0; i < s->count; ++i) {
                    const Tensor raw = net_.forward(column_from(s->input(i)));
                    const Tensor pred = loss_.activate(raw);
                    preds.write(JsonLine{}
                                    .add("step", step)
                                    .add("split", std::string_view{s->name})
                                    .add("id", s->ids[i])
                                    .add("target", s->target(i)[0])
                                    .add("pred", pred.at(0, 0))
                                    .str());
                }
            }
        }
    }

    metrics.flush();
    params_sink.flush();
    preds.flush();

    {
        std::ofstream meta_file(run_dir / "meta.json");
        meta_file << JsonLine{}
                         .add("run_id", std::string_view{run_id_})
                         .add("end", std::string_view{timestamp_utc()})
                         .add("status", std::string_view{"done"})
                         .str()
                  << '\n';
    }

    return run_dir;
}

}  // namespace nn::api
