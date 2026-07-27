#include "api/trainer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
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

std::string safe_label(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char c : value) {
        out.push_back(std::isalnum(c) || c == '-' || c == '_' ? static_cast<char>(c) : '_');
    }
    return out.empty() ? "train" : out;
}

void observe_classification(const Tensor& prediction, const Tensor& target,
                            Trainer::SplitMetrics& metrics) {
    bool all_right = true;
    for (std::size_t k = 0; k < prediction.size(); ++k) {
        if ((prediction.cspan()[k] >= 0.5) != (target.cspan()[k] >= 0.5)) {
            all_right = false;
            break;
        }
    }
    if (all_right) ++metrics.correct;

    if (prediction.size() != 1) return;
    const bool predicted_prime = prediction.cspan()[0] >= 0.5;
    const bool is_prime = target.cspan()[0] >= 0.5;
    if (predicted_prime && is_prime) {
        ++metrics.true_positive;
    } else if (!predicted_prime && !is_prime) {
        ++metrics.true_negative;
    } else if (predicted_prime) {
        ++metrics.false_positive;
    } else {
        ++metrics.false_negative;
    }
}

void finish_metrics(Trainer::SplitMetrics& metrics, std::size_t count,
                    bool classification, bool binary_classification) {
    if (count == 0) return;
    const double n = static_cast<double>(count);
    metrics.loss /= n;
    if (!classification) return;

    metrics.accuracy = static_cast<double>(metrics.correct) / n;
    if (!binary_classification) return;

    const auto positive_total = metrics.true_positive + metrics.false_negative;
    const auto negative_total = metrics.true_negative + metrics.false_positive;
    const auto predicted_positive = metrics.true_positive + metrics.false_positive;
    metrics.prime_recall =
        positive_total == 0
            ? 0.0
            : static_cast<double>(metrics.true_positive) /
                  static_cast<double>(positive_total);
    metrics.composite_recall =
        negative_total == 0
            ? 0.0
            : static_cast<double>(metrics.true_negative) /
                  static_cast<double>(negative_total);
    metrics.balanced_accuracy = 0.5 * (metrics.prime_recall + metrics.composite_recall);
    metrics.precision =
        predicted_positive == 0
            ? 0.0
            : static_cast<double>(metrics.true_positive) /
                  static_cast<double>(predicted_positive);
}

void add_classification_metrics(JsonLine& line, const Trainer::SplitMetrics& metrics,
                                bool binary_classification) {
    line.add("accuracy", metrics.accuracy);
    if (!binary_classification) return;
    line.add("balanced_accuracy", metrics.balanced_accuracy)
        .add("prime_recall", metrics.prime_recall)
        .add("composite_recall", metrics.composite_recall)
        .add("precision", metrics.precision)
        .add("true_positive", metrics.true_positive)
        .add("true_negative", metrics.true_negative)
        .add("false_positive", metrics.false_positive)
        .add("false_negative", metrics.false_negative);
}

}  // namespace

Trainer::Trainer(nn::core::Network& net, nn::core::Loss& loss, nn::core::Optimizer& opt,
                 RunConfig cfg)
    : net_(net), loss_(loss), opt_(opt), cfg_(std::move(cfg)) {
    run_id_ = cfg_.run_id.empty()
                  ? timestamp_utc() + "_" + cfg_.name + "_" + short_hex(cfg_.seed)
                  : cfg_.run_id;
}

Trainer::SplitMetrics Trainer::evaluate(const Split& split) {
    SplitMetrics m;
    if (split.count == 0) return m;

    for (std::size_t i = 0; i < split.count; ++i) {
        const Tensor x = column_from(split.input(i));
        const Tensor target = column_from(split.target(i));
        const Tensor y_hat = net_.forward(x);
        m.loss += loss_.value(y_hat, target);

        if (loss_.is_classification()) {
            const Tensor p = loss_.activate(y_hat);
            observe_classification(p, target, m);
        }
    }
    finish_metrics(m, split.count, loss_.is_classification(), split.output_dim == 1);
    return m;
}

std::filesystem::path Trainer::train(const Dataset& data) {
    const std::filesystem::path run_dir = cfg_.runs_dir / run_id_;
    std::filesystem::create_directories(run_dir);

    // One config per phase, plus config.json as the experiment-level entry point.
    {
        JsonLine line;
        line.add("run_id", std::string_view{run_id_})
            .add("name", std::string_view{cfg_.name})
            .add("experiment", std::string_view{cfg_.experiment})
            .add("phase", std::string_view{cfg_.phase})
            .add("model", std::string_view{cfg_.model_description})
            .add("loss_name", std::string_view{cfg_.loss_name})
            .add("optimizer_name", std::string_view{cfg_.optimizer_name})
            .add("seed", static_cast<long long>(cfg_.seed))
            .add("lr", cfg_.lr)
            .add("weight_decay", cfg_.weight_decay)
            .add("steps", cfg_.steps)
            .add("step_offset", cfg_.step_offset)
            .add("total_steps", cfg_.total_experiment_steps > 0
                                    ? cfg_.total_experiment_steps
                                    : cfg_.steps)
            .add("eval_interval", cfg_.eval_interval)
            .add("param_log_interval", cfg_.param_log_interval)
            .add("predict_interval", cfg_.predict_interval)
            .add("flush_every", cfg_.flush_every)
            .add("flush_interval_ms", cfg_.flush_interval_ms)
            .add("positive_class_weight", cfg_.positive_class_weight)
            .add("negative_class_weight", cfg_.negative_class_weight)
            .add("input_dim", data.input_dim())
            .add("output_dim", data.output_dim())
            .add("phase_classification", loss_.is_classification())
            .add("classification",
                 loss_.is_classification() || cfg_.experiment_has_classification);
        // One size_<split> field per split keeps config.json flat and greppable.
        for (const Split& s : data.splits()) {
            line.add("size_" + s.name, s.count);
        }
        const std::string config_json = line.str();
        std::ofstream phase_cfg_file(run_dir /
                                     ("config_" + safe_label(cfg_.phase) + ".json"));
        phase_cfg_file << config_json << '\n';
        if (!cfg_.append_logs) {
            std::ofstream cfg_file(run_dir / "config.json");
            cfg_file << config_json << '\n';
        }
    }
    if (!cfg_.append_logs) {
        std::ofstream meta_file(run_dir / "meta.json");
        meta_file << JsonLine{}
                         .add("run_id", std::string_view{run_id_})
                         .add("start", std::string_view{timestamp_utc()})
                         .add("status", std::string_view{"running"})
                         .str()
                  << '\n';
    }

    JsonlSink metrics(run_dir / "metrics.jsonl", cfg_.flush_every, cfg_.flush_interval_ms,
                      cfg_.append_logs);
    JsonlSink params_sink(run_dir / "params.jsonl", cfg_.flush_every,
                          cfg_.flush_interval_ms, cfg_.append_logs);
    JsonlSink preds(run_dir / "predictions.jsonl", cfg_.flush_every, cfg_.flush_interval_ms,
                    cfg_.append_logs);

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
        const int global_step = cfg_.step_offset + step;
        net_.zero_grad();

        SplitMetrics train_metrics;
        for (std::size_t i = 0; i < train_split.count; ++i) {
            const Tensor x = column_from(train_split.input(i));
            const Tensor target = column_from(train_split.target(i));
            const Tensor y_hat = net_.forward(x);
            train_metrics.loss += loss_.value(y_hat, target);

            if (loss_.is_classification()) {
                const Tensor p = loss_.activate(y_hat);
                observe_classification(p, target, train_metrics);
            }

            // Seed gradient scaled so the objective is the mean over the batch.
            Tensor g = loss_.grad(y_hat, target);
            for (std::size_t k = 0; k < g.size(); ++k) {
                g.span()[k] *= inv_n;
            }
            net_.backward(g);
        }
        finish_metrics(train_metrics, train_split.count, loss_.is_classification(),
                       train_split.output_dim == 1);

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
                .add("step", global_step)
                .add("phase", std::string_view{cfg_.phase})
                .add("split", std::string_view{"train"})
                .add("loss", train_metrics.loss)
                .add("lr", opt_.learning_rate())
                .add("weight_norm", weight_norm)
                .add("grad_norm", grad_norm)
                .add("wall_ms", static_cast<long long>(wall_ms));
            if (loss_.is_classification()) {
                add_classification_metrics(line, train_metrics,
                                           train_split.output_dim == 1);
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
                    .add("step", global_step)
                    .add("phase", std::string_view{cfg_.phase})
                    .add("split", std::string_view{s.name})
                    .add("loss", m.loss);
                if (loss_.is_classification()) {
                    add_classification_metrics(line, m, s.output_dim == 1);
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
                                              .add("step", global_step)
                                              .add("phase", std::string_view{cfg_.phase})
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
        // script serve different sample-based diagnostics.
        if (cfg_.predict_interval > 0 &&
            (step % cfg_.predict_interval == 0 || step == cfg_.steps - 1)) {
            for (const Split* s : predict_targets) {
                for (std::size_t i = 0; i < s->count; ++i) {
                    const Tensor raw = net_.forward(column_from(s->input(i)));
                    const Tensor pred = loss_.activate(raw);
                    preds.write(JsonLine{}
                                    .add("step", global_step)
                                    .add("phase", std::string_view{cfg_.phase})
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

    if (cfg_.finalize_run) {
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
