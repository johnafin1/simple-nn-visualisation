# simple-nn-visualisation

A neural network built **from scratch in C++20** as a learning project, paired with a
Python analysis track for **real-time visualisation** and **video generation** of training.

Everything that matters — forward pass, backpropagation, optimisers, initialisation,
regularisation — is implemented by hand as functions we discuss and build together. No
black-box ML frameworks in the core.

> **Status:** Phase 0 — documentation and scaffolding. No application code yet.

## The flagship experiment: grokking

Train a tiny MLP to regress `f(x) = x^2` on a small sample and keep training until it
**groks** (sudden delayed generalisation, long after it has memorised the training set).
We log richly enough to turn a run into graphs and a video. See
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Design at a glance

- **Language:** C++20 core, Python 3.11 for analysis/video.
- **Layered architecture:** a functional math layer, an object (OO) layer with polymorphism,
  and an API-like utility layer that orchestrates runs. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Logging is first-class:** components implement a `Loggable` abstract base and emit
  structured **JSONL**, queryable with DuckDB/pandas. See [docs/LOGGING.md](docs/LOGGING.md).
- **Parallel experiments:** runs are isolated by `run_id` so we can launch many configs at
  once and compare them.
- **CPU-first:** GPU/threading come later, behind a stable ops API.

## How we work (master & commander, but you steer)

1. **Discuss** a technique in [docs/TECHNIQUES.md](docs/TECHNIQUES.md) — math + where it fits.
2. **Confirm** the object structures / layout via mermaid diagrams before coding.
3. **Implement** it together as named functions/classes.
4. **Visualise** the result before moving on.

## Documentation

| Doc | What it covers |
|-----|----------------|
| [docs/SETUP.md](docs/SETUP.md) | Toolchain, CMake, dependencies, Python env, verify checklist |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layered design, module boundaries, data flow |
| [docs/CPP_CONVENTIONS.md](docs/CPP_CONVENTIONS.md) | Special member functions, polymorphism, per-layer paradigms |
| [docs/LOGGING.md](docs/LOGGING.md) | `Loggable` contract, JSONL schema, run layout, querying, video |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Phased build plan with exit criteria |
| [docs/TECHNIQUES.md](docs/TECHNIQUES.md) | Living catalogue of algorithms we implement |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Grokking `x^2` design + parallel comparison |

## License

MIT — see [LICENSE](LICENSE).
