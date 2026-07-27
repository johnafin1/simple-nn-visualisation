# simple-nn-visualisation

A neural network built **from scratch in C++20** as a learning project, paired with Python tools
for live inspection and visualisation.

Forward passes, backpropagation, losses, optimisers, initialisation, regularisation, dataset
generation, and runtime JSON logging are implemented by hand. The C++ runtime library avoids
third-party dependencies so the neural-network and systems mechanics remain visible; dependency
management is reserved for build/test tooling such as doctest.

> **Status:** Phases 0-5 are complete. Grokking-oriented experiments are active, but no
> successful grok has been observed yet. See [docs/STATUS.md](docs/STATUS.md).

## The experiment focus: grokking

The flagship experiment trains a tiny MLP on `f(x) = x^2` and watches for delayed
generalisation. A second experiment tested primality with bit and one-hot encodings; both
memorised and failed to generalise. Negative results are retained as learning outcomes rather
than rewritten as successes.

The current follow-up pretrains a wider shared encoder on unary modulo-residue targets for the
small prime divisors needed through 500, freezes that encoder, and trains a class-balanced prime
head. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the command, design, and evidence.
Two-input modular addition remains a future textbook-grok comparison.

## Design at a glance

- **Language:** C++20 core; Python 3.11 for analysis and visualisation.
- **Layered architecture:** functional math, OO domain objects, and thin API orchestration.
- **Logging:** hand-written `JsonLine` plus buffered JSONL sinks; no third-party C++ JSON library.
- **Current execution:** `Trainer` runs one isolated experiment at a time.
- **Next phase:** design and implement parallel experiment orchestration.
- **CPU-first:** threading/GPU remain optional future backends.

## How we work

1. Check [docs/STATUS.md](docs/STATUS.md) and the known
   [rectifications](docs/RECTIFICATIONS.md).
2. Discuss a technique in [docs/TECHNIQUES.md](docs/TECHNIQUES.md).
3. Confirm object structures and file layout before coding.
4. Implement and verify the agreed functions/classes.
5. Visualise the result and update status/evidence.

## Documentation

| Doc | What it covers |
|-----|----------------|
| [docs/STATUS.md](docs/STATUS.md) | Canonical current phase, implemented capabilities, and next work |
| [docs/RECTIFICATIONS.md](docs/RECTIFICATIONS.md) | Known gaps broken into tasks and acceptance criteria |
| [docs/SETUP.md](docs/SETUP.md) | Toolchain, dependency policy, Python environment, and verification |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layered design, module boundaries, data flow, and source layout |
| [docs/CPP_CONVENTIONS.md](docs/CPP_CONVENTIONS.md) | Resource management, polymorphism, and per-layer conventions |
| [docs/LOGGING.md](docs/LOGGING.md) | Implemented logging contract, schemas, limitations, and querying |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Phase order and exit criteria |
| [docs/TECHNIQUES.md](docs/TECHNIQUES.md) | Living catalogue of hand-written algorithms |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Experiment designs, commands, and observed results |

## License

MIT — see [LICENSE](LICENSE).
