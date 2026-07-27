# C++ Conventions

This doc is a learning reference as much as a style guide. It captures the core C++ mechanics
we lean on so decisions are explicit before we write classes.

## The special member functions ("the big 6/7")

When you define a class, the compiler may generate up to six *special member functions*. People
often say "7" by counting a normal (parameterised) constructor alongside them.

```cpp
class Widget {
public:
  Widget();                              // 1. default constructor
  Widget(int a, double b);               // (parameterised ctor - the "7th")
  ~Widget();                             // 2. destructor
  Widget(const Widget& other);           // 3. copy constructor
  Widget& operator=(const Widget& rhs);  // 4. copy assignment
  Widget(Widget&& other) noexcept;       // 5. move constructor
  Widget& operator=(Widget&& rhs) noexcept; // 6. move assignment
};
```

### Which rule to follow

- **Rule of Zero (default, prefer this):** if your members are all self-managing types
  (`std::vector`, `std::string`, smart pointers), declare *none* of the six. The compiler-
  generated ones are correct. Most of our classes (`Tensor`, `Network`, layers) aim for this.
  The name is established modern-C++ guidance (popularised in the C++11 era), not a new C++20
  language feature.
- **Rule of Five:** if you manually manage a resource (raw buffer, GPU handle later), you must
  define all five (dtor + both copies + both moves). We expect to hit this only for a future
  GPU buffer type.
- **Rule of Three:** the pre-C++11 subset (dtor + copy ctor + copy assign). We only mention it
  for context; with C++20 we think in terms of Zero/Five.

### Practical guidance we'll apply

- Mark move operations `noexcept` (containers rely on it for efficient reallocation).
- Use `= default` to declare intent explicitly and `= delete` to forbid (e.g. delete copy on a
  type that owns a thread or unique resource).
- Abstract base classes get a `virtual ~Base() = default;` so deleting through a base pointer is
  safe.

## Polymorphism guidelines

We use runtime polymorphism where the set of variants is open and selected at runtime
(different layers, losses, optimisers). Rules:

- Abstract interface = pure virtual methods + virtual destructor.
- Concrete classes use `override` on every overriding method; mark classes/methods `final`
  when no further derivation is intended.
- Prefer composition over deep hierarchies — hierarchies stay shallow (one abstract base, then
  concretes).
- Pass polymorphic objects by reference/pointer (`Layer&`, `std::unique_ptr<Layer>`), never by
  value (avoids slicing).
- Ownership of polymorphic objects: `std::unique_ptr<Base>` in owning containers
  (e.g. `Network` owns `std::vector<std::unique_ptr<Layer>>`).

## Paradigm per layer

| Layer | Paradigm | Rationale |
|-------|----------|-----------|
| `nn::math` | Functional (free, pure functions) | Reusable, testable, parallelisable; no state to reason about |
| `nn::core` | OO + runtime polymorphism | Open-ended families (layers/losses/optimisers), shared behaviour via bases |
| `nn::api` | OO orchestration, thin | Stable surface for experiments; composes core objects |
| `nn::log` | Identity + output utilities | `Loggable` supplies stable IDs; `JsonLine`/`JsonlSink` emit hand-written JSONL |

## General style

- **C++20**, `-Wall -Wextra -Wpedantic`, warnings-as-errors in CI later.
- `const`-correctness everywhere; `[[nodiscard]]` on functions whose result must be used.
- Prefer `std::span` for non-owning array views (this is why the math layer takes spans).
- Naming: `PascalCase` types, `snake_case` functions/variables, `snake_case` files,
  `namespace nn::...`.
- Headers (declarations) in `src/includes/{helpers,classes,api}/`; implementations in the
  mirrored `src/{helpers,classes,api}/`. Header-only only for tiny inline utilities.
- Errors: exceptions for programmer/setup errors; return values/status for expected control
  flow in hot loops.

## Project dependency preference

- Keep the C++ runtime library free of third-party dependencies so NN and systems mechanics stay
  visible.
- Build/test dependency management is allowed when it does not hide runtime behaviour; doctest
  is currently fetched by CMake for tests.
- Discuss any proposed runtime dependency with the user before adding it.
