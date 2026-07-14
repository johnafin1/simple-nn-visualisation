# Phase 0 Plan — Toolchain + Buildable Skeleton

**Goal:** turn the greenfield repo into a project that compiles and runs a trivial C++20 target
via CMake, with the Python analysis environment ready. No neural-network code yet.

**Exit criteria**
- `SETUP.md` verify checklist passes (`g++`, `cmake`, `ninja`, `python`).
- `cmake` configures and builds a trivial target that runs and prints its C++ standard.
- Python venv exists with `requirements.txt` installed.
- Folder scaffold in place; `runs/` gitignored.

## Decisions locked for this phase

- Build: **CMake + Ninja**, MinGW g++ toolchain, `-std=c++20`, warnings `-Wall -Wextra -Wpedantic`.
- **No third-party C++ deps yet.** nlohmann/json + vcpkg are introduced in Phase 3 (logging),
  not now — Phase 0 stays dependency-free so the build is trivial to get green.
- Tests are wired in Phase 1; Phase 0 only proves the toolchain compiles/links/runs.

## Steps

### 1. Install toolchain

```bash
pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja
```

Confirm `C:\msys64\mingw64\bin` is on PATH, then verify:

```powershell
g++ --version; cmake --version; ninja --version; python --version
```

### 2. Folder scaffold

```
src/
  includes/{helpers,classes,api}/  # declarations (.hpp), populated from Phase 1
  {helpers,classes,api}/           # implementations (.cpp), populated from Phase 1
  app/
    smoke/                         # trivial target to prove the build
    python/                        # populated in Phase 4
tests/                             # populated in Phase 1
runs/                              # generated logs (gitignored)
```

### 3. Top-level `CMakeLists.txt`

- `cmake_minimum_required(VERSION 3.20)`, `project(simple_nn_visualisation CXX)`.
- `set(CMAKE_CXX_STANDARD 20)`, `CMAKE_CXX_STANDARD_REQUIRED ON`, `CMAKE_CXX_EXTENSIONS OFF`.
- Global warning flags via an interface target (e.g. `nn_warnings`).
- `nn_headers` interface target adding `src/includes` to the include path.
- `enable_testing()` (used from Phase 1).
- `add_subdirectory(src/app)`.

### 4. Smoke target — `src/app/smoke/main.cpp`

Minimal program that prints `__cplusplus` and a hello line. Purpose: prove configure → build →
run works end to end with C++20 before we write real code.

### 5. `.gitignore` additions

Confirm/append: `build/`, `runs/`, `.venv/`, `vcpkg_installed/`, CMake artefacts (most already
present).

### 6. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib duckdb imageio imageio-ffmpeg
pip freeze > requirements.txt
```

### 7. Verify (the phase gate)

```powershell
cmake -S . -B build -G Ninja
cmake --build build
.\build\src\app\smoke\smoke.exe    # prints C++ standard + hello
```

## What I need you to confirm before executing

1. **Build layout** above (folder names, `src/app/smoke` as the proof target).
2. **Dependency-free Phase 0** (defer vcpkg/json to Phase 3) — agree?
3. Whether to `git commit` at the end of Phase 0 (I won't commit unless you say so).
