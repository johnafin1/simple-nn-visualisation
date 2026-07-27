# Setup

Target platform for now: **Windows + MinGW (MSYS2)**. The source remains portable because the
C++ library does not depend on Windows-specific APIs.

## Verified toolchain

As of 2026-07-26:

- `g++` 13.1.0 from MSYS2 MinGW-w64.
- CMake 3.26.4.
- Ninja 1.11.1.
- Python 3.11.3.

Ensure `C:\msys64\mingw64\bin` is on `PATH` so `g++`, `cmake`, and `ninja` resolve.

If installing a new environment through MSYS2:

```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja
```

## C++ dependency policy

The `nn` runtime library intentionally has **no third-party C++ dependencies**. Neural-network
math, backpropagation, optimisers, datasets, and JSON/JSONL logging are implemented in this
repository for learning and transparency.

Library/dependency management is still acceptable where it supports the project without hiding
NN behaviour:

- CMake uses `FetchContent` to obtain doctest for the test executable.
- Standard-library facilities are used throughout the runtime.
- A future optional backend may introduce a dependency only after explicit design discussion.

There is no `vcpkg.json` and no nlohmann/json dependency. `JsonLine` and `JsonlSink` provide the
small JSON surface the project currently needs.

## Configure, build, and test

```powershell
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=g++
cmake --build build
ctest --test-dir build --output-on-failure
```

The generated `build/` directory is gitignored.

## Python environment

Create an isolated environment and install the analysis dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Runtime analysis packages include DuckDB, pandas, NumPy, matplotlib, imageio, and ffmpeg support.
The development requirements add Streamlit and watchdog for the test dashboard.

Both `.venv/` and generated run/analysis artefacts are gitignored.

## Verify checklist

```powershell
g++ --version
cmake --version
ninja --version
python --version
cmake --build build
ctest --test-dir build --output-on-failure
```

## Not assumed

- No CUDA/GPU toolchain.
- No black-box C++ ML framework.
- No third-party C++ JSON library.

GPU or other compute backends remain optional future implementations behind the math boundary;
see [ARCHITECTURE.md](ARCHITECTURE.md) and [STATUS.md](STATUS.md).
