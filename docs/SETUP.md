# Setup

Target platform for now: **Windows + MinGW (MSYS2)**, which is what's installed. Everything is
chosen to stay portable (Linux/macOS later) since nothing here is Windows-specific.

## What's already present

- `g++` 13.1.0 at `C:\msys64\mingw64\bin\g++.exe` (MSYS2 MinGW-w64). Supports C++20.
- Python 3.11.3.

## What we need to add

### 1. CMake

Not currently on `PATH`. The repo's `.gitignore` already assumes a CMake + vcpkg workflow.

Install via MSYS2 (keeps it in the same toolchain as g++):

```bash
pacman -S mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja
```

Or via winget:

```powershell
winget install Kitware.CMake
```

Ensure `C:\msys64\mingw64\bin` is on `PATH` so `g++`, `cmake`, and `ninja` resolve.

### 2. C++ standard and compiler

- **Standard: C++20** (`-std=c++20`). We use `std::span`, designated initialisers, and
  possibly concepts.
- Generator: **Ninja** with the MinGW g++ toolchain.

### 3. JSON dependency

Logging emits JSON, so we need a JSON library. Decision: **nlohmann/json** (header-only,
ergonomic). Managed via **vcpkg** (manifest mode, `vcpkg.json`) to match the `.gitignore`.

We will add the actual `vcpkg.json` when we start Phase 3 (logging). For now this is just the
recorded decision.

### 4. Python environment (analysis + video)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib duckdb imageio imageio-ffmpeg
```

- `duckdb` — SQL over our JSONL logs, no import step.
- `pandas` / `numpy` — ad-hoc analysis.
- `matplotlib` — plots and live figures.
- `imageio` + `imageio-ffmpeg` — stitch frames into `.mp4`.

A `requirements.txt` will accompany the first Python script.

## Verify checklist

```powershell
g++ --version        # expect 13.x, C++20 capable
cmake --version      # after install
ninja --version      # after install
python --version     # expect 3.11.x
```

## Not assumed

- **No CUDA / GPU.** `nvidia-smi` and `nvcc` are not present. GPU is a later, optional backend
  behind the ops API (see [ARCHITECTURE.md](ARCHITECTURE.md)); we will not depend on it.
