"""Live Streamlit dashboard for the C++ doctest suite.

Watches src/ and tests/ for .cpp/.hpp changes, rebuilds with CMake, runs
nn_tests with the JUnit reporter, and shows per-case green/red results.

Launch from the repo root (or anywhere):
  .\\.venv\\Scripts\\streamlit.exe run src/app/python/test_dashboard.py
"""

from __future__ import annotations

import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import streamlit as st
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_DIR = REPO_ROOT / "build"
XML_PATH = BUILD_DIR / "test_results.xml"
NN_TESTS = BUILD_DIR / "tests" / "nn_tests.exe"
WATCH_DIRS = (REPO_ROOT / "src", REPO_ROOT / "tests")
WATCH_SUFFIXES = {".cpp", ".hpp", ".h", ".cc", ".hh"}
DEBOUNCE_S = 0.5


@dataclass
class TestCaseResult:
    name: str
    status: str  # "passed" | "failed" | "error" | "skipped"
    message: str = ""
    time_s: float = 0.0


@dataclass
class RunSnapshot:
    timestamp: str = ""
    build_ok: bool = False
    build_log: str = ""
    cases: list[TestCaseResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    running: bool = False
    error: str = ""  # high-level status (e.g. missing build dir)


def parse_junit(xml_path: Path) -> list[TestCaseResult]:
    """Parse doctest JUnit XML into per-case results."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cases: list[TestCaseResult] = []

    # doctest emits <testsuites><testsuite><testcase>...
    for tc in root.iter("testcase"):
        name = tc.get("name") or tc.get("classname") or "unknown"
        time_s = float(tc.get("time") or 0.0)
        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")

        if failure is not None:
            msg = failure.get("message") or (failure.text or "")
            cases.append(TestCaseResult(name, "failed", msg.strip(), time_s))
        elif error is not None:
            msg = error.get("message") or (error.text or "")
            cases.append(TestCaseResult(name, "error", msg.strip(), time_s))
        elif skipped is not None:
            cases.append(TestCaseResult(name, "skipped", "", time_s))
        else:
            cases.append(TestCaseResult(name, "passed", "", time_s))
    return cases


class _DebouncedHandler(FileSystemEventHandler):
    def __init__(self, runner: "TestRunner") -> None:
        super().__init__()
        self._runner = runner
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_any_event(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.is_directory:
            return
        path = Path(getattr(event, "src_path", "") or "")
        if path.suffix.lower() not in WATCH_SUFFIXES:
            return
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_S, self._runner.request_run)
            self._timer.daemon = True
            self._timer.start()


class TestRunner:
    """Singleton that owns the file watcher and the latest run snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = RunSnapshot(error="Waiting for first run…")
        self._run_requested = threading.Event()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="test-runner", daemon=True)
        self._observer = Observer()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        handler = _DebouncedHandler(self)
        for d in WATCH_DIRS:
            if d.is_dir():
                self._observer.schedule(handler, str(d), recursive=True)
        self._observer.start()
        self._worker.start()
        self.request_run()  # initial run on launch

    def stop(self) -> None:
        self._stop.set()
        self._run_requested.set()
        try:
            self._observer.stop()
            self._observer.join(timeout=2)
        except Exception:
            pass

    def request_run(self) -> None:
        self._run_requested.set()

    def get_latest(self) -> RunSnapshot:
        with self._lock:
            return self._latest

    def _set_latest(self, snap: RunSnapshot) -> None:
        with self._lock:
            self._latest = snap

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._run_requested.wait(timeout=1.0)
            if self._stop.is_set():
                break
            if not self._run_requested.is_set():
                continue
            self._run_requested.clear()
            self._execute_once()

    def _execute_once(self) -> None:
        prev = self.get_latest()
        running = RunSnapshot(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            running=True,
            build_ok=prev.build_ok,
            cases=prev.cases,
            total=prev.total,
            passed=prev.passed,
            failed=prev.failed,
            duration_s=prev.duration_s,
            build_log=prev.build_log,
        )
        self._set_latest(running)

        if not BUILD_DIR.is_dir():
            self._set_latest(
                RunSnapshot(
                    timestamp=running.timestamp,
                    error=(
                        f"Build directory missing: {BUILD_DIR}\n"
                        "Configure once with:\n"
                        "  cmake -S . -B build -G Ninja"
                    ),
                )
            )
            return

        t0 = time.perf_counter()
        build = subprocess.run(
            ["cmake", "--build", str(BUILD_DIR)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        build_log = (build.stdout or "") + (build.stderr or "")
        if build.returncode != 0:
            self._set_latest(
                RunSnapshot(
                    timestamp=running.timestamp,
                    build_ok=False,
                    build_log=build_log,
                    duration_s=time.perf_counter() - t0,
                    error="BUILD FAILED",
                )
            )
            return

        if not NN_TESTS.is_file():
            self._set_latest(
                RunSnapshot(
                    timestamp=running.timestamp,
                    build_ok=True,
                    build_log=build_log,
                    duration_s=time.perf_counter() - t0,
                    error=f"Test binary not found: {NN_TESTS}",
                )
            )
            return

        XML_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Remove stale XML so a crash doesn't leave old green results.
        if XML_PATH.exists():
            try:
                XML_PATH.unlink()
            except OSError:
                pass

        test = subprocess.run(
            [
                str(NN_TESTS),
                "--reporters=junit",
                f"--out={XML_PATH}",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        duration = time.perf_counter() - t0

        if not XML_PATH.is_file():
            self._set_latest(
                RunSnapshot(
                    timestamp=running.timestamp,
                    build_ok=True,
                    build_log=build_log
                    + "\n--- test stdout/stderr ---\n"
                    + (test.stdout or "")
                    + (test.stderr or ""),
                    duration_s=duration,
                    error="Tests ran but no JUnit XML was produced.",
                )
            )
            return

        try:
            cases = parse_junit(XML_PATH)
        except ET.ParseError as exc:
            self._set_latest(
                RunSnapshot(
                    timestamp=running.timestamp,
                    build_ok=True,
                    build_log=build_log,
                    duration_s=duration,
                    error=f"Failed to parse JUnit XML: {exc}",
                )
            )
            return

        passed = sum(1 for c in cases if c.status == "passed")
        failed = sum(1 for c in cases if c.status in ("failed", "error"))
        self._set_latest(
            RunSnapshot(
                timestamp=running.timestamp,
                build_ok=True,
                build_log=build_log,
                cases=cases,
                total=len(cases),
                passed=passed,
                failed=failed,
                duration_s=duration,
            )
        )


@st.cache_resource
def get_runner() -> TestRunner:
    runner = TestRunner()
    runner.start()
    return runner


def _status_emoji(status: str) -> str:
    return {
        "passed": "✅",
        "failed": "❌",
        "error": "💥",
        "skipped": "⏭",
    }.get(status, "•")


def render_ui(snap: RunSnapshot, runner: TestRunner) -> None:
    st.title("C++ doctest dashboard")
    st.caption(f"Watching `{REPO_ROOT / 'src'}` and `{REPO_ROOT / 'tests'}` (*.cpp / *.hpp)")

    col_btn, col_ts = st.columns([1, 4])
    with col_btn:
        if st.button("Run now", type="primary", disabled=snap.running):
            runner.request_run()
    with col_ts:
        if snap.running:
            st.info("Running build + tests…")
        elif snap.timestamp:
            st.write(f"Last run: **{snap.timestamp}** ({snap.duration_s:.2f}s)")

    if snap.error and not snap.cases and not snap.build_ok and "BUILD FAILED" not in snap.error:
        st.error(snap.error)
        if snap.build_log:
            with st.expander("Build log"):
                st.code(snap.build_log)
        return

    if not snap.build_ok and snap.error == "BUILD FAILED":
        st.error("BUILD FAILED")
        st.code(snap.build_log or "(no compiler output captured)")
        return

    if snap.error and snap.build_ok and not snap.cases:
        st.warning(snap.error)
        if snap.build_log:
            with st.expander("Log"):
                st.code(snap.build_log)
        return

    if snap.failed == 0 and snap.total > 0:
        st.success(f"ALL PASSING ({snap.total} cases)")
    elif snap.failed > 0:
        st.error(f"{snap.failed} FAILED / {snap.total} total")
    else:
        st.info("No test results yet.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", snap.total)
    m2.metric("Passed", snap.passed)
    m3.metric("Failed", snap.failed)
    m4.metric("Duration (s)", f"{snap.duration_s:.2f}")

    if snap.cases:
        rows = [
            {
                "status": f"{_status_emoji(c.status)} {c.status}",
                "name": c.name,
                "time_s": round(c.time_s, 4),
                "message": c.message[:200] if c.message else "",
            }
            for c in snap.cases
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        failed_cases = [c for c in snap.cases if c.status in ("failed", "error")]
        if failed_cases:
            st.subheader("Failures")
            for c in failed_cases:
                with st.expander(f"{_status_emoji(c.status)} {c.name}", expanded=True):
                    st.code(c.message or "(no message)")

    with st.expander("Last build log"):
        st.code(snap.build_log or "(empty)")


def main() -> None:
    # Must be the first Streamlit call and run exactly once, so keep it out of the
    # auto-rerunning fragment below.
    st.set_page_config(page_title="nn tests", page_icon="🧪", layout="wide")
    runner = get_runner()

    @st.fragment(run_every="1s")
    def live() -> None:
        render_ui(runner.get_latest(), runner)

    live()


if __name__ == "__main__":
    main()
