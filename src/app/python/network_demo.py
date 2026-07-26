"""Streamlit visualiser for one forward+backward pass of the C++ network.

Reads the snapshot emitted by the `demo` executable
(`runs/demo/snapshot.jsonl`) and draws the 1 -> 4 (tanh) -> 1 network: nodes
coloured by activation, edges coloured/annotated by weight (or gradient), plus the
underlying linear-algebra tables (W, b, z = Wx + b, a = tanh(z), and their grads).

Launch from the repo root:
  .\\.venv\\Scripts\\streamlit.exe run src/app/python/network_demo.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_DIR = REPO_ROOT / "build"
DEMO_EXE = BUILD_DIR / "src" / "app" / "demo" / "demo.exe"
SNAPSHOT = REPO_ROOT / "runs" / "demo" / "snapshot.jsonl"


def load_snapshot(path: Path) -> list[dict]:
    """Parse the JSONL snapshot into a list of records."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def regenerate() -> tuple[bool, str]:
    """Rebuild the project and re-run the demo to refresh the snapshot."""
    build = subprocess.run(
        ["cmake", "--build", str(BUILD_DIR)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    log = (build.stdout or "") + (build.stderr or "")
    if build.returncode != 0:
        return False, "BUILD FAILED\n" + log
    if not DEMO_EXE.is_file():
        return False, f"demo executable not found: {DEMO_EXE}\n" + log
    run = subprocess.run(
        [str(DEMO_EXE)], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    return run.returncode == 0, log + (run.stdout or "") + (run.stderr or "")


def split_records(records: list[dict]) -> dict:
    """Group records by type into convenient structures."""
    nodes: dict[tuple[str, int], dict] = {}
    edges: list[dict] = []
    biases: dict[tuple[str, int], dict] = {}
    loss: dict = {}
    for r in records:
        kind = r.get("type")
        if kind == "node":
            nodes[(r["layer"], int(r["idx"]))] = r
        elif kind == "edge":
            edges.append(r)
        elif kind == "bias":
            biases[(r["layer"], int(r["idx"]))] = r
        elif kind == "loss":
            loss = r
    return {"nodes": nodes, "edges": edges, "biases": biases, "loss": loss}


def _hidden_indices(nodes: dict) -> list[int]:
    return sorted(idx for (layer, idx) in nodes if layer == "dense.0.act")


def draw_network(data: dict, mode: str) -> plt.Figure:
    """Draw the network graph. mode is 'weight' or 'grad' for edge colouring."""
    nodes = data["nodes"]
    edges = data["edges"]
    hidden = _hidden_indices(nodes)
    n_hidden = len(hidden)

    # Node positions: input (x=0), hidden (x=1), output (x=2).
    def y_positions(n: int) -> list[float]:
        if n == 1:
            return [0.0]
        top = (n - 1) / 2.0
        return [top - i for i in range(n)]

    pos: dict[tuple[str, int], tuple[float, float]] = {}
    pos[("input", 0)] = (0.0, 0.0)
    for r, y in zip(hidden, y_positions(n_hidden)):
        pos[("hidden", r)] = (1.0, y)
    pos[("output", 0)] = (2.0, 0.0)

    fig, ax = plt.subplots(figsize=(7, 5))

    edge_vals = [e[mode] for e in edges] or [0.0]
    vmax = max(abs(v) for v in edge_vals) or 1.0
    cmap = plt.get_cmap("coolwarm")

    def edge_endpoints(e: dict) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if e["layer"] == "dense.0":  # input[col] -> hidden[row]
            a = pos.get(("input", int(e["col"])))
            b = pos.get(("hidden", int(e["row"])))
        elif e["layer"] == "dense.1":  # hidden[col] -> output[row]
            a = pos.get(("hidden", int(e["col"])))
            b = pos.get(("output", int(e["row"])))
        else:
            return None
        if a is None or b is None:
            return None
        return a, b

    for e in edges:
        ends = edge_endpoints(e)
        if ends is None:
            continue
        (x0, y0), (x1, y1) = ends
        val = e[mode]
        colour = cmap(0.5 + 0.5 * (val / vmax))
        ax.plot([x0, x1], [y0, y1], color=colour, linewidth=0.5 + 3.5 * abs(val) / vmax,
                zorder=1)

    # Nodes coloured by activation value.
    def node_records() -> list[tuple[tuple[float, float], float, float, str]]:
        out = []
        n_in = nodes.get(("input", 0))
        if n_in:
            out.append((pos[("input", 0)], n_in["value"], n_in["grad"], "x"))
        for r in hidden:
            nrec = nodes.get(("dense.0.act", r))
            if nrec:
                out.append((pos[("hidden", r)], nrec["value"], nrec["grad"], f"h{r}"))
        n_out = nodes.get(("output", 0))
        if n_out:
            out.append((pos[("output", 0)], n_out["value"], n_out["grad"], "y"))
        return out

    recs = node_records()
    node_vals = [v for (_, v, _, _) in recs] or [0.0]
    nvmax = max(abs(v) for v in node_vals) or 1.0
    node_cmap = plt.get_cmap("viridis")
    for (x, y), value, grad, label in recs:
        ax.scatter([x], [y], s=1400, c=[node_cmap(0.5 + 0.5 * value / nvmax)],
                   edgecolors="black", zorder=2)
        ax.text(x, y, f"{label}\n{value:.2f}", ha="center", va="center",
                fontsize=8, color="white", zorder=3)

    ax.set_title(f"1 -> {n_hidden} (tanh) -> 1   |   edges coloured by {mode}")
    ax.set_xlim(-0.5, 2.5)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def weight_matrix(edges: list[dict], layer: str, field: str) -> pd.DataFrame:
    rows = [e for e in edges if e["layer"] == layer]
    if not rows:
        return pd.DataFrame()
    n_rows = max(int(e["row"]) for e in rows) + 1
    n_cols = max(int(e["col"]) for e in rows) + 1
    m = np.zeros((n_rows, n_cols))
    for e in rows:
        m[int(e["row"]), int(e["col"])] = e[field]
    return pd.DataFrame(
        m,
        index=[f"out{r}" for r in range(n_rows)],
        columns=[f"in{c}" for c in range(n_cols)],
    )


def main() -> None:
    st.set_page_config(page_title="nn network demo", page_icon="🧠", layout="wide")
    st.title("Network forward / backward visualiser")
    st.caption(
        "One pass through a 1 -> 4 (tanh) -> 1 net. Source of truth is the C++ "
        "snapshot at `runs/demo/snapshot.jsonl`."
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("Rebuild + regenerate", type="primary"):
            with st.spinner("Building and running demo…"):
                ok, log = regenerate()
            if ok:
                st.success("Snapshot regenerated.")
            else:
                st.error("Failed to regenerate snapshot.")
            with st.expander("Log"):
                st.code(log or "(empty)")

    if not SNAPSHOT.is_file():
        st.warning(
            f"No snapshot found at `{SNAPSHOT}`.\n\n"
            "Build the project and run the demo, or click "
            "**Rebuild + regenerate** above."
        )
        return

    data = split_records(load_snapshot(SNAPSHOT))
    loss = data["loss"]

    if loss:
        m1, m2, m3 = st.columns(3)
        out_node = data["nodes"].get(("output", 0), {})
        m1.metric("y_hat", f"{out_node.get('value', float('nan')):.4f}")
        m2.metric("target", f"{loss.get('target', float('nan')):.4f}")
        m3.metric("MSE loss", f"{loss.get('value', float('nan')):.4f}")

    mode = st.radio("Edge colouring", ["weight", "grad"], horizontal=True)
    st.pyplot(draw_network(data, mode))

    st.subheader("The linear algebra")
    st.caption("Node values come from z = Wx + b then a = tanh(z); grads are dL/d(param).")

    for layer in ("dense.0", "dense.1"):
        st.markdown(f"**{layer}**")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("weights W")
            st.dataframe(weight_matrix(data["edges"], layer, "weight"))
        with c2:
            st.caption("weight grads dL/dW")
            st.dataframe(weight_matrix(data["edges"], layer, "grad"))

        biases = sorted(
            ((idx, rec) for (lyr, idx), rec in data["biases"].items() if lyr == layer),
            key=lambda t: t[0],
        )
        if biases:
            bias_df = pd.DataFrame(
                {
                    "bias": [rec["value"] for _, rec in biases],
                    "grad": [rec["grad"] for _, rec in biases],
                },
                index=[f"out{idx}" for idx, _ in biases],
            )
            st.caption("bias b and dL/db")
            st.dataframe(bias_df)


if __name__ == "__main__":
    main()
