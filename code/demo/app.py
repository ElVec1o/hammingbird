"""pigeon — interactive benchmark demo.

Run with:
    streamlit run code/demo/app.py

Picks a dataset + methods + k, runs the selected benchmark, shows wall-clock
and recall, appends one structured JSON record to logs/demo/ for later analysis.

Pre-defined combos live in `presets.py`. Order to run them in is in
`RUN_ORDER.md`.

Memory: rough peak estimate is shown before you click Run. The app does
NOT enforce ceilings — you're driving. Hit Ctrl-C in the terminal or close
the browser tab if it gets out of hand.
"""
from __future__ import annotations
import json
import os
import platform
import resource
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import DATASETS, estimate_memory_mb, load_cifar10_phashes  # noqa: E402
from methods import METHODS, list_methods, compute_recall  # noqa: E402
from presets import PRESETS, by_id  # noqa: E402


LOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "logs", "demo")
)
os.makedirs(LOG_DIR, exist_ok=True)


def _machine_fingerprint() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }


def _save_log(preset_id: str | None, config: dict, rows: list[dict]) -> str:
    """Persist a structured JSON record of one Run."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{preset_id or 'custom'}.json"
    path = os.path.join(LOG_DIR, name)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preset_id": preset_id,
        "machine": _machine_fingerprint(),
        "config": config,
        "results": [
            {k: v for k, v in r.items() if k != "_pairs_set"} for r in rows
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


# ---------- page setup --------------------------------------------------------
st.set_page_config(page_title="pigeon — benchmark demo", layout="wide")
st.title("pigeon — interactive benchmark")
st.caption(
    "Compare pigeon variants against FAISS, usearch, and Annoy. Wall-clock + "
    "correctness on real data. You drive `n`, `d`, `k`. Each run is logged to "
    "`logs/demo/*.json` for later review."
)

if "history" not in st.session_state:
    st.session_state.history = []
if "preset_id" not in st.session_state:
    st.session_state.preset_id = None
if "applied_preset_id" not in st.session_state:
    st.session_state.applied_preset_id = None


# ---------- sidebar controls --------------------------------------------------
with st.sidebar:
    st.header("Preset")
    preset_options = ["Custom (manual)"] + [p.label for p in PRESETS]
    preset_choice = st.selectbox(
        "Pre-defined combo", preset_options, index=0,
        help="Pick a question to answer. See RUN_ORDER.md for the recommended order.",
    )

    chosen_preset = None
    if preset_choice != "Custom (manual)":
        chosen_preset = next(p for p in PRESETS if p.label == preset_choice)
        st.caption(f"**Question:** {chosen_preset.question}")
        st.caption(f"**Expected runtime:** {chosen_preset.rough_runtime}")
        with st.expander("How to interpret"):
            st.write(chosen_preset.notes)

    st.markdown("---")
    st.header("Dataset")

    # Apply preset → seed defaults.
    if chosen_preset is not None:
        ds_default = chosen_preset.dataset
        n_default = chosen_preset.n
        d_bits_default = chosen_preset.d_bits
    else:
        ds_default = list(DATASETS.keys())[0]
        n_default = 50_000
        d_bits_default = 256

    dataset_name = st.selectbox(
        "Generator", list(DATASETS.keys()),
        index=list(DATASETS.keys()).index(ds_default),
    )
    is_real_phash = dataset_name.startswith("Real")
    if is_real_phash:
        st.info("CIFAR-10 pHashes: n is fixed at the dataset size, d_bytes=8.")
        n = st.slider("n (rows)", 100, 50_000, min(n_default, 50_000), step=100)
        d_bytes = 8
    else:
        n = st.number_input(
            "n (rows)", min_value=100, max_value=50_000_000,
            value=n_default, step=1000,
            help="No hard ceiling. Watch the memory estimate.",
        )
        d_bits_options = [64, 128, 256, 512, 1024]
        d_bits = st.radio(
            "d (bits)", d_bits_options, horizontal=True,
            index=d_bits_options.index(d_bits_default)
            if d_bits_default in d_bits_options else 2,
        )
        d_bytes = d_bits // 8

    st.markdown("---")
    st.header("Search")
    k_max = max(1, d_bytes * 8 - 1)
    if chosen_preset is not None and chosen_preset.k_values:
        st.caption("This preset sweeps multiple k values, run sequentially.")
        k_values = st.multiselect(
            "k values to run", list(range(0, k_max + 1)),
            default=chosen_preset.k_values,
        )
    else:
        k_single = st.slider("k", 0, k_max, 2)
        k_values = [k_single]

    # n_sweep — when the preset declares one, expose it as a multi-pick so the
    # whole scaling curve gets built in one Run click.
    n_values: list[int] = [int(n)]  # default: single n from above
    if chosen_preset is not None and chosen_preset.n_sweep:
        st.caption(
            "This preset sweeps multiple n values. They will all be run "
            "(slowest first → fastest, so you see results trickle in)."
        )
        n_values = st.multiselect(
            "n values to sweep", chosen_preset.n_sweep,
            default=chosen_preset.n_sweep,
        )

    st.markdown("---")
    st.header("Methods")
    available = list_methods(only_available=True)
    available_names = [m["name"] for m in available]
    if chosen_preset is not None:
        defaults = [n for n in chosen_preset.methods if n in available_names]
    else:
        defaults = [n for n in available_names if "pigeon" in n.lower() or "Flat" in n]
    selected_names = st.multiselect("Methods to run", available_names, default=defaults)

    st.markdown("---")
    st.header("Run")

    mem_mb = estimate_memory_mb(int(n), int(d_bytes), max(k_values) if k_values else 0, selected_names)
    color = "🟢" if mem_mb < 1024 else "🟡" if mem_mb < 4096 else "🟠" if mem_mb < 12000 else "🔴"
    st.metric("Estimated peak memory", f"{color} ~{mem_mb:.0f} MB")
    if mem_mb > 4000:
        st.warning(
            "Peak estimate over 4 GB — close other apps, save your work, "
            "and keep the terminal visible so you can Ctrl-C if needed."
        )

    run = st.button("▶ Run benchmark", type="primary", use_container_width=True)
    clear = st.button("🗑 Clear session history", use_container_width=True)
    if clear:
        st.session_state.history = []
        st.rerun()


# ---------- main body --------------------------------------------------------
left, right = st.columns([1, 1])
with left:
    st.subheader("Current configuration")
    st.json(
        {
            "preset": chosen_preset.id if chosen_preset else "custom",
            "dataset": dataset_name,
            "n": int(n),
            "d_bits": int(d_bytes * 8),
            "k_values": k_values,
            "methods": selected_names,
        }
    )
with right:
    st.subheader("Library availability")
    st.dataframe(
        pd.DataFrame(
            [
                {"name": m["name"], "category": m["category"], "available": "✅" if m["available"] else "❌"}
                for m in METHODS
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )


def _build_data(name: str, n_: int, d_bytes_: int) -> np.ndarray:
    gen = DATASETS[name]
    if name.startswith("Real"):
        data = load_cifar10_phashes(n_)
        if data is None:
            st.error(
                "CIFAR-10 pHashes not found. Generate them with "
                "`python3 code/experiments/gen_real_phashes.py` first."
            )
            st.stop()
        return data
    return gen(n_, d_bytes_)


def _run_one_config(
    A: np.ndarray,
    k_: int,
    selected_methods: list[dict],
    status_panel,                # st.status() instance, updated live
    completed_lines: list[str],  # mutable list of completion log lines
    config_idx: int,
    config_total: int,
    method_idx_offset: int,
    method_total: int,
    run_start_time: float,
) -> tuple[list[dict], dict | None]:
    """Run all selected methods for a single (n, k) config, updating
    `status_panel` and `completed_lines` between methods so the user has
    real-time visibility into what's running, what's done, and elapsed time."""
    rows = []
    truth_priority = [
        "FAISS IndexBinaryFlat (brute force, exact)",
        "pigeon — find_pairs_self (byte-aligned)",
        "pigeon — find_pairs_self_bit (bit-aligned)",
    ]
    truth_method = None
    for name in truth_priority:
        for m in selected_methods:
            if m["name"] == name:
                truth_method = m
                break
        if truth_method:
            break

    truth_pairs = None
    for i, m in enumerate(selected_methods):
        global_idx = method_idx_offset + i + 1
        elapsed_total = time.perf_counter() - run_start_time
        # Live status — what's running RIGHT NOW.
        status_panel.update(
            label=(
                f"⏳ [{global_idx}/{method_total}] **{m['name']}** "
                f"@ n={A.shape[0]}, d_bits={A.shape[1]*8}, k={k_}  "
                f"(elapsed {elapsed_total:.1f}s)"
            ),
            state="running",
            expanded=True,
        )
        # Re-render the completion log so the user sees the growing list.
        status_panel.empty()
        for line in completed_lines:
            status_panel.write(line)
        status_panel.write(f"⏳ **{m['name']}** — running...")

        t0 = time.perf_counter()
        # NOTE: BaseException, not Exception. PyO3 PanicException inherits
        # from BaseException (so it propagates more aggressively); if we
        # only catch Exception, a Rust panic in the wheel will hang the UI
        # instead of surfacing as a ❌ row. v0.4.3 regression where d=512
        # bit-path panicked and the demo spinner spun forever.
        try:
            out = m["run"](A, int(k_))
            elapsed_method = time.perf_counter() - t0
            row = {
                "method": m["name"],
                "category": m["category"],
                "time_s": round(out["time"], 6),
                "pairs": len(out["pairs"]),
                "n": int(A.shape[0]),
                "d_bits": int(A.shape[1] * 8),
                "k": int(k_),
                "_pairs_set": out["pairs"],
                "extra": out.get("extra", {}),
            }
            rows.append(row)
            if truth_method is not None and m["name"] == truth_method["name"]:
                truth_pairs = out["pairs"]
            line = (
                f"✅ [{global_idx}/{method_total}] **{m['name']}** "
                f"— {out['time']*1000:.1f} ms  ({len(out['pairs'])} pairs)"
            )
        except BaseException as e:
            err_msg = f"{type(e).__name__}: {str(e)[:200]}"
            rows.append({
                "method": m["name"], "category": m["category"],
                "time_s": None, "pairs": None,
                "n": int(A.shape[0]), "d_bits": int(A.shape[1] * 8), "k": int(k_),
                "_pairs_set": set(),
                "extra": {"error": err_msg},
            })
            line = f"❌ [{global_idx}/{method_total}] **{m['name']}** — error: {err_msg[:80]}"
            # Don't re-raise. KeyboardInterrupt and SystemExit will still
            # propagate via Streamlit's own machinery if the user hits Stop.
        completed_lines.append(line)

    for r in rows:
        if truth_pairs is not None and r["_pairs_set"] is not None:
            recall = compute_recall(r["_pairs_set"], truth_pairs)
            r["recall_vs_truth"] = round(recall, 4) if recall is not None else None
        else:
            r["recall_vs_truth"] = None

    completed_lines.append(
        f"━━ config {config_idx}/{config_total} done "
        f"(n={A.shape[0]}, k={k_}) ━━"
    )
    return rows, truth_pairs


# ---------- run ---------------------------------------------------------------
if run:
    if not selected_names:
        st.error("Pick at least one method."); st.stop()
    if not k_values:
        st.error("Pick at least one k value."); st.stop()
    if not n_values:
        st.error("Pick at least one n value."); st.stop()

    selected_methods = [m for m in METHODS if m["name"] in selected_names]

    # Build the full (n, k) config list. For multi-n runs we iterate
    # SMALLEST n FIRST so the user sees results trickle in quickly,
    # rather than waiting on the slow one upfront.
    sorted_ns = sorted(n_values)
    configs = [(int(nn), int(kk)) for nn in sorted_ns for kk in k_values]
    method_total = len(configs) * len(selected_methods)

    run_start = time.perf_counter()
    completed_lines: list[str] = []
    status_panel = st.status(
        f"▶ Starting benchmark — {method_total} method-runs across "
        f"{len(configs)} config(s)...",
        expanded=True, state="running",
    )

    all_rows: list[dict] = []
    method_idx_offset = 0
    for ci, (n_, k_) in enumerate(configs, start=1):
        # Regenerate data only when n changes (cheap when same).
        gen_t0 = time.perf_counter()
        completed_lines.append(
            f"🔧 generating dataset for config {ci}/{len(configs)} "
            f"(n={n_}, k={k_})..."
        )
        status_panel.empty()
        for line in completed_lines:
            status_panel.write(line)
        A = np.ascontiguousarray(_build_data(dataset_name, n_, int(d_bytes)))
        completed_lines[-1] = (
            f"✓ dataset ready for config {ci}/{len(configs)} "
            f"(n={n_}, k={k_}, gen took {time.perf_counter()-gen_t0:.1f}s)"
        )

        rows, _ = _run_one_config(
            A, k_, selected_methods,
            status_panel=status_panel,
            completed_lines=completed_lines,
            config_idx=ci,
            config_total=len(configs),
            method_idx_offset=method_idx_offset,
            method_total=method_total,
            run_start_time=run_start,
        )
        method_idx_offset += len(selected_methods)
        all_rows.extend(rows)

    total_elapsed = time.perf_counter() - run_start

    # Persist session history (no pair sets).
    for r in all_rows:
        r_for_session = {kk: vv for kk, vv in r.items() if kk != "_pairs_set"}
        r_for_session["dataset"] = dataset_name
        st.session_state.history.append(r_for_session)

    # Write JSON log file.
    config = {
        "dataset": dataset_name,
        "n_values": [int(x) for x in sorted_ns],
        "d_bits": int(d_bytes * 8),
        "k_values": list(map(int, k_values)),
        "methods": selected_names,
        "estimated_memory_mb": round(mem_mb, 1),
        "total_elapsed_s": round(total_elapsed, 2),
    }
    log_path = _save_log(
        preset_id=(chosen_preset.id if chosen_preset is not None else None),
        config=config,
        rows=all_rows,
    )

    # Final state: collapse the live panel, show big banner + toast.
    status_panel.update(
        label=f"✅ Complete — {method_total} runs in {total_elapsed:.1f}s",
        state="complete",
        expanded=False,
    )
    n_ok = sum(1 for r in all_rows if r.get("time_s") is not None)
    n_err = len(all_rows) - n_ok
    st.balloons()
    st.toast(f"✅ Benchmark complete — {n_ok}/{len(all_rows)} methods, {total_elapsed:.1f}s", icon="✅")
    msg = (
        f"### ✅ Run complete\n"
        f"**{n_ok}/{len(all_rows)}** method-runs OK"
        + (f", **{n_err}** errored" if n_err else "")
        + f"  •  total wall-clock: **{total_elapsed:.1f}s**"
        f"  •  log file: `{os.path.relpath(log_path)}`\n\n"
        f"Results below ↓"
    )
    st.success(msg)


# ---------- results -----------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("Results — current session")
    df = pd.DataFrame(st.session_state.history)
    cols = ["dataset", "n", "d_bits", "k", "method", "category",
            "time_s", "pairs", "recall_vs_truth", "extra"]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

    # Bar chart of last config's wall-clock.
    last_config = df.tail(20)
    if last_config["time_s"].notna().any():
        st.subheader("Wall-clock (latest runs)")
        chart_df = (
            last_config.dropna(subset=["time_s"])
            .assign(label=lambda d: d["method"].str.slice(0, 40) + " (k=" + d["k"].astype(str) + ")")
            .set_index("label")[["time_s"]]
        )
        st.bar_chart(chart_df)

    # Scaling curve when ≥2 n values exist for the same method/d/k.
    grouped = df.dropna(subset=["time_s"]).groupby(["method", "d_bits", "k"])
    scaling = grouped.filter(lambda g: g["n"].nunique() >= 2)
    if not scaling.empty:
        st.subheader("Scaling curves")
        for (method, d_bits_, k_), g in scaling.groupby(["method", "d_bits", "k"]):
            g = g.sort_values("n")
            st.caption(f"{method} • d_bits={d_bits_} • k={k_}")
            st.line_chart(g[["n", "time_s"]].set_index("n"), height=160)

    st.download_button(
        "⬇ Download session results as CSV",
        df[cols].to_csv(index=False),
        file_name="pigeon_demo_session.csv",
        mime="text/csv",
    )
else:
    st.info(
        "No runs yet. Pick a **Preset** in the sidebar (or use Custom) and hit "
        "**Run benchmark**. See `code/demo/RUN_ORDER.md` for the recommended sequence."
    )

st.markdown("---")
st.caption(
    f"hammingbird 0.4.3 • Logs land in `logs/demo/`. "
    f"Recommended run order: `code/demo/RUN_ORDER.md`."
)
