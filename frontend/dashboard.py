"""
Streamlit Interactive Dashboard for EdgeTune Model Compression Tradeoffs.
"""

import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="EdgeTune - PEFT + Quantization & Pruning Pipeline",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ EdgeTune: LLM PEFT, Quantization & Pruning Tradeoff Dashboard")
st.markdown(
    "**End-to-end compression pipeline evaluating ROUGE accuracy, disk size, peak memory, and TTFT latency across model variants.**"
)

# Load benchmark results
RESULTS_FILE = "results/benchmark_results.json"


@st.cache_data
def load_data():
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)


results = load_data()

if results is None:
    st.warning("⚠️ Benchmark results not found. Run `python scripts/run_full_benchmark_sweep.py` first to generate benchmark metrics.")
else:
    df = pd.DataFrame(results)

    # Top KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    baseline_size = df[df["variant_name"].str.contains("Baseline")]["disk_size_mb"].values[0] if len(df[df["variant_name"].str.contains("Baseline")]) > 0 else df["disk_size_mb"].max()
    max_comp = df["compression_ratio"].max()
    max_tps = df["tokens_per_sec"].max()
    best_rouge = df["rougeL"].max()

    col1.metric("Max Compression Ratio", f"{max_comp:.2f}x", delta="Vs Baseline FP16")
    col2.metric("Peak Generation Speed", f"{max_tps:.1f} tok/s")
    col3.metric("Best ROUGE-L Score", f"{best_rouge:.2f}%")
    col4.metric("Baseline Model Size", f"{baseline_size:.1f} MB")

    st.divider()

    # Main Visual Charts
    tab1, tab2, tab3 = st.tabs(["📊 Pareto Tradeoff Frontiers", "📋 Detailed Benchmark Table", "💬 Live Model Testing"])

    with tab1:
        st.subheader("Pareto Tradeoff Frontiers Across Compression Variants")
        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            for i, row in df.iterrows():
                ax1.scatter(row["disk_size_mb"], row["rougeL"], color=colors[i % len(colors)], s=120, label=row["variant_name"])
                ax1.annotate(row["variant_name"], (row["disk_size_mb"], row["rougeL"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
            ax1.set_xlabel("Disk Size (MB)")
            ax1.set_ylabel("ROUGE-L Score (%)")
            ax1.set_title("Accuracy vs. Disk Footprint")
            st.pyplot(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            for i, row in df.iterrows():
                ax2.scatter(row["peak_memory_mb"], row["tokens_per_sec"], color=colors[i % len(colors)], s=120, label=row["variant_name"])
                ax2.annotate(row["variant_name"], (row["peak_memory_mb"], row["tokens_per_sec"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
            ax2.set_xlabel("Peak Memory (MB)")
            ax2.set_ylabel("Throughput (Tokens/sec)")
            ax2.set_title("Generation Speed vs. Memory Footprint")
            st.pyplot(fig2)

    with tab2:
        st.subheader("Complete Model Variant Benchmark Comparison Table")
        st.dataframe(
            df[["variant_name", "rouge1", "rouge2", "rougeL", "disk_size_mb", "peak_memory_mb", "ttft_ms", "tokens_per_sec", "compression_ratio"]],
            use_container_width=True,
        )

    with tab3:
        st.subheader("Interactive Generation Endpoint Test")
        prompt_input = st.text_area("Input Prompt for Dialogue Summarization:", "Summarize dialogue:\nJohn: Can we reschedule our meeting to 3 PM?\nSarah: Sure, 3 PM works for me.\n\nSummary:")
        if st.button("Generate Summary"):
            st.info("Querying FastAPI serving endpoint at http://localhost:8000/generate ...")
            try:
                import requests
                resp = requests.post("http://localhost:8000/generate", json={"prompt": prompt_input, "max_new_tokens": 48})
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"**Generated Text:** {data['generated_text']}")
                    st.json(data)
                else:
                    st.error(f"API Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.warning(f"Could not connect to FastAPI endpoint: {e}. Start server via `python api/main.py`.")
