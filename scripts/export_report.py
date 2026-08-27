#!/usr/bin/env python3
"""
Generates visual comparison charts (Pareto frontiers), CSV metrics summaries,
and markdown results tables from benchmark outputs.
"""

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def load_benchmark_results(json_path: str = "results/benchmark_results.json") -> list[dict[str, Any]]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Benchmark results file not found at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_csv_report(results: list[dict[str, Any]], csv_path: str = "results/benchmark_results.csv") -> pd.DataFrame:
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[ExportReport] CSV benchmark metrics exported to: {csv_path}")
    return df


def generate_markdown_table(results: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("| Model Variant | ROUGE-L | Size (MB) | Peak Memory (MB) | TTFT (ms) | Tokens / sec | Compression Ratio |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in results:
        v_name = r.get("variant_name", "Unknown")
        rouge_l = r.get("rougeL", 0.0)
        size_mb = r.get("disk_size_mb", 0.0)
        mem_mb = r.get("peak_memory_mb", 0.0)
        ttft = r.get("ttft_ms", 0.0)
        tps = r.get("tokens_per_sec", 0.0)
        ratio = r.get("compression_ratio", 1.0)
        lines.append(f"| **{v_name}** | {rouge_l:.2f}% | {size_mb:.1f} MB | {mem_mb:.1f} MB | {ttft:.1f} ms | {tps:.1f} tok/s | **{ratio:.2f}x** |")

    table_md = "\n".join(lines)
    print("\nGenerated Markdown Results Table:\n")
    print(table_md)
    return table_md


def generate_pareto_chart(results: list[dict[str, Any]], output_img_path: str = "results/comparison_chart.png"):
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    df = pd.DataFrame(results)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    # --- Plot 1: Accuracy (ROUGE-L) vs Model Size (MB) ---
    ax1.set_title("EdgeTune: ROUGE-L Quality vs. Model Size (MB)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Model Footprint on Disk (MB)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("ROUGE-L Score (%)", fontsize=11, fontweight="bold")

    for i, row in df.iterrows():
        color = colors[i % len(colors)]
        ax1.scatter(row["disk_size_mb"], row["rougeL"], color=color, s=140, alpha=0.9, edgecolors="black", zorder=4)
        ax1.annotate(
            row["variant_name"],
            (row["disk_size_mb"], row["rougeL"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="semibold",
        )

    # --- Plot 2: Throughput (Tokens/sec) vs Peak Memory (MB) ---
    ax2.set_title("EdgeTune: Generation Speed vs. Peak Memory Footprint", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Peak Memory Allocation (MB)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Inference Throughput (Tokens / sec)", fontsize=11, fontweight="bold")

    for i, row in df.iterrows():
        color = colors[i % len(colors)]
        ax2.scatter(row["peak_memory_mb"], row["tokens_per_sec"], color=color, s=140, alpha=0.9, edgecolors="black", zorder=4)
        ax2.annotate(
            row["variant_name"],
            (row["peak_memory_mb"], row["tokens_per_sec"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="semibold",
        )

    plt.tight_layout()
    fig.savefig(output_img_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ExportReport] High-res Pareto tradeoff chart saved to: {output_img_path}")


def main():
    json_path = "results/benchmark_results.json"
    if not os.path.exists(json_path):
        print(f"No results found at {json_path}. Run benchmark sweep first.")
        return

    results = load_benchmark_results(json_path)
    generate_csv_report(results)
    generate_markdown_table(results)
    generate_pareto_chart(results)


if __name__ == "__main__":
    main()
