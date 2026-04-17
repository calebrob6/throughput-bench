#!/usr/bin/env python3
"""Generate publication-quality charts from GeoSpeedy benchmark results.

Usage:
    python visualize.py                           # uses results/benchmark_results.csv
    python visualize.py --input results/my.csv    # custom CSV
    python visualize.py --output figures/          # custom output dir
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns

from models import FAMILY_COLORS, MODEL_REGISTRY

sns.set_theme(style="whitegrid", font_scale=1.1)


def load_results(path: str) -> pd.DataFrame:
    """Load and clean benchmark CSV."""
    df = pd.read_csv(path)
    # Drop OOM rows for plotting
    df = df[df["throughput_mean"] != "OOM"].copy()
    for col in ["throughput_mean", "throughput_std", "throughput_median",
                 "pixels_per_sec", "params_M", "macs_G", "peak_memory_mb",
                 "latency_mean_ms", "batch_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def bubble_chart(df: pd.DataFrame, output_dir: Path,
                 task: str = "classification", precision: str = "amp",
                 batch_size: int = 32, compiled: bool = False):
    """Hero chart: MACs vs throughput, bubbles sized by params, colored by family."""
    mask = (
        (df["task"] == task)
        & (df["precision"] == precision)
        & (df["batch_size"] == batch_size)
        & (df["compiled"].astype(str).str.lower() == str(compiled).lower())
        & (df["device"] == "cuda")
    )
    sub = df[mask].copy()
    if sub.empty:
        print(f"  ⚠ No data for {task}/{precision}/bs{batch_size}/compiled={compiled}")
        return

    sub = sub.drop_duplicates(subset=["model_name"], keep="last")
    sub = sub[sub["macs_G"] > 0]

    fig, ax = plt.subplots(figsize=(14, 9))

    # Draw lines connecting models within the same family
    families = sub["model_family"].unique()
    for fam in families:
        fam_data = sub[sub["model_family"] == fam].sort_values("macs_G")
        if len(fam_data) > 1:
            ax.plot(
                fam_data["macs_G"], fam_data["throughput_mean"],
                color=FAMILY_COLORS.get(fam, "#999999"),
                linewidth=1.5, alpha=0.5, zorder=1,
            )

    # Scatter bubbles
    for _, row in sub.iterrows():
        color = FAMILY_COLORS.get(row["model_family"], "#999999")
        size = max(row["params_M"] * 3, 30)
        ax.scatter(
            row["macs_G"], row["throughput_mean"],
            s=size, c=color, alpha=0.8, edgecolors="white",
            linewidth=0.8, zorder=2,
        )
        ax.annotate(
            row["display_name"],
            (row["macs_G"], row["throughput_mean"]),
            textcoords="offset points", xytext=(6, 6),
            fontsize=7, alpha=0.85,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MACs (GFLOPs)", fontsize=13)
    ax.set_ylabel("Throughput (images/sec)", fontsize=13)
    compile_label = " + torch.compile" if compiled else ""
    ax.set_title(
        f"Inference Throughput vs Compute Cost — "
        f"{task.title()} | {precision.upper()}{compile_label} | "
        f"batch={batch_size}",
        fontsize=14, fontweight="bold",
    )

    # Legend for families
    handles = []
    for fam in sorted(FAMILY_COLORS.keys()):
        if fam in sub["model_family"].values:
            handles.append(
                mlines.Line2D([], [], color=FAMILY_COLORS[fam],
                              marker="o", linestyle="None", markersize=8,
                              label=fam)
            )
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              title="Model Family", title_fontsize=10)

    plt.tight_layout()
    fname = f"bubble_{task}_{precision}_bs{batch_size}"
    if compiled:
        fname += "_compiled"
    fig.savefig(output_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / f"{fname}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


def speedup_chart(df: pd.DataFrame, output_dir: Path,
                  task: str = "classification", batch_size: int = 32):
    """Bar chart: throughput across precision modes for each model."""
    mask = (
        (df["task"] == task)
        & (df["batch_size"] == batch_size)
        & (df["compiled"].astype(str).str.lower() == "false")
        & (df["device"] == "cuda")
    )
    sub = df[mask].copy()
    if sub.empty:
        return

    # Pivot: model × precision → throughput
    pivot = sub.pivot_table(
        index="display_name", columns="precision",
        values="throughput_mean", aggfunc="first",
    )
    # Sort by fp32 throughput
    if "fp32" in pivot.columns:
        pivot = pivot.sort_values("fp32", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.4)))
    pivot.plot.barh(ax=ax, width=0.75)
    ax.set_xlabel("Throughput (images/sec)", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(
        f"Precision Speedup — {task.title()} | batch={batch_size}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(title="Precision", fontsize=9)
    plt.tight_layout()

    fname = f"speedup_{task}_bs{batch_size}"
    fig.savefig(output_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


def compile_chart(df: pd.DataFrame, output_dir: Path,
                  task: str = "classification", batch_size: int = 32,
                  precision: str = "amp"):
    """Bar chart: compiled vs not compiled for each model."""
    mask = (
        (df["task"] == task)
        & (df["batch_size"] == batch_size)
        & (df["precision"] == precision)
        & (df["device"] == "cuda")
    )
    sub = df[mask].copy()
    if sub.empty:
        return

    sub["compiled_label"] = sub["compiled"].astype(str).str.lower().map(
        {"true": "compiled", "false": "eager"}
    )
    pivot = sub.pivot_table(
        index="display_name", columns="compiled_label",
        values="throughput_mean", aggfunc="first",
    )
    if "eager" in pivot.columns:
        pivot = pivot.sort_values("eager", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.4)))
    pivot.plot.barh(ax=ax, width=0.75, color=["#4C72B0", "#DD8452"])
    ax.set_xlabel("Throughput (images/sec)", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(
        f"torch.compile Effect — {task.title()} | {precision.upper()} | "
        f"batch={batch_size}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(title="Mode", fontsize=9)
    plt.tight_layout()

    fname = f"compile_{task}_{precision}_bs{batch_size}"
    fig.savefig(output_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


def batch_scaling_chart(df: pd.DataFrame, output_dir: Path,
                        task: str = "classification", precision: str = "amp"):
    """Line chart: throughput vs batch size for each model."""
    mask = (
        (df["task"] == task)
        & (df["precision"] == precision)
        & (df["compiled"].astype(str).str.lower() == "false")
        & (df["device"] == "cuda")
    )
    sub = df[mask].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    for name in sub["display_name"].unique():
        m = sub[sub["display_name"] == name].sort_values("batch_size")
        color = FAMILY_COLORS.get(m.iloc[0]["model_family"], "#999999")
        ax.plot(m["batch_size"], m["throughput_mean"], marker="o",
                label=name, color=color, linewidth=1.5, markersize=5)

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Throughput (images/sec)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(
        f"Batch Size Scaling — {task.title()} | {precision.upper()}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    plt.tight_layout()

    fname = f"batch_scaling_{task}_{precision}"
    fig.savefig(output_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


def cnn_vs_vit_summary(df: pd.DataFrame, output_dir: Path,
                       batch_size: int = 32, precision: str = "amp"):
    """Side-by-side comparison of CNN vs ViT throughput."""
    mask = (
        (df["batch_size"] == batch_size)
        & (df["precision"] == precision)
        & (df["compiled"].astype(str).str.lower() == "false")
        & (df["device"] == "cuda")
    )
    sub = df[mask].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i, task in enumerate(["classification", "segmentation"]):
        ax = axes[i]
        tsub = sub[sub["task"] == task].copy()
        if tsub.empty:
            ax.set_title(f"{task.title()} — No data")
            continue

        tsub = tsub.sort_values("throughput_mean", ascending=True)
        colors = [FAMILY_COLORS.get(f, "#999999") for f in tsub["model_family"]]
        ax.barh(tsub["display_name"], tsub["throughput_mean"], color=colors)
        ax.set_xlabel("Throughput (images/sec)", fontsize=11)
        ax.set_title(f"{task.title()}", fontsize=13, fontweight="bold")

    fig.suptitle(
        f"CNN vs Transformer Throughput — {precision.upper()} | "
        f"batch={batch_size}",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()

    fname = f"cnn_vs_vit_{precision}_bs{batch_size}"
    fig.savefig(output_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


def main():
    parser = argparse.ArgumentParser(description="GeoSpeedy visualization")
    parser.add_argument("--input", default="results/benchmark_results.csv")
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Loading results from {args.input}")
    df = load_results(args.input)
    print(f"   {len(df)} rows loaded")
    print()

    # Determine available batch sizes and precisions
    batch_sizes = sorted(df["batch_size"].dropna().unique().astype(int))
    best_bs = 32 if 32 in batch_sizes else batch_sizes[-1]

    print("📈 Generating bubble charts...")
    for task in ["classification", "segmentation"]:
        for prec in ["fp32", "amp"]:
            for compiled in [False, True]:
                bubble_chart(df, output_dir, task=task, precision=prec,
                             batch_size=best_bs, compiled=compiled)

    print("\n📊 Generating speedup charts...")
    for task in ["classification", "segmentation"]:
        speedup_chart(df, output_dir, task=task, batch_size=best_bs)

    print("\n📊 Generating compile charts...")
    for task in ["classification", "segmentation"]:
        compile_chart(df, output_dir, task=task, batch_size=best_bs)

    print("\n📊 Generating batch scaling charts...")
    for task in ["classification", "segmentation"]:
        batch_scaling_chart(df, output_dir, task=task, precision="amp")

    print("\n📊 Generating CNN vs ViT summary...")
    cnn_vs_vit_summary(df, output_dir, batch_size=best_bs)

    print(f"\n✅ All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
