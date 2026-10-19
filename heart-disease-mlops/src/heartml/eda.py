"""Exploratory data analysis: saves publication-quality figures + a JSON summary."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heartml import config  # noqa: E402

PALETTE = {0: "#4C78A8", 1: "#E45756"}
LABELS = {0: "No disease", 1: "Disease"}

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(df, out_dir):
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v} ({v / counts.sum():.0%})",
                ha="center", fontsize=10)
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, out_dir, "class_balance.png")


def plot_missing(raw, out_dir):
    miss = raw.isna().sum()
    miss = miss[miss > 0].sort_values()
    fig, ax = plt.subplots(figsize=(5, 3))
    if miss.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    else:
        ax.barh(miss.index, miss.values, color="#72B7B2")
        for i, v in enumerate(miss.values):
            ax.text(v + 0.1, i, str(v), va="center")
        ax.set_xlabel("Missing values")
    ax.set_title("Missing values in raw data")
    return _save(fig, out_dir, "missing_values.png")


def plot_numeric_histograms(df, out_dir):
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), cols):
        for t in (0, 1):
            sns.histplot(df.loc[df[config.TARGET] == t, col].dropna(), ax=ax,
                         color=PALETTE[t], label=LABELS[t], kde=True,
                         stat="density", alpha=0.45, bins=18)
        ax.set_title(config.FEATURE_DESCRIPTIONS[col], fontsize=10)
        ax.set_xlabel(col)
    axes[0, 0].legend()
    fig.suptitle("Distribution of numeric features by diagnosis", fontsize=13)
    return _save(fig, out_dir, "numeric_histograms.png")


def plot_categorical(df, out_dir):
    cols = config.CATEGORICAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.ravel(), cols):
        rate = df.groupby(col)[config.TARGET].agg(["mean", "size"]).reset_index()
        ax.bar(rate[col].astype(int).astype(str), rate["mean"], color="#E45756", alpha=0.8)
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(col, fontsize=11)
        ax.set_ylabel("Disease rate")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Heart-disease rate per category", fontsize=13)
    return _save(fig, out_dir, "categorical_disease_rate.png")


def plot_correlation(df, out_dir):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = corr.where(~pd.DataFrame(
        [[i < j for j in range(len(corr))] for i in range(len(corr))],
        index=corr.index, columns=corr.columns)).isna()
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.7},
                annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation heatmap (Pearson)")
    return _save(fig, out_dir, "correlation_heatmap.png")


def plot_target_correlation(df, out_dir):
    corr = df.corr(numeric_only=True)[config.TARGET].drop(config.TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(corr.index, corr.values, color=["#4C78A8" if v < 0 else "#E45756" for v in corr])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Correlation of each feature with the target")
    return _save(fig, out_dir, "target_correlation.png")


def plot_boxplots(df, out_dir):
    long = df.melt(id_vars=config.TARGET, value_vars=config.NUMERIC_FEATURES[:5])
    long[config.TARGET] = long[config.TARGET].map(LABELS)
    g = sns.catplot(data=long, x=config.TARGET, y="value", col="variable", kind="box",
                    sharey=False, height=3.4, aspect=0.75,
                    palette=[PALETTE[0], PALETTE[1]], hue=config.TARGET, legend=False)
    g.set_titles("{col_name}")
    g.set_xlabels("")
    g.figure.suptitle("Numeric features by diagnosis (outlier view)", y=1.04)
    path = out_dir / "boxplots.png"
    g.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)
    return path


def summarize(raw, df) -> dict:
    return {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(df)),
        "n_features": len(config.FEATURES),
        "missing_raw": {k: int(v) for k, v in raw.isna().sum().items() if v},
        "missing_after_cleaning": {k: int(v) for k, v in df.isna().sum().items() if v},
        "class_counts": {LABELS[int(k)]: int(v)
                         for k, v in df[config.TARGET].value_counts().sort_index().items()},
        "numeric_summary": df[config.NUMERIC_FEATURES].describe().round(2).to_dict(),
        "top_target_correlations": df.corr(numeric_only=True)[config.TARGET]
        .drop(config.TARGET).abs().sort_values(ascending=False).head(5).round(3).to_dict(),
    }


def run_eda(raw: pd.DataFrame, df: pd.DataFrame, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (config.FIGURES_DIR / "eda")
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        plot_class_balance(df, out_dir),
        plot_missing(raw, out_dir),
        plot_numeric_histograms(df, out_dir),
        plot_categorical(df, out_dir),
        plot_correlation(df, out_dir),
        plot_target_correlation(df, out_dir),
        plot_boxplots(df, out_dir),
    ]
    summary = summarize(raw, df)
    summary["figures"] = [str(p.relative_to(config.PROJECT_ROOT)) if p.is_relative_to(
        config.PROJECT_ROOT) else str(p) for p in figures]
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
