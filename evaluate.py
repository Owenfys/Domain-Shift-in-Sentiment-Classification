"""
evaluate.py
-----------
Evaluation utilities: metrics, confusion matrices, comparison tables, and plots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, label: str = "") -> dict:
    """
    Compute accuracy, precision, recall, F1 and return as a dict.
    Also prints a short summary.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if label:
        print(f"\n[evaluate] {label}")
    for k, v in metrics.items():
        print(f"  {k:>10s}: {v:.4f}")
    return metrics


def full_classification_report(y_true, y_pred, label: str = "") -> str:
    """Print and return sklearn's full classification report."""
    report = classification_report(y_true, y_pred, target_names=["negative", "positive"])
    if label:
        print(f"\n[evaluate] Classification report — {label}")
    print(report)
    return report


# ── Confusion matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
    figsize: tuple = (6, 5),
):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Saved confusion matrix → {save_path}")
    plt.close(fig)
    return fig


# ── Comparison table ─────────────────────────────────────────────────────────

def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a nested dict of results.

    Parameters
    ----------
    results : dict
        {model_name: {metric_name: value, ...}, ...}

    Returns
    -------
    pd.DataFrame with models as rows, metrics as columns, sorted by F1 desc.
    """
    df = pd.DataFrame(results).T
    df.index.name = "model"
    df = df.sort_values("f1", ascending=False)
    return df.round(4)


# ── Bar-plot: cross-domain performance drop ──────────────────────────────────

def plot_performance_drop(
    in_domain: dict[str, dict],
    cross_domain: dict[str, dict],
    metric: str = "f1",
    title: str = "Cross-Domain Performance Drop",
    save_path: str | None = None,
):
    """
    Bar chart comparing in-domain vs cross-domain performance for each model.

    Parameters
    ----------
    in_domain    : {model_name: {metric: value, ...}}
    cross_domain : same structure
    metric       : which metric to compare (default: f1)
    """
    models = sorted(set(in_domain) & set(cross_domain))
    in_vals = [in_domain[m][metric] for m in models]
    cross_vals = [cross_domain[m][metric] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, in_vals, width, label="In-Domain (IMDB)", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, cross_vals, width, label="Cross-Domain (Financial)", color="#DD8452")

    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars1, fmt="%.2f", fontsize=8)
    ax.bar_label(bars2, fmt="%.2f", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Saved performance drop plot → {save_path}")
    plt.close(fig)
    return fig


# ── Fine-tuning improvement comparison ────────────────────────────────────────

def plot_finetune_improvement(
    cross_domain: dict[str, dict],
    finetuned: dict[str, dict],
    metric: str = "f1",
    title: str = "Fine-Tuning Improvement on Financial Data",
    save_path: str | None = None,
):
    """
    Bar chart: cross-domain performance vs fine-tuned on financial data.
    """
    models = sorted(set(cross_domain) & set(finetuned))
    cross_vals = [cross_domain[m][metric] for m in models]
    ft_vals = [finetuned[m][metric] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, cross_vals, width, label="Cross-Domain (no fine-tune)", color="#DD8452")
    bars2 = ax.bar(x + width / 2, ft_vals, width, label="Fine-Tuned on Financial", color="#55A868")

    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars1, fmt="%.2f", fontsize=8)
    ax.bar_label(bars2, fmt="%.2f", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Saved fine-tune improvement plot → {save_path}")
    plt.close(fig)
    return fig


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Convert unified records to a flat metrics DataFrame.
    """
    rows = []
    for r in records:
        m = r.get("metrics", {})
        rows.append(
            {
                "experiment": r.get("experiment"),
                "domain_setup": r.get("domain_setup"),
                "model": r.get("model"),
                "seed": r.get("seed"),
                "accuracy": m.get("accuracy"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "macro_f1": m.get("macro_f1"),
            }
        )
    return pd.DataFrame(rows)


def summarize_records_mean_std(records: list[dict]) -> pd.DataFrame:
    """
    Summarize multi-seed records as mean±std by experiment/domain/model.
    """
    df = records_to_dataframe(records)
    if df.empty:
        return df
    metric_cols = ["accuracy", "precision", "recall", "f1", "macro_f1"]
    grouped = df.groupby(["experiment", "domain_setup", "model"], as_index=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in grouped.columns
    ]
    return grouped.round(4)
