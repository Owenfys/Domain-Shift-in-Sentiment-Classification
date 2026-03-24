"""
run_classical.py
----------------
Main experiment runner for classical models (Naive Bayes + Logistic Regression).
Runs the three experiments from the project proposal:
    1. In-domain:    Train on IMDB → Test on IMDB
    2. Cross-domain: Train on IMDB → Test on Financial PhraseBank
    3. Fine-tuned:   Train on Financial PhraseBank → Test on Financial PhraseBank

Usage
-----
    python run_classical.py                   # run all 3 experiments
    python run_classical.py --experiment 1    # run only experiment 1
"""

import argparse
import os
import json

from data_loader import load_imdb, load_financial_phrasebank
from features import fit_transform_tfidf, transform_tfidf
from models import train_all_classical
from evaluate import (
    compute_metrics,
    full_classification_report,
    plot_confusion_matrix,
    build_comparison_table,
    plot_performance_drop,
    plot_finetune_improvement,
)

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_results(results: dict, filename: str):
    """Persist results dict as JSON."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[run] Saved results → {path}")


# ── Experiment 1: In-Domain (IMDB → IMDB) ────────────────────────────────────

def experiment_1():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: In-Domain — Train on IMDB, Test on IMDB")
    print("=" * 70)

    imdb = load_imdb()
    X_train, X_test, vectorizer = fit_transform_tfidf(imdb["X_train"], imdb["X_test"])
    models = train_all_classical(X_train, imdb["y_train"])

    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        metrics = compute_metrics(imdb["y_test"], preds, label=f"Exp1 | {name} | IMDB→IMDB")
        full_classification_report(imdb["y_test"], preds, label=name)
        plot_confusion_matrix(
            imdb["y_test"], preds,
            title=f"Exp1 — {name} (IMDB→IMDB)",
            save_path=os.path.join(OUTPUT_DIR, f"exp1_cm_{name}.png"),
        )
        results[name] = metrics

    table = build_comparison_table(results)
    print("\n[Exp1] Comparison Table — In-Domain (IMDB→IMDB)")
    print(table.to_string())
    table.to_csv(os.path.join(OUTPUT_DIR, "exp1_comparison.csv"))
    save_results(results, "exp1_results.json")

    return results, vectorizer, models


# ── Experiment 2: Cross-Domain (IMDB → Financial PhraseBank) ─────────────────

def experiment_2(vectorizer=None, models=None):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Cross-Domain — Train on IMDB, Test on Financial PhraseBank")
    print("=" * 70)

    # If models/vectorizer not passed, retrain on IMDB
    if vectorizer is None or models is None:
        imdb = load_imdb()
        X_train_imdb, _, vectorizer = fit_transform_tfidf(imdb["X_train"], imdb["X_test"])
        models = train_all_classical(X_train_imdb, imdb["y_train"])

    fpb = load_financial_phrasebank()

    # Transform financial texts using IMDB-fitted vectorizer
    X_test_fpb = transform_tfidf(fpb["X_test"], vectorizer)
    # Also transform the full FPB set for a more comprehensive evaluation
    X_all_fpb = transform_tfidf(fpb["X_train"] + fpb["X_test"], vectorizer)
    y_all_fpb = fpb["y_train"] + fpb["y_test"]

    results = {}
    for name, model in models.items():
        preds = model.predict(X_all_fpb)
        metrics = compute_metrics(y_all_fpb, preds, label=f"Exp2 | {name} | IMDB→FPB")
        full_classification_report(y_all_fpb, preds, label=name)
        plot_confusion_matrix(
            y_all_fpb, preds,
            title=f"Exp2 — {name} (IMDB→Financial)",
            save_path=os.path.join(OUTPUT_DIR, f"exp2_cm_{name}.png"),
        )
        results[name] = metrics

    table = build_comparison_table(results)
    print("\n[Exp2] Comparison Table — Cross-Domain (IMDB→Financial)")
    print(table.to_string())
    table.to_csv(os.path.join(OUTPUT_DIR, "exp2_comparison.csv"))
    save_results(results, "exp2_results.json")

    return results


# ── Experiment 3: Fine-Tuned (Financial PhraseBank → Financial PhraseBank) ───

def experiment_3():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Fine-Tuned — Train on FPB, Test on FPB")
    print("=" * 70)

    fpb = load_financial_phrasebank()
    X_train, X_test, vectorizer = fit_transform_tfidf(fpb["X_train"], fpb["X_test"])
    models = train_all_classical(X_train, fpb["y_train"])

    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        metrics = compute_metrics(fpb["y_test"], preds, label=f"Exp3 | {name} | FPB→FPB")
        full_classification_report(fpb["y_test"], preds, label=name)
        plot_confusion_matrix(
            fpb["y_test"], preds,
            title=f"Exp3 — {name} (FPB→FPB)",
            save_path=os.path.join(OUTPUT_DIR, f"exp3_cm_{name}.png"),
        )
        results[name] = metrics

    table = build_comparison_table(results)
    print("\n[Exp3] Comparison Table — Fine-Tuned (FPB→FPB)")
    print(table.to_string())
    table.to_csv(os.path.join(OUTPUT_DIR, "exp3_comparison.csv"))
    save_results(results, "exp3_results.json")

    return results


# ── Summary visualizations ────────────────────────────────────────────────────

def generate_summary_plots(exp1_results, exp2_results, exp3_results):
    """Generate cross-experiment comparison plots."""
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY VISUALIZATIONS")
    print("=" * 70)

    # Performance drop: in-domain vs cross-domain
    plot_performance_drop(
        exp1_results, exp2_results,
        metric="f1",
        title="F1 Drop: In-Domain (IMDB) vs Cross-Domain (Financial)",
        save_path=os.path.join(OUTPUT_DIR, "summary_performance_drop.png"),
    )

    # Fine-tuning improvement
    plot_finetune_improvement(
        exp2_results, exp3_results,
        metric="f1",
        title="F1 Improvement: Cross-Domain vs Fine-Tuned on Financial",
        save_path=os.path.join(OUTPUT_DIR, "summary_finetune_improvement.png"),
    )

    # Combined comparison table
    combined = {}
    for name in exp1_results:
        combined[name] = {
            "IMDB→IMDB (F1)": exp1_results[name]["f1"],
            "IMDB→FPB (F1)": exp2_results.get(name, {}).get("f1", None),
            "FPB→FPB (F1)": exp3_results.get(name, {}).get("f1", None),
        }
        # Calculate drop
        if combined[name]["IMDB→FPB (F1)"] is not None:
            combined[name]["Domain Shift Drop"] = round(
                combined[name]["IMDB→IMDB (F1)"] - combined[name]["IMDB→FPB (F1)"], 4
            )

    import pandas as pd
    summary_df = pd.DataFrame(combined).T.round(4)
    summary_df.index.name = "model"
    print("\n[Summary] Combined Results Across All Experiments")
    print(summary_df.to_string())
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_all_experiments.csv"))
    print(f"\n[run] All outputs saved to '{OUTPUT_DIR}/' directory.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run classical NLP experiments")
    parser.add_argument(
        "--experiment", type=int, choices=[1, 2, 3], default=None,
        help="Run a specific experiment (1, 2, or 3). Default: run all.",
    )
    args = parser.parse_args()

    if args.experiment == 1:
        experiment_1()
    elif args.experiment == 2:
        experiment_2()
    elif args.experiment == 3:
        experiment_3()
    else:
        # Run all three + summary
        exp1_results, vectorizer, models = experiment_1()
        exp2_results = experiment_2(vectorizer=vectorizer, models=models)
        exp3_results = experiment_3()
        generate_summary_plots(exp1_results, exp2_results, exp3_results)


if __name__ == "__main__":
    main()
