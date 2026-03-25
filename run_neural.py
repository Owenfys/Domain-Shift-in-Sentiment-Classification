"""
run_neural.py
-------------
Run neural experiments (LSTM + GloVe, BERT fine-tuning) with
deterministic splits, unified schema, and multi-seed reporting.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any
from sklearn.metrics import f1_score

from data_loader import load_financial_phrasebank_with_val, load_imdb_with_val
from evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    records_to_dataframe,
    summarize_records_mean_std,
)
from neural_features import (
    build_bert_datasets,
    build_lstm_dataloaders,
    load_glove_embeddings,
)
from neural_models import (
    predict_bert,
    predict_lstm,
    train_bert_with_search,
    train_lstm,
)
from result_schema import make_result_record, save_result_records


OUTPUT_DIR = "outputs/neural"
os.makedirs(OUTPUT_DIR, exist_ok=True)


DEFAULT_SEEDS = [42, 52, 62]
DEFAULT_BERT_SEARCH = {
    "learning_rate": [2e-5, 3e-5],
    "batch_size": [8, 16],
    "num_train_epochs": [2, 3],
}


def _save_json(path: str, payload: dict | list) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _run_lstm(
    experiment: str,
    domain_setup: str,
    seed: int,
    split_train: dict,
    split_test: dict,
    glove_path: str,
) -> dict[str, Any]:
    lstm_data = build_lstm_dataloaders(
        X_train=split_train["X_train"],
        y_train=split_train["y_train"],
        X_val=split_train["X_val"],
        y_val=split_train["y_val"],
        X_test=split_test["X_test"],
        y_test=split_test["y_test"],
        batch_size=64,
    )
    emb_matrix, glove_stats = load_glove_embeddings(
        glove_path=glove_path,
        vocab=lstm_data["vocab"],
        embedding_dim=100,
        seed=seed,
    )
    model, train_info = train_lstm(
        train_loader=lstm_data["train_loader"],
        val_loader=lstm_data["val_loader"],
        vocab_size=len(lstm_data["vocab"]),
        embedding_dim=100,
        hidden_dim=128,
        num_layers=1,
        bidirectional=True,
        dropout=0.3,
        lr=1e-3,
        epochs=5,
        embedding_matrix=emb_matrix,
        finetune_embeddings=True,
        seed=seed,
    )
    y_true, y_pred, y_prob = predict_lstm(model, lstm_data["test_loader"])
    metrics = compute_metrics(y_true, y_pred, label=f"{experiment} | LSTM_GloVe | seed={seed}")
    metrics["macro_f1"] = float(metrics["f1"])  # binary macro_f1 aligns here
    record = make_result_record(
        experiment=experiment,
        model="LSTM_GloVe",
        seed=seed,
        domain_setup=domain_setup,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        metrics=metrics,
        train_config={
            "embedding_dim": 100,
            "hidden_dim": 128,
            "epochs": 5,
            "lr": 1e-3,
            "max_len": lstm_data["max_len"],
            "oov_rate_val": lstm_data["oov_rate_val"],
            "oov_rate_test": lstm_data["oov_rate_test"],
            "glove": glove_stats,
        },
        notes="LSTM with train-only vocabulary and optional GloVe initialization.",
    )
    record["train_info"] = train_info
    return record


def _run_bert(
    experiment: str,
    domain_setup: str,
    seed: int,
    split_train: dict,
    split_test: dict,
    best_hparams: dict | None,
    do_search: bool,
) -> tuple[dict[str, Any] | None, dict | None]:
    datasets = build_bert_datasets(
        X_train=split_train["X_train"],
        y_train=split_train["y_train"],
        X_val=split_train["X_val"],
        y_val=split_train["y_val"],
        X_test=split_test["X_test"],
        y_test=split_test["y_test"],
        tokenizer_name="bert-base-uncased",
        max_len=256,
    )
    if do_search:
        trainer, search_info = train_bert_with_search(
            model_name="bert-base-uncased",
            train_dataset=datasets["train_dataset"],
            val_dataset=datasets["val_dataset"],
            search_space=DEFAULT_BERT_SEARCH,
            seed=seed,
            output_dir=os.path.join(OUTPUT_DIR, f"bert_search_{experiment}_seed{seed}"),
        )
        best_hparams = search_info["best_hparams"]
    else:
        # If no pre-selected hparams were passed (e.g., running a single experiment),
        # use a stable default config without launching a full search.
        if best_hparams is None:
            best_hparams = {
                "learning_rate": 3e-5,
                "batch_size": 16,
                "num_train_epochs": 3,
            }
        # Reuse selected hparams; single run with fixed config.
        trainer, search_info = train_bert_with_search(
            model_name="bert-base-uncased",
            train_dataset=datasets["train_dataset"],
            val_dataset=datasets["val_dataset"],
            search_space={
                "learning_rate": [best_hparams["learning_rate"]],
                "batch_size": [best_hparams["batch_size"]],
                "num_train_epochs": [best_hparams["num_train_epochs"]],
            },
            seed=seed,
            output_dir=os.path.join(OUTPUT_DIR, f"bert_fixed_{experiment}_seed{seed}"),
        )
    y_true, y_pred, y_prob = predict_bert(trainer, datasets["test_dataset"])
    metrics = compute_metrics(y_true, y_pred, label=f"{experiment} | BERT_base | seed={seed}")
    # Compute explicit macro_f1 for cross-domain main metric.
    
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    record = make_result_record(
        experiment=experiment,
        model="BERT_base",
        seed=seed,
        domain_setup=domain_setup,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        metrics=metrics,
        best_hparams=best_hparams,
        train_config={"tokenizer": "bert-base-uncased", "max_len": 256},
        notes="Validation-only hyperparameter selection; test evaluated once.",
    )
    return record, best_hparams


def _save_experiment_outputs(experiment: str, records: list[dict]) -> None:
    exp_dir = os.path.join(OUTPUT_DIR, experiment.lower())
    os.makedirs(exp_dir, exist_ok=True)
    save_result_records(records, os.path.join(exp_dir, "records.json"))
    raw_df = records_to_dataframe(records)
    raw_df.to_csv(os.path.join(exp_dir, "raw_metrics_by_seed.csv"), index=False)
    summary_df = summarize_records_mean_std(records)
    summary_df.to_csv(os.path.join(exp_dir, "summary_mean_std.csv"), index=False)
    # Per-experiment and per-model confusion matrix from first seed for quick inspection.
    for model_name in sorted({r["model"] for r in records}):
        pick = next(r for r in records if r["model"] == model_name)
        plot_confusion_matrix(
            pick["y_true"],
            pick["y_pred"],
            title=f"{experiment} — {model_name} (seed={pick['seed']})",
            save_path=os.path.join(exp_dir, f"cm_{model_name}.png"),
        )


def experiment_1(seeds: list[int], glove_path: str, run_bert: bool = True) -> tuple[list[dict], dict | None]:
    split = load_imdb_with_val(random_state=42, clean=False, val_size=0.1)
    records = []
    best_hparams = None
    for i, seed in enumerate(seeds):
        records.append(
            _run_lstm(
                experiment="Exp1",
                domain_setup="IMDB->IMDB",
                seed=seed,
                split_train=split,
                split_test=split,
                glove_path=glove_path,
            )
        )
        if run_bert:
            bert_record, best_hparams = _run_bert(
                experiment="Exp1",
                domain_setup="IMDB->IMDB",
                seed=seed,
                split_train=split,
                split_test=split,
                best_hparams=best_hparams,
                do_search=(i == 0),
            )
            if bert_record is not None:
                records.append(bert_record)
    _save_experiment_outputs("Exp1", records)
    return records, best_hparams


def experiment_2(seeds: list[int], glove_path: str, best_hparams: dict | None, run_bert: bool = True) -> list[dict]:
    imdb = load_imdb_with_val(random_state=42, clean=False, val_size=0.1)
    fpb = load_financial_phrasebank_with_val(random_state=42, clean=False, val_size=0.1)
    records = []
    for seed in seeds:
        records.append(
            _run_lstm(
                experiment="Exp2",
                domain_setup="IMDB->FPB",
                seed=seed,
                split_train=imdb,
                split_test=fpb,
                glove_path=glove_path,
            )
        )
        if run_bert:
            bert_record, _ = _run_bert(
                experiment="Exp2",
                domain_setup="IMDB->FPB",
                seed=seed,
                split_train=imdb,
                split_test=fpb,
                best_hparams=best_hparams,
                do_search=False,
            )
            if bert_record is not None:
                records.append(bert_record)
    _save_experiment_outputs("Exp2", records)
    return records


def experiment_3(seeds: list[int], glove_path: str, best_hparams: dict | None, run_bert: bool = True) -> list[dict]:
    fpb = load_financial_phrasebank_with_val(random_state=42, clean=False, val_size=0.1)
    records = []
    for seed in seeds:
        records.append(
            _run_lstm(
                experiment="Exp3",
                domain_setup="FPB->FPB",
                seed=seed,
                split_train=fpb,
                split_test=fpb,
                glove_path=glove_path,
            )
        )
        if run_bert:
            bert_record, _ = _run_bert(
                experiment="Exp3",
                domain_setup="FPB->FPB",
                seed=seed,
                split_train=fpb,
                split_test=fpb,
                best_hparams=best_hparams,
                do_search=False,
            )
            if bert_record is not None:
                records.append(bert_record)
    _save_experiment_outputs("Exp3", records)
    return records


def run_all(seeds: list[int], glove_path: str, run_bert: bool = True) -> None:
    exp1, best_hparams = experiment_1(seeds=seeds, glove_path=glove_path, run_bert=run_bert)
    exp2 = experiment_2(seeds=seeds, glove_path=glove_path, best_hparams=best_hparams, run_bert=run_bert)
    exp3 = experiment_3(seeds=seeds, glove_path=glove_path, best_hparams=best_hparams, run_bert=run_bert)
    all_records = exp1 + exp2 + exp3
    save_result_records(all_records, os.path.join(OUTPUT_DIR, "all_records.json"))
    summary_df = summarize_records_mean_std(all_records)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "all_summary_mean_std.csv"), index=False)
    if best_hparams is not None:
        _save_json(os.path.join(OUTPUT_DIR, "best_bert_hparams.json"), best_hparams)


def main():
    parser = argparse.ArgumentParser(description="Run neural NLP experiments")
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--glove-path", type=str, default="")
    parser.add_argument("--skip-bert", action="store_true", help="Run only LSTM pipeline.")
    args = parser.parse_args()
    run_bert = not args.skip_bert

    best_hparams = None
    if args.experiment == 1:
        experiment_1(seeds=args.seeds, glove_path=args.glove_path, run_bert=run_bert)
    elif args.experiment == 2:
        experiment_2(seeds=args.seeds, glove_path=args.glove_path, best_hparams=best_hparams, run_bert=run_bert)
    elif args.experiment == 3:
        experiment_3(seeds=args.seeds, glove_path=args.glove_path, best_hparams=best_hparams, run_bert=run_bert)
    else:
        run_all(seeds=args.seeds, glove_path=args.glove_path, run_bert=run_bert)


if __name__ == "__main__":
    main()
