"""
result_schema.py
----------------
Unified experiment result schema helpers.
"""

from __future__ import annotations

from typing import Any
import json


def make_result_record(
    experiment: str,
    model: str,
    seed: int,
    domain_setup: str,
    y_true: list[int],
    y_pred: list[int],
    metrics: dict[str, float],
    y_prob: list[float] | None = None,
    best_hparams: dict[str, Any] | None = None,
    train_config: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """
    Build a unified result record consumed by evaluate/reporting utilities.
    """
    record = {
        "experiment": experiment,
        "model": model,
        "seed": seed,
        "domain_setup": domain_setup,
        "y_true": list(y_true),
        "y_pred": list(y_pred),
        "metrics": metrics,
        "train_config": train_config or {},
        "notes": notes,
    }
    if y_prob is not None:
        record["y_prob"] = list(y_prob)
    if best_hparams is not None:
        record["best_hparams"] = best_hparams
    return record


def save_result_records(records: list[dict], path: str) -> None:
    """Persist a list of result records as JSON."""
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
