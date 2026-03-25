"""
neural_models.py
----------------
Neural sentiment models:
- LSTM classifier (PyTorch)
- BERT fine-tuning with lightweight hyperparameter search
"""

from __future__ import annotations

import itertools
import os
import random
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification


def resolve_device() -> torch.device:
    """
    Device priority:
    1) CUDA GPU
    2) Apple MPS
    3) CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pad_idx: int = 0,
        embedding_matrix: np.ndarray | None = None,
        finetune_embeddings: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = finetune_embeddings

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, 2)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            feats = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            feats = h_n[-1]
        feats = self.dropout(feats)
        return self.classifier(feats)


def _classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_lstm_model(model: LSTMClassifier, dataloader, device: torch.device) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, lengths)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_prob.extend(probs.cpu().tolist())
    return all_true, all_pred, all_prob


def train_lstm(
    train_loader,
    val_loader,
    vocab_size: int,
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    num_layers: int = 1,
    bidirectional: bool = True,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 5,
    embedding_matrix: np.ndarray | None = None,
    finetune_embeddings: bool = True,
    seed: int = 42,
) -> tuple[LSTMClassifier, dict[str, Any]]:
    set_global_seed(seed)
    device = resolve_device()
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        embedding_matrix=embedding_matrix,
        finetune_embeddings=finetune_embeddings,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_macro_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_true, val_pred, _ = evaluate_lstm_model(model, val_loader, device)
        val_metrics = _classification_metrics(val_true, val_pred)
        avg_loss = epoch_loss / max(1, len(train_loader))
        history.append({"epoch": epoch, "train_loss": avg_loss, **val_metrics})
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_info = {
        "best_val_macro_f1": float(best_macro_f1),
        "epochs": epochs,
        "lr": lr,
        "device": str(device),
        "history": history,
    }
    return model, train_info


def predict_lstm(model: LSTMClassifier, dataloader) -> tuple[list[int], list[int], list[float]]:
    device = resolve_device()
    model = model.to(device)
    return evaluate_lstm_model(model, dataloader, device)


def _eval_bert(model, dataloader, device: torch.device) -> tuple[dict[str, float], list[int], list[int], list[float]]:
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop("labels")
            out = model(**inputs)
            logits = out.logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_prob.extend(probs.cpu().tolist())
    return _classification_metrics(all_true, all_pred), all_true, all_pred, all_prob


def train_bert_with_search(
    model_name: str,
    train_dataset,
    val_dataset,
    search_space: dict[str, list],
    seed: int = 42,
    output_dir: str = "outputs/neural/bert_search",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Hyperparameter search over (learning_rate, batch_size, num_train_epochs).
    Selection metric: validation macro-F1.
    """
    set_global_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    combos = list(
        itertools.product(
            search_space["learning_rate"],
            search_space["batch_size"],
            search_space["num_train_epochs"],
        )
    )
    device = resolve_device()
    best = {"score": -1.0, "hparams": None, "state_dict": None}
    all_trials = []

    for trial_idx, (lr, bs, n_epochs) in enumerate(combos, start=1):
        run_dir = tempfile.mkdtemp(prefix=f"bert_trial_{trial_idx}_", dir=output_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        best_val = -1.0
        best_trial_state = None
        for _ in range(int(n_epochs)):
            model.train()
            for batch in train_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                out = model(**inputs)
                loss = out.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_metrics, _, _, _ = _eval_bert(model, val_loader, device)
            if val_metrics["macro_f1"] > best_val:
                best_val = val_metrics["macro_f1"]
                best_trial_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        score = float(best_val)
        trial = {"learning_rate": lr, "batch_size": bs, "num_train_epochs": n_epochs, "val_macro_f1": score}
        all_trials.append(trial)
        if score > best["score"]:
            best = {"score": score, "hparams": trial, "state_dict": best_trial_state}
        # Save trial metrics for traceability.
        with open(os.path.join(run_dir, "trial_metrics.json"), "w") as f:
            import json
            json.dump(trial, f, indent=2)

    if best["state_dict"] is None:
        raise RuntimeError("BERT hyperparameter search failed to produce a model.")
    best_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    best_model.load_state_dict(best["state_dict"])
    best_info = {"best_val_macro_f1": best["score"], "best_hparams": best["hparams"], "all_trials": all_trials}
    return best_model, best_info


def predict_bert(model: torch.nn.Module, dataset, batch_size: int = 16) -> tuple[list[int], list[int], list[float]]:
    device = resolve_device()
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    _, y_true, y_pred, y_prob = _eval_bert(model, dataloader, device)
    return y_true, y_pred, y_prob
