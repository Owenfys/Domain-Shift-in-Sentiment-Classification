"""
neural_features.py
------------------
Feature and data pipeline utilities for neural experiments:
- LSTM + GloVe pipeline
- BERT tokenizer pipeline
"""

from __future__ import annotations

from collections import Counter
import os
import re
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def simple_tokenize(text: str) -> list[str]:
    """Lightweight tokenizer for LSTM baseline."""
    return re.findall(r"\b\w+\b", text.lower())


def text_length_stats(texts: list[str]) -> dict[str, float]:
    """Compute simple token length stats to choose max_len more reasonably."""
    lengths = [len(simple_tokenize(t)) for t in texts]
    if not lengths:
        return {"count": 0, "mean": 0, "p90": 0, "p95": 0, "max": 0}
    arr = np.asarray(lengths)
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def choose_max_len(train_texts: list[str], cap: int = 512) -> int:
    """Pick max_len from train distribution; capped for memory control."""
    stats = text_length_stats(train_texts)
    max_len = int(stats["p95"]) if stats["p95"] > 0 else 128
    return max(16, min(max_len, cap))


def build_vocab(
    texts: list[str],
    min_freq: int = 2,
    max_size: int = 50_000,
) -> dict[str, int]:
    """Build vocabulary from train texts only."""
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> tuple[list[int], int]:
    tokens = simple_tokenize(text)
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens[:max_len]]
    seq_len = len(ids)
    if seq_len < max_len:
        ids.extend([vocab[PAD_TOKEN]] * (max_len - seq_len))
    return ids, seq_len


def encode_texts(
    texts: list[str],
    vocab: dict[str, int],
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_ids, lengths = [], []
    for text in texts:
        ids, seq_len = encode_text(text, vocab, max_len=max_len)
        all_ids.append(ids)
        lengths.append(seq_len)
    return torch.tensor(all_ids, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def compute_oov_rate(texts: list[str], vocab: dict[str, int]) -> float:
    total, oov = 0, 0
    for text in texts:
        tokens = simple_tokenize(text)
        total += len(tokens)
        oov += sum(1 for tok in tokens if tok not in vocab)
    return float(oov / total) if total > 0 else 0.0


class LSTMDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, lengths: torch.Tensor, labels: list[int]):
        self.input_ids = input_ids
        self.lengths = lengths
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "lengths": self.lengths[idx],
            "labels": self.labels[idx],
        }


def build_lstm_dataloaders(
    X_train: list[str],
    y_train: list[int],
    X_val: list[str],
    y_val: list[int],
    X_test: list[str],
    y_test: list[int],
    batch_size: int = 64,
    min_freq: int = 2,
    max_vocab_size: int = 50_000,
    max_len: int | None = None,
) -> dict[str, Any]:
    """Construct vocabulary, encoded tensors, and train/val/test dataloaders."""
    vocab = build_vocab(X_train, min_freq=min_freq, max_size=max_vocab_size)
    if max_len is None:
        max_len = choose_max_len(X_train)

    train_ids, train_len = encode_texts(X_train, vocab, max_len=max_len)
    val_ids, val_len = encode_texts(X_val, vocab, max_len=max_len)
    test_ids, test_len = encode_texts(X_test, vocab, max_len=max_len)

    train_ds = LSTMDataset(train_ids, train_len, y_train)
    val_ds = LSTMDataset(val_ids, val_len, y_val)
    test_ds = LSTMDataset(test_ids, test_len, y_test)

    return {
        "vocab": vocab,
        "max_len": max_len,
        "oov_rate_val": compute_oov_rate(X_val, vocab),
        "oov_rate_test": compute_oov_rate(X_test, vocab),
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }


def load_glove_embeddings(
    glove_path: str,
    vocab: dict[str, int],
    embedding_dim: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Build embedding matrix from a GloVe text file.
    If file does not exist, returns random matrix and notes fallback stats.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.05, size=(len(vocab), embedding_dim)).astype(np.float32)
    matrix[vocab[PAD_TOKEN]] = np.zeros(embedding_dim, dtype=np.float32)
    found = 0

    if not glove_path or not os.path.exists(glove_path):
        stats = {
            "glove_found": 0.0,
            "vocab_size": float(len(vocab)),
            "coverage": 0.0,
            "fallback_random": 1.0,
        }
        return matrix, stats

    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= embedding_dim:
                continue
            token = parts[0]
            if token not in vocab:
                continue
            vec = np.asarray(parts[1:1 + embedding_dim], dtype=np.float32)
            if vec.shape[0] != embedding_dim:
                continue
            matrix[vocab[token]] = vec
            found += 1

    coverage = found / max(1, len(vocab) - 2)
    stats = {
        "glove_found": float(found),
        "vocab_size": float(len(vocab)),
        "coverage": float(coverage),
        "fallback_random": 0.0,
    }
    return matrix, stats


class BertTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer_name: str,
        max_len: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_bert_datasets(
    X_train: list[str],
    y_train: list[int],
    X_val: list[str],
    y_val: list[int],
    X_test: list[str],
    y_test: list[int],
    tokenizer_name: str = "bert-base-uncased",
    max_len: int = 256,
) -> dict[str, Dataset]:
    return {
        "train_dataset": BertTextDataset(X_train, y_train, tokenizer_name=tokenizer_name, max_len=max_len),
        "val_dataset": BertTextDataset(X_val, y_val, tokenizer_name=tokenizer_name, max_len=max_len),
        "test_dataset": BertTextDataset(X_test, y_test, tokenizer_name=tokenizer_name, max_len=max_len),
    }
