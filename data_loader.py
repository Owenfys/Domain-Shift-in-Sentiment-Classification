"""
data_loader.py
--------------
Handles loading and preprocessing of IMDB Reviews and Financial PhraseBank datasets.
Provides unified interfaces for train/test splits and cross-domain evaluation setups.
"""

import re
import zipfile
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip HTML, remove non-alpha chars."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)          # strip HTML tags (common in IMDB)
    text = re.sub(r"[^a-z\s]", " ", text)       # keep only letters + spaces
    text = re.sub(r"\s+", " ", text).strip()     # collapse whitespace
    return text


def split_train_val_test(
    texts: list[str],
    labels: list[int],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> dict:
    """
    Deterministic stratified split into train/val/test.

    Parameters
    ----------
    test_size : float
        Fraction of total data reserved for test split.
    val_size : float
        Fraction of total data reserved for validation split.
    """
    if test_size <= 0 or val_size < 0 or (test_size + val_size) >= 1:
        raise ValueError("Require 0 < test_size and 0 <= val_size and test_size + val_size < 1.")

    if len(texts) != len(labels):
        raise ValueError("texts and labels must have equal length.")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    if val_size == 0:
        return {
            "X_train": X_trainval,
            "X_val": [],
            "X_test": X_test,
            "y_train": y_trainval,
            "y_val": [],
            "y_test": y_test,
        }

    # Convert global val fraction to fraction of the remaining train+val partition.
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_trainval,
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


# ── IMDB ──────────────────────────────────────────────────────────────────────

def load_imdb(test_size: float = 0.2, random_state: int = 42, clean: bool = True) -> dict:
    """
    Load the IMDB Reviews dataset (50 000 reviews, pos/neg).

    Returns
    -------
    dict with keys:
        X_train, X_test   – lists of review strings
        y_train, y_test   – lists of int labels (0 = negative, 1 = positive)
    """
    print("[data_loader] Loading IMDB dataset from HuggingFace …")
    ds = load_dataset("imdb")

    # HuggingFace IMDB already has a train/test split (25 k each)
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]

    if clean:
        print("[data_loader] Cleaning IMDB texts …")
        train_texts = [clean_text(t) for t in train_texts]
        test_texts = [clean_text(t) for t in test_texts]

    print(f"[data_loader] IMDB loaded — train: {len(train_texts)}, test: {len(test_texts)}")
    return {
        "X_train": train_texts,
        "X_test": test_texts,
        "y_train": train_labels,
        "y_test": test_labels,
    }


# ── Financial PhraseBank ──────────────────────────────────────────────────────

LABEL_MAP_FPB = {"positive": 1, "negative": 0, "neutral": 2}

def load_financial_phrasebank(
    agreement: str = "sentences_allagree",
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
    remove_neutral: bool = True,
) -> dict:
    """
    Load the Financial PhraseBank dataset.

    Parameters
    ----------
    agreement : str
        Which agreement subset to use.  Options:
        'sentences_allagree', 'sentences_75agree',
        'sentences_66agree', 'sentences_50agree'
    remove_neutral : bool
        If True, drop neutral examples so labels align with binary IMDB.

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test
    """
    print(f"[data_loader] Loading Financial PhraseBank ({agreement}) …")

    # The repo only contains a zip file; download it and extract the right .txt
    _AGREEMENT_FILENAMES = {
        "sentences_allagree": "Sentences_AllAgree.txt",
        "sentences_75agree":  "Sentences_75Agree.txt",
        "sentences_66agree":  "Sentences_66Agree.txt",
        "sentences_50agree":  "Sentences_50Agree.txt",
    }
    target_txt = _AGREEMENT_FILENAMES[agreement]

    zip_path = hf_hub_download(
        repo_id="takala/financial_phrasebank",
        filename="data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )

    # Parse: each line is  "sentence text@label"  (label = positive|negative|neutral)
    _LABEL_STR_MAP = {"negative": 0, "neutral": 1, "positive": 2}
    texts, labels = [], []
    with zipfile.ZipFile(zip_path) as zf:
        # Find the entry matching the target filename (path inside zip may vary)
        entry = next((n for n in zf.namelist() if n.endswith(target_txt)), None)
        if entry is None:
            raise FileNotFoundError(f"{target_txt} not found in zip. Contents: {zf.namelist()}")
        with zf.open(entry) as fh:
            for line in fh:
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                idx = line.rfind("@")
                if idx == -1:
                    continue
                sentence = line[:idx].strip()
                label_str = line[idx + 1:].strip().lower()
                if label_str in _LABEL_STR_MAP:
                    texts.append(sentence)
                    labels.append(_LABEL_STR_MAP[label_str])

    # Remap: original encoding is 0=neg, 1=neutral, 2=pos
    # We want 0=negative, 1=positive (dropping neutral)
    remapped_texts = []
    remapped_labels = []
    for t, l in zip(texts, labels):
        if l == 1 and remove_neutral:
            continue                       # skip neutral
        new_label = 1 if l == 2 else 0     # 2 → 1 (pos), 0 stays 0 (neg)
        remapped_texts.append(t)
        remapped_labels.append(new_label)

    if clean:
        print("[data_loader] Cleaning Financial PhraseBank texts …")
        remapped_texts = [clean_text(t) for t in remapped_texts]

    X_train, X_test, y_train, y_test = train_test_split(
        remapped_texts, remapped_labels,
        test_size=test_size, random_state=random_state, stratify=remapped_labels,
    )

    n_pos = sum(remapped_labels)
    n_neg = len(remapped_labels) - n_pos
    print(f"[data_loader] Financial PhraseBank loaded — total: {len(remapped_labels)} "
          f"(pos: {n_pos}, neg: {n_neg}), train: {len(X_train)}, test: {len(X_test)}")
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ── Convenience: get all data for the three experiments ───────────────────────

def load_all_data(clean: bool = True, random_state: int = 42) -> dict:
    """
    Load both datasets and return a single dict for easy access.

    Returns
    -------
    dict with keys 'imdb' and 'fpb', each containing the standard split dict.
    """
    imdb = load_imdb(clean=clean, random_state=random_state)
    fpb = load_financial_phrasebank(clean=clean, random_state=random_state)
    return {"imdb": imdb, "fpb": fpb}


def load_imdb_with_val(
    random_state: int = 42,
    clean: bool = False,
    val_size: float = 0.1,
) -> dict:
    """
    Load IMDB with train/val/test splits.

    Notes
    -----
    Uses the original HuggingFace train+test partitions as one pool, then applies
    a deterministic stratified split to keep all model families comparable.
    """
    print("[data_loader] Loading IMDB with train/val/test split …")
    ds = load_dataset("imdb")
    texts = list(ds["train"]["text"]) + list(ds["test"]["text"])
    labels = list(ds["train"]["label"]) + list(ds["test"]["label"])
    if clean:
        texts = [clean_text(t) for t in texts]
    split = split_train_val_test(
        texts=texts,
        labels=labels,
        test_size=0.2,
        val_size=val_size,
        random_state=random_state,
    )
    print(
        f"[data_loader] IMDB split — train: {len(split['X_train'])}, "
        f"val: {len(split['X_val'])}, test: {len(split['X_test'])}"
    )
    return split


def load_financial_phrasebank_with_val(
    agreement: str = "sentences_allagree",
    random_state: int = 42,
    clean: bool = False,
    remove_neutral: bool = True,
    val_size: float = 0.1,
) -> dict:
    """
    Load Financial PhraseBank with deterministic train/val/test splits.
    """
    base = load_financial_phrasebank(
        agreement=agreement,
        random_state=random_state,
        clean=clean,
        remove_neutral=remove_neutral,
    )
    texts = base["X_train"] + base["X_test"]
    labels = base["y_train"] + base["y_test"]
    split = split_train_val_test(
        texts=texts,
        labels=labels,
        test_size=0.2,
        val_size=val_size,
        random_state=random_state,
    )
    print(
        f"[data_loader] FPB split — train: {len(split['X_train'])}, "
        f"val: {len(split['X_val'])}, test: {len(split['X_test'])}"
    )
    return split


# ── Quick sanity-check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_all_data()
    for name, d in data.items():
        print(f"\n{'='*40}")
        print(f"Dataset: {name}")
        print(f"  Train size : {len(d['X_train'])}")
        print(f"  Test size  : {len(d['X_test'])}")
        print(f"  Sample text: {d['X_train'][0][:120]}…")
        print(f"  Sample label: {d['y_train'][0]}")
