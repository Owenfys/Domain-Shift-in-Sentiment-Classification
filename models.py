"""
models.py
---------
Classical sentiment classifiers: Naive Bayes and Logistic Regression.
Each function returns a fitted model ready for prediction.
"""

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
import scipy.sparse as sp


# ── Naive Bayes ───────────────────────────────────────────────────────────────

def train_naive_bayes(
    X_train: sp.csr_matrix,
    y_train: list[int],
    variant: str = "multinomial",
    alpha: float = 1.0,
) -> MultinomialNB | ComplementNB:
    """
    Train a Naive Bayes classifier.

    Parameters
    ----------
    variant : 'multinomial' or 'complement'
        ComplementNB can be more robust under class imbalance.
    alpha : float
        Laplace smoothing parameter.
    """
    cls = MultinomialNB if variant == "multinomial" else ComplementNB
    model = cls(alpha=alpha)
    print(f"[models] Training {variant.title()} Naive Bayes (alpha={alpha}) …")
    model.fit(X_train, y_train)
    print(f"[models] NB training complete.")
    return model


# ── Logistic Regression ──────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: sp.csr_matrix,
    y_train: list[int],
    C: float = 1.0,
    class_weight: str | dict | None = None,
    max_iter: int = 1000,
    solver: str = "lbfgs",
) -> LogisticRegression:
    """
    Train a Logistic Regression classifier with configurable regularization.

    Parameters
    ----------
    C : float
        Inverse regularization strength (smaller = stronger reg).
    class_weight : None, 'balanced', or dict
        Adjusts weights inversely proportional to class frequencies if 'balanced'.
    """
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        solver=solver,
        random_state=42,
    )
    cw_str = class_weight if class_weight else "none"
    print(f"[models] Training Logistic Regression (C={C}, class_weight={cw_str}) …")
    model.fit(X_train, y_train)
    print(f"[models] LR training complete.")
    return model


# ── Convenience: train all classical model variants ──────────────────────────

DEFAULT_CONFIGS = {
    "NB_multinomial": {"variant": "multinomial", "alpha": 1.0},
    "NB_complement": {"variant": "complement", "alpha": 1.0},
    "LR_C1": {"C": 1.0, "class_weight": None},
    "LR_C0.1": {"C": 0.1, "class_weight": None},
    "LR_C10": {"C": 10.0, "class_weight": None},
    "LR_C1_balanced": {"C": 1.0, "class_weight": "balanced"},
}


def train_all_classical(
    X_train: sp.csr_matrix,
    y_train: list[int],
    configs: dict | None = None,
) -> dict:
    """
    Train all classical model variants defined in configs.

    Returns
    -------
    dict mapping model_name → fitted sklearn model
    """
    if configs is None:
        configs = DEFAULT_CONFIGS

    trained = {}
    for name, params in configs.items():
        print(f"\n{'─'*50}")
        if name.startswith("NB"):
            trained[name] = train_naive_bayes(X_train, y_train, **params)
        else:
            trained[name] = train_logistic_regression(X_train, y_train, **params)
    return trained
