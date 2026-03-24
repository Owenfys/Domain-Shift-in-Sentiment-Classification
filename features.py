"""
features.py
-----------
TF-IDF feature extraction with configurable parameters.
Provides fit/transform interfaces for in-domain and cross-domain experiments.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp


def build_tfidf_vectorizer(
    max_features: int = 30_000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with sensible defaults for sentiment analysis."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )


def fit_transform_tfidf(
    train_texts: list[str],
    test_texts: list[str],
    vectorizer: TfidfVectorizer | None = None,
    **kwargs,
) -> tuple[sp.csr_matrix, sp.csr_matrix, TfidfVectorizer]:
    """
    Fit TF-IDF on train_texts, transform both train and test.

    Parameters
    ----------
    train_texts : list[str]
    test_texts  : list[str]
    vectorizer  : optional pre-configured vectorizer; if None, one is built with **kwargs
    **kwargs    : forwarded to build_tfidf_vectorizer if vectorizer is None

    Returns
    -------
    X_train_tfidf, X_test_tfidf, fitted_vectorizer
    """
    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer(**kwargs)

    print(f"[features] Fitting TF-IDF on {len(train_texts)} documents …")
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    print(f"[features] Vocabulary size: {len(vectorizer.vocabulary_)}, "
          f"feature matrix: {X_train.shape}")
    return X_train, X_test, vectorizer


def transform_tfidf(
    texts: list[str],
    fitted_vectorizer: TfidfVectorizer,
) -> sp.csr_matrix:
    """Transform new texts using an already-fitted vectorizer (for cross-domain)."""
    X = fitted_vectorizer.transform(texts)
    print(f"[features] Transformed {len(texts)} documents → {X.shape}")
    return X
