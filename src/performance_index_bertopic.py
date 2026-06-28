"""
result_analysis.py
──────────────────
Performance index functions for evaluating BERTopic results.

Each function takes inputs, returns a value. No plotting, no file I/O.

Metrics:
  - C_V coherence
  - Topic diversity
  - Jaccard similarity (single pair)
  - Pairwise Jaccard matrix (across multiple runs)
  - Cluster stats (n_topics, outlier_pct, DBCV)
  - Summary table across multiple runs

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary


# ══════════════════════════════════════════════════════════════════════════════
# Tokenisation — build once, reuse across all coherence calls
# ══════════════════════════════════════════════════════════════════════════════

def build_tokeniser(
    docs: list[str],
    stop_words: list[str] | None = None,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[list[list[str]], Dictionary]:
    """
    Tokenise documents using the same CountVectorizer settings as BERTopic,
    so topic keywords and tokenised docs share the same token space.

    Call once per experiment and pass the result into every compute_coherence()
    call — tokenising the full corpus is expensive.

    Parameters
    ----------
    docs : list of raw document strings.
    stop_words : same stop word list passed to make_vectorizer().
    min_df : same min_df passed to make_vectorizer().
    ngram_range : same ngram_range passed to make_vectorizer().

    Returns
    -------
    tokenised : list of token lists, one per document.
    dictionary : gensim Dictionary built from the tokenised docs.
    """
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        min_df=min_df,
        ngram_range=ngram_range,
    )
    vectorizer.fit(docs)
    tokenised  = [vectorizer.build_analyzer()(doc) for doc in docs]
    dictionary = Dictionary(tokenised)
    return tokenised, dictionary


# ══════════════════════════════════════════════════════════════════════════════
# Keyword extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_topic_keywords(
    model,
    top_n: int = 10,
) -> dict[int, set[str]]:
    """
    Extract the top-n keywords per topic from a fitted BERTopic model.

    Parameters
    ----------
    model : fitted BERTopic instance (output of run_bertopic()).
    top_n : number of keywords to extract per topic.

    Returns
    -------
    Dict mapping topic_id -> set of keyword strings.
    Outlier topic (-1) is excluded.
    """
    return {
        tid: set(w for w, _ in model.get_topic(tid)[:top_n])
        for tid in model.get_topics()
        if tid != -1
    }


# ══════════════════════════════════════════════════════════════════════════════
# Cluster stats
# ══════════════════════════════════════════════════════════════════════════════

def compute_cluster_stats(
    model,
    topics: list[int],
) -> dict[str, float | int]:
    """
    Compute basic cluster quality stats from a fitted BERTopic model.

    Parameters
    ----------
    model : fitted BERTopic instance.
    topics : list of topic IDs returned by fit_transform() (-1 = outlier).

    Returns
    -------
    Dict with keys:
      n_topics    : number of discovered topics (excluding outlier)
      n_outliers  : number of documents assigned to topic -1
      outlier_pct : percentage of documents assigned to topic -1
      dbcv        : HDBSCAN relative validity score (DBCV)
    """
    topic_keywords = extract_topic_keywords(model)
    n_topics       = len(topic_keywords)
    n_outliers     = sum(1 for t in topics if t == -1)
    outlier_pct    = round(100 * n_outliers / len(topics), 1)
    dbcv           = round(float(model.hdbscan_model.relative_validity_), 4)

    return {
        "n_topics":    n_topics,
        "n_outliers":  n_outliers,
        "outlier_pct": outlier_pct,
        "dbcv":        dbcv,
    }


# ══════════════════════════════════════════════════════════════════════════════
# C_V Coherence
# ══════════════════════════════════════════════════════════════════════════════

def compute_coherence(
    topic_keywords: dict[int, set[str]],
    tokenised: list[list[str]],
    dictionary: Dictionary,
) -> float | None:
    """
    Compute C_V coherence for a set of topics.

    Parameters
    ----------
    topic_keywords : dict mapping topic_id -> set of keyword strings.
                     Output of extract_topic_keywords().
    tokenised : list of token lists. Output of build_tokeniser().
    dictionary : gensim Dictionary. Output of build_tokeniser().

    Returns
    -------
    C_V coherence score (float), or None if no valid topics.
    """
    topics_as_lists = [
        list(words) for words in topic_keywords.values()
        if len(words) >= 2
    ]
    if not topics_as_lists:
        return None

    cm = CoherenceModel(
        topics=topics_as_lists,
        texts=tokenised,
        dictionary=dictionary,
        coherence="c_v",
    )
    return round(cm.get_coherence(), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Topic Diversity
# ══════════════════════════════════════════════════════════════════════════════

def compute_diversity(
    topic_keywords: dict[int, set[str]],
) -> float | None:
    """
    Compute topic diversity — proportion of unique words across all topics.

    1.0 = every keyword is unique across all topics (fully diverse).
    0.0 = all topics share the same keywords (no diversity).

    Parameters
    ----------
    topic_keywords : dict mapping topic_id -> set of keyword strings.

    Returns
    -------
    Diversity score (float), or None if no keywords.
    """
    all_words = [w for words in topic_keywords.values() for w in words]
    if not all_words:
        return None
    return round(len(set(all_words)) / len(all_words), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Jaccard similarity
# ══════════════════════════════════════════════════════════════════════════════

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two keyword sets."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def pairwise_jaccard(
    kws_a: dict[int, set[str]],
    kws_b: dict[int, set[str]],
) -> float:
    """
    Symmetric mean best-match Jaccard between two configurations.

    For each topic in A, finds the most similar topic in B, then averages.
    Runs in both directions and averages to make it symmetric.

    Parameters
    ----------
    kws_a, kws_b : dicts mapping topic_id -> set of keyword strings.

    Returns
    -------
    Symmetric mean best-match Jaccard score (float, 0–1).
    """
    def one_direction(source, target):
        scores = [
            max((jaccard(s, t) for t in target.values()), default=0.0)
            for s in source.values()
        ]
        return np.mean(scores) if scores else 0.0

    return round((one_direction(kws_a, kws_b) + one_direction(kws_b, kws_a)) / 2, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Aggregations across multiple runs
# ══════════════════════════════════════════════════════════════════════════════

def build_jaccard_matrix(
    results: dict[str, dict],
) -> pd.DataFrame:
    """
    Build a symmetric pairwise Jaccard similarity matrix across all runs.

    Parameters
    ----------
    results : dict mapping run label -> result dict with key "keywords"
              (dict[int, set[str]]).

    Returns
    -------
    DataFrame of shape (n_runs, n_runs). Diagonal is 1.0.
    """
    labels = list(results.keys())
    n      = len(labels)
    matrix = np.zeros((n, n))

    for i, label_a in enumerate(labels):
        for j, label_b in enumerate(labels):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                score = pairwise_jaccard(
                    results[label_a]["keywords"],
                    results[label_b]["keywords"],
                )
                matrix[i, j] = score
                matrix[j, i] = score

    return pd.DataFrame(matrix, index=labels, columns=labels)


def build_summary_table(
    results: dict[str, dict],
    tokenised: list[list[str]],
    dictionary: Dictionary,
) -> pd.DataFrame:
    """
    Build a summary DataFrame of all metrics across multiple runs.

    Parameters
    ----------
    results : dict mapping run label -> result dict with keys:
                "keywords"    : dict[int, set[str]]
                "n_topics"    : int
                "outlier_pct" : float
                "dbcv"        : float
                "params"      : dict (from cfg.to_dict())
    tokenised : output of build_tokeniser().
    dictionary : output of build_tokeniser().

    Returns
    -------
    DataFrame with one row per run and columns:
      run, min_cluster_size, n_neighbors, min_samples,
      n_topics, outlier_pct, dbcv, coherence_cv, diversity
    """
    rows = []
    for label, res in results.items():
        p = res["params"]
        rows.append({
            "run":              label,
            "min_cluster_size": p.get("min_cluster_size"),
            "n_neighbors":      p.get("n_neighbors"),
            "min_samples":      p.get("min_samples"),
            "n_topics":         res["n_topics"],
            "outlier_pct":      round(res["outlier_pct"], 1),
            "dbcv":             res.get("dbcv"),
            "coherence_cv":     compute_coherence(res["keywords"], tokenised, dictionary),
            "diversity":        compute_diversity(res["keywords"]),
        })

    return pd.DataFrame(rows)
