"""
bertopic_components.py
──────────────────────
Two layers:

  1. make_* factory functions — one per BERTopic component.
     Call with no arguments to get the project defaults, or override
     individual parameters as needed.

  2. BERTopicConfig dataclass + run_bertopic() — the experiment layer.
     Define one config per experiment file, call run_bertopic(), get back
     a fitted model and topic assignments. No boilerplate repeated across files.

Three training modes (set via BERTopicConfig.mode):
  - "unsupervised"   : standard BERTopic, no label guidance
  - "semi_supervised": passes integer class labels (y) to UMAP so the
                       embedding geometry is biased toward known accident
                       situation classes (main_situation_class)
  - "guided"         : passes seed_topic_list to BERTopic so topic
                       discovery is biased toward user-defined keyword sets

──────────────────────────────────────────────────────────────────────────────
Typical usage
──────────────────────────────────────────────────────────────────────────────

    from bertopic_components import (
        BERTopicConfig,
        run_bertopic,
    )
    from embeddings import get_embeddings
    from stop_words import get_stop_words

    danish_stop_words = get_stop_words()
    embedding_model, embeddings = get_embeddings(docs)

    # ── Unsupervised (default) ──
    cfg = BERTopicConfig(stop_words=danish_stop_words)
    model, topics, probs = run_bertopic(cfg, docs, embeddings)

    # ── Semi-supervised ──
    cfg = BERTopicConfig(
        mode="semi_supervised",
        labels=df["main_situation_class"].tolist(),
        stop_words=danish_stop_words,
        n_neighbors=75,
    )
    model, topics, probs = run_bertopic(cfg, docs, embeddings, embedding_model)

    # ── Guided ──
    cfg = BERTopicConfig(
        mode="guided",
        seed_topic_list=[
            ["fodgænger", "fortov", "kryds"],
            ["cyklist", "cykelsti", "cykel"],
            ["spiritus", "promille", "alkohol"],
        ],
        stop_words=danish_stop_words,
    )
    model, topics, probs = run_bertopic(cfg, docs, embeddings, embedding_model)
"""

from __future__ import annotations

import random
import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Literal

import umap
import hdbscan


# ── Reproducibility ───────────────────────────────────────────────────────────

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch globally.

    Call once at the start of each experiment file to ensure reproducibility
    across UMAP, HDBSCAN, and any sampling steps.

    Parameters
    ----------
    seed : integer seed. Defaults to the project-wide SEED (42).
           Pass a different value to run a reproducibility check.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()  # applied once on import

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Component factories
# ══════════════════════════════════════════════════════════════════════════════

def load_stop_words(filepath: str) -> list[str]:
    """
    Read a plain-text stop word file (one word per line).

    Parameters
    ----------
    filepath : path to the .txt file

    Returns
    -------
    List of stripped, non-empty stop word strings.
    """
    with open(filepath, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def make_umap(
    n_neighbors: int = 50,
    n_components: int = 10,
    min_dist: float = 0.01,
    metric: str = "cosine",
    random_state: int = 42,
    target_weight: float | None = None,
    target_metric: str = "categorical",
) -> umap.UMAP:
    """
    Build a UMAP dimensionality reduction model.

    Parameters
    ----------
    n_neighbors : controls local vs global structure.
                  Lower = more local detail, higher = more global.
    n_components : target dimensionality for clustering (not visualisation).
                   Keep >=5 here; use 2 only for 2D scatter plots.
    min_dist : how tightly points are packed in the reduced space.
               Keep low (0.0–0.05) so HDBSCAN sees separable regions.
    metric : distance metric applied to the input embeddings.
    random_state : reproducibility seed.
    target_weight : strength of label supervision in semi-supervised mode.
                    0.0 = fully unsupervised, 1.0 = fully label-driven.
                    Only applied when not None (i.e. semi_supervised mode).
    target_metric : distance metric used for the label side of UMAP.
                    'categorical' is correct for integer class labels.
    """
    kwargs = dict(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    if target_weight is not None:
        kwargs["target_weight"] = target_weight
        kwargs["target_metric"] = target_metric
    return umap.UMAP(**kwargs)


def make_hdbscan(
    min_cluster_size: int = 100,
    min_samples: int = 10,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    gen_min_span_tree: bool = True,
    prediction_data: bool = True,
) -> hdbscan.HDBSCAN:
    """
    Build an HDBSCAN clustering model.

    Parameters
    ----------
    min_cluster_size : minimum documents required to form a topic.
                       Increase for fewer, broader topics.
    min_samples : controls outlier sensitivity.
                  Higher = more points assigned to noise (-1).
                  Should be <= min_cluster_size.
    metric : distance metric on the UMAP-reduced space.
             'euclidean' is standard after UMAP compression.
    cluster_selection_method : 'eom' (excess of mass) handles varied
                                cluster sizes better than 'leaf'.
    gen_min_span_tree : required for the DBCV score (relative_validity_).
    prediction_data : required for approximate_predict() on new documents.
    """
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        gen_min_span_tree=gen_min_span_tree,
        prediction_data=prediction_data,
    )


def make_vectorizer(
    stop_words: list[str] | None = None,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (1, 2),
) -> CountVectorizer:
    """
    Build a CountVectorizer for BERTopic topic representation.

    Parameters
    ----------
    stop_words : words to exclude from topic keywords.
                 Pass the output of load_stop_words() here.
    min_df : minimum document frequency for a token to be retained.
    ngram_range : (min_n, max_n). (1, 2) captures single words and bigrams,
                  e.g. 'rød lys' as one unit.
    """
    return CountVectorizer(
        stop_words=stop_words,
        min_df=min_df,
        ngram_range=ngram_range,
    )


def make_ctfidf(
    reduce_frequent_words: bool = True,
) -> ClassTfidfTransformer:
    """
    Build a c-TF-IDF transformer for topic keyword extraction.

    Parameters
    ----------
    reduce_frequent_words : down-weights words that appear across many topics,
                            making per-topic keywords more distinctive.
    """
    return ClassTfidfTransformer(reduce_frequent_words=reduce_frequent_words)


def make_representation() -> KeyBERTInspired:
    """
    Build a KeyBERT-inspired representation model.

    Refines topic keywords using embedding similarity on top of c-TF-IDF,
    producing more semantically coherent keyword sets.
    """
    return KeyBERTInspired()


def make_bertopic(
    embedding_model=None,
    umap_model: umap.UMAP | None = None,
    hdbscan_model: hdbscan.HDBSCAN | None = None,
    vectorizer_model: CountVectorizer | None = None,
    ctfidf_model: ClassTfidfTransformer | None = None,
    representation_model: KeyBERTInspired | None = None,
    seed_topic_list: list[list[str]] | None = None,
    nr_topics: str | int = "auto",
    top_n_words: int = 20,
    verbose: bool = True,
) -> BERTopic:
    """
    Assemble a BERTopic model from components.

    Any component left as None is replaced by the project default.
    Pass pre-built components to override specific parts.

    Parameters
    ----------
    embedding_model : SentenceTransformer loaded from embeddings.py.
                      Pass the model so BERTopic can encode new documents
                      after fitting — but embeddings are always pre-computed
                      and passed directly to fit_transform().
    umap_model : UMAP instance (unfitted).
    hdbscan_model : HDBSCAN instance (unfitted).
    vectorizer_model : CountVectorizer for keyword extraction.
    ctfidf_model : ClassTfidfTransformer.
    representation_model : KeyBERTInspired or similar for keyword refinement.
    seed_topic_list : list of keyword lists, one per desired seed topic.
                      Only relevant in guided mode — pass None otherwise.
    nr_topics : 'auto' = BERTopic merges similar topics automatically.
                Integer = fix the final number of topics.
    top_n_words : number of keywords stored and displayed per topic.
    verbose : print fitting progress.
    """
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model or make_umap(),
        hdbscan_model=hdbscan_model or make_hdbscan(),
        vectorizer_model=vectorizer_model or make_vectorizer(),
        ctfidf_model=ctfidf_model or make_ctfidf(),
        representation_model=representation_model or make_representation(),
        seed_topic_list=seed_topic_list,
        nr_topics=nr_topics,
        top_n_words=top_n_words,
        verbose=verbose,
    )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Experiment config + runner
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BERTopicConfig:
    """
    Full configuration for one BERTopic experiment.

    Define one instance per experiment file and pass it to run_bertopic().
    All parameters have project defaults so you only set what differs.

    Training modes
    --------------
    mode = "unsupervised"  (default)
        Standard BERTopic. No label guidance.

    mode = "semi_supervised"
        Passes integer class labels to UMAP via the y argument, biasing the
        embedding geometry toward known accident situation classes.
        Requires: labels — list of ints, one per document.

    mode = "guided"
        Passes seed_topic_list to BERTopic, biasing topic discovery toward
        user-defined keyword groups.
        Requires: seed_topic_list — list of keyword lists, one per seed topic.
    """

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode: Literal["unsupervised", "semi_supervised", "guided"] = "unsupervised"

    # ── UMAP ──────────────────────────────────────────────────────────────────
    n_neighbors: int = 50
    n_components: int = 10
    min_dist: float = 0.01
    umap_metric: str = "cosine"
    target_weight: float | None = None   # semi_supervised only; None = off
    target_metric: str = "categorical"   # metric for label side of UMAP

    # ── HDBSCAN ───────────────────────────────────────────────────────────────
    min_cluster_size: int = 100
    min_samples: int = 10
    hdbscan_metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    gen_min_span_tree: bool = True
    prediction_data: bool = True

    # ── Vectorizer ────────────────────────────────────────────────────────────
    stop_words: list[str] | None = None
    min_df: int = 2
    ngram_range: tuple[int, int] = (1, 2)

    # ── c-TF-IDF ──────────────────────────────────────────────────────────────
    reduce_frequent_words: bool = True

    # ── BERTopic ──────────────────────────────────────────────────────────────
    nr_topics: str | int = "auto"
    top_n_words: int = 20
    verbose: bool = True
    representation_model: object = None

    # ── Mode-specific ─────────────────────────────────────────────────────────
    labels: list[int] | None = None
    seed_topic_list: list[list[str]] | None = field(default=None)

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42

    def __post_init__(self) -> None:
        if self.mode == "semi_supervised" and self.labels is None:
            raise ValueError(
                "mode='semi_supervised' requires labels. "
                "Pass labels=df['main_situation_class'].tolist()"
            )
        if self.mode == "guided" and self.seed_topic_list is None:
            raise ValueError(
                "mode='guided' requires seed_topic_list "
                "(list of keyword lists, one per seed topic)."
            )

    def to_dict(self) -> dict:
        """
        Serialise the config to a plain dict for logging alongside results.
        Excludes labels and seed_topic_list (large / variable-length objects).
        """
        return {
            "mode":                     self.mode,
            "n_neighbors":              self.n_neighbors,
            "n_components":             self.n_components,
            "min_dist":                 self.min_dist,
            "umap_metric":              self.umap_metric,
            "target_weight":            self.target_weight,
            "target_metric":            self.target_metric,
            "min_cluster_size":         self.min_cluster_size,
            "min_samples":              self.min_samples,
            "hdbscan_metric":           self.hdbscan_metric,
            "cluster_selection_method": self.cluster_selection_method,
            "gen_min_span_tree":        self.gen_min_span_tree,
            "prediction_data":          self.prediction_data,
            "min_df":                   self.min_df,
            "ngram_range":              self.ngram_range,
            "reduce_frequent_words":    self.reduce_frequent_words,
            "nr_topics":                self.nr_topics,
            "top_n_words":              self.top_n_words,
            "seed":                     self.seed,
        }


def run_bertopic(
    cfg: BERTopicConfig,
    docs: list[str],
    embeddings,
    embedding_model=None,
) -> tuple[BERTopic, list[int], list[float]]:
    """
    Build and fit a BERTopic model from a BERTopicConfig.

    Parameters
    ----------
    cfg : BERTopicConfig defining the full experiment setup.
    docs : list of raw document strings (police narratives).
    embeddings : pre-computed numpy array of shape (n_docs, hidden_dim).
                 Always pass pre-computed embeddings — avoids re-encoding
                 across multiple experiments.
    embedding_model : SentenceTransformer loaded from embeddings.py.
                      Passed through to BERTopic so it can encode new
                      documents after fitting. Not used during fit_transform
                      since embeddings are always pre-computed.

    Returns
    -------
    model : fitted BERTopic instance
    topics : list of topic IDs, one per document (-1 = outlier)
    probs : list of assignment probabilities, one per document
    """
    model = make_bertopic(
        embedding_model=embedding_model,
        umap_model=make_umap(
            n_neighbors=cfg.n_neighbors,
            n_components=cfg.n_components,
            min_dist=cfg.min_dist,
            metric=cfg.umap_metric,
            random_state=cfg.seed,
            target_weight=cfg.target_weight if cfg.mode == "semi_supervised" else None,
            target_metric=cfg.target_metric,
        ),
        hdbscan_model=make_hdbscan(
            min_cluster_size=cfg.min_cluster_size,
            min_samples=cfg.min_samples,
            metric=cfg.hdbscan_metric,
            cluster_selection_method=cfg.cluster_selection_method,
            gen_min_span_tree=cfg.gen_min_span_tree,
            prediction_data=cfg.prediction_data,
        ),
        vectorizer_model=make_vectorizer(
            stop_words=cfg.stop_words,
            min_df=cfg.min_df,
            ngram_range=cfg.ngram_range,
        ),
        ctfidf_model=make_ctfidf(
            reduce_frequent_words=cfg.reduce_frequent_words,
        ),
        representation_model=cfg.representation_model or make_representation(),
        seed_topic_list=cfg.seed_topic_list if cfg.mode == "guided" else None,
        nr_topics=cfg.nr_topics,
        top_n_words=cfg.top_n_words,
        verbose=cfg.verbose,
    )

    # fit_transform call differs only for semi_supervised (y argument)
    if cfg.mode == "semi_supervised":
        topics, probs = model.fit_transform(
            documents=docs,
            embeddings=embeddings,
            y=cfg.labels,
        )
    else:
        # unsupervised: no y
        # guided: seed_topic_list is already set on the model above
        topics, probs = model.fit_transform(
            documents=docs,
            embeddings=embeddings,
        )

    return model, topics, probs