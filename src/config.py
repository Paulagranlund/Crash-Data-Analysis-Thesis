"""Central configuration for Track 1 (supervised BERT classification).

All tunable constants and shared paths live here so that the training
script and the analysis notebook read from a single source of truth.
"""
import os

# Repository root, so every path below is absolute regardless of the working
# directory a notebook is launched from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "Maltehb/danish-bert-botxo"  # Danish BERT backbone, lowercased text
MAX_LENGTH = 256                          # maximum tokens processed per narrative

# Columns
TEXT_COL = "police_narrative"
LABEL_COL = "main_situation_class"

# Paths (relative to the repository root).
# The pipeline points at the synthetic dataset by default so the repository
# is reproducible without access to the restricted VD database.
DATA_FOLDER = os.path.join(_REPO_ROOT, "data", "synthetic")
MODEL_DIR = os.path.join(_REPO_ROOT, "models", "track1_classification")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results", "track1_classification")


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for random, numpy and torch (CPU and CUDA)."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Track 2: BERTopic data paths and configuration selection ──────────────────
# Centralised so every analysis notebook reads its data through one place.
# Point these at data/raw/... to run on the real VD data instead of synthetic.

_SYNTH = os.path.join(_REPO_ROOT, "data", "synthetic")

BASE_DATA_FOLDER = os.path.join(_SYNTH, "base")          # police narratives (input)
ELEMENT_DATA_FOLDER = os.path.join(_SYNTH, "element")    # element/person rows (input)
VD_FIELDS_FILES = [os.path.join(_SYNTH, f"lable_analysis_{i}.xlsx") for i in [1, 2, 3, 4]]
SEVERITY_FILE = os.path.join(_SYNTH, "bad_uheld.xlsx")   # PERSONSKADE per element (input)
COORDS_FILE = os.path.join(_SYNTH, "df_coords.parquet")  # kommune + x/y (input)

# BERTopic outputs are RESULTS, so they live under results/, not data/.
# Produced by analysis/track2_bertopic/fit_bertopic_models.ipynb (or the
# synthetic stand-ins from make_synthetic_data.py).
RESULTS_SEMI_DIR = os.path.join(_REPO_ROOT, "results", "track2_bertopic", "results_semi")

# Semi-supervised topic configurations -> results_semi subfolder
CONFIG_DIRS = {
    "main_0.3": "main_0.3",
    "report_accident_0.25": "report_accident_0.25",
    "all_0.2": "all_0.2",
}

# One-off case-study inputs
DRUNK_FLAG_FILE = os.path.join(_SYNTH, "drunk_driving_data.xlsx")
CRIME_RATES_FILE = os.path.join(_SYNTH, "criminal_rates_with_codes.xlsx")
