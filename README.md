# Crash-Narratives NLP

Code for the master's thesis *Information Extraction and Pattern Discovery of
Danish Police-Written Crash Narratives Using Domain-Specific Language Models and
Semantic Topic Modelling* (DTU).

The work has two tracks:

- **Track 1 — classification.** A fine-tuned Danish BERT (`danish-bert-botxo`)
  for supervised classification of the main crash situation.
- **Track 2 — topic modelling.** A semi-supervised BERTopic pipeline for
  unsupervised pattern discovery.

## Layout

- `src/` — reusable toolbox. These files are imported, never run directly.
- `analysis/` — the notebooks you actually run, one per track.
- `data/` — `raw/` holds the restricted VD extract (never committed);
  `synthetic/` holds a safe-to-share fake dataset that everything runs against
  by default.
- `models/` — saved model weights (written when a notebook runs; gitignored).
- `results/` — figures, tables, and metrics written by the notebooks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Track 1: open `analysis/track1_classification/bert_classification.ipynb` and run
the cells top to bottom. It loads the data, fine-tunes the model, saves it to
`models/track1_classification/`, and writes results to
`results/track1_classification/`.

By default the pipeline reads from `data/synthetic/`, so the repository is
reproducible without access to the restricted VD database. To run on the real
data, point `DATA_FOLDER` in `src/config.py` at `data/raw/`.

## Note on data

The real crash narratives come from the Vejdirektoratet (VD) crash database and
are access-restricted, so they are not included here. The synthetic dataset
mirrors the real schema column-for-column so the code runs unchanged.

## Track 2 — BERTopic (running on synthetic data)

Every Track 2 notebook reads its data through one configurable merger, so the
data paths live in `src/config.py` rather than inside the notebooks.

First generate the synthetic stand-in data (schema-correct, random values, so
the notebooks run but produce no real findings):

```bash
python data/synthetic/make_synthetic_data.py
```

Then open any notebook under `analysis/track2_bertopic/`. The case studies and
the general cluster analysis build their dataframe with:

```python
from analysis_data import build_analysis_dataframe
df_acc   = build_analysis_dataframe(config="main_0.3")   # accident level
df_merged = build_analysis_dataframe(level="person")     # person level
```

`config` selects which semi-supervised run under `results_semi/` is merged on
(`main_0.3`, `report_accident_0.25`, `all_0.2`). To run on the real VD data,
repoint the paths in `src/config.py` from `data/synthetic/...` to `data/raw/...`.

The notebooks use flat imports (`from analysis_data import ...`); each case study
begins with a short setup cell that puts `src/` on the path automatically.

### Producing `results_semi/` (the topic models)

`results_semi/` lives under `results/track2_bertopic/`, not `data/`, because the
topic assignments are model output. They are produced by:

```
analysis/track2_bertopic/fit_bertopic_models.ipynb
```

which fits one fully unsupervised model (`unsupervised/`) and the semi-supervised
runs (`main_0.3/`, `report_accident_0.25/`, `all_0.2/`) and writes the
`document_topics.csv` / `topic_info.csv` / `topic_words.json` each analysis
notebook reads. `make_synthetic_data.py` also writes these same folders with
stand-in values, so the analysis notebooks run without fitting first; run
`fit_bertopic_models.ipynb` to regenerate them for real.

Track 2 run order: `make_synthetic_data.py` → (optionally) `fit_bertopic_models.ipynb`
→ any analysis notebook.
