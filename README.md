# Crash-Narratives NLP

Information extraction and pattern discovery from Danish police-written crash
narratives, using domain-specific language models and semantic topic
modelling. This is the code accompanying the DTU master's thesis *Information
Extraction and Pattern Discovery of Danish Police-Written Crash Narratives Using
Domain-Specific Language Models and Semantic Topic Modelling*.

The narratives, a free-text field written by police officers, and structured data has been accessed from Vejman.dk. 
The project asks what the narratives reveal that the structured fields do not, and approaches it from two directions.

## Overview

The work is organised into two tracks that share a common data layer.

**Track 1 — supervised classification.** A fine-tuned Danish BERT
(`Maltehb/danish-bert-botxo`) predicts the main crash situation
(`main_situation_class`) directly from the narrative text. Integrated Gradients
attribution shows which words drive each prediction, so the model's behaviour can
be inspected rather than taken on trust.

**Track 2 — semantic topic modelling.** A BERTopic pipeline
(`KennethTM/MiniLM-L6-danish-encoder` embeddings, UMAP, HDBSCAN, and c-TF-IDF), run both fully
unsupervised and semi-supervised, discovers latent structure in the narratives.
The standout finding, an active hit-and-run cluster with no corresponding
structured VD field, lives in the Track 2 case studies.

The two tracks are complementary. The narrative methods enrich and validate the
structured data; they do not replace it.

> **Important about the committed results:** the data and result files included in
> this public repository are synthetic stand-ins. Any figures, topic labels,
> metrics, topic assignments, or case-study outputs produced from them are only
> examples showing that the code runs. They are **not** the thesis findings and
> should not be interpreted as crash-safety evidence.

## 📂 Project structure

```
crash-narratives-nlp/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                         # real VD data (gitignored, never shared)
│   ├── synthetic/                   # schema-correct stand-ins (safe to commit)
│   │   ├── make_synthetic_data.py   # regenerates everything below
│   │   ├── base/                    # police narratives
│   │   ├── element/                 # element / person rows (age, type)
│   │   ├── lable_analysis_1..4.xlsx # VD accident-level fields
│   │   ├── bad_uheld.xlsx           # injury severity per element
│   │   ├── df_coords.parquet        # kommune code + coordinates
│   │   ├── drunk_driving_data.xlsx  # alcohol flag (case study)
│   │   └── criminal_rates_with_codes.xlsx
│   └── README.md                    # post-load schema
│
├── src/                             # the toolbox: imported, never run directly
│   ├── config.py                    # paths, seed, configuration selection
│   ├── data_load.py                 # load + clean accident data
│   ├── split_data.py                # stratified train/val/test split
│   ├── embedding.py                 # embedding + classifier loaders
│   ├── analysis_data.py             # the merger: build_analysis_dataframe()
│   ├── bertopic_components.py       # BERTopic builders + run_bertopic()
│   ├── performance_index_bertopic.py# coherence, diversity, Jaccard metrics
│   ├── sensitivity_analysis.py      # weight / label sensitivity
│   ├── topic_vd_analysis.py         # topic vs VD-field added-value analysis
│   ├── rear_end_analysis.py         # rear-end case-study logic
│   ├── stop_words.py                # loads final_stopwords.txt
│   ├── final_stopwords.txt          # Danish stop word list
│   └── classification/              # Track 1 internals
│       ├── dataset.py               # tokenisation (HuggingFace datasets)
│       ├── trainer.py               # class weights, weighted loss, metrics
│       └── attribution.py           # Integrated Gradients
│
├── analysis/                        # the notebooks you actually run
│   ├── track1_classification/
│   │   └── bert_classification.ipynb        # train + analyse, one file
│   └── track2_bertopic/
│       ├── fit_bertopic_models.ipynb        # produces results_semi/
│       ├── mteb_embedding_benchmark.ipynb   # Danish vs English encoder
│       ├── general_cluster_analysis.ipynb   # all-clusters appendix
│       ├── hyperparameter_analysis/
│       │   ├── engineering_exploration.ipynb   # UMAP / HDBSCAN settings
│       │   └── sensitivity_analysis.ipynb      # semi-supervised weights/labels
│       └── case_studies/
│           ├── topic11_hit_and_run.ipynb
│           ├── rear_end.ipynb
│           ├── drunk_driving.ipynb
│           └── solo_bicycle_fall.ipynb
│
├── models/                          # saved model weights (gitignored)
└── results/
    ├── track1_classification/       # metrics, confusion matrix, IG tables
    └── track2_bertopic/
        └── results_semi/            # topic outputs, one folder per model
            ├── unsupervised/
            ├── main_0.3/
            ├── report_accident_0.25/
            └── all_0.2/
```

The layout follows one rule that keeps it legible: every folder has a single
role.

`src/` is the **toolbox**. These modules are imported, never run on their own.
The three loaders at the top (`data_load`, `split_data`, `embedding`) are shared
by both tracks; `analysis_data` is the merger both tracks read through; the
remaining modules hold the BERTopic and case-study logic; and `classification/`
holds the Track 1 internals.

`analysis/` is what you **run**. Each notebook opens, runs top to bottom, and
produces output. They stay short because the heavy logic lives in `src/`.

`data/`, `models/` and `results/` are **input and output**. You do not edit them
by hand; the notebooks read from `data/` and write into `models/` and
`results/`. In this public copy, the committed files under `data/synthetic/` and
`results/` are execution fixtures: they preserve the expected schema and folder
layout, but their values are randomly generated stand-ins rather than valid
research results.

## 🧩 The two tracks

### Track 1 — classification

`analysis/track1_classification/bert_classification.ipynb` is the only file you
run for Track 1. It loads and cleans the data, builds a stratified split,
fine-tunes the class-weighted classifier, saves the best checkpoint to
`models/track1_classification/`, and writes the confusion matrix, per-class
metrics, and Integrated Gradients token tables into
`results/track1_classification/`.

### Track 2 — BERTopic

Every Track 2 analysis reads topic assignments produced by
`fit_bertopic_models.ipynb`, which fits one fully unsupervised model and the
semi-supervised runs and writes them to `results/track2_bertopic/results_semi/`.
The remaining notebooks consume those outputs: the embedding benchmark, the
hyperparameter and sensitivity exploration, the all-clusters appendix, and the
four case studies.

## 🗄️ Data and the merger

The real narratives come from the VD crash database and are access-restricted, so
`data/raw/` ships empty and gitignored. Everything runs by default against
`data/synthetic/`, a set of schema-correct stand-in files with random values. The
synthetic files are designed to exercise joins, notebooks, plots, and output
writers; they are not statistically meaningful. Results generated from these
files are demonstration outputs, not correct thesis results.

```bash
python data/synthetic/make_synthetic_data.py     # (re)generate the stand-ins
```

Every analysis notebook builds its dataframe through one configurable merger
rather than loading data inline, so all paths live in `src/config.py`:

```python
from analysis_data import build_analysis_dataframe

df_acc    = build_analysis_dataframe(config="main_0.3")   # one row per accident
df_merged = build_analysis_dataframe(level="person")      # person-level rows
```

`config` selects which semi-supervised run is merged on (`main_0.3`,
`report_accident_0.25`, `all_0.2`). To run on the real data, repoint the paths in
`src/config.py` from `data/synthetic/...` to `data/raw/...`; nothing else
changes. The post-load schema is documented in `data/README.md`.

## 🛠️ Installation

```bash
conda create -n crash-nlp python=3.10
conda activate crash-nlp
pip install -r requirements.txt
```

`requirements.txt` covers both tracks: `torch`, `transformers`, `datasets`,
`captum` (Track 1); `bertopic`, `umap-learn`, `hdbscan`, `sentence-transformers`,
`gensim`, `mteb` (Track 2); and the shared `scikit-learn`, `pandas`, `numpy`,
`scipy`, `matplotlib`, `seaborn`, `ruptures`, `statsmodels`, `openpyxl`,
`pyarrow`.

## 🚀 Running

**Track 1.** Open `analysis/track1_classification/bert_classification.ipynb` and
run it top to bottom.

**Track 2.** Run in this order:

| Step | File | Produces |
| ---- | ---- | -------- |
| 1 | `data/synthetic/make_synthetic_data.py` | schema-correct synthetic input data + stand-in topic outputs |
| 2 | `analysis/track2_bertopic/fit_bertopic_models.ipynb` | regenerated `results_semi/` topic outputs for whichever data source is configured |
| 3 | any notebook under `analysis/track2_bertopic/` | analysis outputs for the configured data source |

Step 2 is optional on synthetic data, since step 1 already writes stand-in topic
outputs so the analysis notebooks run without fitting. If you run Step 2 while
`src/config.py` still points at `data/synthetic/`, the regenerated models are
still synthetic demonstration results. To reproduce the thesis analyses, first
point `src/config.py` at the restricted real VD data and then regenerate the
models and downstream outputs.

The notebooks use flat imports (`from analysis_data import ...`); each one begins
with a short setup cell that puts `src/` on the path automatically, so they run
correctly from any working directory.

## 📊 Outputs

* **Model checkpoints** → `models/`
* **Track 1 metrics, confusion matrix, IG tables** → `results/track1_classification/`
* **Track 2 topic assignments** → `results/track2_bertopic/results_semi/<config>/`
  (`document_topics.csv`, `topic_info.csv`, `topic_words.json`)
* **Case-study figures and tables** → written by each notebook into `results/`

When the repository is run with the default synthetic configuration, these
outputs are placeholders. They confirm that the pipeline executes and that files
are written in the expected format; they do not contain the validated numbers,
figures, clusters, or conclusions from the thesis.

## 📦 `src/` module reference

| Module | Role |
| ------ | ---- |
| `config.py` | Central paths, seed, and configuration selection |
| `data_load.py` | Load and clean the raw accident data |
| `split_data.py` | Stratified train / validation / test split |
| `embedding.py` | Load the sentence-embedding and classifier models |
| `analysis_data.py` | The merger: builds the accident- or person-level dataframe |
| `bertopic_components.py` | BERTopic component factories and `run_bertopic()` |
| `performance_index_bertopic.py` | Coherence, diversity, and Jaccard metrics |
| `sensitivity_analysis.py` | Semi-supervised weight and label sensitivity |
| `topic_vd_analysis.py` | Compares topics against structured VD fields |
| `rear_end_analysis.py` | Profile, trend, and changepoint logic for the rear-end study |
| `stop_words.py` | Loads the Danish stop word list |
| `classification/dataset.py` | Tokenisation via the HuggingFace `datasets` library |
| `classification/trainer.py` | Class weights, weighted loss, and metrics |
| `classification/attribution.py` | Integrated Gradients token attribution |

## A note on data

The real crash narratives are restricted and are not included. The synthetic
dataset mirrors the real schema closely enough for the code to run unchanged;
only the paths in `src/config.py` differ between the two. Because the synthetic
values are random and intentionally artificial, any “results” shown from the
default repository state are examples of file structure and workflow only, not
correct empirical findings.
