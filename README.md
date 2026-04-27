# Olfactory Intelligence

A machine learning project that predicts fragrance success using the Fragrantica dataset.  
Given a perfume's composition (notes, accords, metadata), the model predicts whether it will be a top-30% commercial success.

---

## Project Goals

- Build a binary classification model that predicts perfume success from composition alone
- Understand which fragrance notes and accords most strongly drive success (SHAP explainability)

---

## Dataset

**Source:** Fragrantica web-scraped dataset (`data/raw/fragrantica_raw.csv`)  
**Format:** CSV
**Size:** ~23,846 perfumes after cleaning  
**Key columns:**

| Column | Description |
|---|---|
| `url` | Fragrantica page URL |
| `Perfume` | Perfume name |
| `Brand` | Brand name |
| `Country` | Brand country of origin |
| `Gender` | Target gender (Men / Women / Unisex) |
| `Rating Value` | Average user rating (0–5) |
| `Rating Count` | Number of ratings |
| `Year` | Release year |
| `Top` / `Middle` / `Base` | Fragrance notes (pipe-separated strings) |
| `Perfumer1` / `Perfumer2` | Perfumer names |
| `mainaccord1`–`mainaccord5` | Up to 5 dominant accords |

> **The raw data is not committed to this repository.** Download it separately and place it at `data/raw/fragrantica_raw.csv`.

---

## Directory Structure

```
olfactory-intelligence/
│
├── data/
│   ├── raw/                        # Original Fragrantica CSV (git-ignored)
│   ├── interim/                    # Intermediate transformations (git-ignored)
│   └── processed/                  # Clean + feature-engineered data (git-ignored)
│       ├── fragrantica_clean.csv   # Output of notebook 02
│       ├── fragrantica_features.csv
│       └── fragrantica_features.parquet   # Output of notebook 03 (model input)
│
├── models/
│   ├── rf_composition.pkl          # Trained Composition-only RF (git-ignored)
│   ├── rf_full.pkl                 # Trained Full RF model (git-ignored)
│   └── model_config.json           # Best threshold config (tracked)
│
├── notebooks/
│   ├── 01_data_audit.ipynb         # Initial exploration of raw data
│   ├── 02_data_cleaning.ipynb      # Cleaning pipeline
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb           # Model training, evaluation, threshold tuning
│   └── 05_shap_analysis.ipynb      # SHAP explainability analysis
│
├── reports/
│   ├── figures/                    # All plots (git-ignored)
│   └── results/                    # CSVs + JSON artifacts (git-ignored)
│
├── src/
│   ├── data/
│   │   ├── clean_data.py           # Cleaning functions (used by notebook 02)
│   │   ├── build_features.py       # Feature pipeline script (scaffolded)
│   │   └── load_data.py            # Data loader (scaffolded)
│   ├── models/
│   │   ├── __init__.py             # Exports modeling functions
│   │   ├── modeling.py             # RF training, evaluation, threshold search
│   │   ├── train_model.py          # CLI training script (scaffolded)
│   │   ├── evaluate_model.py       # CLI evaluation script (scaffolded)
│   │   └── explain_model.py        # CLI SHAP script (scaffolded)
│   ├── graph/
│   │   ├── build_graph.py          # Ingredient co-occurrence graph (scaffolded)
│   │   └── graph_features.py       # Node2Vec graph embeddings (scaffolded)
│   └── utils/
│       ├── paths.py                # Centralised path constants
│       └── helpers.py              # Shared utilities (scaffolded)
│
├── app/
│   └── streamlit_app.py            # Fragrance recommender UI (scaffolded)
│
├── requirements.txt
├── .gitignore
├── CLAUDE.md                       # AI assistant context document
└── README.md
```

---

## Pipeline Overview

```
data/raw/fragrantica_raw.csv
        │
        ▼  notebook 02 / src/data/clean_data.py
data/processed/fragrantica_clean.csv
        │
        ▼  notebook 03
data/processed/fragrantica_features.parquet
        │
        ├──▶  notebook 04  ──▶  models/rf_*.pkl
        │                  ──▶  reports/results/model_comparison.csv
        │                  ──▶  reports/figures/rf_*_confusion_matrix.png
        │                  ──▶  reports/figures/rf_*_top25_importance.png
        │
        └──▶  notebook 05  ──▶  reports/figures/shap_summary_*.png
                           ──▶  reports/figures/shap_dependence_*.png
                           ──▶  reports/figures/waterfall_*.png
                           ──▶  reports/results/shap_*.csv
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/olfactory-intelligence.git
cd olfactory-intelligence

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the raw data
# Copy fragrantica_raw.csv into data/raw/

# 5. Run notebooks in order
jupyter notebook
```

---

## Model Results

**Task:** Binary classification — top 30% of `rating_value × log(rating_count)` = "successful"  
**Success threshold:** 22.42 (score value)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Composition Only (notes + accords) | 71.7% | 56.4% | 25.6% | 35.3% |
| Full Model (default threshold 0.50) | 74.9% | 61.7% | 43.3% | 50.9% |
| **Full Model (tuned threshold 0.39)** | **70.7%** | **50.8%** | **71.7%** | **59.5%** |

**Feature engineering:** 367 features — one-hot notes, accords, gender, metadata (age, name length, note counts), brand aggregates  
**Algorithm:** Random Forest (300 estimators, balanced class weights, 80/20 train-test split)

---

## Generated Outputs

### Figures (`reports/figures/`)

| File | Description |
|---|---|
| `success_formula_comparison_kde.png` | KDE comparison of two success score definitions |
| `rf_composition_confusion_matrix.png` | Confusion matrix for composition-only model |
| `rf_composition_top25_importance.png` | Top 25 feature importances (composition model) |
| `rf_full_confusion_matrix.png` | Confusion matrix for full model (default threshold) |
| `rf_full_top25_importance.png` | Top 25 feature importances (full model) |
| `rf_full_tuned_confusion_matrix.png` | Confusion matrix for tuned threshold |
| `shap_summary_beeswarm_full_model.png` | SHAP beeswarm — feature impact distribution |
| `shap_summary_bar_full_model.png` | SHAP bar — mean absolute SHAP per feature |
| `shap_dependence_*.png` | SHAP dependence plots for top-3 features |
| `waterfall_*.png` | Per-perfume waterfall explanation |

### Results (`reports/results/`)

| File | Description |
|---|---|
| `model_comparison.csv` | Accuracy / precision / recall / F1 for all 3 models |
| `modeling_artifacts.json` | Feature column lists, train/test indices, thresholds |
| `rf_full_feature_importance.csv` | Gini feature importances for the full model |
| `rf_composition_feature_importance.csv` | Gini feature importances for composition model |
| `test_predictions.csv` | Predicted probabilities on the test set |
| `shap_feature_importance_full_model.csv` | Mean absolute SHAP for all features |
| `shap_top20_features_full_model.csv` | Top 20 SHAP features |
| `shap_report_table_top15.csv` | Ranked top-15 for report writing |
| `shap_prediction_sample_full_model.csv` | Probabilities + labels for the SHAP sample |

---

## Tech Stack

| Library | Purpose |
|---|---|
| pandas, numpy | Data manipulation |
| scikit-learn | Random Forest, preprocessing, metrics |
| shap | Model explainability |
| matplotlib, plotly | Visualisation |
| networkx, python-louvain, node2vec | Graph analysis (planned) |
| sentence-transformers | Text embeddings (planned) |
| umap-learn, hdbscan | Dimensionality reduction / clustering (planned) |
| streamlit | Web app (planned) |
| joblib | Model serialisation |
