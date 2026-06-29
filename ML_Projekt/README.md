# Toxic ML Project — Binary Classification

Binary classification on a deliberately "toxified" dataset. The grading metric is the
**F1-score of the positive class** on a held-out exam set. The goal is both the best
possible predictions and a defensible explanation of every modeling choice.

## Problem at a glance

| Property | Value |
|----------|-------|
| Train rows | 8000 (16 features `f00`–`f15`, binary `label`) |
| Exam rows | 2000 (same features, no label — we predict these) |
| Class balance | ~10% positive (9:1 imbalance) |
| Metric | F1 of the positive class |
| Missing values | `f11` (~15.5%) and `f15` (~15.4%) — **MNAR** |

## Key insight (reverse-engineered)

The data was generated from clean `N(0,1)` features plus three "toxic" injections:

1. **MNAR missingness** on `f11`/`f15` — missingness is label-dependent
   (~27% positive when missing vs ~7% when present). **This is the strongest signal.**
2. **Linear scaling** on `f10`/`f13`/`f14` — signal preserved, just a distraction.
3. **Heavy-tail injection** on `f09`/`f15` — destroys the original signal (use the
   missing indicator, not the values).

`f04`, `f08`, `f12` are never touched; `f12` is the strongest *clean* predictor.
Full analysis in [docs/2_data_understanding.md](docs/2_data_understanding.md).

## Repository structure (CRISP-DM)

The repo is organised along the six CRISP-DM phases, so every decision is traceable from the
business goal down to the deployed prediction.

```
ML_Projekt/
├── README.md                     # this file — project hub + how to run
├── requirements.txt              # Python dependencies
├── predictions.csv               # FINAL exam predictions (single column "label")
├── data/                         # datasets + data dictionary (see data/README.md)
├── docs/                         # CRISP-DM narrative — one markdown file per phase
│   ├── 1_business_understanding.md
│   ├── 2_data_understanding.md
│   ├── 3_data_preparation.md
│   ├── 4_modeling.md
│   ├── 5_evaluation.md
│   ├── 6_deployment.md
│   └── references/               # original brief, deep-dive notes, reference write-up
├── src/                          # code grouped by CRISP-DM phase
│   ├── data_understanding/       # eda, cross-dataset compare, reverse-engineering, diagnostics
│   ├── modeling/                 # training pipelines + ensembling + TabPFN finaliser
│   └── evaluation/               # honest CV comparison + final result report
└── artifacts/                    # tuned params + cached OOF predictions
```

## CRISP-DM phase map

| Phase | Document | Code | What happens |
|-------|----------|------|--------------|
| 1 · Business Understanding | [docs/1_business_understanding.md](docs/1_business_understanding.md) | — | goal, F1 metric, constraints |
| 2 · Data Understanding | [docs/2_data_understanding.md](docs/2_data_understanding.md) | `src/data_understanding/` | EDA, missingness, reverse-engineered generator |
| 3 · Data Preparation | [docs/3_data_preparation.md](docs/3_data_preparation.md) | feature engineering in pipelines | 13 engineered features, imputation, scaling |
| 4 · Modeling | [docs/4_modeling.md](docs/4_modeling.md) | `src/modeling/` | Optuna-tuned XGB/LGBM/Cat + MLP/RF/TabPFN stack |
| 5 · Evaluation | [docs/5_evaluation.md](docs/5_evaluation.md) | `src/evaluation/` | honest repeated CV, threshold tuning, sanity checks |
| 6 · Deployment | [docs/6_deployment.md](docs/6_deployment.md) | `finalize_with_tabpfn.py` | refit on all data → predictions.csv |

**Headline:** honest cross-validated **F1 = 0.6938 ± 0.026** (6-model stack with TabPFN), up
from the ~0.654 engineered-XGB baseline.

## Setup

```powershell
# from the project root, with the venv activated
python -m pip install -r requirements.txt
```

**The datasets are not in the repo** — they are distributed as a separate ZIP (attached to
the repo's GitHub **Releases**) to keep clones small. Download it and extract into `data/`;
see [data/README.md](data/README.md#getting-the-data). The `01111` pair alone is enough to
run the full pipeline.

## How to run

All scripts resolve paths from their own location, so they run from any working directory.

```powershell
# 2 · Data Understanding
python src\data_understanding\eda_analysis.py        # EDA console report
python src\data_understanding\reverse_engineer.py    # decode the injection pattern
python src\data_understanding\compare_datasets.py    # cross-dataset comparison

# 4 · Modeling — main 5-model pipeline -> predictions.csv (honest CV 0.6811)
python src\modeling\train_ultra.py

# 6 · Deployment — committed best: 6-model + TabPFN -> predictions.csv (honest CV 0.6938)
python src\modeling\finalize_with_tabpfn.py

# 5 · Evaluation — headline metrics + live verification of predictions.csv
python src\evaluation\final_result.py
```

## Methodology summary

1. **EDA** — shapes, missingness, class balance, per-feature distributions, mutual
   information vs. label.
2. **Feature engineering** — binary `f11_missing` / `f15_missing` indicators (top
   signal), keep clean `f12`/`f08`, `sign·log1p` transform for heavy-tailed `f09`/`f15`.
3. **Modeling** — Logistic Regression, Random Forest, Gradient Boosting, MLP, and
   XGBoost/LightGBM, all evaluated with the same stratified 5-fold CV and F1 metric.
4. **Threshold tuning** — sweep the decision threshold on out-of-fold predictions
   (default 0.5 is suboptimal at 10% positives).
5. **Final fit** — refit the winning model/threshold on all training data before
   predicting the exam set. The exam file is never used during training or threshold
   selection.

`random_state=42` is fixed everywhere for reproducibility.
