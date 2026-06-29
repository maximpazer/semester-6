# Phase 4 — Modeling

> CRISP-DM Phase 4. *Which models, how tuned, how combined?*
> Code: [`src/modeling/`](../src/modeling/).
> Source: [references/maximize_performance_results.md](references/maximize_performance_results.md).

## 4.1 Validation protocol (the foundation)

Everything is judged by **the same** honest cross-validation so scores are comparable:

- `StratifiedKFold(k=5)` for out-of-fold (OOF) predictions.
- `RepeatedStratifiedKFold` (e.g. 5×10) for the **noise gate**: a change is kept *only* if it
  beats the previous score by more than the fold std. This is what makes chasing small gains
  safe on a 2 000-row exam.
- Class imbalance handled with `scale_pos_weight ≈ 9` (trees) / `class_weight='balanced'` (linear).

## 4.2 Base learners

| Model | Role | OOF F1 |
|-------|------|--------|
| Logistic Regression | baseline / linear reference | — |
| Random Forest | half-credit benchmark | 0.5702 |
| **XGBoost** | gradient boosting | 0.6685 |
| **LightGBM** | gradient boosting | 0.6721 |
| **CatBoost** | gradient boosting (best single) | **0.6837** |
| MLP | neural upper benchmark | 0.6435 |
| **TabPFN** | pretrained tabular foundation model | 0.6667 |

XGB / LGBM / CatBoost were tuned with **Optuna** (40 / 40 / 25 trials), each maximising
5-fold OOF best-F1. This hyper-parameter search produced **the bulk of the lift** over the
~0.654 baseline.

## 4.3 Stacking (the winning combination)

The OOF predictions of the base learners are fed to a **LogisticRegression meta-learner**
(via `cross_val_predict`, so it stays honest):

| Combiner | OOF F1 |
|----------|--------|
| best single (CatBoost) | 0.6837 |
| equal-weight blend | 0.6739 |
| **5-model stack** | **0.6866** |
| **6-model stack (+ TabPFN)** | **0.6945** |

Stacking beats both the best single model and the naive blend because the meta-learner
**weights** the bases and exploits their disagreement.

## 4.4 TabPFN as a 6th base learner

TabPFN is individually *weaker* than CatBoost (0.6667 vs 0.6837) but makes **different
errors**, so the meta-learner extracts genuine diversity. Added under the same meta-learner it
lifted the honest stack **0.6811 → 0.6938** (paired Δ = +0.0127, ~9× SE — significant). This is
the current **committed** model ([`finalize_with_tabpfn.py`](../src/modeling/finalize_with_tabpfn.py)).

> Use open-weights `tabpfn==2.2.1` (newer versions gate weights behind a login and crash on
> Windows). CPU on > 1000 rows needs `TABPFN_ALLOW_CPU_LARGE_DATASET=1`.

## 4.5 Pseudo-labelling

Confident exam rows (`P > 0.90` positive / `< 0.03` negative) were added to training. It
helped the 5-model stack (+0.004) but was **dropped** for the committed 6-model stack, which
already dominates it.

## 4.6 Scripts

| Script | Purpose |
|--------|---------|
| [`train_model.py`](../src/modeling/train_model.py) | first full pipeline (LR/RF/GB/MLP), didactic baseline |
| [`train_max.py`](../src/modeling/train_max.py) | tuned ensemble + weighted blend |
| **[`train_ultra.py`](../src/modeling/train_ultra.py)** | **main pipeline**: features → Optuna → 5-model stack → threshold → predictions (CV 0.6811) |
| [`try_tabpfn.py`](../src/modeling/try_tabpfn.py) | TabPFN experiment, caches OOF |
| **[`finalize_with_tabpfn.py`](../src/modeling/finalize_with_tabpfn.py)** | **committed best**: 6-model stack → predictions.csv (CV 0.6938) |
| [`reproduce_stack.py`](../src/modeling/reproduce_stack.py) | minimal reproduction of the stack |

→ Continue with **[Phase 5 — Evaluation](5_evaluation.md)**.
