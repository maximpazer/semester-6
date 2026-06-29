# Phase 6 — Deployment

> CRISP-DM Phase 6. *How is the model delivered and reproduced?*
> Producer: [`finalize_with_tabpfn.py`](../src/modeling/finalize_with_tabpfn.py).

## 6.1 The deliverable

[`../predictions.csv`](../predictions.csv) — the graded artifact.

| Property | Value |
|----------|-------|
| Rows | 2 000 (same order as `toxic_exam_01111.csv`) |
| Columns | single column `label` (0/1) |
| Predicted positive | 232 (11.6 %) |
| Decision threshold | 0.295 (global) |
| Model | 6-model stack (XGB, LGBM, CatBoost, MLP, RF, TabPFN) → LogisticRegression meta |
| Honest CV F1 | 0.6938 ± 0.026 |

## 6.2 How it is produced

The final fit follows the no-leakage rule: meta-learner and threshold are chosen on
**training OOF only**, then the bases are refit on **all** training data and applied **once**
to the exam set.

```powershell
# 1. install dependencies (one-time)
python -m pip install -r requirements.txt
python -m pip install "tabpfn==2.2.1"     # open-weights build

# 2. regenerate predictions.csv (reuses cached OOF + tuned params; ~8 min)
python src\modeling\finalize_with_tabpfn.py
```

It reuses the cached artifacts so it does **not** re-run Optuna or recompute every OOF:

| Artifact | Contents |
|----------|----------|
| `artifacts/tuned_params.json` | Optuna-tuned XGB/LGBM/CatBoost configs |
| `artifacts/base_oof.npz` | tuned 5-model OOF predictions (+ y) — for meta + threshold |
| `artifacts/tabpfn_oof.npz` | TabPFN OOF predictions — the 6th base |

Only the final exam predictions are computed fresh (5 base refits + 1 TabPFN fit ≈ 474 s CPU).

## 6.3 Verify a delivered `predictions.csv`

```powershell
python src\evaluation\final_result.py
```

This prints the headline metrics and live-verifies that the committed predictions still match
the MNAR signal (missing `f11`/`f15` → far higher positive rate — see [Phase 5](5_evaluation.md)).

## 6.4 Reproducibility notes

- `random_state = 42` everywhere.
- The cached `artifacts/` (`tuned_params.json` + `*_oof.npz`, ~0.5 MB total) are committed so
  this step reproduces in ~8 min. Delete the `*_oof.npz` to force a full recompute
  (Optuna + OOF, ~30 min).
- The exam file is never touched before this final step.

## 6.5 Optional future levers

- Bake TabPFN into `train_ultra.py` canonically (with OOF cache reuse).
- Re-add pseudo-labelling on top of the 6-model stack.
- Feed TabPFN the raw NaN-native features (it handles missing values natively) instead of the
  median-imputed 13.

← Back to **[README](../README.md)** · **[Phase 1](1_business_understanding.md)**.
