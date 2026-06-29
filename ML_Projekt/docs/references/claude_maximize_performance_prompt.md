# Prompt for Claude — Maximize F1 on the "Toxic" Binary Classification Task

> Copy everything below the line into Claude. Attach the data files
> (`toxic_data_01111.csv`, `toxic_exam_01111.csv`) and, if allowed, the scripts
> `src/train_model.py` and `src/train_max.py`.

---

## Role

You are a senior ML competition specialist. Your single objective is to **maximize the
F1-score of the POSITIVE class** on a held-out exam set. I am graded purely on that
number, so squeeze out every point. Be rigorous, quantitative, and skeptical — prefer
cross-validated evidence over intuition, and never leak the exam set.

## The task

- Binary classification. Metric: **F1 of the positive class** (class 1).
- `toxic_data_01111.csv`: 8000 rows, 16 features `f00`–`f15`, column `label` (0/1).
- `toxic_exam_01111.csv`: 2000 rows, same features, **no label** — I must predict these.
- Output: a CSV with a single column `label` (0/1), same row order as the exam file.
- Class imbalance: ~10% positive (800/8000), 9:1.
- Everything must be reproducible (`random_state=42`). The exam file must **never** be
  used during training, CV, or threshold selection — only for the final prediction.

## What I already reverse-engineered about the data (verified across 6 sibling datasets)

The data is synthetic. All 16 features start as independent `N(0,1)`; the label is a
function of the *clean* features; then three "toxic" injections are applied:

1. **MNAR missingness (the strongest signal).** Exactly two features are made missing
   (~15%) in a **label-dependent** way. In my dataset these are `f11` and `f15`:
   - Positive rate ≈ 27% when the value is missing vs ≈ 7% when present (chi² p ≈ 1e-100).
   - So `f11_missing` and `f15_missing` binary indicators are top predictors.
2. **Linear scaling (distraction).** `f10`, `f13`, `f14` are `a*x + b` rescaled
   (std ~50–200). Signal is **preserved** — just scaled.
3. **Heavy-tail injection (signal destroyed).** `f09`, `f15` get kurtosis ~17 with
   extreme outliers (±63,000). The *values* are mostly noise; for `f15` use the missing
   indicator, not the magnitude.

Never-touched / clean features: `f04`, `f08`, `f12`. **`f12` is the strongest clean
predictor** (MI ≈ 0.035). Features `f00`–`f07` are clean N(0,1) noise (MI ≈ 0).

Mutual information ranking (with missing indicators): `f11_missing` ≈ `f12` > `f15_missing`
> `f08` > `f13` > everything else ≈ 0.

## What I've already built and the results so far

Pipeline (stratified 5-fold CV, F1 of positive class, threshold tuned on out-of-fold
predictions, refit on all data, then predict exam):

- Feature engineering: `f11_missing`, `f15_missing`, `both_missing`, raw `f12`, `f08`,
  `f10`, `f13`, `f14`, `f11_value`, `sign(x)*log1p(|x|)` transforms of `f09`/`f15`,
  plus `f04`, `f03`. Median imputation. RobustScaler for linear/MLP.
- Models tried and **CV F1 (positive class)**:
  | Model | CV F1 |
  |-------|-------|
  | Logistic Regression (balanced) | 0.416 |
  | MLP (64,32) | 0.545 |
  | Gradient Boosting | 0.555 |
  | Random Forest | 0.566 |
  | **XGBoost (scale_pos_weight≈9)** | **0.650** |
- Best so far: **XGBoost, CV F1 ≈ 0.650**, tuned threshold 0.53 (OOF F1 ≈ 0.652).
  Exam predictions: 241/2000 positive (~12%).
- A larger ensemble script (`train_max.py`) blends XGB + MLP + RF + LightGBM with a
  weighted average and aggressive threshold search, but I have not confirmed it beats
  single XGBoost out-of-fold.

XGBoost shows a large train/CV gap (train F1 ≈ 0.96 vs CV ≈ 0.65), i.e. it is
**overfitting** — there is likely room to improve generalization.

## What I want from you

Give me a concrete, prioritized plan to push the **cross-validated** F1 as high as
possible, then the exact code to implement it. Specifically:

1. **Diagnose the ceiling.** Given the label is generated from clean features
   (`f12`, the missing indicators, maybe weak `f08`/`f13`), estimate a realistic upper
   bound on achievable F1 and tell me whether I'm near it. Is the remaining error
   irreducible noise (Bayes error) or model/feature gap?

2. **Try to recover the exact label rule.** The label is a deterministic-ish function of
   clean features. Probe whether it's a logistic/linear threshold on
   `f12` (+ the MNAR mechanism), a small decision rule, or an interaction. If we can
   approximate the generating rule, we beat any black-box model. Show the analysis
   (e.g., fit on the "neither-missing" subset, inspect `f12` distribution by label,
   look for a threshold).

3. **Reduce XGBoost overfitting** without hurting recall: regularization
   (`min_child_weight`, `gamma`, `reg_lambda`, `reg_alpha`, lower `max_depth`,
   subsampling), early stopping, and proper `scale_pos_weight` vs. threshold trade-off.
   Tune with Optuna (objective = CV F1 of positive class), not a hand grid.

4. **Feature engineering ideas** I may have missed: interactions with the missing
   indicators (`f12 * any_missing`, etc.), binning `f12`, dropping pure-noise features
   to cut variance, whether `f11_value` (after imputation) actually helps or hurts.

5. **Threshold + calibration.** Best practice for choosing the F1-optimal threshold so
   it generalizes (nested CV / repeated CV to avoid optimistic threshold selection),
   and whether probability calibration (Platt/Isotonic) changes the optimal threshold.

6. **Robust evaluation.** Recommend repeated stratified k-fold (e.g., 5×5) and report
   mean ± std so I don't chase CV noise. Tell me how big a CV F1 difference is actually
   meaningful given 800 positives.

7. **Ensembling** — only if it provably beats the best single model out-of-fold. Show
   how to validate the blend weights without leakage.

## Constraints & rules

- Fixed `random_state=42` everywhere; fully reproducible.
- Exam set used **only** for the final `predict` — never for fitting, CV, or threshold
  choice.
- If torn between two reasonable choices, run **both** and compare on CV F1 — don't guess.
- Every non-obvious modeling decision needs a one-line justification I can defend orally.
- Deliverables I need from you:
  1. A ranked list of changes with **expected CV-F1 impact**.
  2. Complete, runnable Python (sklearn + xgboost/lightgbm + optuna) implementing the
     top recommendations, including the CV/threshold/refit/predict flow and the exam CSV
     export.
  3. A short note on what to report at an oral exam (why each choice was made).

Ask me for anything you need (e.g., I can paste `describe()`, per-feature MI, or the
`f12`-by-label histogram). Start by telling me your hypotheses about the label rule and
the single highest-leverage change.
