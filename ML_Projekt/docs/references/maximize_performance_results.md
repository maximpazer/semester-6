# Maximizing F1 on `toxic_exam_01111` — Results

**Goal:** push the positive-class F1 above the long-standing ~0.654 ceiling.
**Outcome:** honest cross-validated F1 = **0.6938 ± 0.026** with TabPFN added as a 6th base
learner (was **0.6811 ± 0.027** for the 5-model stack; genuine +0.040 over the 0.654 baseline).
`predictions.csv` regenerated. Pipelines: `src/train_ultra.py` (5-model) →
`src/finalize_with_tabpfn.py` (current committed 6-model). See the TabPFN section at the end.

---

## TL;DR — where the gain came from

| Lever | Honest effect | Kept? |
|---|---|---|
| Feature engineering (forward-selected) | +0.003 (`f11_abs` only) | yes, tiny |
| **Optuna tuning of XGB / LGBM / CatBoost** | **the bulk of the lift** | **yes** |
| **Stacking (LogisticRegression meta-learner)** | OOF 0.6866 > best single 0.6837 > blend 0.6739 | **yes** |
| **TabPFN as 6th base learner** | **+0.0127 paired CV (0.6811 → 0.6938, ~9× SE)** | **yes (committed)** |
| Per-stratum thresholds | −0.008 (overfits small strata) | **no** |
| Prior-aware top-K threshold | ties global | no (global used) |
| Pseudo-labeling (confident exam rows) | +0.004 | yes |
| External / enriched / Lethal data | proven 0 or negative | **no** (excluded) |

The breakthrough was **not** a hidden data leak — it was disciplined hyper-parameter
search + model stacking. The label is intrinsically probabilistic (max `P(pos|f12) ≈ 0.45`),
so there is a real ceiling; we got close to it honestly.

---

## Method (validated against a repeated-CV noise gate)

Every change had to beat the previous score by more than the 5×N-fold CV std, otherwise it
was discarded as fold noise. This is what makes "chasing every 0.005" safe on a 2000-row exam.

1. **Diagnostics** (`src/diagnostics.py`) — missingness audit (only `f11`,`f15`), per-class
   variance/nonlinear leak hunt, residual-correlation hunt, Bayes-ceiling estimate
   (internal repeated-CV ≈ 0.62–0.65; external `g(f12)` adds nothing).
2. **Forward feature selection** — start from the proven 12-feature set, add a candidate only
   if repeated-CV (250-tree XGB, 5×2) gain > 0.0015. Result: only `f11_abs` qualified →
   **13 features**. Confirms the feature frontier is essentially exhausted.
3. **Optuna** — 40 trials XGB, 40 LGBM, 25 CatBoost, each maximizing 5-fold OOF best-F1.
4. **Stacking** — OOF of {XGB, LGBM, CatBoost, MLP, RF} → LogisticRegression meta
   (cross-val-predicted, so honest). Stack beat the best single model and the equal blend.
5. **Threshold strategy** — honest repeated-CV (5×10) comparison of global vs per-stratum
   (missing/present) vs prior-aware top-K. **Global won**; per-stratum overfit the small strata.
6. **Pseudo-labeling** — add confident exam rows (prob > 0.90 pos / < 0.03 neg) to training;
   kept only because repeated-CV improved (+0.004).

## Final selected features (13)
`m11, m15, both_miss, f12, f12_sq, f08, f10, f13, f14, f11_val, f15_slog, f09_slog, f11_abs`

## Final numbers
- Optuna OOF best-F1: XGB 0.6685 · LGBM 0.6721 · CatBoost 0.6837
- OOF F1: MLP 0.6435 · RF 0.5702 · **STACK 0.6866** · blend 0.6739
- Honest threshold CV: **global 0.6811 ± 0.0270** · prior 0.6803 · stratum 0.6730
- Pseudo-labeling: 0.6779 vs 0.6735 → used
- `predictions.csv`: 247 positive (12.35%), global threshold 0.295
- Verification (matches the MNAR signal): pred-pos | `f11` missing = 39.0% ·
  `f11` present = 7.8% · `f15` missing = 38.4%

## Honesty caveat
The 0.6811 threshold selection is cross-validated and honest, but the Optuna hyper-parameters
were chosen to maximize the OOF best-F1 (in-sample threshold), which adds mild optimism.
Realistic graded exam F1 is therefore likely **~0.66–0.68** — still a clear, real gain over 0.654.

## Dead ends (do not revisit)
External/enriched datasets, cross-dataset row lookup, the Lethal track, bloated transform
features (the 8 pure-noise clean feats `f00–f07` plus most magnitude/sign transforms), and
per-stratum thresholds.

---

## Update (2026-06-15) — TabPFN as a 6th base learner (current committed best)

Adding **TabPFN** (a pretrained tabular foundation model) as a 6th base learner under the same
LogisticRegression meta lifted the honest stack from **0.6811 → 0.6938**. This is the result a
friend's "GitHub model ~0.69" hinted at, reproduced and validated here.

### Why it works
TabPFN is individually *weaker* than CatBoost (solo OOF 0.6667 vs 0.6837) but makes **different
errors**, so the meta-learner extracts genuine diversity rather than redundancy.

### Honest validation (paired repeated 5×10 CV on the tuned OOF, `src/eval_tabpfn_combo.py`)
| Stack | OOF best-F1 | Honest CV |
|---|---|---|
| STACK 5 (tuned, no TabPFN) | 0.6866 | 0.6811 ± 0.0270 |
| **STACK 6 (tuned + TabPFN)** | **0.6945** | **0.6938 ± 0.0257** |

- Paired delta **+0.0127** (SE 0.0014, n=50) → **~9× SE, significant**; variance also slightly lower.
- Single-model OOF: XGB 0.6685 · LGBM 0.6721 · CatBoost 0.6837 · MLP 0.6435 · RF 0.5702 · **TabPFN 0.6667**.

### Committed `predictions.csv` (6-model, no pseudo-labeling)
- **232 positive (11.6%)** / 1768 neg, 2000 rows, global threshold 0.295.
- MNAR signal intact: pred-pos | `f11` missing 36.6% (present 7.3%) · `f15` missing 36.4% (present 7.2%).
- Pseudo-labeling dropped here: 6-model-no-pseudo (0.6938) already dominates the old 5-model+pseudo (~0.685).

### Reproduce
```powershell
# one-time, open-weights build (downgrades sklearn→1.6.1, pandas→2.3.3):
.\.venv\Scripts\python.exe -m pip install "tabpfn==2.2.1"
# regenerate predictions.csv (reuses cached OOF + tuned params; ~8 min, TabPFN CPU fit ≈ 474s):
.\.venv\Scripts\python.exe src\finalize_with_tabpfn.py
```
Reuses `artifacts/{base_oof.npz, tabpfn_oof.npz, tuned_params.json}` for the meta + threshold;
only the final exam predictions are computed fresh (5 base refits + 1 TabPFN fit).

### Notes / gotchas
- Must use **open-weights `tabpfn==2.2.1`** — v6+/v8 gate weights behind a PriorLabs login and
  crash on Windows stdin. CPU on >1000 rows needs `TABPFN_ALLOW_CPU_LARGE_DATASET=1` +
  `ignore_pretraining_limits=True`. The posthog SSL telemetry errors are harmless.
- Same honesty caveat as before applies: Optuna picked params on the in-sample OOF threshold,
  so realistic graded exam F1 is likely a touch below the CV figure — but the +0.0127 TabPFN
  gain is measured on a *paired* basis and is robust.

### Optional future levers
Bake TabPFN into `train_ultra.py` canonically (with OOF cache reuse); re-add pseudo-labeling on
top of the 6-model stack; or feed TabPFN the raw NaN-native features (it handles missing natively)
instead of the median-imputed 13.
