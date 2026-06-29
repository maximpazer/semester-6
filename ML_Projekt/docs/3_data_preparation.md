# Phase 3 — Data Preparation

> CRISP-DM Phase 3. *How is the raw data turned into model-ready features?*
> Implemented as `engineer_*()` functions inside the pipelines in
> [`src/modeling/`](../src/modeling/) (e.g. `train_ultra.py`, `finalize_with_tabpfn.py`).

## 3.1 Guiding principle

Phase 2 showed that the signal lives in **(a) the missingness pattern** of `f11`/`f15`,
**(b) the clean driver `f12`**, and **(c) weak residuals** in a few scaled features.
Everything else is noise. Preparation therefore *extracts* those signals explicitly rather
than feeding raw columns to the model.

## 3.2 The final 13-feature set

Forward-selected (a candidate is kept only if repeated-CV F1 improves by > 0.0015):

```
m11, m15, both_miss, f12, f12_sq, f08, f10, f13, f14, f11_val, f15_slog, f09_slog, f11_abs
```

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `m11`, `m15` | `f11.isna()`, `f15.isna()` (0/1) | **top signal** — the MNAR indicators |
| `both_miss` | `m11 & m15` | interaction (both missing → even more positive) |
| `f12` | raw `f12` | the clean causal driver |
| `f12_sq` | `f12²` | captures the **U-shape** of `P(pos\|f12)` (non-monotone) |
| `f08` | raw `f08` | clean weak signal |
| `f10`, `f13`, `f14` | raw (scaled) | residual signal survives scaling; trees are scale-invariant |
| `f11_val` | raw `f11` | present-value carries extra signal (corr ≈ −0.115) |
| `f11_abs` | `\|f11\|` | the only forward-selected *addition* that beat noise (+0.003) |
| `f15_slog` | `sign(f15)·log1p(\|f15\|)` | tames the heavy tail without dropping the column |
| `f09_slog` | `sign(f09)·log1p(\|f09\|)` | same treatment for the other heavy-tailed feature |

## 3.3 Imputation

`f11`/`f15` are imputed with the **median** *after* the missing indicators are created — so
the information in the missingness is never lost. We compared median vs model-based
imputation; median + indicator generalised at least as well and is simpler to defend.

## 3.4 Scaling

- **Tree models** (XGB, LGBM, CatBoost, RF): no scaling — they are invariant to monotone
  transforms.
- **Linear / MLP**: `RobustScaler` (robust to the heavy tails) on the numeric features.
- The signed-log transform (`f15_slog`, `f09_slog`) is itself a robust outlier treatment,
  chosen over winsorising/clipping because it preserves ordering and is differentiable.

## 3.5 What was deliberately dropped

| Dropped | Why |
|---------|-----|
| `f00`–`f07` raw | clean `N(0,1)` with ~zero mutual information — pure noise that only adds variance |
| `f09` raw value | heavy-tail injection destroyed the signal (kept only as `f09_slog`, marginal) |
| External / enriched / Lethal data | proven to add **0 or negative** value (see [Phase 5](5_evaluation.md)) |

> The feature frontier is essentially **exhausted**: of all engineered candidates, only
> `f11_abs` survived the noise gate. Further gains came from *modelling*, not features.

→ Continue with **[Phase 4 — Modeling](4_modeling.md)**.
