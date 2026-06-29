# Toxic ML Project — Reverse Engineering Findings

## Dataset Overview

| Property | Value |
|----------|-------|
| Training rows | 8000 |
| Exam rows | 2000 |
| Features | 16 (f00–f15) |
| Label | Binary (0/1) |
| Positive rate | 10% (800 samples) |
| Imbalance ratio | 9:1 |

---

## Professor's Injection Method (Reverse-Engineered)

Verified across 6 datasets: `00000`, `00001`, `00010`, `00011`, `00100`, `01111` (ours).

All datasets share the same structure — "same, same but different." The binary dataset ID controls **which features** receive each injection, but the **method is identical**.

### Base Data Generation

1. All 16 features start as **independent N(0,1)** draws (8000 samples).
2. The **label** (0/1, 10% positive) is generated as a function of some subset of clean features.
3. Then three types of "toxic" injections are applied to specific features.

---

## The 3 Injection Types

### 1. MNAR Missingness (Missing Not At Random)

| Property | Value |
|----------|-------|
| Always applied to | Exactly 2 features |
| Missing rate | ~15% |
| Label-dependent? | **YES** — this is the key |
| Positive rate when missing | ~27% |
| Positive rate when present | ~7% |
| Chi-square p-value | ~10⁻¹⁰⁰ |

**This is the strongest predictive signal.** The missing indicator alone has MI ~0.02–0.04 with label.

**Per dataset:**
- `01111` (ours): f11, f15
- `00000`: f00, f03
- `00001`: f03, f07
- `00010`: f00, f03
- `00011`: f03, f07
- `00100`: f00, f03

### 2. Linear Scaling

| Property | Value |
|----------|-------|
| Transform | `new = a * original + b` |
| Result | std goes from ~1 to 50–200 |
| Destroys signal? | **No** — correlation with label is preserved |
| Purpose | Red herring / distraction |

Features look scary (large values, big range) but the information is intact — just scaled.

**Our dataset:** f10, f13, f14 are linearly scaled.

### 3. Heavy-Tail Injection

| Property | Value |
|----------|-------|
| Result | Kurtosis ~17 (normal = 0) |
| Extreme outliers | e.g., ±63,000 for f15 in our dataset |
| Destroys signal? | **Yes** — original information is overwritten |
| Distribution | Likely t-distribution (df ≈ 2) or Cauchy-contaminated |

**Our dataset:** f09, f15 have heavy tails. Note: f15 has **both** MNAR + heavy tails.

---

## Feature State Matrix

```
Feature  00000        00001        00010        00011        00100        01111 (ours)
─────────────────────────────────────────────────────────────────────────────────────
f00      MNAR         SCALED       MNAR         SCALED       MNAR         CLEAN
f01      HEAVY_TAIL   HEAVY_TAIL   HEAVY_TAIL   HEAVY_TAIL   HEAVY_TAIL   CLEAN
f02      HEAVY_TAIL   HEAVY_TAIL   CLEAN        CLEAN        HEAVY_TAIL   CLEAN
f03      MNAR         MNAR         MNAR         MNAR         MNAR         CLEAN
f04      CLEAN        CLEAN        CLEAN        CLEAN        CLEAN        CLEAN
f05      SCALED       SCALED       SCALED       SCALED       SCALED       CLEAN
f06      SCALED       SCALED       SCALED       SCALED       SCALED       CLEAN
f07      SCALED       MNAR         HEAVY_TAIL   MNAR         SCALED       CLEAN
f08      CLEAN        CLEAN        CLEAN        CLEAN        CLEAN        CLEAN
f09      CLEAN        CLEAN        CLEAN        CLEAN        CLEAN        HEAVY_TAIL
f10      CLEAN        CLEAN        CLEAN        CLEAN        CLEAN        SCALED
f11      CLEAN        CLEAN        CLEAN        CLEAN        CLEAN        MNAR
f12      CLEAN        CLEAN        CLEAN        CLEAN        CLEAN        CLEAN
f13      CLEAN        CLEAN        CLEAN        CLEAN        CLEAN        SCALED
f14      CLEAN        CLEAN        CLEAN        CLEAN        CLEAN        SCALED
f15      CLEAN        CLEAN        CLEAN        CLEAN        CLEAN        MNAR
```

**Key observation:** Our dataset (01111) has injections only in f09–f15, while peers have them in f00–f07. Features **f04, f08, f12 are NEVER touched** in any dataset.

---

## Feature Importance for Our Dataset (01111)

### Top Predictors (by Mutual Information)

| Rank | Feature | MI Score | Why it works |
|------|---------|----------|--------------|
| 1 | f11 | 0.0396 | MNAR — missing indicator is the signal |
| 2 | f12 | 0.0348 | Clean signal — directly predicts label |
| 3 | f15 | 0.0314 | MNAR — missing indicator is the signal |
| 4 | f08 | 0.0083 | Clean, weak signal |
| 5 | f13 | 0.0070 | Scaled, weak residual signal |

### Noise Features (MI ≈ 0)

f00, f01, f02, f03, f04, f05, f06, f07 — all clean N(0,1) with no label correlation.

### What This Means for Modeling

1. **Create `f11_missing` and `f15_missing` binary indicators** — these are top-2 features.
2. **f12 is the strongest "real" feature** — use it as-is.
3. **f09 (heavy-tail) is useless** — signal destroyed by injection.
4. **f10, f13, f14 (scaled)** — the signal may survive after proper scaling.
5. **f15** has both MNAR + heavy tails — use the missing indicator, not the values.

---

## Skewness & Kurtosis (Our Dataset)

| Feature | Skewness | Kurtosis | Interpretation |
|---------|----------|----------|----------------|
| f15 | -0.68 | **16.4** | Heavy-tailed, extreme outliers |
| f09 | -0.15 | **16.9** | Heavy-tailed, extreme outliers |
| f11 | 0.10 | 0.21 | Normal shape (but has MNAR) |
| f14 | 0.08 | -0.03 | Normal (scaled) |
| f10 | -0.03 | -0.03 | Normal (scaled) |
| f12 | 0.05 | -0.09 | Normal (clean, predictive!) |
| f04 | 0.02 | 0.03 | Normal (clean, noise) |
| f08 | -0.04 | -0.04 | Normal (clean, weak signal) |

---

## Implications for the Modeling Pipeline

1. **Imputation strategy:** Don't blindly impute f11/f15 — the missingness IS the feature. Always add binary missing indicators.

2. **Scaling:** RobustScaler for f09/f15 (heavy tails), StandardScaler for the rest. Tree-based models don't need scaling.

3. **Feature engineering:**
   - `f11_missing = f11.isna().astype(int)` ← strong predictor
   - `f15_missing = f15.isna().astype(int)` ← strong predictor
   - Consider dropping f09 entirely (signal destroyed)

4. **Model selection:**
   - Tree-based models (RF, XGBoost) can naturally split on missingness patterns
   - Linear models need the explicit missing indicators
   - MLP needs proper scaling + missing indicators

5. **Threshold tuning:** With 10% positive rate, default 0.5 threshold is suboptimal. Sweep for max F1.

---

## Verification

The pattern holds across **all 6 datasets**:
- Same 10% positive rate
- Same 3 injection types
- Same label-dependent missingness mechanism (chi² p < 10⁻⁸⁰)
- Same kurtosis ~17 for heavy-tail features
- f04, f08, f12 always untouched

This is not a coincidence — it's the professor's systematic data generation pipeline.
