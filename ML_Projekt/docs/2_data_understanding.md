# Phase 2 — Data Understanding

> CRISP-DM Phase 2. *What is in the data, and what generated it?*
> Code: [`src/data_understanding/`](../src/data_understanding/).
> Deep-dive sources: [references/findings.md](references/findings.md),
> [references/enriched_data_findings.md](references/enriched_data_findings.md).

## 2.1 Dataset overview

| Property | Value |
|----------|-------|
| Training rows | 8 000 |
| Exam rows | 2 000 |
| Features | 16 (`f00`–`f15`) |
| Label | binary (0/1) |
| Positive rate | ~10 % (≈ 800 positives) |
| Imbalance | 9:1 |
| Missing values | only `f11` (~15.5 %) and `f15` (~15.4 %), in **both** train and exam |

Run [`eda_analysis.py`](../src/data_understanding/eda_analysis.py) for the full console
report (shapes, dtypes, `describe()`, class balance, mutual information).

## 2.2 The three "toxic" injections (reverse-engineered)

The data starts as independent `N(0,1)` draws; the professor then applies three corruption
types. Decoded and verified across multiple peer datasets
([`reverse_engineer.py`](../src/data_understanding/reverse_engineer.py),
[`compare_datasets.py`](../src/data_understanding/compare_datasets.py)):

### 1. MNAR missingness — *the strongest signal*
Missing-**N**ot-**A**t-**R**andom: missingness depends on the label.
- `f11` missing → **26.8 %** positive vs **6.9 %** when present.
- `f15` missing → **26.4 %** positive vs **7.0 %** when present.
- χ² p-value ≈ 10⁻¹⁰⁰. **The missing *indicator* is the feature, not the value.**

### 2. Linear scaling — *a distraction*
`new = a·x + b`, blowing std up to ~50–200. The signal is **preserved**, just rescaled.
Applied to `f10`, `f13`, `f14`.

### 3. Heavy-tail injection — *destroys signal*
Kurtosis ≈ 17, extreme outliers (e.g. `f15` reaches ±63 000). The original information is
**overwritten**. Applied to `f09`, `f15` (note `f15` is *both* MNAR **and** heavy-tailed →
use its missing indicator, not its values).

## 2.3 Feature state matrix for `01111`

| Feature | State | Use it? |
|---------|-------|---------|
| `f11`, `f15` | **MNAR missing** | ✅ via missing indicator (top signal) |
| `f12` | **clean** | ✅ strongest *real* predictor (causal driver) |
| `f08` | clean | ✅ weak signal |
| `f10`, `f13`, `f14` | scaled | ✅ weak residual signal after scaling |
| `f09` | heavy-tail | ⚠️ value is noise (signed-log only) |
| `f00`–`f07` | clean `N(0,1)` | ❌ no label correlation (pure noise) |

**`f04`, `f08`, `f12` are never touched in any dataset.**

## 2.4 The generator rule (from the enriched peer data)

- **Rows are independent across dataset IDs** — no cross-dataset row lookup is possible
  (f12 correlation `00000` vs `01111` ≈ 0.014). The peer data is useful *structurally*, not row-wise.
- The **first two ID bits select the causal driver feature**: `00→f04`, `01→f12`, `10→f08`,
  `11→f00`. `01111` starts with `01` → **the driver is `f12`** (confirmed independently).
- The label is **probabilistic**: `P(positive | f12)` is a skewed U-shape, peaking at only
  **~45 %** near `f12 ≈ +2`. This sets a **hard F1 ceiling around ~0.65–0.69** — no model can
  separate classes that genuinely overlap.

## 2.5 Predictive ranking (mutual information vs label)

| Rank | Feature | Why it works |
|------|---------|--------------|
| 1 | `f11` (missing) | MNAR indicator |
| 2 | `f12` | clean causal driver |
| 3 | `f15` (missing) | MNAR indicator |
| 4 | `f08` | clean, weak |
| 5 | `f13` | scaled, weak residual |

[`diagnostics.py`](../src/data_understanding/diagnostics.py) confirms the label is
**multi-feature** (`|f12|` correlation ≈ 0.24, `f13` ≈ 0.13, `f08` magnitude) and finds no
further hidden leaks.

> ⚠️ `diagnostics.py` reads the 360 MB `data/download/` dump (git-ignored). See
> [../data/README.md](../data/README.md) to obtain it; the other scripts only need the
> committed `01111` files.

→ Continue with **[Phase 3 — Data Preparation](3_data_preparation.md)**.
