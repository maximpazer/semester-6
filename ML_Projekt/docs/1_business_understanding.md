# Phase 1 — Business Understanding

> CRISP-DM Phase 1. *What are we trying to achieve, and how is success measured?*
> Source brief: [references/Instruction.md](references/Instruction.md).

## 1.1 Objective

Build a binary classifier for a deliberately **"toxified"** tabular dataset
(`toxic_*_01111`) and predict the label for a held-out **exam** set. The project has two
equally weighted deliverables:

1. **The best possible predictions** on the exam set.
2. **A defensible explanation** of every modelling choice — the work is defended in an
   **oral exam**, so each non-obvious decision must be justifiable.

## 1.2 Success criterion / metric

| | |
|---|---|
| **Primary metric** | **F1-score of the positive class** on the held-out exam set |
| Why F1 (not accuracy) | The data is ~**10 % positive** (9:1 imbalance). Accuracy is misleading — a trivial all-negative classifier scores 90 % accuracy but F1 = 0. |
| Half-credit benchmark | Beat a **Random Forest** baseline. |
| Upper benchmark | Approach an **MLP** / strong ensemble. |

Because F1 balances precision and recall of the minority class, the **decision threshold**
is a first-class modelling decision (the default 0.5 is provably suboptimal at a 10 % prior).

## 1.3 Data provided

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| `toxic_data_01111.csv` | 8 000 | `f00`–`f15` + `label` | training data |
| `toxic_exam_01111.csv` | 2 000 | `f00`–`f15` (no label) | we must predict these |

The `01111` suffix is the dataset ID — it encodes *which* features were corrupted (see
[Phase 2](2_data_understanding.md)). Peer datasets (`00000`, `00001`, …) were provided for
cross-dataset analysis only.

## 1.4 Constraints and rules

- **No exam leakage.** The exam file is used *only* for the final prediction — never during
  training, feature selection, or threshold tuning.
- **Reproducibility.** `random_state = 42` is fixed everywhere; results must be repeatable.
- **Honest validation.** When two reasonable choices exist, both are run and compared on
  cross-validated F1 rather than guessed. Every reported gain must beat fold noise.
- **Explainability.** Each modelling step must be defensible verbally.

## 1.5 Definition of done

- A `predictions.csv` with one column `label` (0/1), same row order as the exam file.
- A reproducible pipeline and a phase-by-phase decision log (this `docs/` folder).
- A realistic, cross-validated estimate of the graded F1 — **not** an optimistic in-sample number.

→ Continue with **[Phase 2 — Data Understanding](2_data_understanding.md)**.
