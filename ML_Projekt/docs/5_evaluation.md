# Phase 5 — Evaluation

> CRISP-DM Phase 5. *How good is the model really, and where are the traps?*
> Code: [`src/evaluation/`](../src/evaluation/).
> Source: [references/final_result.txt](references/final_result.txt),
> [references/maximize_performance_results.md](references/maximize_performance_results.md).

## 5.1 Headline

| Model | Honest CV F1 (positive class) |
|-------|-------------------------------|
| Engineered XGB baseline (single global threshold) | ~0.654 |
| 5-model stack ([`train_ultra.py`](../src/modeling/train_ultra.py)) | **0.6811 ± 0.027** |
| **6-model stack + TabPFN** ([`finalize_with_tabpfn.py`](../src/modeling/finalize_with_tabpfn.py)) | **0.6938 ± 0.026** |

A genuine **+0.040** over the baseline, every step validated by repeated CV.

## 5.2 Threshold tuning

At a 10 % prior the default 0.5 threshold is wrong. The decision threshold is swept on OOF
predictions and compared honestly (repeated 5×10 CV):

| Strategy | CV F1 | Verdict |
|----------|-------|---------|
| **Global threshold (0.295)** | **0.6811 ± 0.027** | ✅ chosen |
| Prior-aware top-K | 0.6803 ± 0.026 | ties global |
| Per-stratum (missing/present) | 0.6730 ± 0.028 | ❌ overfits the small strata |

The committed predictions use the **global threshold 0.295**.

## 5.3 Honest validation & the overfitting caveat

- Train-vs-CV gap was monitored; no large gap (the noise gate rejects fold-noise gains).
- **Caveat:** the Optuna hyper-parameters were chosen to maximise the in-sample OOF best-F1,
  which adds mild optimism. The realistic **graded exam F1 is therefore likely ~0.66–0.68** —
  still a clear, real gain over 0.654. We report this honestly rather than the rosier CV number.
- The **probabilistic label** (max `P(pos|f12) ≈ 0.45`) imposes a hard ceiling — we are near it.

## 5.4 Sanity check on the predictions

The committed `predictions.csv` (232 positives, 11.6 %) preserves the MNAR signal that Phase 2
identified — strong evidence the model learned the *right* thing:

| | predicted-positive rate |
|---|---|
| `f11` **missing** | 36.6 % |
| `f11` present | 7.3 % |
| `f15` **missing** | 36.4 % |
| `f15` present | 7.2 % |

## 5.5 Interpretation (feature importance)

Tree importance + permutation importance agree with Phase 2: the **missing indicators** of
`f11`/`f15` and the clean driver **`f12`** dominate; `f00`–`f07` contribute nothing.

## 5.6 Dead ends (validated, do not revisit)

External / enriched datasets, cross-dataset row lookup, the **Lethal** track (a separate,
harder 24 000-row problem with no missingness leak), per-stratum thresholds, and the
pure-noise clean features `f00`–`f07`. Each was measured and rejected.

Reproduce the comparison with
[`eval_tabpfn_combo.py`](../src/evaluation/eval_tabpfn_combo.py) and the headline report with
[`final_result.py`](../src/evaluation/final_result.py).

→ Continue with **[Phase 6 — Deployment](6_deployment.md)**.
