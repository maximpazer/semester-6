r"""
eval_tabpfn_combo.py - Does TabPFN improve the FULLY-TUNED stack?

Uses the cached OOF from the real pipeline run:
  * artifacts/base_oof.npz   -> Optuna-tuned XGB/LGBM/Cat/MLP/RF OOF (+ y)
  * artifacts/tabpfn_oof.npz -> TabPFN OOF (+ y)
Both were produced with StratifiedKFold(5, shuffle, random_state=42) on the same
13 features in the same row order, so the OOF columns are row-aligned. We just
re-run the logistic stacker with vs without TabPFN and compare honestly. <1s.
"""
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

RS = 42
ROOT = Path(__file__).resolve().parent.parent.parent
b = np.load(ROOT / 'artifacts' / 'base_oof.npz')
t = np.load(ROOT / 'artifacts' / 'tabpfn_oof.npz')

y = b['y']
assert np.array_equal(y, t['y']), "y mismatch -> caches are not aligned, re-run both"
tab = t['tab']

base_names = ['XGB', 'LGBM', 'Cat', 'MLP', 'RF']
base_cols = [b['xgb'], b['lgbm'], b['cat'], b['mlp'], b['rf']]
base5 = np.column_stack(base_cols)
base6 = np.column_stack(base_cols + [tab])


def best_thr(p, yt):
    bv, bt = 0.0, 0.5
    for th in np.arange(0.02, 0.95, 0.005):
        f = f1_score(yt, (p >= th).astype(int))
        if f > bv:
            bv, bt = f, th
    return bt, bv


def stack(base):
    meta = LogisticRegression(max_iter=2000, C=1.0)
    return cross_val_predict(meta, base, y, cv=StratifiedKFold(5, shuffle=True, random_state=RS),
                             method='predict_proba')[:, 1]


def per_fold_f1(oof, splits):
    out = []
    for trn, va in splits:
        th = best_thr(oof[trn], y[trn])[0]
        out.append(f1_score(y[va], (oof[va] >= th).astype(int)))
    return np.array(out)


print("Single-model OOF best-F1 (TUNED bases):")
for n, c in zip(base_names + ['TabPFN'], base_cols + [tab]):
    print(f"  {n:<7} {best_thr(c, y)[1]:.4f}")

s5, s6 = stack(base5), stack(base6)
print("\nEnsemble OOF best-F1:")
print(f"  STACK 5 (tuned, no TabPFN) {best_thr(s5, y)[1]:.4f}")
print(f"  STACK 6 (tuned + TabPFN)   {best_thr(s6, y)[1]:.4f}")
print(f"  BLEND 6 (equal)            {best_thr(base6.mean(1), y)[1]:.4f}")

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RS)
splits = list(rcv.split(s5.reshape(-1, 1), y))
f5, f6 = per_fold_f1(s5, splits), per_fold_f1(s6, splits)
d = f6 - f5
se = d.std(ddof=1) / np.sqrt(len(d))
print("\nHonest paired repeated 5x10 CV (global threshold):")
print(f"  STACK 5  {f5.mean():.4f} +/- {f5.std():.4f}")
print(f"  STACK 6  {f6.mean():.4f} +/- {f6.std():.4f}")
print(f"  paired delta = {d.mean():+.4f}  (SE {se:.4f}, n={len(d)})")
print("  VERDICT:", "TabPFN HELPS the tuned stack (delta > 2*SE)" if d.mean() > 2 * se
      else "within noise on the tuned stack - not worth the cost")
