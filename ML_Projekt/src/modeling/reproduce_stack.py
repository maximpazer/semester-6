r"""
reproduce_stack.py - Reproduce the STACK out-of-fold F1 (0.6866) in <1 second.

It loads the base-model OOF probabilities cached by src/train_ultra.py and
re-runs ONLY the logistic-regression stacking + threshold sweep. No models are
retrained, so the result is bit-exact to the run that produced the cache.

If the cache is missing, run the full pipeline once to create it (it now saves
the cache automatically):
    .\.venv\Scripts\python.exe src\train_ultra.py
"""
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

RS = 42
ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / 'artifacts' / 'base_oof.npz'


def best_thr(p, yt):
    b, bt = 0.0, 0.5
    for t in np.arange(0.02, 0.95, 0.005):
        f = f1_score(yt, (p >= t).astype(int))
        if f > b:
            b, bt = f, t
    return bt, b


if not CACHE.exists():
    print(f"cache not found: {CACHE.relative_to(ROOT)}")
    print("Run the full pipeline once to create it (it caches automatically):")
    print(r"    .\.venv\Scripts\python.exe src\train_ultra.py")
    sys.exit(1)

d = np.load(CACHE)
y = d['y']
base_oof = np.column_stack([d['xgb'], d['lgbm'], d['cat'], d['mlp'], d['rf']])
names = ['XGB', 'LGBM', 'Cat', 'MLP', 'RF']

print("Reproducing ensemble OOF F1 from cached base predictions")
print("-" * 56)
for n, col in zip(names, base_oof.T):
    print(f"  {n:<5} OOF F1 = {best_thr(col, y)[1]:.4f}")

# identical meta-learner and CV as src/train_ultra.py
meta = LogisticRegression(max_iter=2000, C=1.0)
stack_oof = cross_val_predict(
    meta, base_oof, y,
    cv=StratifiedKFold(5, shuffle=True, random_state=RS),
    method='predict_proba')[:, 1]
blend_oof = base_oof.mean(axis=1)

print("-" * 56)
print(f"  STACK OOF F1 = {best_thr(stack_oof, y)[1]:.4f}   <- target 0.6866")
print(f"  BLEND OOF F1 = {best_thr(blend_oof, y)[1]:.4f}")
