r"""
finalize_with_tabpfn.py - Regenerate predictions.csv with TabPFN as a 6th base learner.

WHY: The 5-model tuned stack scores honest CV 0.6811. Adding TabPFN as a 6th base
lifts it to 0.6938 (paired delta +0.0127, ~9x SE) - a real, significant improvement.
This script commits that improvement to predictions.csv.

REUSES the expensive cached artifacts so this runs in ~8 min (no Optuna, no OOF recompute):
  * artifacts/tuned_params.json -> Optuna-tuned XGB/LGBM/Cat configs
  * artifacts/base_oof.npz      -> tuned 5-model OOF (+ y)   [meta fit + threshold]
  * artifacts/tabpfn_oof.npz    -> TabPFN OOF                [meta fit + threshold]

Only the final EXAM predictions are computed fresh:
  5 GBM/NN refits on full train (~1 min) + 1 TabPFN fit/predict (CPU, ~5-7 min).

NOTE: no pseudo-labeling here (kept clean). TabPFN's +0.0127 already dominates the
~+0.004 the old 5-model pseudo step gave, so 6-model-no-pseudo > 5-model-with-pseudo.
"""
from pathlib import Path
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
RS = 42
np.random.seed(RS)
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
ART = ROOT / 'artifacts'

# ---------------------------------------------------------------- data + features
df_train = pd.read_csv(DATA / 'toxic_data_01111.csv')
df_exam = pd.read_csv(DATA / 'toxic_exam_01111.csv')
y = df_train['label'].values

# the 13 forward-selected features (matches train_ultra `selected` + the cached OOF)
SELECTED = ['m11', 'm15', 'both_miss', 'f12', 'f12_sq', 'f08', 'f10', 'f13', 'f14',
            'f11_val', 'f15_slog', 'f09_slog', 'f11_abs']


def engineer(df):
    X = pd.DataFrame(index=df.index)
    m11 = df['f11'].isna().astype(int)
    m15 = df['f15'].isna().astype(int)
    X['m11'], X['m15'] = m11, m15
    X['both_miss'] = (m11 & m15).astype(int)
    X['f12'] = df['f12']
    X['f12_sq'] = df['f12'] ** 2
    X['f08'] = df['f08']
    X['f10'] = df['f10']
    X['f13'] = df['f13']
    X['f14'] = df['f14']
    X['f11_val'] = df['f11']
    X['f11_abs'] = df['f11'].abs()
    X['f15_slog'] = np.sign(df['f15']) * np.log1p(df['f15'].abs())
    X['f09_slog'] = np.sign(df['f09']) * np.log1p(df['f09'].abs())
    return X[SELECTED]


Xtr_raw = engineer(df_train)
Xex_raw = engineer(df_exam)
imp = SimpleImputer(strategy='median').fit(Xtr_raw)
Xtr = pd.DataFrame(imp.transform(Xtr_raw), columns=SELECTED, index=Xtr_raw.index)
Xex = pd.DataFrame(imp.transform(Xex_raw), columns=SELECTED, index=Xex_raw.index)
sc = RobustScaler().fit(Xtr)
Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=SELECTED, index=Xtr.index)
Xex_s = pd.DataFrame(sc.transform(Xex), columns=SELECTED, index=Xex.index)

# ---------------------------------------------------------------- model factories
p = json.loads((ART / 'tuned_params.json').read_text())


def make_xgb():
    return XGBClassifier(**p['xgb'], random_state=RS, eval_metric='logloss', n_jobs=-1)


def make_lgbm():
    return LGBMClassifier(**p['lgbm'], random_state=RS, verbose=-1, n_jobs=-1)


def make_cat():
    return CatBoostClassifier(**p['cat'], random_state=RS, verbose=0, thread_count=-1)


def make_mlp():
    return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.001, learning_rate_init=0.001,
                         max_iter=1000, early_stopping=True, validation_fraction=0.15,
                         learning_rate='adaptive', batch_size=128, random_state=RS)


def make_rf():
    return RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3,
                                  class_weight='balanced_subsample', random_state=RS, n_jobs=-1)


def best_thr(prob, yt):
    b, bt = 0.0, 0.5
    for t in np.arange(0.02, 0.95, 0.005):
        f = f1_score(yt, (prob >= t).astype(int))
        if f > b:
            b, bt = f, t
    return bt, b


# ---------------------------------------------------------------- cached OOF (meta + threshold)
bo = np.load(ART / 'base_oof.npz')
to = np.load(ART / 'tabpfn_oof.npz')
assert np.array_equal(bo['y'], y) and np.array_equal(to['y'], y), \
    "cache row-misaligned with current data; re-run train_ultra.py + try_tabpfn.py"
base6_oof = np.column_stack([bo['xgb'], bo['lgbm'], bo['cat'], bo['mlp'], bo['rf'], to['tab']])

# ---------------------------------------------------------------- fresh exam predictions
print("Refitting 5 tuned base models on full train...")
t0 = time.time()
ex_xgb = make_xgb().fit(Xtr, y).predict_proba(Xex)[:, 1]
ex_lgbm = make_lgbm().fit(Xtr, y).predict_proba(Xex)[:, 1]
ex_cat = make_cat().fit(Xtr, y).predict_proba(Xex)[:, 1]
ex_mlp = make_mlp().fit(Xtr_s, y).predict_proba(Xex_s)[:, 1]
ex_rf = make_rf().fit(Xtr, y).predict_proba(Xex)[:, 1]
print(f"  base refit done in {time.time() - t0:.0f}s")

print("Fitting TabPFN on full train + predicting exam (CPU, ~5-7 min)...")
os.environ.setdefault('TABPFN_ALLOW_CPU_LARGE_DATASET', '1')
from tabpfn import TabPFNClassifier  # noqa: E402


def make_tabpfn():
    for kw in (dict(device='cpu', ignore_pretraining_limits=True, random_state=RS),
               dict(device='cpu', ignore_pretraining_limits=True),
               dict(device='cpu'), dict()):
        try:
            return TabPFNClassifier(**kw)
        except TypeError:
            continue
    return TabPFNClassifier()


t0 = time.time()
ex_tab = make_tabpfn().fit(Xtr.values, y).predict_proba(Xex.values)[:, 1]
print(f"  TabPFN exam done in {time.time() - t0:.0f}s")

exam_base6 = np.column_stack([ex_xgb, ex_lgbm, ex_cat, ex_mlp, ex_rf, ex_tab])

# ---------------------------------------------------------------- stack + threshold
meta = LogisticRegression(max_iter=2000, C=1.0)
s6 = cross_val_predict(meta, base6_oof, y,
                       cv=StratifiedKFold(5, shuffle=True, random_state=RS),
                       method='predict_proba')[:, 1]
thr, oof_f1 = best_thr(s6, y)
meta_full = LogisticRegression(max_iter=2000, C=1.0).fit(base6_oof, y)
exam_prob = meta_full.predict_proba(exam_base6)[:, 1]
exam_pred = (exam_prob >= thr).astype(int)

pd.DataFrame({'label': exam_pred}).to_csv(ROOT / 'predictions.csv', index=False)

# ---------------------------------------------------------------- report
print("\n" + "=" * 58)
print("FINAL  6-model stack  (5 tuned GBM/NN + TabPFN)")
print("=" * 58)
print(f"  stack OOF best-F1 ...... {oof_f1:.4f}   (5-model stack was 0.6866)")
print(f"  honest CV (paired) ..... 0.6938         (5-model stack was 0.6811)")
print(f"  global threshold ....... {thr:.3f}")
print(f"  predictions ............ {exam_pred.sum()} pos "
      f"({exam_pred.mean() * 100:.2f}%) / {len(exam_pred) - exam_pred.sum()} neg")
m11 = df_exam['f11'].isna().values
m15 = df_exam['f15'].isna().values
print(f"  pred pos | f11 missing . {exam_pred[m11].mean() * 100:.1f}%  "
      f"(present {exam_pred[~m11].mean() * 100:.1f}%)")
print(f"  pred pos | f15 missing . {exam_pred[m15].mean() * 100:.1f}%  "
      f"(present {exam_pred[~m15].mean() * 100:.1f}%)")
print("  saved -> predictions.csv")
print("DONE.")
