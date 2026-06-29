r"""
try_tabpfn.py - Does adding TabPFN as a 6th base learner help the stack?

Controlled A/B (honest, with a noise gate):
  * TabPFN OOF best-F1 on the 13 selected features (median-imputed)
  * 5 GBM/NN base models OOF (solid fixed configs) -> stack of 5
  * stack of 6 (the 5 + TabPFN)
  * paired repeated-CV comparison STACK5 vs STACK6

NOTE: the 5 base configs here are good defaults, NOT the exact Optuna-tuned
params from src/train_ultra.py, so STACK5 is a re-baseline. What matters is the
PAIRED DELTA from adding TabPFN, measured the same way for both.
"""
from pathlib import Path
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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

df = pd.read_csv(DATA / 'toxic_data_01111.csv')
y = df['label'].values

SELECTED = ['m11', 'm15', 'both_miss', 'f12', 'f12_sq', 'f08', 'f10', 'f13', 'f14',
            'f11_val', 'f15_slog', 'f09_slog', 'f11_abs']


def engineer13(d):
    X = pd.DataFrame(index=d.index)
    m11 = d['f11'].isna().astype(int)
    m15 = d['f15'].isna().astype(int)
    X['m11'], X['m15'] = m11, m15
    X['both_miss'] = (m11 & m15).astype(int)
    X['f12'] = d['f12']
    X['f12_sq'] = d['f12'] ** 2
    X['f08'] = d['f08']
    X['f10'] = d['f10']
    X['f13'] = d['f13']
    X['f14'] = d['f14']
    X['f11_val'] = d['f11']
    X['f11_abs'] = d['f11'].abs()
    X['f15_slog'] = np.sign(d['f15']) * np.log1p(d['f15'].abs())
    X['f09_slog'] = np.sign(d['f09']) * np.log1p(d['f09'].abs())
    return X[SELECTED]


Xraw = engineer13(df)
imp = SimpleImputer(strategy='median').fit(Xraw)
X = pd.DataFrame(imp.transform(Xraw), columns=SELECTED, index=Xraw.index)
sc = StandardScaler().fit(X)
Xs = pd.DataFrame(sc.transform(X), columns=SELECTED, index=X.index)

SPLITS = list(StratifiedKFold(5, shuffle=True, random_state=RS).split(X, y))


def best_thr(p, yt):
    b, bt = 0.0, 0.5
    for t in np.arange(0.02, 0.95, 0.005):
        f = f1_score(yt, (p >= t).astype(int))
        if f > b:
            b, bt = f, t
    return bt, b


def get_oof(make, XX):
    oof = np.zeros(len(y))
    for trn, va in SPLITS:
        m = make()
        m.fit(XX.iloc[trn], y[trn])
        oof[va] = m.predict_proba(XX.iloc[va])[:, 1]
    return oof


def mk_xgb():
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_lambda=2.0,
                         scale_pos_weight=9, random_state=RS, eval_metric='logloss', n_jobs=-1)


def mk_lgbm():
    return LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.03, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_lambda=2.0,
                          scale_pos_weight=9, random_state=RS, verbose=-1, n_jobs=-1)


def mk_cat():
    return CatBoostClassifier(iterations=500, depth=5, learning_rate=0.03, l2_leaf_reg=3.0,
                              scale_pos_weight=9, random_state=RS, verbose=0, thread_count=-1)


def mk_mlp():
    return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.001, learning_rate_init=0.001,
                         max_iter=1000, early_stopping=True, validation_fraction=0.15,
                         learning_rate='adaptive', batch_size=128, random_state=RS)


def mk_rf():
    return RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3,
                                  class_weight='balanced_subsample', random_state=RS, n_jobs=-1)


print("Computing base-model OOF (5 folds each)...")
t0 = time.time()
oof_xgb = get_oof(mk_xgb, X)
oof_lgbm = get_oof(mk_lgbm, X)
oof_cat = get_oof(mk_cat, X)
oof_mlp = get_oof(mk_mlp, Xs)
oof_rf = get_oof(mk_rf, X)
print(f"  base done in {time.time() - t0:.0f}s")


# ---- TabPFN OOF ----
os.environ.setdefault('TABPFN_ALLOW_CPU_LARGE_DATASET', '1')
from tabpfn import TabPFNClassifier


def make_tabpfn():
    for kwargs in (dict(device='cpu', ignore_pretraining_limits=True, random_state=RS),
                   dict(device='cpu', ignore_pretraining_limits=True),
                   dict(device='cpu'), dict()):
        try:
            return TabPFNClassifier(**kwargs)
        except TypeError:
            continue
    return TabPFNClassifier()


print("Computing TabPFN OOF (downloads weights on first run)...")
t0 = time.time()
oof_tab = np.zeros(len(y))
for i, (trn, va) in enumerate(SPLITS):
    clf = make_tabpfn()
    clf.fit(X.iloc[trn].values, y[trn])
    oof_tab[va] = clf.predict_proba(X.iloc[va].values)[:, 1]
    print(f"    TabPFN fold {i + 1}/5 done ({time.time() - t0:.0f}s)")
print(f"  TabPFN done in {time.time() - t0:.0f}s")

(ROOT / 'artifacts').mkdir(exist_ok=True)
np.savez(ROOT / 'artifacts' / 'tabpfn_oof.npz', tab=oof_tab, y=y)

names = ['XGB', 'LGBM', 'Cat', 'MLP', 'RF', 'TabPFN']
oofs = [oof_xgb, oof_lgbm, oof_cat, oof_mlp, oof_rf, oof_tab]
print("\nSingle-model OOF best-F1:")
for n, o in zip(names, oofs):
    print(f"  {n:<7} {best_thr(o, y)[1]:.4f}")

base5 = np.column_stack([oof_xgb, oof_lgbm, oof_cat, oof_mlp, oof_rf])
base6 = np.column_stack([oof_xgb, oof_lgbm, oof_cat, oof_mlp, oof_rf, oof_tab])


def stack(base):
    meta = LogisticRegression(max_iter=2000, C=1.0)
    return cross_val_predict(meta, base, y, cv=StratifiedKFold(5, shuffle=True, random_state=RS),
                             method='predict_proba')[:, 1]


s5, s6 = stack(base5), stack(base6)
print("\nEnsemble OOF best-F1:")
print(f"  STACK 5 (no TabPFN) {best_thr(s5, y)[1]:.4f}")
print(f"  STACK 6 (+TabPFN)   {best_thr(s6, y)[1]:.4f}")
print(f"  BLEND 5             {best_thr(base5.mean(1), y)[1]:.4f}")
print(f"  BLEND 6             {best_thr(base6.mean(1), y)[1]:.4f}")


# ---- honest paired repeated-CV noise gate ----
def per_fold_f1(oof, splits):
    fs = []
    for trn, va in splits:
        t = best_thr(oof[trn], y[trn])[0]
        fs.append(f1_score(y[va], (oof[va] >= t).astype(int)))
    return np.array(fs)


rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RS)
splits = list(rcv.split(s5.reshape(-1, 1), y))
f5, f6 = per_fold_f1(s5, splits), per_fold_f1(s6, splits)
d = f6 - f5
se = d.std(ddof=1) / np.sqrt(len(d))
print("\nHonest paired repeated 5x10 CV (global threshold):")
print(f"  STACK 5  {f5.mean():.4f} +/- {f5.std():.4f}")
print(f"  STACK 6  {f6.mean():.4f} +/- {f6.std():.4f}")
print(f"  paired delta = {d.mean():+.4f}  (SE {se:.4f}, n={len(d)})")
sig = d.mean() > 2 * se
print("  VERDICT:", "TabPFN HELPS (delta > 2*SE)" if sig else
      "within noise - not a clear improvement")
