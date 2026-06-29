"""
Phase 1 diagnostics for toxic_data_01111.
Goal: find any UNEXPLOITED signal/leak and estimate the achievable F1 ceiling
before investing in heavy modeling.

Sections:
  1. Missingness audit (all 16 features, train + exam)
  2. Per-class variance / nonlinear leak (resolves the "why do scaled feats help?" clue)
  3. Best OOF model + residual hunt (is there leftover linear signal?)
  4. Multi-feature test (does a combo of clean feats beat the single f12 driver?)
  5. Bayes ceiling estimate (pooled g(f12) from the eight 01xxx datasets)
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RS = 42
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
DL = DATA / 'download'

tr = pd.read_csv(DATA / 'toxic_data_01111.csv')
ex = pd.read_csv(DATA / 'toxic_exam_01111.csv')
feats = [c for c in tr.columns if c != 'label']
y = tr['label'].values
print(f"train {tr.shape}  exam {ex.shape}  prior={y.mean():.4f}")


def best_f1(proba, yt):
    b, bt = 0.0, 0.5
    for t in np.arange(0.02, 0.95, 0.005):
        f = f1_score(yt, (proba >= t).astype(int))
        if f > b:
            b, bt = f, t
    return b, bt


# ============================================================ 1. MISSINGNESS AUDIT
print("\n" + "=" * 70 + "\n1. MISSINGNESS AUDIT (all features, train + exam)\n" + "=" * 70)
for c in feats:
    mt, me = tr[c].isna().mean(), ex[c].isna().mean()
    if mt > 0 or me > 0:
        pr = y[tr[c].isna().values].mean() if tr[c].isna().sum() > 0 else float('nan')
        prp = y[tr[c].notna().values].mean()
        print(f"  {c}: train_miss={mt:.3f} exam_miss={me:.3f} | pos|miss={pr:.3f} pos|present={prp:.3f}")

# ===================================================== 2. PER-CLASS VARIANCE / NONLINEAR LEAK
print("\n" + "=" * 70 + "\n2. PER-CLASS VARIANCE & NONLINEAR LEAK (corr with label)\n" + "=" * 70)
print(f"  {'feat':<5} {'std0':>6} {'std1':>6} {'r(v)':>7} {'r(|v|)':>7} {'r(v^2)':>7}")
for c in feats:
    v = tr[c]
    v0, v1 = v[y == 0].dropna(), v[y == 1].dropna()
    vm = v.fillna(v.median())
    cv = np.corrcoef(vm, y)[0, 1]
    ca = np.corrcoef(vm.abs(), y)[0, 1]
    cs = np.corrcoef(vm ** 2, y)[0, 1]
    flag = "  <-- LEAK?" if max(abs(cv), abs(ca), abs(cs)) > 0.04 else ""
    print(f"  {c:<5} {v0.std():>6.2f} {v1.std():>6.2f} {cv:>+7.3f} {ca:>+7.3f} {cs:>+7.3f}{flag}")


# =================================================== 3. BEST OOF MODEL + RESIDUAL HUNT
def engineer(df):
    X = pd.DataFrame(index=df.index)
    X['m11'] = df['f11'].isna().astype(int)
    X['m15'] = df['f15'].isna().astype(int)
    X['both'] = (X['m11'] & X['m15']).astype(int)
    X['anym'] = (X['m11'] | X['m15']).astype(int)
    X['mc'] = X['m11'] + X['m15']
    X['f12'] = df['f12']
    X['f12sq'] = df['f12'] ** 2
    X['f12a'] = df['f12'].abs()
    X['f12xm'] = df['f12'] * X['anym']
    X['f08'] = df['f08']
    X['f11v'] = df['f11']
    X['f15sl'] = np.sign(df['f15']) * np.log1p(df['f15'].abs())
    X['f09sl'] = np.sign(df['f09']) * np.log1p(df['f09'].abs())
    for f in ['f00', 'f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f10', 'f13', 'f14']:
        X[f] = df[f]
    return X


X = engineer(tr).fillna(engineer(tr).median())
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
oof = np.zeros(len(y))
for trn, va in cv.split(X, y):
    m = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, scale_pos_weight=9,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1,
                      random_state=RS, eval_metric='logloss').fit(X.iloc[trn], y[trn])
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]
f1b, tb = best_f1(oof, y)
print("\n" + "=" * 70 + f"\n3. BEST OOF F1={f1b:.4f} @ t={tb:.3f} (pred_pos={ (oof>=tb).mean():.3f})\n" + "=" * 70)

resid = y - oof
print("RESIDUAL HUNT - corr(residual, transform); >0.03 = missed signal:")
hits = []
for c in feats:
    v = tr[c].fillna(tr[c].median())
    for nm, tv in [('raw', v), ('abs', v.abs()), ('sq', v ** 2), ('sign', np.sign(v))]:
        r = np.corrcoef(tv, resid)[0, 1]
        if abs(r) > 0.03:
            hits.append((abs(r), f"{c}.{nm}", r))
if hits:
    for _, n, r in sorted(hits, reverse=True)[:15]:
        print(f"    {n:<12} corr_resid={r:+.4f}")
else:
    print("    (nothing > 0.03 - no obvious missed linear signal)")

# ===================================================== 4. MULTI-FEATURE TEST (clean block)
print("\n" + "=" * 70 + "\n4. MULTI-FEATURE TEST - does a clean-feature combo beat f12 alone?\n" + "=" * 70)
clean = ['f00', 'f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f08', 'f12']


def poly(df, cols):
    Z = df[cols].fillna(df[cols].median())
    out = [Z]
    out.append(Z.pow(2).rename(columns=lambda c: c + '_sq'))
    return pd.concat(out, axis=1)


def cv_logit(Xz):
    oofp = np.zeros(len(y))
    for trn, va in cv.split(Xz, y):
        sc = StandardScaler().fit(Xz.iloc[trn])
        lr = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        lr.fit(sc.transform(Xz.iloc[trn]), y[trn])
        oofp[va] = lr.predict_proba(sc.transform(Xz.iloc[va]))[:, 1]
    return best_f1(oofp, y)[0]


print(f"  logit f12 (+sq)         : {cv_logit(poly(tr, ['f12'])):.4f}")
print(f"  logit all clean (+sq)   : {cv_logit(poly(tr, clean)):.4f}")
print(f"  logit f12+f04+f08 (+sq) : {cv_logit(poly(tr, ['f12', 'f04', 'f08'])):.4f}")

# ===================================================== 5. BAYES CEILING via pooled g(f12)
print("\n" + "=" * 70 + "\n5. BAYES CEILING - pooled g(f12) from eight 01xxx datasets\n" + "=" * 70)


def load_dl(kind, split, did):
    folder = DL / f'{kind.capitalize()}_{split.capitalize()}'
    for n in [f'{kind}_{split}_{did}.csv', f'{kind}_{split}_{did} (1).csv']:
        p = folder / n
        if p.exists():
            return pd.read_csv(p)
    return None


ids01 = ['01000', '01001', '01010', '01011', '01100', '01101', '01110']  # exclude 01111
Z, Yp = [], []
for d in ids01:
    dd = load_dl('toxic', 'data', d)
    if dd is not None:
        Z.append(dd['f12'].values)
        Yp.append(dd['label'].values)
if Z:
    z = np.concatenate(Z)
    yp = np.concatenate(Yp)
    def fz(a):
        return np.column_stack([a, a ** 2, a ** 3, np.abs(a)])
    g = LogisticRegression(max_iter=2000).fit(fz(z), yp)
    print(f"  pooled rows for g(f12): {len(z)}")
    g01 = g.predict_proba(fz(tr['f12'].values))[:, 1]
    # ceiling: strong model WITH external g vs WITHOUT, repeated CV
    Xg = X.copy(); Xg['g_ext'] = g01
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
    def rep_oof(Xv):
        scores = []
        for trn, va in rcv.split(Xv, y):
            m = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, scale_pos_weight=9,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1,
                              random_state=RS, eval_metric='logloss').fit(Xv.iloc[trn], y[trn])
            p = m.predict_proba(Xv.iloc[va])[:, 1]
            scores.append(best_f1(p, y[va])[0])
        return np.mean(scores), np.std(scores)
    m0, s0 = rep_oof(X)
    m1, s1 = rep_oof(Xg)
    print(f"  repeated-CV F1  internal      : {m0:.4f} +/- {s0:.4f}")
    print(f"  repeated-CV F1  internal+g_ext: {m1:.4f} +/- {s1:.4f}")
    print(f"  --> per-fold honest F1 ~ {m0:.3f}; in-sample tuned ~ {f1b:.3f}")
else:
    print("  (no 01xxx datasets found in data/download)")

print("\nDONE.")
