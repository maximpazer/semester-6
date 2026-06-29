"""
train_ultra.py - Maximum-performance pipeline for toxic_data_01111.

Built on Phase-1 diagnostics (src/diagnostics.py):
  - Label is MULTI-feature: f12 (|f12| strongest), f13 (raw), f10, f08 (magnitude).
  - MNAR missingness of f11/f15 is a strong label leak.
  - Residual signal in f11.sign, f15.sign, f12.abs/sq, f13.raw/sign.

Pipeline:
  1. Rich feature engineering (residual-hunt driven)
  2. Feature-set validation (repeated CV: baseline vs rich)   [Phase 2]
  3. Optuna-tuned XGB / LGBM / CatBoost + fixed MLP / RF       [Phase 4]
  4. Out-of-fold stacking (logistic meta-learner)             [Phase 4]
  5. Threshold strategy search: global / per-stratum / prior  [Phase 3]
  6. Pseudo-labeling (transductive) - kept only if it helps   [Phase 5]
  7. Final fit -> predictions.csv + verification              [Phase 6]

All gains validated with RepeatedStratifiedKFold so we only keep what beats noise.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RS = 42
np.random.seed(RS)
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'

df_train = pd.read_csv(DATA / 'toxic_data_01111.csv')
df_exam = pd.read_csv(DATA / 'toxic_exam_01111.csv')
y = df_train['label'].values
PRIOR = y.mean()
print(f"train={df_train.shape} exam={df_exam.shape} prior={PRIOR:.4f}")


# ============================================================ FEATURE ENGINEERING
def engineer(df):
    X = pd.DataFrame(index=df.index)
    # --- MNAR missingness leak (strongest single signal) ---
    m11 = df['f11'].isna().astype(int)
    m15 = df['f15'].isna().astype(int)
    X['m11'], X['m15'] = m11, m15
    X['both_miss'] = (m11 & m15).astype(int)
    X['any_miss'] = (m11 | m15).astype(int)
    X['miss_count'] = m11 + m15
    # --- f12: strongest driver, U-shaped (magnitude is key: |f12| corr 0.24) ---
    f12 = df['f12']
    X['f12'] = f12
    X['f12_abs'] = f12.abs()
    X['f12_sq'] = f12 ** 2
    X['f12_cube'] = f12 ** 3
    X['f12_pos'] = f12.clip(lower=0)
    X['f12_neg'] = (-f12).clip(lower=0)
    # --- f13: genuine multi-feature signal (raw corr 0.134) ---
    f13 = df['f13']
    X['f13'] = f13
    X['f13_abs'] = f13.abs()
    X['f13_sq'] = f13 ** 2
    X['f13_sign'] = np.sign(f13)
    # --- f10: raw signal (corr 0.065) ---
    X['f10'] = df['f10']
    X['f10_abs'] = df['f10'].abs()
    # --- f14: weak ---
    X['f14'] = df['f14']
    # --- f08: magnitude leak (std differs by class, |f08| corr 0.069) ---
    f08 = df['f08']
    X['f08'] = f08
    X['f08_abs'] = f08.abs()
    X['f08_sq'] = f08 ** 2
    # --- f11: value + magnitude + sign leak (present values) ---
    X['f11_val'] = df['f11']
    X['f11_abs'] = df['f11'].abs()
    X['f11_sign'] = np.sign(df['f11'])
    # --- f15: sign leak + heavy-tail (compressed) ---
    X['f15_slog'] = np.sign(df['f15']) * np.log1p(df['f15'].abs())
    X['f15_sign'] = np.sign(df['f15'])
    X['f15_abslog'] = np.log1p(df['f15'].abs())
    # --- f09: heavy-tail (compressed, weak) ---
    X['f09_slog'] = np.sign(df['f09']) * np.log1p(df['f09'].abs())
    # --- interactions ---
    X['f12_x_f13'] = f12 * f13
    X['f12_x_miss'] = f12 * X['any_miss']
    X['f12abs_x_miss'] = f12.abs() * X['any_miss']
    X['f13_x_miss'] = f13 * X['any_miss']
    # --- remaining clean features (little signal, cheap to include) ---
    for f in ['f00', 'f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07']:
        X[f] = df[f]
    return X


# build full candidate feature matrix (rich), imputed
Xtr_full_raw = engineer(df_train)
Xex_full_raw = engineer(df_exam)
imp = SimpleImputer(strategy='median').fit(Xtr_full_raw)
Xtr_full = pd.DataFrame(imp.transform(Xtr_full_raw), columns=Xtr_full_raw.columns, index=Xtr_full_raw.index)
Xex_full = pd.DataFrame(imp.transform(Xex_full_raw), columns=Xex_full_raw.columns, index=Xex_full_raw.index)
print(f"candidate features: {Xtr_full.shape[1]}")


# strata for per-stratum thresholding
def strata_of(df):
    m11, m15 = df['f11'].isna().values, df['f15'].isna().values
    return np.where(m11 & m15, 'both', np.where(m11 | m15, 'one', 'none'))
strata_tr = strata_of(df_train)
strata_ex = strata_of(df_exam)


# ============================================================ HELPERS
def best_thr(proba, yt):
    b, bt = 0.0, 0.5
    for t in np.arange(0.02, 0.95, 0.005):
        f = f1_score(yt, (proba >= t).astype(int))
        if f > b:
            b, bt = f, t
    return bt, b


def repeated_f1_cols(cols, n_estimators=250, n_repeats=2):
    """Repeated-CV best-F1 for an XGB on a given feature subset (used for selection)."""
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RS)
    Xc = Xtr_full[cols]
    fs = []
    for trn, va in rcv.split(Xc, y):
        m = XGBClassifier(n_estimators=n_estimators, max_depth=4, learning_rate=0.05, scale_pos_weight=9,
                          subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1,
                          random_state=RS, eval_metric='logloss', n_jobs=-1)
        m.fit(Xc.iloc[trn], y[trn])
        fs.append(best_thr(m.predict_proba(Xc.iloc[va])[:, 1], y[va])[1])
    return float(np.mean(fs))


# ============================================================ 2. FORWARD FEATURE SELECTION
print("\n" + "=" * 70 + "\n2. FORWARD FEATURE SELECTION (greedy, repeated CV, beat noise margin)\n" + "=" * 70)
BASELINE = ['m11', 'm15', 'both_miss', 'f12', 'f12_sq', 'f08', 'f10', 'f13', 'f14',
            'f11_val', 'f15_slog', 'f09_slog']
CANDIDATES = ['f12_abs', 'f12_pos', 'f13_abs', 'f13_sign', 'f08_abs', 'f08_sq',
              'f11_abs', 'f11_sign', 'f15_sign', 'f12_x_f13', 'f12_x_miss', 'f12abs_x_miss']
MARGIN = 0.0015
selected = list(BASELINE)
current = repeated_f1_cols(selected)
print(f"  baseline ({len(selected)} feats): {current:.4f}")
remaining = list(CANDIDATES)
while remaining:
    gains = sorted(((repeated_f1_cols(selected + [c]) - current, c) for c in remaining), reverse=True)
    g, c = gains[0]
    if g > MARGIN:
        selected.append(c)
        remaining.remove(c)
        current += g
        print(f"  + {c:<14} -> {current:.4f} (+{g:.4f})")
    else:
        print(f"  stop: best remaining {c} gain {g:+.4f} < {MARGIN}")
        break
print(f"  FINAL: {len(selected)} feats, repeated-CV F1={current:.4f}")
print(f"  selected: {selected}")

# lock in selected feature matrices for the rest of the pipeline
Xtr = Xtr_full[selected].copy()
Xex = Xex_full[selected].copy()
sc = RobustScaler().fit(Xtr)
Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=Xtr.columns, index=Xtr.index)
Xex_s = pd.DataFrame(sc.transform(Xex), columns=Xex.columns, index=Xex.index)
SPLITS = list(StratifiedKFold(5, shuffle=True, random_state=RS).split(Xtr, y))


def get_oof(make_model, X):
    oof = np.zeros(len(y))
    for trn, va in SPLITS:
        m = make_model()
        m.fit(X.iloc[trn], y[trn])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


# ============================================================ 3. OPTUNA TUNING
print("\n" + "=" * 70 + "\n3. OPTUNA TUNING (maximize OOF best-F1)\n" + "=" * 70)


def tune_xgb(n_trials=50):
    def obj(tr):
        p = dict(
            n_estimators=tr.suggest_int('n_estimators', 200, 700, step=50),
            max_depth=tr.suggest_int('max_depth', 3, 7),
            learning_rate=tr.suggest_float('learning_rate', 0.01, 0.15, log=True),
            subsample=tr.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=tr.suggest_float('colsample_bytree', 0.5, 1.0),
            min_child_weight=tr.suggest_int('min_child_weight', 1, 8),
            gamma=tr.suggest_float('gamma', 0.0, 0.4),
            reg_lambda=tr.suggest_float('reg_lambda', 0.5, 5.0),
            scale_pos_weight=tr.suggest_float('scale_pos_weight', 5.0, 12.0),
        )
        oof = get_oof(lambda: XGBClassifier(**p, random_state=RS, eval_metric='logloss', n_jobs=-1), Xtr)
        return best_thr(oof, y)[1]
    s = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RS))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return s.best_params, s.best_value


def tune_lgbm(n_trials=50):
    def obj(tr):
        p = dict(
            n_estimators=tr.suggest_int('n_estimators', 200, 700, step=50),
            max_depth=tr.suggest_int('max_depth', 3, 8),
            learning_rate=tr.suggest_float('learning_rate', 0.01, 0.15, log=True),
            num_leaves=tr.suggest_int('num_leaves', 15, 80),
            subsample=tr.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=tr.suggest_float('colsample_bytree', 0.5, 1.0),
            min_child_samples=tr.suggest_int('min_child_samples', 5, 40),
            reg_lambda=tr.suggest_float('reg_lambda', 0.0, 5.0),
            scale_pos_weight=tr.suggest_float('scale_pos_weight', 5.0, 12.0),
        )
        oof = get_oof(lambda: LGBMClassifier(**p, random_state=RS, verbose=-1, n_jobs=-1), Xtr)
        return best_thr(oof, y)[1]
    s = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RS))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return s.best_params, s.best_value


def tune_cat(n_trials=30):
    def obj(tr):
        p = dict(
            iterations=tr.suggest_int('iterations', 200, 600, step=50),
            depth=tr.suggest_int('depth', 3, 7),
            learning_rate=tr.suggest_float('learning_rate', 0.01, 0.15, log=True),
            l2_leaf_reg=tr.suggest_float('l2_leaf_reg', 1.0, 10.0),
            scale_pos_weight=tr.suggest_float('scale_pos_weight', 5.0, 12.0),
        )
        oof = get_oof(lambda: CatBoostClassifier(**p, random_state=RS, verbose=0, thread_count=-1), Xtr)
        return best_thr(oof, y)[1]
    s = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RS))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return s.best_params, s.best_value


xgb_p, xgb_f1 = tune_xgb(40); print(f"  XGB  best OOF F1={xgb_f1:.4f}")
lgbm_p, lgbm_f1 = tune_lgbm(40); print(f"  LGBM best OOF F1={lgbm_f1:.4f}")
cat_p, cat_f1 = tune_cat(25); print(f"  Cat  best OOF F1={cat_f1:.4f}")


def make_xgb():
    return XGBClassifier(**xgb_p, random_state=RS, eval_metric='logloss', n_jobs=-1)
def make_lgbm():
    return LGBMClassifier(**lgbm_p, random_state=RS, verbose=-1, n_jobs=-1)
def make_cat():
    return CatBoostClassifier(**cat_p, random_state=RS, verbose=0, thread_count=-1)
def make_mlp():
    return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.001, learning_rate_init=0.001,
                         max_iter=1000, early_stopping=True, validation_fraction=0.15,
                         learning_rate='adaptive', batch_size=128, random_state=RS)
def make_rf():
    return RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3,
                                  class_weight='balanced_subsample', random_state=RS, n_jobs=-1)


# ============================================================ 4. OOF + STACKING
print("\n" + "=" * 70 + "\n4. OUT-OF-FOLD STACKING\n" + "=" * 70)
oof_xgb = get_oof(make_xgb, Xtr)
oof_lgbm = get_oof(make_lgbm, Xtr)
oof_cat = get_oof(make_cat, Xtr)
oof_mlp = get_oof(make_mlp, Xtr_s)
oof_rf = get_oof(make_rf, Xtr)
base_oof = np.column_stack([oof_xgb, oof_lgbm, oof_cat, oof_mlp, oof_rf])
names = ['XGB', 'LGBM', 'Cat', 'MLP', 'RF']
for n, o in zip(names, base_oof.T):
    print(f"  {n:<5} OOF F1={best_thr(o, y)[1]:.4f}")

meta = LogisticRegression(max_iter=2000, C=1.0)
stack_oof = cross_val_predict(meta, base_oof, y, cv=StratifiedKFold(5, shuffle=True, random_state=RS),
                              method='predict_proba')[:, 1]
print(f"  STACK OOF F1={best_thr(stack_oof, y)[1]:.4f}")
# also equal blend for reference
blend_oof = base_oof.mean(axis=1)
print(f"  BLEND OOF F1={best_thr(blend_oof, y)[1]:.4f}")

# cache base OOF + tuned params so the stack number can be reproduced instantly
# and bit-exactly via src/reproduce_stack.py (no Optuna, no retraining)
import json as _json
ART = ROOT / 'artifacts'; ART.mkdir(exist_ok=True)
np.savez(ART / 'base_oof.npz', xgb=oof_xgb, lgbm=oof_lgbm, cat=oof_cat,
         mlp=oof_mlp, rf=oof_rf, y=y)
(ART / 'tuned_params.json').write_text(_json.dumps(
    {'xgb': xgb_p, 'lgbm': lgbm_p, 'cat': cat_p,
     'oof_f1': {'xgb': xgb_f1, 'lgbm': lgbm_f1, 'cat': cat_f1}}, indent=2),
    encoding='utf-8')
print(f"  cached base OOF + params -> {ART.name}/")

ens_oof = stack_oof if best_thr(stack_oof, y)[1] >= best_thr(blend_oof, y)[1] else blend_oof
ens_kind = 'stack' if ens_oof is stack_oof else 'blend'
print(f"  -> using {ens_kind} as ensemble")


# ============================================================ 5. THRESHOLD STRATEGY
print("\n" + "=" * 70 + "\n5. THRESHOLD STRATEGY (honest repeated CV on OOF)\n" + "=" * 70)


def eval_strategy(oof, strata, kind, n_repeats=10):
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RS)
    fs = []
    for trn, va in rcv.split(oof.reshape(-1, 1), y):
        if kind == 'global':
            t = best_thr(oof[trn], y[trn])[0]
            pred = (oof[va] >= t).astype(int)
        elif kind == 'stratum':
            gt = best_thr(oof[trn], y[trn])[0]
            pred = np.zeros(len(va), dtype=int)
            for s in ['both', 'one', 'none']:
                mtr, mva = strata[trn] == s, strata[va] == s
                t = best_thr(oof[trn][mtr], y[trn][mtr])[0] if (mtr.sum() > 40 and y[trn][mtr].sum() > 4) else gt
                pred[mva] = (oof[va][mva] >= t).astype(int)
        elif kind == 'prior':
            bk, bt = PRIOR, 0.5
            best_local = -1
            for K in np.arange(0.07, 0.16, 0.005):
                t = np.quantile(oof[trn], 1 - K)
                f = f1_score(y[trn], (oof[trn] >= t).astype(int))
                if f > best_local:
                    best_local, bk = f, K
            t = np.quantile(oof[trn], 1 - bk)
            pred = (oof[va] >= t).astype(int)
        fs.append(f1_score(y[va], pred))
    return np.mean(fs), np.std(fs)


strat_results = {}
for kind in ['global', 'stratum', 'prior']:
    m, s = eval_strategy(ens_oof, strata_tr, kind)
    strat_results[kind] = m
    print(f"  {kind:<8}: {m:.4f} +/- {s:.4f}")
best_strategy = max(strat_results, key=strat_results.get)
print(f"  -> best threshold strategy: {best_strategy} ({strat_results[best_strategy]:.4f})")


# ============================================================ 6. PSEUDO-LABELING (transductive)
print("\n" + "=" * 70 + "\n6. PSEUDO-LABELING (kept only if repeated CV improves)\n" + "=" * 70)
# train ensemble on full data, score exam, take confident rows as pseudo-labels
exam_xgb = make_xgb().fit(Xtr, y).predict_proba(Xex)[:, 1]
exam_lgbm = make_lgbm().fit(Xtr, y).predict_proba(Xex)[:, 1]
exam_cat = make_cat().fit(Xtr, y).predict_proba(Xex)[:, 1]
exam_blend_prob = np.column_stack([exam_xgb, exam_lgbm, exam_cat]).mean(axis=1)
hi = exam_blend_prob > 0.90
lo = exam_blend_prob < 0.03
print(f"  confident exam rows: {hi.sum()} pos, {lo.sum()} neg")


def repeated_f1_pseudo(n_repeats=3):
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RS)
    fs_base, fs_pseudo = [], []
    Xpl = pd.concat([Xex[hi], Xex[lo]])
    ypl = np.concatenate([np.ones(hi.sum(), int), np.zeros(lo.sum(), int)])
    for trn, va in rcv.split(Xtr, y):
        m = make_xgb(); m.fit(Xtr.iloc[trn], y[trn])
        fs_base.append(best_thr(m.predict_proba(Xtr.iloc[va])[:, 1], y[va])[1])
        Xa = pd.concat([Xtr.iloc[trn], Xpl]); ya = np.concatenate([y[trn], ypl])
        m2 = make_xgb(); m2.fit(Xa, ya)
        fs_pseudo.append(best_thr(m2.predict_proba(Xtr.iloc[va])[:, 1], y[va])[1])
    return np.mean(fs_base), np.mean(fs_pseudo)


pb, pp = repeated_f1_pseudo()
use_pseudo = pp > pb + 0.001
print(f"  XGB no-pseudo={pb:.4f}  with-pseudo={pp:.4f}  -> {'USE' if use_pseudo else 'SKIP'} pseudo-labels")


# ============================================================ 7. FINAL FIT + PREDICTIONS
print("\n" + "=" * 70 + "\n7. FINAL PREDICTIONS\n" + "=" * 70)
Xfit, yfit = Xtr, y
Xfit_s = Xtr_s
if use_pseudo:
    Xpl = pd.concat([Xex[hi], Xex[lo]]); ypl = np.concatenate([np.ones(hi.sum(), int), np.zeros(lo.sum(), int)])
    Xfit = pd.concat([Xtr, Xpl]); yfit = np.concatenate([y, ypl])
    Xfit_s = pd.DataFrame(sc.transform(Xfit), columns=Xfit.columns, index=Xfit.index)

# base models on full (possibly augmented) train
fit_xgb = make_xgb().fit(Xfit, yfit)
fit_lgbm = make_lgbm().fit(Xfit, yfit)
fit_cat = make_cat().fit(Xfit, yfit)
fit_mlp = make_mlp().fit(Xfit_s, yfit)
fit_rf = make_rf().fit(Xfit, yfit)
exam_base = np.column_stack([
    fit_xgb.predict_proba(Xex)[:, 1], fit_lgbm.predict_proba(Xex)[:, 1],
    fit_cat.predict_proba(Xex)[:, 1], fit_mlp.predict_proba(Xex_s)[:, 1],
    fit_rf.predict_proba(Xex)[:, 1]])

if ens_kind == 'stack':
    meta_full = LogisticRegression(max_iter=2000, C=1.0).fit(base_oof, y)
    exam_prob = meta_full.predict_proba(exam_base)[:, 1]
else:
    exam_prob = exam_base.mean(axis=1)

# apply chosen threshold strategy (fit on full OOF, apply to exam)
if best_strategy == 'global':
    t = best_thr(ens_oof, y)[0]
    exam_pred = (exam_prob >= t).astype(int)
    print(f"  global threshold={t:.3f}")
elif best_strategy == 'stratum':
    gt = best_thr(ens_oof, y)[0]
    exam_pred = np.zeros(len(exam_prob), dtype=int)
    for s in ['both', 'one', 'none']:
        mtr, mva = strata_tr == s, strata_ex == s
        t = best_thr(ens_oof[mtr], y[mtr])[0] if (mtr.sum() > 40 and y[mtr].sum() > 4) else gt
        exam_pred[mva] = (exam_prob[mva] >= t).astype(int)
        print(f"  stratum {s:<5} threshold={t:.3f} (n_exam={mva.sum()})")
else:  # prior
    best_local, bk = -1, PRIOR
    for K in np.arange(0.07, 0.16, 0.005):
        t = np.quantile(ens_oof, 1 - K)
        f = f1_score(y, (ens_oof >= t).astype(int))
        if f > best_local:
            best_local, bk = f, K
    t = np.quantile(ens_oof, 1 - bk)
    exam_pred = (exam_prob >= t).astype(int)
    print(f"  prior-aware K={bk:.3f} threshold={t:.3f}")

pd.DataFrame({'label': exam_pred}).to_csv(ROOT / 'predictions.csv', index=False)
print(f"\n  predictions: {exam_pred.sum()} pos ({exam_pred.mean()*100:.1f}%) / {len(exam_pred)-exam_pred.sum()} neg")
print(f"  saved -> predictions.csv")

# verification
print(f"\n  --- verification ---")
print(f"  pred pos | f11 missing : {exam_pred[df_exam['f11'].isna().values].mean()*100:.1f}%")
print(f"  pred pos | f11 present : {exam_pred[df_exam['f11'].notna().values].mean()*100:.1f}%")
print(f"  pred pos | f15 missing : {exam_pred[df_exam['f15'].isna().values].mean()*100:.1f}%")
print(f"  best ensemble OOF F1   : {best_thr(ens_oof, y)[1]:.4f}")
print(f"  honest strategy F1     : {strat_results[best_strategy]:.4f} ({best_strategy})")
print("\nDONE.")
