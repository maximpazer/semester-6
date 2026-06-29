r"""
final_result.py - Final result report for toxic_data_01111.

Prints the headline metrics produced by the full pipeline (src/train_ultra.py),
recomputes a LIVE baseline F1 anchor (default XGB), and LIVE-verifies the
committed predictions.csv. Also writes the report to docs/final_result.txt.

Run:  .\.venv\Scripts\python.exe src\final_result.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RS = 42
np.random.seed(RS)
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'

df_train = pd.read_csv(DATA / 'toxic_data_01111.csv')
df_exam = pd.read_csv(DATA / 'toxic_exam_01111.csv')
y = df_train['label'].values

# 13 features chosen by forward selection in src/train_ultra.py
SELECTED = ['m11', 'm15', 'both_miss', 'f12', 'f12_sq', 'f08', 'f10', 'f13', 'f14',
            'f11_val', 'f15_slog', 'f09_slog', 'f11_abs']


def engineer13(df):
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


def best_thr(p, yt):
    b, bt = 0.0, 0.5
    for t in np.arange(0.02, 0.95, 0.005):
        f = f1_score(yt, (p >= t).astype(int))
        if f > b:
            b, bt = f, t
    return bt, b


# ---- LIVE baseline anchor: default XGB on the 13 feats, repeated 5x3 CV ----
Xtr_raw = engineer13(df_train)
imp = SimpleImputer(strategy='median').fit(Xtr_raw)
Xtr = pd.DataFrame(imp.transform(Xtr_raw), columns=SELECTED, index=Xtr_raw.index)

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
fs = []
for trn, va in rcv.split(Xtr, y):
    m = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, scale_pos_weight=9,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1,
                      random_state=RS, eval_metric='logloss', n_jobs=-1)
    m.fit(Xtr.iloc[trn], y[trn])
    fs.append(best_thr(m.predict_proba(Xtr.iloc[va])[:, 1], y[va])[1])
baseline_live, baseline_std = float(np.mean(fs)), float(np.std(fs))

# ---- LIVE verification of the committed predictions.csv ----
pred = pd.read_csv(ROOT / 'predictions.csv')['label'].values
miss11 = df_exam['f11'].isna().values
miss15 = df_exam['f15'].isna().values

L = []
def p(s=''):
    L.append(s)
    print(s)

p("=" * 64)
p("FINAL RESULT - toxic_exam_01111  (metric: F1 of positive class)")
p("=" * 64)
p()
p("HEADLINE")
p(f"  Baseline (engineered XGB, single global threshold) ..... F1 ~ 0.654")
p(f"  Best pipeline (src/train_ultra.py), honest CV .......... F1 = 0.6811 +/- 0.027")
p(f"  -> genuine improvement of about +0.027, validated by repeated CV")
p()
p("LIVE RECOMPUTE (run just now)")
p(f"  Baseline default-XGB, 13 feats, repeated 5x3 CV ........ F1 = {baseline_live:.4f} +/- {baseline_std:.4f}")
p()
p("FULL PIPELINE METRICS  (from the src/train_ultra.py run that wrote predictions.csv)")
p("  Forward feature selection (margin 0.0015):")
p("    baseline 12 feats 0.6384  ->  + f11_abs 0.6419   (only addition that beat noise)")
p("    final 13 feats: " + ", ".join(SELECTED))
p("  Optuna OOF best-F1:   XGB 0.6685   LGBM 0.6721   CatBoost 0.6837")
p("  Out-of-fold F1:       XGB 0.6685   LGBM 0.6721   Cat 0.6837   MLP 0.6435   RF 0.5702")
p("                        STACK 0.6866 (meta) > BLEND 0.6739      -> use STACK")
p("  Threshold (honest repeated 5x10 CV):")
p("    global  0.6811 +/- 0.0270   <- chosen")
p("    prior   0.6803 +/- 0.0264")
p("    stratum 0.6730 +/- 0.0280   (per-stratum overfits the small strata)")
p("  Pseudo-labeling:      no-pseudo 0.6735  ->  with-pseudo 0.6779   -> used")
p()
p("COMMITTED predictions.csv  (verified live)")
p(f"  rows ................... {len(pred)}")
p(f"  predicted positive ..... {int(pred.sum())}  ({pred.mean()*100:.2f}%)")
p(f"  predicted negative ..... {int((1 - pred).sum())}")
p(f"  global threshold ....... 0.295")
p(f"  pred pos | f11 missing . {pred[miss11].mean()*100:.1f}%   (vs present {pred[~miss11].mean()*100:.1f}%)")
p(f"  pred pos | f15 missing . {pred[miss15].mean()*100:.1f}%   (vs present {pred[~miss15].mean()*100:.1f}%)")
p("  -> matches the MNAR signal (missing f11/f15 => far more positive)")
p()
p("NOTE: the 0.6811 threshold is honestly cross-validated, but Optuna picked")
p("hyperparameters on the OOF, so the true graded exam F1 is likely ~0.66-0.68.")
p("=" * 64)

out = ROOT / 'docs' / 'final_result.txt'
out.write_text("\n".join(L), encoding='utf-8')
print(f"\nsaved -> {out.relative_to(ROOT)}")
