"""ExtraTrees as 7th base OOF column (13 engineered features, median imputation)."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent.parent
RS   = 42

ID         = "extratrees_13feat"
HYPOTHESIS = "ExtraTrees (500 trees, balanced) auf 13 engineered features als 7. Stack-Spalte"
TIMEOUT_S  = 180

SELECTED = ['m11','m15','both_miss','f12','f12_sq','f08','f10','f13','f14',
            'f11_val','f15_slog','f09_slog','f11_abs']


def engineer(d: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    m11, m15 = d['f11'].isna().astype(int), d['f15'].isna().astype(int)
    X['m11'], X['m15'] = m11, m15
    X['both_miss'] = (m11 & m15).astype(int)
    X['f12']      = d['f12']
    X['f12_sq']   = d['f12'] ** 2
    X['f08']      = d['f08']
    X['f10']      = d['f10']
    X['f13']      = d['f13']
    X['f14']      = d['f14']
    X['f11_val']  = d['f11']
    X['f11_abs']  = d['f11'].abs()
    X['f15_slog'] = np.sign(d['f15']) * np.log1p(d['f15'].abs())
    X['f09_slog'] = np.sign(d['f09']) * np.log1p(d['f09'].abs())
    return X[SELECTED]


def run(run_dir: Path) -> np.ndarray:
    # Load data
    for p in [ROOT/"data"/"data"/"toxic_data_01111.csv", ROOT/"data"/"toxic_data_01111.csv"]:
        if p.exists():
            df = pd.read_csv(p); break
    else:
        raise FileNotFoundError("Training data not found")

    Xraw = engineer(df)
    y    = df['label'].values
    oof  = np.zeros(len(y), dtype=np.float64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    for trn, val in skf.split(Xraw, y):
        imp = SimpleImputer(strategy='median').fit(Xraw.iloc[trn])
        Xtr = imp.transform(Xraw.iloc[trn])
        Xva = imp.transform(Xraw.iloc[val])
        m = ExtraTreesClassifier(
            n_estimators=500,
            max_features='sqrt',
            class_weight='balanced_subsample',
            n_jobs=-1,
            random_state=RS,
        )
        m.fit(Xtr, y[trn])
        oof[val] = m.predict_proba(Xva)[:, 1]

    return oof
