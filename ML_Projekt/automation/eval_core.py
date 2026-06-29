"""
Standalone paired evaluation engine. Imported by all experiment scripts.
No side effects at import time.
"""
import json
import warnings
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
RS = 42

# Fixed CV splitters
DEV_CV  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
CONF_CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RS)


def load_baseline():
    """Returns (S6 [8000,6], y [8000])."""
    b = np.load(ROOT / "artifacts" / "base_oof.npz")
    t = np.load(ROOT / "artifacts" / "tabpfn_oof.npz")
    y = b["y"]
    assert np.array_equal(y, t["y"])
    S6 = np.column_stack([b["xgb"], b["lgbm"], b["cat"], b["mlp"], b["rf"], t["tab"]])
    return S6, y


def make_meta():
    """StandardScaler + LogReg meta-learner (fixes overflow with extreme OOF values)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=RS)),
    ])


def best_thr(prob, yt):
    bv, bt = 0.0, 0.5
    for th in np.arange(0.02, 0.95, 0.005):
        f = f1_score(yt, prob >= th)
        if f > bv:
            bv, bt = f, th
    return bt, bv


def stack_oof(X, y, cv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_val_predict(make_meta(), X, y, cv=cv, method="predict_proba")[:, 1]


def per_fold_f1(oof, y, splits):
    out = []
    for trn, va in splits:
        th = best_thr(oof[trn], y[trn])[0]
        out.append(f1_score(y[va], oof[va] >= th))
    return np.array(out)


def oof_corr(new_col, S6):
    """Pearson correlation of new_col against each S6 column."""
    return np.array([float(np.corrcoef(new_col, S6[:, i])[0, 1]) for i in range(6)])


def evaluate_meta_variant(make_meta_fn, run_dir: Path, transform_fn=None) -> dict:
    """
    Paired eval for a different meta-learner or input transform on fixed S6.
    make_meta_fn(): returns a new unfitted sklearn estimator.
    transform_fn(S): optional, transforms S6 before fitting meta-learner.
    Baseline is always StandardScaler+LogReg(C=1) on raw S6.
    """
    S6, y = load_baseline()
    run_dir = Path(run_dir)

    S_in = transform_fn(S6) if transform_fn else S6

    def stack_variant(X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cross_val_predict(
                make_meta_fn(), X, y, cv=DEV_CV, method="predict_proba"
            )[:, 1]

    dev_splits = list(DEV_CV.split(S6[:, :1], y))
    oof6 = stack_oof(S6, y, DEV_CV)        # baseline
    oof_c = stack_variant(S_in)            # candidate

    f6 = per_fold_f1(oof6, y, dev_splits)
    fc = per_fold_f1(oof_c, y, dev_splits)
    d = fc - f6
    se = d.std(ddof=1) / np.sqrt(len(d))
    passes = bool(d.mean() > 0 and (d.mean() > se))

    dev = {
        "phase": "development", "n_folds": 5,
        "baseline_mean": float(f6.mean()), "baseline_std": float(f6.std()),
        "candidate_mean": float(fc.mean()), "candidate_std": float(fc.std()),
        "delta_mean": float(d.mean()), "delta_se": float(se),
        "oof_corr_max": 1.0,  # same features
        "passes_to_confirmatory": passes,
        "fold_deltas": d.tolist(),
    }
    (run_dir / "dev_metrics.json").write_text(json.dumps(dev, indent=2), encoding="utf-8")

    result = {"dev": dev, "conf": None}
    if not passes:
        return result

    # Confirmatory
    conf_splits = list(CONF_CV.split(S6[:, :1], y))
    f6c = per_fold_f1(oof6, y, conf_splits)  # NOTE: oof6 was computed on dev_cv folds only
    # Recompute oof6 and oof_c on conf cv properly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof6_full  = cross_val_predict(make_meta(), S6,   y, cv=CONF_CV, method="predict_proba")[:, 1]
        oof_c_full = cross_val_predict(make_meta_fn(), S_in, y, cv=CONF_CV, method="predict_proba")[:, 1]
    f6c  = per_fold_f1(oof6_full,  y, conf_splits)
    fcc  = per_fold_f1(oof_c_full, y, conf_splits)
    dc   = fcc - f6c
    sec  = dc.std(ddof=1) / np.sqrt(len(dc))
    collapse  = int((dc < -0.04).sum())
    accepted  = bool(dc.mean() > 2 * sec and collapse <= 5)

    conf = {
        "phase": "confirmatory", "n_folds": 50,
        "baseline_mean": float(f6c.mean()), "baseline_std": float(f6c.std()),
        "candidate_mean": float(fcc.mean()), "candidate_std": float(fcc.std()),
        "delta_mean": float(dc.mean()), "delta_se": float(sec),
        "fold_collapse_count": collapse, "accepted": accepted,
        "fold_deltas": dc.tolist(),
    }
    (run_dir / "conf_metrics.json").write_text(json.dumps(conf, indent=2), encoding="utf-8")
    result["conf"] = conf
    return result


def evaluate(new_cols, run_dir: Path, label: str = "candidate") -> dict:
    """
    Full paired evaluation of new_cols (shape [8000] or [8000,K]) vs S6 baseline.
    Writes dev_metrics.json, oof_correlation.csv, and optionally conf_metrics.json.
    Returns metrics dict.
    """
    S6, y = load_baseline()
    run_dir = Path(run_dir)

    if new_cols.ndim == 1:
        new_cols = new_cols.reshape(-1, 1)
    S_cand = np.column_stack([S6, new_cols])

    # OOF correlation (first new column vs S6)
    corr = oof_corr(new_cols[:, 0], S6)
    corr_max = float(corr.max())
    col_names = ["xgb", "lgbm", "cat", "mlp", "rf", "tabpfn"]
    (run_dir / "oof_correlation.csv").write_text(
        "col,pearson_r\n" + "\n".join(f"{n},{v:.6f}" for n, v in zip(col_names, corr)),
        encoding="utf-8"
    )

    # Development eval (5-fold)
    dev_splits = list(DEV_CV.split(S6[:, :1], y))
    oof6 = stack_oof(S6, y, DEV_CV)
    oof_c = stack_oof(S_cand, y, DEV_CV)
    f6 = per_fold_f1(oof6, y, dev_splits)
    fc = per_fold_f1(oof_c, y, dev_splits)
    d = fc - f6
    se = d.std(ddof=1) / np.sqrt(len(d))

    passes = bool(d.mean() > 0 and (d.mean() > se or corr_max < 0.80))

    dev = {
        "phase": "development", "n_folds": 5,
        "baseline_mean": float(f6.mean()), "baseline_std": float(f6.std()),
        "candidate_mean": float(fc.mean()), "candidate_std": float(fc.std()),
        "delta_mean": float(d.mean()), "delta_se": float(se),
        "oof_corr_max": corr_max, "oof_corr_per_col": corr.tolist(),
        "passes_to_confirmatory": passes,
        "fold_deltas": d.tolist(),
    }
    (run_dir / "dev_metrics.json").write_text(json.dumps(dev, indent=2), encoding="utf-8")

    result = {"dev": dev, "conf": None}

    if not passes:
        return result

    # Confirmatory eval (5×10 = 50 folds)
    conf_splits = list(CONF_CV.split(S6[:, :1], y))
    f6c = per_fold_f1(oof6, y, conf_splits)
    fcc = per_fold_f1(oof_c, y, conf_splits)
    dc = fcc - f6c
    sec = dc.std(ddof=1) / np.sqrt(len(dc))
    collapse = int((dc < -0.04).sum())
    accepted = bool(dc.mean() > 2 * sec and collapse <= 5)

    conf = {
        "phase": "confirmatory", "n_folds": 50,
        "baseline_mean": float(f6c.mean()), "baseline_std": float(f6c.std()),
        "candidate_mean": float(fcc.mean()), "candidate_std": float(fcc.std()),
        "delta_mean": float(dc.mean()), "delta_se": float(sec),
        "fold_collapse_count": collapse, "accepted": accepted,
        "fold_deltas": dc.tolist(),
    }
    (run_dir / "conf_metrics.json").write_text(json.dumps(conf, indent=2), encoding="utf-8")

    # fold_scores.csv
    lines = ["fold,f1_baseline,f1_candidate,delta"]
    for i, (a, b_, c) in enumerate(zip(f6c, fcc, dc)):
        lines.append(f"{i},{a:.6f},{b_:.6f},{c:+.6f}")
    (run_dir / "fold_scores.csv").write_text("\n".join(lines), encoding="utf-8")

    result["conf"] = conf
    return result
