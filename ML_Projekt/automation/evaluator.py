"""
Evaluation logic: Development (5-fold) and Confirmatory (5x10 repeated) paired tests.
Functions copied from src/evaluation/eval_tabpfn_combo.py to avoid side-effect imports.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

RS = 42
ROOT = Path(__file__).resolve().parent.parent

# ── Shared primitives ─────────────────────────────────────────────────────────

def best_thr(p: np.ndarray, yt: np.ndarray) -> tuple[float, float]:
    bv, bt = 0.0, 0.5
    for th in np.arange(0.02, 0.95, 0.005):
        f = f1_score(yt, (p >= th).astype(int))
        if f > bv:
            bv, bt = f, th
    return bt, bv


def stack_oof(base_matrix: np.ndarray, y: np.ndarray, cv) -> np.ndarray:
    meta = LogisticRegression(max_iter=2000, C=1.0)
    return cross_val_predict(
        meta, base_matrix, y,
        cv=cv, method="predict_proba"
    )[:, 1]


def per_fold_f1(oof: np.ndarray, y: np.ndarray, splits: list) -> np.ndarray:
    out = []
    for trn, va in splits:
        th = best_thr(oof[trn], y[trn])[0]
        out.append(f1_score(y[va], (oof[va] >= th).astype(int)))
    return np.array(out)


def oof_correlations(new_col: np.ndarray, base6: np.ndarray) -> np.ndarray:
    """Pearson correlation of new_col against each of the 6 baseline OOF columns."""
    return np.array([
        float(np.corrcoef(new_col, base6[:, i])[0, 1])
        for i in range(base6.shape[1])
    ])


# ── Load baseline OOF (read-only) ─────────────────────────────────────────────

def load_baseline() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (base6_matrix [8000,6], y [8000], tab_col [8000])."""
    b = np.load(ROOT / "artifacts" / "base_oof.npz")
    t = np.load(ROOT / "artifacts" / "tabpfn_oof.npz")
    y = b["y"]
    assert np.array_equal(y, t["y"]), "y mismatch between base_oof and tabpfn_oof"
    base6 = np.column_stack([b["xgb"], b["lgbm"], b["cat"], b["mlp"], b["rf"], t["tab"]])
    return base6, y, t["tab"]


# ── Development evaluation (5-fold, fast) ─────────────────────────────────────

def evaluate_dev(new_oof_path: Path, run_dir: Path) -> dict:
    """
    Phase A: 5-fold development evaluation.
    Compares Stack-6 baseline against Stack-7 (baseline + new_oof).
    Returns metrics dict and writes dev_metrics.json.
    """
    base6, y, _ = load_baseline()
    new_col = np.load(new_oof_path)

    if new_col.ndim == 2:
        # Multiple new columns
        stack7 = np.column_stack([base6, new_col])
    else:
        stack7 = np.column_stack([base6, new_col])

    # OOF correlations
    corr = oof_correlations(new_col if new_col.ndim == 1 else new_col[:, 0], base6)
    corr_max = float(corr.max())

    dev_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    dev_splits = list(dev_cv.split(base6[:, :1], y))

    oof6 = stack_oof(base6, y, dev_cv)
    oof7 = stack_oof(stack7, y, dev_cv)

    f6 = per_fold_f1(oof6, y, dev_splits)
    f7 = per_fold_f1(oof7, y, dev_splits)
    d = f7 - f6
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0

    # Continuation criterion: delta_mean > 0 AND (delta_mean > delta_se OR corr_max < 0.80)
    passes = bool(d.mean() > 0 and (d.mean() > se or corr_max < 0.80))

    metrics = {
        "phase": "development",
        "n_folds": int(len(d)),
        "baseline_mean": float(f6.mean()),
        "baseline_std": float(f6.std()),
        "candidate_mean": float(f7.mean()),
        "candidate_std": float(f7.std()),
        "delta_mean": float(d.mean()),
        "delta_se": float(se),
        "oof_corr_max": corr_max,
        "oof_corr_per_col": corr.tolist(),
        "passes_to_confirmatory": passes,
        "fold_deltas": d.tolist(),
    }
    (run_dir / "dev_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Save OOF correlations CSV
    col_names = ["xgb", "lgbm", "cat", "mlp", "rf", "tabpfn"]
    csv_lines = ["col,pearson_r"] + [f"{n},{v:.6f}" for n, v in zip(col_names, corr)]
    (run_dir / "oof_correlation.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    return metrics


# ── Confirmatory evaluation (5×10 repeated, expensive) ────────────────────────

def evaluate_confirmatory(new_oof_path: Path, run_dir: Path) -> dict:
    """
    Phase B: 50-fold paired confirmatory evaluation.
    Identical protocol to eval_tabpfn_combo.py.
    Acceptance: delta_mean > 2*delta_se AND no fold collapse.
    """
    base6, y, _ = load_baseline()
    new_col = np.load(new_oof_path)

    if new_col.ndim == 2:
        stack7 = np.column_stack([base6, new_col])
    else:
        stack7 = np.column_stack([base6, new_col])

    conf_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RS)
    conf_splits = list(conf_cv.split(base6[:, :1], y))

    dev_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    oof6 = stack_oof(base6, y, dev_cv)
    oof7 = stack_oof(stack7, y, dev_cv)

    f6 = per_fold_f1(oof6, y, conf_splits)
    f7 = per_fold_f1(oof7, y, conf_splits)
    d = f7 - f6
    se = d.std(ddof=1) / np.sqrt(len(d))

    # Fold collapse check: not more than 5/50 folds with delta < -0.04
    fold_collapse = int((d < -0.04).sum())
    no_collapse = fold_collapse <= 5

    accepted = bool(d.mean() > 2 * se and no_collapse)

    metrics = {
        "phase": "confirmatory",
        "n_folds": int(len(d)),
        "baseline_mean": float(f6.mean()),
        "baseline_std": float(f6.std()),
        "candidate_mean": float(f7.mean()),
        "candidate_std": float(f7.std()),
        "delta_mean": float(d.mean()),
        "delta_se": float(se),
        "fold_collapse_count": fold_collapse,
        "no_fold_collapse": no_collapse,
        "accepted": accepted,
        "fold_deltas": d.tolist(),
    }
    (run_dir / "conf_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # fold_scores.csv: all 50 per-fold values
    csv_lines = ["fold,f1_baseline,f1_candidate,delta"]
    for i, (fv6, fv7, dv) in enumerate(zip(f6, f7, d)):
        csv_lines.append(f"{i},{fv6:.6f},{fv7:.6f},{dv:+.6f}")
    (run_dir / "fold_scores.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    return metrics


# ── Leakage heuristic ─────────────────────────────────────────────────────────

def leakage_suspected(dev_metrics: dict) -> bool:
    """Flag suspiciously large dev gains or implausibly high F1."""
    return (
        dev_metrics["candidate_mean"] > 0.76
        or dev_metrics["delta_mean"] > 0.04
    )
