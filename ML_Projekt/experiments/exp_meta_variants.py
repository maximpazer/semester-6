"""
Stack-Meta-Varianten: teste verschiedene Meta-Learner-Konfigurationen auf S6.
Standalone: .venv/bin/python experiments/exp_meta_variants.py
"""
import json
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
RS   = 42
import sys; sys.path.insert(0, str(ROOT))
from automation.eval_core import (
    load_baseline, per_fold_f1, best_thr, DEV_CV, CONF_CV, make_meta
)

RUNS_DIR = ROOT / "runs"
LB_FILE  = RUNS_DIR / "LEADERBOARD.md"
BASELINE = 0.6938


def stack_with(meta_fn, X, y, cv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_val_predict(meta_fn(), X, y, cv=cv, method="predict_proba")[:, 1]


def logit(S, eps=1e-6):
    S_c = np.clip(S, eps, 1 - eps)
    return np.log(S_c / (1 - S_c))


def rank_norm(S):
    from scipy.stats import rankdata
    return np.column_stack([rankdata(S[:, i]) / len(S) for i in range(S.shape[1])])


def write_results(rid, run_dir, hyp, dev, conf=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "hypothesis.md").write_text(f"# {hyp}\n", encoding="utf-8")
    (run_dir / "dev_metrics.json").write_text(json.dumps(dev, indent=2), encoding="utf-8")
    decision_lines = [f"# Decision: {dev.get('decision', '?')}",
                      f"**Reason:** {dev.get('reason', '')}",
                      "",
                      f"Dev delta: {dev['delta_mean']:+.4f} (SE {dev['delta_se']:.4f})",
                      f"Passes: {dev['passes_to_confirmatory']}"]
    if conf:
        (run_dir / "conf_metrics.json").write_text(json.dumps(conf, indent=2), encoding="utf-8")
        decision_lines += [f"Conf delta: {conf['delta_mean']:+.4f} (SE {conf['delta_se']:.4f})",
                           f"Accepted: {conf['accepted']}"]
    (run_dir / "decision.md").write_text("\n".join(decision_lines), encoding="utf-8")

    # Leaderboard
    conf_d = f"{conf['delta_mean']:+.4f}" if conf else "—"
    conf_se = f"{conf['delta_se']:.4f}" if conf else "—"
    dec = conf.get('decision', dev.get('decision', '?')) if conf else dev.get('decision', '?')
    entry = (f"| {rid} | {hyp[:40]} | {dev['delta_mean']:+.4f} | {dev['delta_se']:.4f} "
             f"| {conf_d} | {conf_se} | {dec} |\n")
    if not LB_FILE.exists():
        header = ("# Leaderboard\n\nBaseline F1: 0.6938\n\n"
                  "| Run | Hypothesis | Dev Δ | Dev SE | Conf Δ | Conf SE | Decision |\n"
                  "|-----|-----------|-------|--------|--------|---------|----------|\n")
        LB_FILE.write_text(header + entry, encoding="utf-8")
    else:
        LB_FILE.write_text(LB_FILE.read_text(encoding="utf-8") + entry, encoding="utf-8")


def run_variant(name, hyp, meta_fn, S_dev, S_conf, y, S6_raw):
    """S6_raw: always raw S6 for baseline. S_dev/S_conf: candidate's (possibly transformed) input."""
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = f"{ts}_{name}"
    run_dir = RUNS_DIR / rid

    dev_splits  = list(DEV_CV.split(S6_raw[:, :1], y))
    conf_splits = list(CONF_CV.split(S6_raw[:, :1], y))

    oof6  = stack_with(make_meta, S6_raw, y, DEV_CV)  # baseline always on raw S6
    oof_c = stack_with(meta_fn,   S_dev,  y, DEV_CV)  # candidate

    f6 = per_fold_f1(oof6,  y, dev_splits)
    fc = per_fold_f1(oof_c, y, dev_splits)
    d  = fc - f6
    se = d.std(ddof=1) / np.sqrt(len(d))
    passes = bool(d.mean() > 0 and d.mean() > se)

    dev = {
        "delta_mean": float(d.mean()), "delta_se": float(se),
        "baseline_mean": float(f6.mean()), "candidate_mean": float(fc.mean()),
        "passes_to_confirmatory": passes, "fold_deltas": d.tolist(),
    }
    print(f"[DEV] {name}: delta={d.mean():+.4f} (SE={se:.4f}) passes={passes}")

    conf = None
    if passes:
        oof6_c  = stack_with(make_meta, S6_raw, y, CONF_CV)
        oof_c_c = stack_with(meta_fn,   S_conf, y, CONF_CV)
        f6c = per_fold_f1(oof6_c,  y, conf_splits)
        fcc = per_fold_f1(oof_c_c, y, conf_splits)
        dc  = fcc - f6c
        sec = dc.std(ddof=1) / np.sqrt(len(dc))
        col = int((dc < -0.04).sum())
        acc = bool(dc.mean() > 2 * sec and col <= 5)
        conf = {
            "delta_mean": float(dc.mean()), "delta_se": float(sec),
            "baseline_mean": float(f6c.mean()), "candidate_mean": float(fcc.mean()),
            "fold_collapse_count": col, "accepted": acc,
            "decision": "KEEP" if acc else "REJECT",
        }
        print(f"[CONF] {name}: delta={dc.mean():+.4f} (SE={sec:.4f}) accepted={acc}")
    else:
        dev["decision"] = "REJECT"
        dev["reason"]   = f"Dev delta non-positive or ≤ SE"

    write_results(rid, run_dir, hyp, dev, conf)
    return dev, conf


def main():
    S6, y = load_baseline()

    S6_logit = logit(S6)
    S6_rank  = rank_norm(S6)

    # === VARIANT 1: Logit-transform inputs to meta-learner ===
    run_variant(
        "meta_logit",
        "Logit-Transform S6 → LogReg(C=1) meta",
        lambda: Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=RS))]),
        S_dev=S6_logit, S_conf=S6_logit, y=y, S6_raw=S6
    )

    # === VARIANT 2: LogReg C=0.1 (stronger regularization) ===
    run_variant(
        "meta_C01",
        "LogReg(C=0.1) meta auf S6 (stärkere Regularisierung)",
        lambda: Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(C=0.1, max_iter=2000, random_state=RS))]),
        S_dev=S6, S_conf=S6, y=y, S6_raw=S6
    )

    # === VARIANT 3: LogReg C=0.01 ===
    run_variant(
        "meta_C001",
        "LogReg(C=0.01) meta auf S6",
        lambda: Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(C=0.01, max_iter=2000, random_state=RS))]),
        S_dev=S6, S_conf=S6, y=y, S6_raw=S6
    )

    # === VARIANT 4: Rank-normalized S6 + LogReg ===
    run_variant(
        "meta_rank",
        "Rank-normierte S6-Inputs → LogReg(C=1) meta",
        lambda: Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=RS))]),
        S_dev=S6_rank, S_conf=S6_rank, y=y, S6_raw=S6
    )

    # === VARIANT 5: Logit + C=0.1 ===
    run_variant(
        "meta_logit_C01",
        "Logit-Transform + LogReg(C=0.1) meta",
        lambda: Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(C=0.1, max_iter=2000, random_state=RS))]),
        S_dev=S6_logit, S_conf=S6_logit, y=y, S6_raw=S6
    )


if __name__ == "__main__":
    main()
