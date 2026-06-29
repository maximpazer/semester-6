"""
TabPFN on raw 16 features + 3 missing indicators (NaN kept as-is).
Tests if TabPFN's native missing-value handling provides different signal
from the current tabpfn_oof.npz (which uses 13 imputed engineered features).
Standalone: TABPFN_ALLOW_CPU_LARGE_DATASET=1 .venv/bin/python experiments/exp_tabpfn_raw.py
"""
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

# Allow large dataset on CPU
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

ROOT = Path(__file__).resolve().parent.parent
RS   = 42
import sys; sys.path.insert(0, str(ROOT))
from automation.eval_core import load_baseline, evaluate

RUNS_DIR = ROOT / "runs"
LB_FILE  = RUNS_DIR / "LEADERBOARD.md"
warnings.filterwarnings("ignore")


def make_tabpfn():
    from tabpfn import TabPFNClassifier
    for kwargs in [
        {"device": "cpu", "ignore_pretraining_limits": True, "random_state": RS},
        {"device": "cpu", "ignore_pretraining_limits": True},
        {"device": "cpu"},
        {},
    ]:
        try:
            return TabPFNClassifier(**kwargs)
        except TypeError:
            continue
    return TabPFNClassifier()


def get_raw_features(df: pd.DataFrame) -> np.ndarray:
    """16 raw features + 3 missing indicators; NaN kept."""
    cols = ['f00','f01','f02','f03','f04','f05','f06','f07',
            'f08','f09','f10','f11','f12','f13','f14','f15']
    X = df[cols].copy()
    X['m11']       = df['f11'].isna().astype(float)
    X['m15']       = df['f15'].isna().astype(float)
    X['both_miss'] = (df['f11'].isna() & df['f15'].isna()).astype(float)
    return X.values.astype(np.float32)


def main():
    for p in [ROOT/"data"/"data"/"toxic_data_01111.csv", ROOT/"data"/"toxic_data_01111.csv"]:
        if p.exists(): df = pd.read_csv(p); break

    Xraw = get_raw_features(df)
    y    = df['label'].values
    oof  = np.zeros(len(y), dtype=np.float64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    t0  = time.time()
    for i, (trn, val) in enumerate(skf.split(Xraw, y)):
        clf = make_tabpfn()
        clf.fit(Xraw[trn], y[trn])
        oof[val] = clf.predict_proba(Xraw[val])[:, 1]
        print(f"  TabPFN-raw fold {i+1}/5 ({time.time()-t0:.0f}s)")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid     = f"{ts}_tabpfn_raw"
    run_dir = RUNS_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "new_oof.npy", oof)
    (run_dir / "hypothesis.md").write_text(
        "# tabpfn_raw\n\nTabPFN auf 16 Rohfeatures + 3 Missing-Indikatoren, NaN nativ\n",
        encoding="utf-8"
    )

    print(f"\nTabPFN-raw OOF fertig ({time.time()-t0:.0f}s). Evaluiere...")
    result = evaluate(oof, run_dir)
    dev    = result["dev"]
    conf   = result.get("conf")

    print(f"[DEV]  delta={dev['delta_mean']:+.4f} (SE={dev['delta_se']:.4f})  "
          f"corr_max={dev['oof_corr_max']:.3f}  passes={dev['passes_to_confirmatory']}")
    if conf:
        print(f"[CONF] delta={conf['delta_mean']:+.4f} (SE={conf['delta_se']:.4f})  "
              f"accepted={conf['accepted']}")

    decision = ("KEEP" if conf and conf["accepted"] else
                "REJECT" if (conf and not conf["accepted"]) or not dev["passes_to_confirmatory"] else
                "FOLLOW-UP")
    reason = (f"Conf delta={conf['delta_mean']:+.4f}" if conf else
              f"Dev delta={dev['delta_mean']:+.4f}")

    (run_dir / "decision.md").write_text(
        "\n".join([f"# Decision: {decision}", f"**Reason:** {reason}",
                   f"Dev delta: {dev['delta_mean']:+.4f} (SE {dev['delta_se']:.4f})",
                   f"OOF corr max: {dev['oof_corr_max']:.3f}"] +
                  ([f"Conf delta: {conf['delta_mean']:+.4f}"] if conf else [])),
        encoding="utf-8"
    )
    print(f"[{decision}] {reason}")

    conf_d  = f"{conf['delta_mean']:+.4f}" if conf else "—"
    conf_se = f"{conf['delta_se']:.4f}"    if conf else "—"
    entry = (f"| {rid} | tabpfn_raw | {dev['delta_mean']:+.4f} | {dev['delta_se']:.4f} "
             f"| {conf_d} | {conf_se} | {decision} |\n")
    if not LB_FILE.exists():
        LB_FILE.write_text(
            "# Leaderboard\n\nBaseline F1: 0.6938\n\n"
            "| Run | Hypothesis | Dev Δ | Dev SE | Conf Δ | Conf SE | Decision |\n"
            "|-----|-----------|-------|--------|--------|---------|----------|\n" + entry,
            encoding="utf-8"
        )
    else:
        LB_FILE.write_text(LB_FILE.read_text(encoding="utf-8") + entry, encoding="utf-8")

    return oof, run_dir


if __name__ == "__main__":
    main()
