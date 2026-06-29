"""
LightGBM native NaN: raw 16 features + missing indicators, no imputation.
Tests if NaN-native split logic captures signal beyond imputed version.
Standalone: .venv/bin/python experiments/exp_lgbm_native_nan.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
RS   = 42
import sys; sys.path.insert(0, str(ROOT))
from automation.eval_core import load_baseline, evaluate

RUNS_DIR = ROOT / "runs"
LB_FILE  = RUNS_DIR / "LEADERBOARD.md"


def get_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """16 raw features + 3 missing indicators. NaN kept as-is for LGBM."""
    X = df[['f00','f01','f02','f03','f04','f05','f06','f07',
             'f08','f09','f10','f11','f12','f13','f14','f15']].copy()
    X['m11']       = df['f11'].isna().astype(float)
    X['m15']       = df['f15'].isna().astype(float)
    X['both_miss'] = (df['f11'].isna() & df['f15'].isna()).astype(float)
    return X


def main():
    # Load data
    for p in [ROOT/"data"/"data"/"toxic_data_01111.csv", ROOT/"data"/"toxic_data_01111.csv"]:
        if p.exists(): df = pd.read_csv(p); break

    Xraw = get_raw_features(df)
    y    = df['label'].values
    oof  = np.zeros(len(y), dtype=np.float64)

    # Tuned LGBM params (from artifacts/tuned_params.json)
    params = json.loads((ROOT / "artifacts" / "tuned_params.json").read_text())["lgbm"]
    model_params = {
        "n_estimators":      params["n_estimators"],
        "max_depth":         params["max_depth"],
        "learning_rate":     params["learning_rate"],
        "num_leaves":        params["num_leaves"],
        "subsample":         params["subsample"],
        "colsample_bytree":  params["colsample_bytree"],
        "min_child_samples": params["min_child_samples"],
        "reg_lambda":        params["reg_lambda"],
        "scale_pos_weight":  params["scale_pos_weight"],
        "verbose": -1, "n_jobs": -1, "random_state": RS,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    for trn, val in skf.split(Xraw, y):
        m = lgb.LGBMClassifier(**model_params)
        m.fit(Xraw.iloc[trn], y[trn])
        oof[val] = m.predict_proba(Xraw.iloc[val])[:, 1]

    # Save and evaluate
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = f"{ts}_lgbm_native_nan"
    run_dir = RUNS_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "new_oof.npy", oof)
    (run_dir / "hypothesis.md").write_text(
        "# lgbm_native_nan\n\nLGBM nativ-NaN auf 16 Rohfeatures + 3 Missing-Indikatoren\n",
        encoding="utf-8"
    )

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
              f"Dev delta={dev['delta_mean']:+.4f} non-positive / low diversity")

    dec_lines = [f"# Decision: {decision}", f"**Reason:** {reason}", "",
                 f"Dev delta: {dev['delta_mean']:+.4f} (SE {dev['delta_se']:.4f})",
                 f"OOF corr max: {dev['oof_corr_max']:.3f}"]
    if conf:
        dec_lines.append(f"Conf delta: {conf['delta_mean']:+.4f} (SE {conf['delta_se']:.4f})")
    (run_dir / "decision.md").write_text("\n".join(dec_lines), encoding="utf-8")
    print(f"[{decision}] {reason}")

    entry = (f"| {rid} | lgbm_native_nan | {dev['delta_mean']:+.4f} | {dev['delta_se']:.4f} "
             f"| {conf['delta_mean']:+.4f} | {conf['delta_se']:.4f} | {decision} |\n"
             if conf else
             f"| {rid} | lgbm_native_nan | {dev['delta_mean']:+.4f} | {dev['delta_se']:.4f} "
             f"| — | — | {decision} |\n")
    if not LB_FILE.exists():
        header = ("# Leaderboard\n\nBaseline F1: 0.6938\n\n"
                  "| Run | Hypothesis | Dev Δ | Dev SE | Conf Δ | Conf SE | Decision |\n"
                  "|-----|-----------|-------|--------|--------|---------|----------|\n")
        LB_FILE.write_text(header + entry, encoding="utf-8")
    else:
        LB_FILE.write_text(LB_FILE.read_text(encoding="utf-8") + entry, encoding="utf-8")


if __name__ == "__main__":
    main()
