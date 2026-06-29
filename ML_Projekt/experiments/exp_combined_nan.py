"""
Combined native-NaN experiment: [S6 + LGBM_NaN + CatBoost_NaN] vs S6.
Reuses saved OOF predictions from previous runs.
Also tests [S6 + LGBM_NaN + XGB_NaN] if XGB OOF available.
Standalone: .venv/bin/python experiments/exp_combined_nan.py
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
RS   = 42
import sys; sys.path.insert(0, str(ROOT))
from automation.eval_core import load_baseline, evaluate

RUNS_DIR = ROOT / "runs"
LB_FILE  = RUNS_DIR / "LEADERBOARD.md"

LGBM_NAN_OOF = RUNS_DIR / "20260629_162030_lgbm_native_nan" / "new_oof.npy"
CAT_NAN_OOF  = RUNS_DIR / "20260629_162158_catboost_native_nan" / "new_oof.npy"


def write_entry(rid, hyp, dev, conf):
    decision = ("KEEP" if conf and conf["accepted"] else
                "REJECT" if (conf and not conf["accepted"]) or not dev["passes_to_confirmatory"] else
                "FOLLOW-UP")
    reason = (f"Conf delta={conf['delta_mean']:+.4f}" if conf else
              f"Dev delta={dev['delta_mean']:+.4f}")

    run_dir = RUNS_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "hypothesis.md").write_text(f"# {hyp}\n", encoding="utf-8")
    (run_dir / "dev_metrics.json").write_text(json.dumps(dev, indent=2), encoding="utf-8")
    if conf:
        (run_dir / "conf_metrics.json").write_text(json.dumps(conf, indent=2), encoding="utf-8")
    (run_dir / "decision.md").write_text(
        "\n".join([f"# Decision: {decision}", f"**Reason:** {reason}",
                   f"Dev delta: {dev['delta_mean']:+.4f} (SE {dev['delta_se']:.4f})"] +
                  ([f"Conf delta: {conf['delta_mean']:+.4f}"] if conf else [])),
        encoding="utf-8"
    )

    print(f"[DEV]  delta={dev['delta_mean']:+.4f} (SE={dev['delta_se']:.4f})  "
          f"corr_max={dev['oof_corr_max']:.3f}  passes={dev['passes_to_confirmatory']}")
    if conf:
        print(f"[CONF] delta={conf['delta_mean']:+.4f} (SE={conf['delta_se']:.4f})  "
              f"accepted={conf['accepted']}")
    print(f"[{decision}] {reason}")

    conf_d  = f"{conf['delta_mean']:+.4f}" if conf else "—"
    conf_se = f"{conf['delta_se']:.4f}"    if conf else "—"
    entry = (f"| {rid} | {hyp[:40]} | {dev['delta_mean']:+.4f} | {dev['delta_se']:.4f} "
             f"| {conf_d} | {conf_se} | {decision} |\n")
    if not LB_FILE.exists():
        header = ("# Leaderboard\n\nBaseline F1: 0.6938\n\n"
                  "| Run | Hypothesis | Dev Δ | Dev SE | Conf Δ | Conf SE | Decision |\n"
                  "|-----|-----------|-------|--------|--------|---------|----------|\n")
        LB_FILE.write_text(header + entry, encoding="utf-8")
    else:
        LB_FILE.write_text(LB_FILE.read_text(encoding="utf-8") + entry, encoding="utf-8")

    return decision


def main():
    lgbm_oof = np.load(LGBM_NAN_OOF)
    cat_oof  = np.load(CAT_NAN_OOF)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # COMBINATION A: LGBM_NaN + CatBoost_NaN
    combined_lc = np.column_stack([lgbm_oof, cat_oof])
    rid_a   = f"{ts}_nan_lgbm_cat_combined"
    (RUNS_DIR / rid_a).mkdir(parents=True, exist_ok=True)
    result_a = evaluate(combined_lc, RUNS_DIR / rid_a)
    write_entry(rid_a, "S6 + LGBM_NaN + CatBoost_NaN combined", result_a["dev"], result_a.get("conf"))

    # COMBINATION B: LGBM_NaN averaged with CatBoost_NaN (single ensemble column)
    avg_oof = (lgbm_oof + cat_oof) / 2.0
    rid_b   = f"{ts}_nan_lgbm_cat_avg"
    (RUNS_DIR / rid_b).mkdir(parents=True, exist_ok=True)
    result_b = evaluate(avg_oof, RUNS_DIR / rid_b)
    write_entry(rid_b, "S6 + avg(LGBM_NaN, CatBoost_NaN)", result_b["dev"], result_b.get("conf"))


if __name__ == "__main__":
    main()
