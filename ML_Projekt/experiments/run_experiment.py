"""
Lightweight experiment runner. Called directly:
  .venv/bin/python experiments/run_experiment.py <experiment_module>

Each experiment module must define:
  ID          = "slug"
  HYPOTHESIS  = "one line"
  TIMEOUT_S   = 300
  def run(run_dir: Path) -> np.ndarray:  # returns new_oof shape [8000]
"""
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from automation.eval_core import evaluate

RUNS_DIR = ROOT / "runs"
LB_FILE  = ROOT / "runs" / "LEADERBOARD.md"
BASELINE = 0.6938


def _run_dir(slug: str) -> tuple:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = f"{ts}_{slug}"
    d = RUNS_DIR / rid
    d.mkdir(parents=True, exist_ok=True)
    return rid, d


def _write_decision(run_dir: Path, decision: str, reason: str,
                    dev: dict, conf=None):
    lines = [f"# Decision: {decision}", "", f"**Reason:** {reason}", ""]
    m = dev
    lines += [
        "## Development (5-fold)",
        f"- Baseline: {m['baseline_mean']:.4f} ± {m['baseline_std']:.4f}",
        f"- Candidate: {m['candidate_mean']:.4f} ± {m['candidate_std']:.4f}",
        f"- Delta: {m['delta_mean']:+.4f} (SE {m['delta_se']:.4f})",
        f"- OOF corr max: {m['oof_corr_max']:.3f}",
        "",
    ]
    if conf:
        m = conf
        lines += [
            "## Confirmatory (5×10 = 50 folds)",
            f"- Baseline: {m['baseline_mean']:.4f} ± {m['baseline_std']:.4f}",
            f"- Candidate: {m['candidate_mean']:.4f} ± {m['candidate_std']:.4f}",
            f"- Delta: {m['delta_mean']:+.4f} (SE {m['delta_se']:.4f})",
            f"- Fold collapses: {m['fold_collapse_count']}/50",
            f"- Accepted: {m['accepted']}",
            "",
        ]
    (run_dir / "decision.md").write_text("\n".join(lines), encoding="utf-8")


def _update_leaderboard(rid: str, hyp: str, dev: dict,
                        conf, decision: str, runtime_s: float):
    entry = (
        f"| {rid} | {hyp[:40]} | "
        f"{dev['delta_mean']:+.4f} | {dev['delta_se']:.4f} | "
        f"{conf['delta_mean']:+.4f} | {conf['delta_se']:.4f} | "
        f"{decision} | {runtime_s:.0f}s |"
        if conf else
        f"| {rid} | {hyp[:40]} | "
        f"{dev['delta_mean']:+.4f} | {dev['delta_se']:.4f} | "
        f"— | — | {decision} | {runtime_s:.0f}s |"
    )
    header = (
        "| Run | Hypothesis | Dev Δ | Dev SE | Conf Δ | Conf SE | Decision | Time |\n"
        "|-----|-----------|-------|--------|--------|---------|---------|------|\n"
    )
    if not LB_FILE.exists():
        LB_FILE.write_text(f"# Leaderboard\n\nBaseline F1: {BASELINE}\n\n{header}{entry}\n",
                           encoding="utf-8")
    else:
        existing = LB_FILE.read_text(encoding="utf-8")
        LB_FILE.write_text(existing + entry + "\n", encoding="utf-8")


def main(exp_path: str):
    # Load experiment module
    spec = importlib.util.spec_from_file_location("exp", exp_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    slug  = mod.ID
    hyp   = mod.HYPOTHESIS
    print(f"\n[RUN] {slug}")
    print(f"      {hyp}")

    rid, run_dir = _run_dir(slug)
    (run_dir / "hypothesis.md").write_text(
        f"# {slug}\n\n{hyp}\n", encoding="utf-8"
    )

    # Run the experiment
    t0 = time.monotonic()
    new_oof = mod.run(run_dir)
    runtime_s = time.monotonic() - t0
    print(f"[RUN] Finished in {runtime_s:.1f}s. Evaluating...")

    import numpy as np
    np.save(run_dir / "new_oof.npy", new_oof.astype(np.float64))

    # Evaluate
    result = evaluate(new_oof, run_dir)
    dev  = result["dev"]
    conf = result.get("conf")

    print(f"[DEV]  delta={dev['delta_mean']:+.4f} (SE={dev['delta_se']:.4f})  "
          f"corr_max={dev['oof_corr_max']:.3f}  passes={dev['passes_to_confirmatory']}")

    if conf:
        print(f"[CONF] delta={conf['delta_mean']:+.4f} (SE={conf['delta_se']:.4f})  "
              f"accepted={conf['accepted']}  collapse={conf['fold_collapse_count']}/50")

    # Decision
    if conf and conf["accepted"]:
        decision = "KEEP"
        reason   = f"Conf delta={conf['delta_mean']:+.4f} > 2×SE={2*conf['delta_se']:.4f}"
    elif conf and not conf["accepted"]:
        decision = "REJECT"
        reason   = f"Conf delta={conf['delta_mean']:+.4f} not > 2×SE"
    elif dev["passes_to_confirmatory"]:
        decision = "FOLLOW-UP"
        reason   = "Dev passes but conf not run"
    else:
        decision = "REJECT"
        reason   = f"Dev delta={dev['delta_mean']:+.4f} non-positive / low diversity"

    _write_decision(run_dir, decision, reason, dev, conf)
    _update_leaderboard(rid, hyp, dev, conf, decision, runtime_s)

    print(f"[{decision}] {reason}")
    print(f"[RUN] dir: {run_dir}")
    return decision, dev, conf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiments/run_experiment.py experiments/<exp>.py")
        sys.exit(1)
    main(sys.argv[1])
