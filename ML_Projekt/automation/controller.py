"""
Main orchestration loop for the ML optimization pipeline.

Usage:
  python automation/controller.py              # full loop (stops on review events)
  python automation/controller.py --run-once   # single experiment then stop
  python automation/controller.py --status     # print state and exit
  python automation/controller.py --dry-run    # generate Qwen request, no subprocess
  python automation/controller.py --eval-only runs/<id>/   # eval existing new_oof.npy
  python automation/controller.py --auto-claude-review     # enable auto Claude review
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
AUTO_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))

from automation import qwen_client, evaluator
from automation.experiment_runner import run_experiment, write_experiment_script
from automation.safety_check import check_script
from automation.preflight import preflight_and_patch

STATE_FILE = AUTO_DIR / "state.json"
QUEUE_FILE = AUTO_DIR / "queue.json"
RUNS_DIR = ROOT / "runs"

BASELINE_F1 = 0.6938
MAX_EXPERIMENTS = 20
MAX_HOURS = 6.0
MAX_CONSECUTIVE_ERRORS = 5
NO_PROGRESS_LIMIT = 8
REVIEW_EVERY_N = 5


# ── State I/O ─────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    state = {
        "start_time": _now_iso(),
        "experiment_count": 0,
        "best_f1_dev": BASELINE_F1,
        "best_f1_conf": BASELINE_F1,
        "best_run_id": None,
        "consecutive_errors": 0,
        "no_progress_count": 0,
        "completed_since_review": 0,
        "needs_review": False,
        "stop_reason": None,
        "auto_claude_review": False,
        "history": [],
    }
    save_state(state)
    return state


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_queue() -> list:
    if QUEUE_FILE.exists():
        return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    return []


def save_queue(queue: list):
    QUEUE_FILE.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Stop condition checks ─────────────────────────────────────────────────────

def check_stop(state: dict) -> Optional[str]:
    if state.get("stop_reason"):
        return state["stop_reason"]
    if state["experiment_count"] >= MAX_EXPERIMENTS:
        return f"max_experiments ({MAX_EXPERIMENTS})"
    if state["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS:
        return f"consecutive_errors ({MAX_CONSECUTIVE_ERRORS})"
    if state["no_progress_count"] >= NO_PROGRESS_LIMIT:
        return f"no_progress ({NO_PROGRESS_LIMIT} experiments)"
    elapsed = (
        datetime.fromisoformat(state["start_time"])
    )
    from datetime import timezone
    now = datetime.now(timezone.utc)
    # Handle both offset-aware and naive
    try:
        hours = (now - elapsed).total_seconds() / 3600
    except TypeError:
        from datetime import datetime as dt
        elapsed_naive = dt.fromisoformat(state["start_time"].replace("Z", ""))
        hours = (dt.utcnow() - elapsed_naive).total_seconds() / 3600
    if hours >= MAX_HOURS:
        return f"max_hours ({MAX_HOURS}h elapsed)"
    return None


def check_f1_target(state: dict) -> bool:
    return state["best_f1_conf"] > 0.70


# ── Run directory creation ────────────────────────────────────────────────────

def make_run_dir(slug: str) -> Tuple[str, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{slug[:40]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


# ── Qwen interaction ──────────────────────────────────────────────────────────

def compact_history(state: dict) -> dict:
    """Build compact history JSON for Qwen context."""
    return {
        "baseline_f1": BASELINE_F1,
        "best_f1_so_far": max(state["best_f1_dev"], BASELINE_F1),
        "completed": [
            {
                "slug": h["run_id"].split("_", 2)[-1] if "_" in h["run_id"] else h["run_id"],
                "type": h.get("experiment_type", "unknown"),
                "delta_dev": h.get("delta_dev"),
                "delta_se_dev": h.get("delta_se_dev"),
                "oof_corr_max": h.get("oof_corr_max"),
                "runtime_s": h.get("runtime_s"),
                "decision": h.get("decision"),
                "reason": h.get("reason", ""),
            }
            for h in state["history"][-10:]  # last 10 only
        ],
        "open_slots": MAX_EXPERIMENTS - state["experiment_count"],
        "budget": "60% exploitation / 30% new_family / 10% risky",
    }


def ask_qwen_for_script(item: dict, run_dir: Path) -> Optional[str]:
    """Ask Qwen to generate an experiment script. Returns Python code or None."""
    user_msg = (
        f"Generate experiment script for hypothesis:\n"
        f"ID: {item['id']}\n"
        f"Type: {item['type']}\n"
        f"Hypothesis: {item['hypothesis_short']}\n\n"
        f"Return JSON with keys: hypothesis, expected_mechanism, risks, script"
    )
    try:
        resp = qwen_client.call(
            qwen_client.SCRIPT_SYSTEM_PROMPT,
            user_msg,
            run_dir,
        )
        content = qwen_client.extract_content(resp)
        parsed = qwen_client.extract_json_block(content)
        return parsed.get("script", "")
    except Exception as e:
        print(f"[QWEN] Script generation failed: {e}")
        return None


def ask_qwen_to_prioritize(state: dict, run_dir: Path) -> list:
    """Ask Qwen for new candidate proposals. Returns list of queue items."""
    history_json = json.dumps(compact_history(state), indent=2)
    system = qwen_client.PRIORITIZE_SYSTEM_PROMPT.format(baseline_f1=BASELINE_F1)
    try:
        resp = qwen_client.call(system, history_json, run_dir)
        content = qwen_client.extract_content(resp)
        parsed = qwen_client.extract_json_block(content)
        return parsed.get("proposals", [])
    except Exception as e:
        print(f"[QWEN] Prioritization failed: {e}")
        return []


# ── Decision helpers ──────────────────────────────────────────────────────────

def write_decision(run_dir: Path, decision: str, reason: str,
                   dev_metrics: Optional[dict] = None,
                   conf_metrics: Optional[dict] = None):
    lines = [f"# Decision: {decision}", f"", f"**Reason:** {reason}", ""]
    if dev_metrics:
        lines += [
            "## Development Metrics",
            f"- Baseline mean F1: {dev_metrics['baseline_mean']:.4f} ± {dev_metrics['baseline_std']:.4f}",
            f"- Candidate mean F1: {dev_metrics['candidate_mean']:.4f} ± {dev_metrics['candidate_std']:.4f}",
            f"- Paired delta: {dev_metrics['delta_mean']:+.4f} (SE {dev_metrics['delta_se']:.4f}, n={dev_metrics['n_folds']})",
            f"- OOF corr max: {dev_metrics['oof_corr_max']:.3f}",
            f"- Passes to confirmatory: {dev_metrics['passes_to_confirmatory']}",
            "",
        ]
    if conf_metrics:
        lines += [
            "## Confirmatory Metrics",
            f"- Baseline mean F1: {conf_metrics['baseline_mean']:.4f} ± {conf_metrics['baseline_std']:.4f}",
            f"- Candidate mean F1: {conf_metrics['candidate_mean']:.4f} ± {conf_metrics['candidate_std']:.4f}",
            f"- Paired delta: {conf_metrics['delta_mean']:+.4f} (SE {conf_metrics['delta_se']:.4f}, n={conf_metrics['n_folds']})",
            f"- Fold collapse count: {conf_metrics['fold_collapse_count']}/50",
            f"- Accepted: {conf_metrics['accepted']}",
            "",
        ]
    (run_dir / "decision.md").write_text("\n".join(lines), encoding="utf-8")


def write_hypothesis(run_dir: Path, item: dict):
    content = (
        f"# Hypothesis: {item['hypothesis_short']}\n\n"
        f"**Type:** {item['type']}\n"
        f"**ID:** {item['id']}\n"
        f"**Expected runtime:** {item.get('expected_runtime', 'unknown')}\n"
    )
    (run_dir / "hypothesis.md").write_text(content, encoding="utf-8")
    config = {
        "experiment_type": item["type"],
        "hypothesis_short": item["hypothesis_short"],
        "timeout_s": item.get("timeout_s", 600),
        "id": item["id"],
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


# ── Review trigger ────────────────────────────────────────────────────────────

def print_review_summary(state: dict, event: str, extra: Optional[dict] = None):
    summary = {
        "event": event,
        "timestamp": _now_iso(),
        "experiment_count": state["experiment_count"],
        "best_f1_dev": state["best_f1_dev"],
        "best_f1_conf": state["best_f1_conf"],
        "best_run_id": state["best_run_id"],
        "consecutive_errors": state["consecutive_errors"],
        "no_progress_count": state["no_progress_count"],
        "recent_history": state["history"][-5:],
    }
    if extra:
        summary.update(extra)
    print("\n" + "=" * 60)
    print(f"[REVIEW NEEDED] Event: {event}")
    print(json.dumps(summary, indent=2))
    print("=" * 60 + "\n")


def maybe_auto_claude_review(state: dict, event: str, extra: Optional[dict] = None) -> Optional[dict]:
    """If --auto-claude-review enabled, call claude CLI. Returns parsed JSON response."""
    if not state.get("auto_claude_review"):
        return None
    import subprocess as sp
    summary = {
        "event": event,
        "best_f1_dev": state["best_f1_dev"],
        "best_f1_conf": state["best_f1_conf"],
        "history": state["history"][-5:],
    }
    if extra:
        summary.update(extra)
    prompt = (
        "You are reviewing an ML optimization loop. "
        "Respond ONLY with JSON: {\"decision\": \"KEEP|REJECT|INVESTIGATE\", \"reason\": \"...\"}.\n\n"
        + json.dumps(summary, indent=2)
    )
    try:
        result = sp.run(
            ["claude", "--print", "--max-turns", "2", prompt],
            capture_output=True, text=True, timeout=120
        )
        return qwen_client.extract_json_block(result.stdout)
    except Exception as e:
        print(f"[AUTO-CLAUDE] Failed: {e}")
        return None


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_one(state: dict, queue: list) -> Tuple[dict, list, bool]:
    """
    Run one experiment from the queue.
    Returns (updated_state, updated_queue, should_stop).
    """
    if not queue:
        print("[CONTROLLER] Queue empty. Asking Qwen to reprioritize...")
        dummy_dir = RUNS_DIR / "_qwen_prioritize"
        dummy_dir.mkdir(exist_ok=True)
        new_items = ask_qwen_to_prioritize(state, dummy_dir)
        if new_items:
            queue.extend(new_items)
            save_queue(queue)
            print(f"[CONTROLLER] Qwen proposed {len(new_items)} new candidates.")
        else:
            print("[CONTROLLER] No new proposals from Qwen. Stopping.")
            state["stop_reason"] = "queue_empty_no_proposals"
            save_state(state)
            return state, queue, True

    item = queue.pop(0)
    save_queue(queue)

    slug = item["id"]
    run_id, run_dir = make_run_dir(slug)
    print(f"\n[RUN] {run_id}")
    print(f"      {item['hypothesis_short']} ({item['type']})")

    write_hypothesis(run_dir, item)

    # Generate script via Qwen — max 2 infra retries before giving up on this item
    script = ask_qwen_for_script(item, run_dir)
    if not script:
        retries = item.get("qwen_retries", 0) + 1
        if retries < 2:
            # Infrastructure failure: push item to queue end, don't count as ML experiment
            item["qwen_retries"] = retries
            queue.append(item)
            save_queue(queue)
            print(f"[QWEN] Infrastructure retry {retries}/2 — '{item['id']}' pushed to queue end.")
            return state, queue, False
        # Exhausted retries: record as infrastructure error (not an ML experiment)
        _record_infra_error(state, run_id, item, "Qwen script generation failed after 2 retries")
        save_state(state)
        return state, queue, False

    write_experiment_script(run_dir, script)

    # Preflight: syntax + auto-fix known Qwen mistakes
    script_path = run_dir / "experiment.py"
    pf_ok, pf_report = preflight_and_patch(script_path)
    for line in pf_report:
        print(f"[PREFLIGHT] {line}")
    (run_dir / "preflight.log").write_text("\n".join(pf_report), encoding="utf-8")
    if not pf_ok:
        _record_infra_error(state, run_id, item,
                            "Preflight syntax error persists after auto-fix")
        save_state(state)
        return state, queue, False

    # Run experiment
    timeout_s = item.get("timeout_s", 600)
    t0 = time.monotonic()
    runner_result = run_experiment(run_dir, timeout_s=timeout_s)
    runtime_s = time.monotonic() - t0

    state["experiment_count"] += 1

    if not runner_result["success"]:
        reason = runner_result.get("error", "unknown error")
        _record_error(state, run_id, item, reason, runtime_s)
        write_decision(run_dir, "ERROR", reason)
        save_state(state)
        _maybe_trigger_review(state, "consecutive_errors")
        return state, queue, state.get("needs_review", False)

    state["consecutive_errors"] = 0

    # Phase A: Development evaluation
    new_oof_path = run_dir / "new_oof.npy"
    try:
        dev_metrics = evaluator.evaluate_dev(new_oof_path, run_dir)
    except Exception as e:
        _record_error(state, run_id, item, f"Dev eval failed: {e}", runtime_s)
        write_decision(run_dir, "ERROR", f"Dev eval: {e}")
        save_state(state)
        return state, queue, False

    # Leakage check
    if evaluator.leakage_suspected(dev_metrics):
        _record_event(state, run_id, item, "SAFETY_BLOCKED",
                      "Leakage suspected", dev_metrics, runtime_s)
        write_decision(run_dir, "SAFETY_BLOCKED",
                       "Leakage suspected: implausibly high dev F1 or delta",
                       dev_metrics=dev_metrics)
        save_state(state)
        print_review_summary(state, "leakage_suspected", {"run_id": run_id})
        state["needs_review"] = True
        return state, queue, True

    passes = dev_metrics["passes_to_confirmatory"]
    delta_dev = dev_metrics["delta_mean"]
    se_dev = dev_metrics["delta_se"]

    print(f"[DEV]  delta={delta_dev:+.4f} (SE={se_dev:.4f})  "
          f"corr_max={dev_metrics['oof_corr_max']:.3f}  passes={passes}")

    if not passes:
        state["no_progress_count"] += 1
        _record_event(state, run_id, item, "REJECT",
                      f"Dev: delta={delta_dev:+.4f} <= 0 and corr_max >= 0.80",
                      dev_metrics, runtime_s)
        write_decision(run_dir, "REJECT",
                       f"Dev delta non-positive and low diversity",
                       dev_metrics=dev_metrics)
        save_state(state)
        _reprioritize_qwen(state, queue, run_dir)
        _maybe_trigger_review(state, "periodic")
        return state, queue, state.get("needs_review", False)

    # Phase B: Confirmatory evaluation
    print("[CONF] Running 5×10 repeated CV...")
    try:
        conf_metrics = evaluator.evaluate_confirmatory(new_oof_path, run_dir)
    except Exception as e:
        _record_error(state, run_id, item, f"Conf eval failed: {e}", runtime_s)
        write_decision(run_dir, "ERROR", f"Conf eval: {e}", dev_metrics=dev_metrics)
        save_state(state)
        return state, queue, False

    accepted = conf_metrics["accepted"]
    delta_conf = conf_metrics["delta_mean"]
    se_conf = conf_metrics["delta_se"]

    print(f"[CONF] delta={delta_conf:+.4f} (SE={se_conf:.4f})  "
          f"accepted={accepted}  collapse={conf_metrics['fold_collapse_count']}/50")

    if accepted:
        state["no_progress_count"] = 0
        state["best_f1_dev"] = max(state["best_f1_dev"], dev_metrics["candidate_mean"])
        state["best_f1_conf"] = max(state["best_f1_conf"], conf_metrics["candidate_mean"])
        state["best_run_id"] = run_id
        decision = "KEEP"
        reason = (f"Conf delta={delta_conf:+.4f} > 2×SE={2*se_conf:.4f}, "
                  f"no fold collapse")
        _record_event(state, run_id, item, decision, reason, dev_metrics, runtime_s,
                      delta_conf=delta_conf, delta_se_conf=se_conf)
        write_decision(run_dir, decision, reason, dev_metrics, conf_metrics)
        save_state(state)
        print(f"[KEEP] New best F1 conf: {state['best_f1_conf']:.4f}")
        # Always trigger review on new winner
        state["needs_review"] = True
        print_review_summary(state, "new_best",
                             {"run_id": run_id, "conf_metrics": conf_metrics})
        maybe_auto_claude_review(state, "new_best",
                                 {"run_id": run_id, "conf_accepted": accepted})
        if check_f1_target(state):
            state["stop_reason"] = f"f1_target_reached ({state['best_f1_conf']:.4f} > 0.70)"
            save_state(state)
            return state, queue, True
    else:
        state["no_progress_count"] += 1
        decision = "REJECT"
        reason = (f"Conf delta={delta_conf:+.4f} not > 2×SE={2*se_conf:.4f} "
                  f"or fold collapse={conf_metrics['fold_collapse_count']}")
        _record_event(state, run_id, item, decision, reason, dev_metrics, runtime_s,
                      delta_conf=delta_conf, delta_se_conf=se_conf)
        write_decision(run_dir, decision, reason, dev_metrics, conf_metrics)
        save_state(state)
        _reprioritize_qwen(state, queue, run_dir)

    state["completed_since_review"] += 1
    _maybe_trigger_review(state, "periodic")
    save_state(state)
    return state, queue, state.get("needs_review", False)


def _record_error(state: dict, run_id: str, item: dict, reason: str,
                  runtime_s: float = 0):
    state["consecutive_errors"] += 1
    state["no_progress_count"] += 1
    state["history"].append({
        "run_id": run_id,
        "hypothesis_short": item.get("hypothesis_short", ""),
        "experiment_type": item.get("type", ""),
        "delta_dev": None,
        "delta_se_dev": None,
        "delta_conf": None,
        "delta_se_conf": None,
        "oof_corr_max": None,
        "runtime_s": round(runtime_s, 1),
        "decision": "ERROR",
        "reason": reason,
    })
    print(f"[ERROR] {reason}")


def _record_infra_error(state: dict, run_id: str, item: dict, reason: str,
                        runtime_s: float = 0):
    """Record a non-ML infrastructure failure. Does NOT increment experiment_count."""
    state["history"].append({
        "run_id": run_id,
        "hypothesis_short": item.get("hypothesis_short", ""),
        "experiment_type": item.get("type", ""),
        "delta_dev": None,
        "delta_se_dev": None,
        "delta_conf": None,
        "delta_se_conf": None,
        "oof_corr_max": None,
        "runtime_s": round(runtime_s, 1),
        "decision": "INFRASTRUCTURE_ERROR",
        "reason": reason,
    })
    print(f"[INFRA-ERROR] {reason}")


def _record_event(state: dict, run_id: str, item: dict, decision: str,
                  reason: str, dev_metrics: dict, runtime_s: float,
                  delta_conf: Optional[float] = None,
                  delta_se_conf: Optional[float] = None):
    state["history"].append({
        "run_id": run_id,
        "hypothesis_short": item.get("hypothesis_short", ""),
        "experiment_type": item.get("type", ""),
        "delta_dev": round(dev_metrics["delta_mean"], 5),
        "delta_se_dev": round(dev_metrics["delta_se"], 5),
        "delta_conf": round(delta_conf, 5) if delta_conf is not None else None,
        "delta_se_conf": round(delta_se_conf, 5) if delta_se_conf is not None else None,
        "oof_corr_max": round(dev_metrics["oof_corr_max"], 4),
        "runtime_s": round(runtime_s, 1),
        "decision": decision,
        "reason": reason,
    })


def _reprioritize_qwen(state: dict, queue: list, run_dir: Path):
    """After a completed run, ask Qwen for 1-2 new proposals if queue has < 3 items."""
    if len(queue) >= 3:
        return
    print("[QWEN] Reprioritizing...")
    new_items = ask_qwen_to_prioritize(state, run_dir)
    if new_items:
        # Deduplicate against existing queue and history
        existing_ids = {h["run_id"].split("_", 2)[-1] for h in state["history"]}
        existing_ids |= {q["id"] for q in queue}
        fresh = [n for n in new_items if n["id"] not in existing_ids]
        queue.extend(fresh)
        save_queue(queue)
        print(f"[QWEN] Added {len(fresh)} new candidates.")


def _maybe_trigger_review(state: dict, event: str):
    if event == "consecutive_errors" and state["consecutive_errors"] >= 3:
        state["needs_review"] = True
        state["completed_since_review"] = 0
        print_review_summary(state, "consecutive_errors")
        maybe_auto_claude_review(state, "consecutive_errors")
    elif event == "periodic" and state["completed_since_review"] >= REVIEW_EVERY_N:
        state["needs_review"] = True
        state["completed_since_review"] = 0
        print_review_summary(state, f"periodic_review_every_{REVIEW_EVERY_N}")
        maybe_auto_claude_review(state, "periodic")


# ── CLI entry points ──────────────────────────────────────────────────────────

def cmd_status(state: dict, queue: list):
    stop = check_stop(state)
    print(json.dumps({
        "experiment_count": state["experiment_count"],
        "best_f1_dev": state["best_f1_dev"],
        "best_f1_conf": state["best_f1_conf"],
        "best_run_id": state["best_run_id"],
        "consecutive_errors": state["consecutive_errors"],
        "no_progress_count": state["no_progress_count"],
        "needs_review": state["needs_review"],
        "stop_reason": state["stop_reason"],
        "stop_check": stop,
        "queue_length": len(queue),
        "queue_next": queue[0]["id"] if queue else None,
        "auto_claude_review": state.get("auto_claude_review", False),
    }, indent=2))


def cmd_eval_only(run_dir_path: str, state: dict):
    run_dir = Path(run_dir_path)
    new_oof = run_dir / "new_oof.npy"
    if not new_oof.exists():
        print(f"[ERROR] {new_oof} does not exist.")
        sys.exit(1)
    print("[DEV] Running development evaluation...")
    dev = evaluator.evaluate_dev(new_oof, run_dir)
    print(json.dumps(dev, indent=2))
    if dev["passes_to_confirmatory"]:
        print("[CONF] Running confirmatory evaluation...")
        conf = evaluator.evaluate_confirmatory(new_oof, run_dir)
        print(json.dumps(conf, indent=2))


def _cmd_dry_run(queue: list):
    """
    State-neutral dry-run: peek at queue[0], generate script, run AST+preflight checks.
    Never modifies queue.json, state.json, or any counters.
    Uses a fixed _dry_run/ directory that is overwritten each time.
    """
    if not queue:
        print("[DRY-RUN] Queue is empty.")
        return

    item = queue[0]  # peek only — no pop
    print(f"\n[DRY-RUN] Peeking at queue[0]: {item['id']}")
    print(f"          {item['hypothesis_short']} ({item['type']})")

    dry_dir = RUNS_DIR / "_dry_run"
    dry_dir.mkdir(parents=True, exist_ok=True)

    script = ask_qwen_for_script(item, dry_dir)
    if not script:
        print("[DRY-RUN] Qwen did not return a script.")
        return

    script_path = dry_dir / "experiment.py"
    write_experiment_script(dry_dir, script)
    print(f"[DRY-RUN] Script written to {script_path}")

    violations = check_script(script_path, "_dry_run")
    if violations:
        print(f"[SAFETY] {len(violations)} violation(s):")
        for v in violations:
            print(f"  ✗ {v}")
    else:
        print("[SAFETY] AST check passed.")

    pf_ok, pf_report = preflight_and_patch(script_path)
    for line in pf_report:
        print(f"[PREFLIGHT] {line}")

    print(f"[DRY-RUN] Queue unchanged. Queue length: {len(queue)}. Next: {queue[0]['id']}")
    print("[DRY-RUN] State unchanged. No training performed.")


def main():
    parser = argparse.ArgumentParser(description="ML Optimization Loop Controller")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", metavar="RUN_DIR", default=None)
    parser.add_argument("--auto-claude-review", action="store_true")
    args = parser.parse_args()

    state = load_state()
    queue = load_queue()

    if args.auto_claude_review:
        state["auto_claude_review"] = True
        save_state(state)

    if args.status:
        cmd_status(state, queue)
        return

    if args.eval_only:
        cmd_eval_only(args.eval_only, state)
        return

    RUNS_DIR.mkdir(exist_ok=True)

    if args.dry_run:
        # State-neutral dry-run: peek at queue[0] without modifying queue or state
        _cmd_dry_run(queue)
        return

    # Reset start_time if this is a fresh start (no experiments yet)
    if state["experiment_count"] == 0 and not state.get("stop_reason"):
        state["start_time"] = _now_iso()
        save_state(state)

    # Main loop
    print(f"[CONTROLLER] Starting. Baseline F1={BASELINE_F1}. Queue: {len(queue)} items.")
    while True:
        stop = check_stop(state)
        if stop:
            state["stop_reason"] = stop
            save_state(state)
            print(f"[CONTROLLER] STOP: {stop}")
            print_review_summary(state, f"stop:{stop}")
            break

        if state.get("needs_review"):
            print("[CONTROLLER] Paused. Set needs_review=false in state.json to continue.")
            break

        state, queue, should_stop = run_one(state, queue)
        save_state(state)
        save_queue(queue)

        if should_stop:
            break

        if args.run_once:
            print("[CONTROLLER] --run-once: stopping after one experiment.")
            break

    print("[CONTROLLER] Done.")


if __name__ == "__main__":
    main()
