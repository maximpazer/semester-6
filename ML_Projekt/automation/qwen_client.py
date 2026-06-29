"""
Thin Ollama HTTP client for qwen3:32b.
All requests and responses are saved to the run directory.
"""
import json
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:32b"
TIMEOUT_S = 300  # 5 min max for Qwen response


def _post(payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _make_payload(system_prompt: str, user_message: str) -> dict:
    return {
        "model": MODEL,
        "stream": False,
        "think": False,       # disable <think> chain-of-thought to prevent token bleed
        "format": "json",     # enforce JSON output at the Ollama level
        "options": {"num_predict": 4096, "temperature": 0.2},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }


def call(system_prompt: str, user_message: str, run_dir: Path) -> dict:
    """
    Send a message to Qwen with automatic retry on empty content or length truncation.
    Saves each attempt as qwen_request_N.json / qwen_response_N.json.
    Returns the first valid (non-empty content) response dict.
    Raises RuntimeError after 2 failed attempts.
    """
    for attempt in range(1, 3):
        payload = _make_payload(system_prompt, user_message)
        request_record = {
            "attempt": attempt,
            "timestamp": datetime.utcnow().isoformat(),
            "model": MODEL,
            "think": False,
            "format": "json",
            "num_predict": 4096,
            "system_prompt_len": len(system_prompt),
            "user_message_len": len(user_message),
        }
        req_file = run_dir / f"qwen_request_{attempt}.json"
        req_file.write_text(
            json.dumps(request_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Keep the canonical name pointing to the latest attempt
        (run_dir / "qwen_request.json").write_text(
            json.dumps(request_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        try:
            response = _post(payload)
        except urllib.error.URLError as e:
            err = {"attempt": attempt, "error": str(e),
                   "timestamp": datetime.utcnow().isoformat()}
            (run_dir / f"qwen_response_{attempt}.json").write_text(
                json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (run_dir / "qwen_response.json").write_text(
                json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise RuntimeError(f"Ollama unreachable: {e}") from e

        response_record = {
            "attempt": attempt,
            "timestamp": datetime.utcnow().isoformat(),
            "done_reason": response.get("done_reason"),
            "eval_count": response.get("eval_count"),
            "response": response,
        }
        (run_dir / f"qwen_response_{attempt}.json").write_text(
            json.dumps(response_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "qwen_response.json").write_text(
            json.dumps(response_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        content = response.get("message", {}).get("content", "")
        done_reason = response.get("done_reason", "")

        # Never extract from message.thinking — only use content
        if content and done_reason != "length":
            return response

        reason = "empty content" if not content else f"done_reason='{done_reason}'"
        print(f"[QWEN] Attempt {attempt} failed ({reason}). "
              + ("Retrying..." if attempt < 2 else "Giving up."))

    raise RuntimeError(
        "Qwen returned empty content or was truncated after 2 attempts. "
        "Check qwen_response_1.json and qwen_response_2.json."
    )


def extract_content(response: dict) -> str:
    """Extract text content from Ollama chat response. Never reads message.thinking."""
    return response.get("message", {}).get("content", "")


def extract_json_block(text: str) -> dict:
    """Extract first JSON object from Qwen's text response."""
    import re
    # Try fenced code block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try raw JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON found in Qwen response: {text[:200]}")


# ── Prompt templates ──────────────────────────────────────────────────────────

SCRIPT_SYSTEM_PROMPT = """You are an ML experiment code generator for a binary classification project.

TASK: Generate a Python experiment script that produces new OOF (out-of-fold) predictions.

RULES (MANDATORY):
1. Output file: new_oof.npy in current working directory (the run folder). Shape: (8000,) float64.
2. Load training data using this exact path resolution (check data/data/ first, then data/):
   ROOT / "data" / "data" / "toxic_data_01111.csv"  (primary)
   ROOT / "data" / "toxic_data_01111.csv"            (fallback)
   Columns: f00, f01, ..., f15, label
3. You MAY load (read-only): artifacts/base_oof.npz, artifacts/tabpfn_oof.npz, artifacts/tuned_params.json
4. NEVER write to: artifacts/base_oof.npz, artifacts/tabpfn_oof.npz, artifacts/tuned_params.json, predictions.csv
5. NEVER use subprocess, os.system, exec, eval.
6. ONLY import from: numpy, pandas, sklearn, xgboost, lightgbm, catboost, tabpfn, scipy, math, os, sys, pathlib, json, warnings, time
7. CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42) — same splits as baseline
8. Set random_state=42 everywhere.
9. The script must be self-contained and run with: python experiment.py (cwd = run folder)
10. ROOT = Path(__file__).resolve().parent.parent.parent  (THREE levels up: experiment.py → run_dir → runs/ → project_root)

PATH TEMPLATE (copy this exactly):
    ROOT = Path(__file__).resolve().parent.parent.parent
    for _p in [ROOT / "data" / "data" / "toxic_data_01111.csv", ROOT / "data" / "toxic_data_01111.csv"]:
        if _p.exists():
            data_path = _p
            break
    else:
        raise FileNotFoundError("Training data not found")

OUTPUT:
- Save OOF probabilities as: numpy.save(Path("new_oof.npy"), oof.astype(numpy.float64))
- oof must have shape (8000,) — one probability per training row

Respond with a JSON object containing exactly these keys:
{"hypothesis": "...", "expected_mechanism": "...", "risks": "...", "script": "...full python code..."}"""

PRIORITIZE_SYSTEM_PROMPT = """You are an ML experiment prioritizer for a binary classification stack.

Baseline F1: {baseline_f1:.4f} (higher is better, target > 0.70).
You will receive a history of completed experiments and must propose 1-3 new candidates.

RULES:
- NEVER repeat a semantically identical experiment (check hypothesis_short in history).
- Label each proposal with category: exploitation or new_family or risky
- Aim for mix: 60% exploitation (variant of promising run), 30% new_family, 10% risky
- Prefer cheap experiments (low timeout_s) when uncertain
- experiment_type must be one of: new_base_oof, feature_variant, seed_bagging, model_hyperparameters, stack_variant, calibration_variant, threshold_variant, pseudo_label_variant

Respond with a JSON object:
{"proposals": [{"id": "slug_no_spaces", "priority": 1, "type": "experiment_type", "hypothesis_short": "one line", "category": "exploitation", "timeout_s": 120, "expected_runtime": "~N min", "rationale": "one sentence why"}]}"""
