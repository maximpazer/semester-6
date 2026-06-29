"""
Runs generated experiment.py scripts in isolation.
Four safety layers: AST check, mtime guard, subprocess with timeout, output validation.
"""
import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

# Resolve the correct Python interpreter: prefer the project venv, fall back to sys.executable
def _find_python() -> str:
    import sys
    candidates = [
        ROOT / ".venv" / "bin" / "python",
        ROOT / ".venv" / "bin" / "python3",
        Path(sys.executable),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable

PYTHON = _find_python()

CANONICAL_ARTIFACTS = [
    ROOT / "artifacts" / "base_oof.npz",
    ROOT / "artifacts" / "tabpfn_oof.npz",
    ROOT / "artifacts" / "tuned_params.json",
    ROOT / "predictions.csv",
]

DEFAULT_TIMEOUT_S = 600
TABPFN_ENV = {**os.environ, "TABPFN_ALLOW_CPU_LARGE_DATASET": "1"}


def resolve_data_path(filename: str) -> Path:
    for candidate in [
        ROOT / "data" / "data" / filename,
        ROOT / "data" / filename,
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{filename} not found under data/data/ or data/"
    )


def _mtimes() -> dict[str, float]:
    result = {}
    for p in CANONICAL_ARTIFACTS:
        result[str(p)] = p.stat().st_mtime if p.exists() else -1.0
    return result


def _check_mtime_unchanged(before: dict[str, float]) -> list[str]:
    after = _mtimes()
    changed = []
    for path, mtime_before in before.items():
        if after[path] != mtime_before:
            changed.append(path)
    return changed


def run_experiment(run_dir: Path, timeout_s: int = DEFAULT_TIMEOUT_S) -> dict:
    """
    Execute experiment.py in run_dir.
    Returns result dict with keys: success, returncode, runtime_s, error, stdout_lines.
    Writes stdout.log.
    """
    from automation.safety_check import check_script

    script = run_dir / "experiment.py"
    if not script.exists():
        return {"success": False, "error": "experiment.py missing", "runtime_s": 0}

    # Layer 1: AST safety check
    violations = check_script(script, run_dir.name)
    if violations:
        error_msg = f"SAFETY_BLOCKED: {'; '.join(violations)}"
        (run_dir / "stdout.log").write_text(error_msg, encoding="utf-8")
        return {"success": False, "error": error_msg, "runtime_s": 0,
                "safety_blocked": True}

    # Layer 2: mtime snapshot of canonical artifacts
    mtime_before = _mtimes()

    # Layer 3: subprocess execution
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [PYTHON, "experiment.py"],
            cwd=run_dir,
            timeout=timeout_s,
            capture_output=True,
            text=True,
            env=TABPFN_ENV,
        )
        runtime_s = time.monotonic() - t0
        stdout = result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else "")
        (run_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        returncode = result.returncode
        success = (returncode == 0)
        error = None if success else f"returncode={returncode}"
    except subprocess.TimeoutExpired:
        runtime_s = time.monotonic() - t0
        error = f"TIMEOUT after {timeout_s}s"
        (run_dir / "stdout.log").write_text(error, encoding="utf-8")
        return {"success": False, "error": error, "runtime_s": runtime_s,
                "returncode": -1}

    # Layer 4: post-run mtime check
    changed = _check_mtime_unchanged(mtime_before)
    if changed:
        error_msg = f"SAFETY_VIOLATION: canonical files modified: {changed}"
        (run_dir / "stdout.log").write_text(
            error_msg + "\n\n" + (run_dir / "stdout.log").read_text(encoding="utf-8"),
            encoding="utf-8"
        )
        return {"success": False, "error": error_msg, "runtime_s": runtime_s,
                "returncode": returncode, "safety_blocked": True}

    # Check expected output
    output_file = run_dir / "new_oof.npy"
    if success and not output_file.exists():
        error = "new_oof.npy not produced"
        success = False

    return {
        "success": success,
        "returncode": returncode,
        "runtime_s": round(runtime_s, 1),
        "error": error,
        "output_exists": output_file.exists(),
    }


def write_experiment_script(run_dir: Path, script_content: str):
    """Write the generated script to run_dir/experiment.py."""
    (run_dir / "experiment.py").write_text(script_content, encoding="utf-8")
