"""
Static preflight checks and auto-fixes for Qwen-generated experiment scripts.

Checks:
  1. Python syntax (ast.parse)
  2. Missing stdlib imports used in code (warnings, time, json, math)
  3. Common Qwen index typos: y[val] / X[val] / y[train] / X[train]
     instead of y[val_idx] / X[val_idx] etc.
  4. Warns about any other obvious NameErrors detectable statically.

Returns a (fixed_source, report) tuple.
fixed_source is the patched script; report lists every change made.
Raises SyntaxError if the source cannot be parsed even after fixes.
"""
import ast
import re
from pathlib import Path
from typing import Tuple

# Stdlib modules whose usage is detectable by name but commonly forgotten
_STDLIB_AUTOFIX = ["warnings", "time", "json", "math", "sys", "os", "copy"]

# Pattern: bare bracket index that is almost certainly a Qwen typo.
# Matches y[val], X[val], y[train], X[train] — but NOT y[val_idx], y[train_idx].
_INDEX_TYPO_RE = re.compile(
    r'\b([A-Za-z_]\w*)\[(val|train)\](?!_idx)'
)


_ROOT_BAD_RE = re.compile(
    r'Path\(__file__\)\.resolve\(\)\.parent\.parent(?!\.parent)'
)
_ROOT_BAD_REPL = "Path(__file__).resolve().parent.parent.parent"


def _fix_root_depth(source: str) -> Tuple[str, list]:
    """Fix ROOT = .parent.parent → .parent.parent.parent (runs/<id>/experiment.py is 3 levels deep)."""
    fixes = []
    new_source = _ROOT_BAD_RE.sub(lambda m: (_ROOT_BAD_REPL, fixes.append(
        "Fixed ROOT depth: .parent.parent → .parent.parent.parent"
    ))[0], source)
    return new_source, fixes


def _fix_missing_imports(source: str) -> Tuple[str, list]:
    """Add stdlib imports that are referenced but not imported."""
    fixes = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, fixes  # let caller handle syntax errors

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])

    # Find which stdlib modules are used as bare names but not imported
    needed = []
    for mod in _STDLIB_AUTOFIX:
        if mod not in imported:
            # Check if the name is used as an attribute base (e.g. warnings.filterwarnings)
            pattern = re.compile(r'\b' + re.escape(mod) + r'\s*\.')
            if pattern.search(source):
                needed.append(mod)

    if needed:
        import_block = "\n".join(f"import {m}" for m in needed)
        # Insert after the last existing import line
        lines = source.splitlines()
        last_import_line = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import_line = i
        lines.insert(last_import_line + 1, import_block)
        source = "\n".join(lines)
        fixes.append(f"Added missing import(s): {', '.join(needed)}")

    return source, fixes


def _fix_index_typos(source: str) -> Tuple[str, list]:
    """Replace bare y[val]/X[val]/y[train]/X[train] with the _idx variants."""
    fixes = []

    def _replacer(m):
        orig = m.group(0)
        arr = m.group(1)
        idx = m.group(2)
        fixed = f"{arr}[{idx}_idx]"
        fixes.append(f"  {orig!r} → {fixed!r}")
        return fixed

    new_source = _INDEX_TYPO_RE.sub(_replacer, source)
    if fixes:
        fixes = [f"Fixed index typo(s):"] + fixes
    return new_source, fixes


def run_preflight(script_path: Path) -> Tuple[str, bool, list]:
    """
    Read script_path, apply auto-fixes, verify syntax.
    Returns (fixed_source, ok, report_lines).
    ok=False means a non-fixable syntax error was found.
    """
    source = script_path.read_text(encoding="utf-8")
    report = []

    # Step 1: Syntax check on original
    try:
        ast.parse(source)
    except SyntaxError as e:
        report.append(f"[PREFLIGHT] SyntaxError in original script: {e}")
        # Attempt fixes anyway — they might not touch the broken part
        # but we still try.

    # Step 2: Fix ROOT depth (.parent.parent → .parent.parent.parent)
    source, root_fixes = _fix_root_depth(source)
    report.extend(root_fixes)

    # Step 3: Fix missing stdlib imports
    source, imp_fixes = _fix_missing_imports(source)
    report.extend(imp_fixes)

    # Step 4: Fix index typos
    source, idx_fixes = _fix_index_typos(source)
    report.extend(idx_fixes)

    # Step 4: Final syntax verification
    try:
        ast.parse(source)
        ok = True
        if report:
            report.insert(0, "[PREFLIGHT] Auto-fixes applied:")
        else:
            report.append("[PREFLIGHT] No issues found.")
    except SyntaxError as e:
        ok = False
        report.append(f"[PREFLIGHT] SyntaxError persists after auto-fix: {e}")

    return source, ok, report


def preflight_and_patch(script_path: Path) -> Tuple[bool, list]:
    """
    Run preflight on script_path, write fixes back in-place.
    Returns (ok, report_lines).
    """
    fixed_source, ok, report = run_preflight(script_path)
    script_path.write_text(fixed_source, encoding="utf-8")
    return ok, report
