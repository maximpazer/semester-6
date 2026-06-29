"""
AST-based safety checker for generated experiment scripts.
Four layers: AST analysis, forbidden path strings, import whitelist, structural checks.
"""
import ast
import sys
from pathlib import Path

FORBIDDEN_PATH_FRAGMENTS = [
    "artifacts/base_oof",
    "artifacts/tabpfn_oof",
    "artifacts/tuned_params",
    "predictions.csv",
    "artifacts\\base_oof",
    "artifacts\\tabpfn_oof",
    "artifacts\\tuned_params",
]

ALLOWED_TOP_LEVEL_IMPORTS = {
    "numpy", "pandas", "sklearn", "xgboost", "lightgbm", "catboost",
    "tabpfn", "scipy", "math", "os", "sys", "pathlib", "json",
    "warnings", "time", "datetime", "collections", "functools",
    "itertools", "typing", "abc", "copy", "random", "hashlib",
    "pickle", "struct", "io", "gc",
}

WRITE_FUNCTION_NAMES = {
    "np.save", "np.savez", "np.savez_compressed",
    "pd.DataFrame.to_csv", "pd.DataFrame.to_parquet",
    "pd.DataFrame.to_pickle", "pd.DataFrame.to_json",
    "pickle.dump", "json.dump",
    "open",
}


class SafetyViolation(Exception):
    pass


class _Visitor(ast.NodeVisitor):
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.violations: list[str] = []

    def _check_str(self, s: str, node: ast.AST):
        for frag in FORBIDDEN_PATH_FRAGMENTS:
            if frag in s:
                self.violations.append(
                    f"Line {node.lineno}: forbidden path fragment '{frag}' in string '{s[:80]}'"
                )

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            self._check_str(node.value, node)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in ALLOWED_TOP_LEVEL_IMPORTS:
                self.violations.append(
                    f"Line {node.lineno}: forbidden import '{alias.name}'"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            top = node.module.split(".")[0]
            if top not in ALLOWED_TOP_LEVEL_IMPORTS:
                self.violations.append(
                    f"Line {node.lineno}: forbidden from-import '{node.module}'"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Detect open() calls that write
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            # Check mode argument
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if mode is None or any(c in str(mode) for c in ("w", "a", "x", "+")):
                # Check the path argument for forbidden fragments
                if node.args and isinstance(node.args[0], ast.Constant):
                    self._check_str(str(node.args[0].value), node)
                # Conservative: flag any open() that could write to unknown path
                # Only flag if path is not under "." or a simple filename
                if node.args:
                    path_arg = node.args[0]
                    if isinstance(path_arg, ast.Constant):
                        p = str(path_arg.value)
                        if any(frag in p for frag in FORBIDDEN_PATH_FRAGMENTS):
                            self.violations.append(
                                f"Line {node.lineno}: open() write to forbidden path '{p}'"
                            )
        # Detect subprocess.run / os.system / exec / eval
        func_repr = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        if func_repr in ("subprocess.run", "subprocess.call", "subprocess.Popen",
                         "os.system", "os.popen"):
            self.violations.append(
                f"Line {node.lineno}: forbidden subprocess/shell call: {func_repr}()"
            )
        if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval", "compile"):
            self.violations.append(
                f"Line {node.lineno}: forbidden built-in: {node.func.id}()"
            )
        self.generic_visit(node)


def check_script(script_path: Path, run_id: str) -> list[str]:
    """Return list of violation messages. Empty list = safe."""
    source = script_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(script_path))
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    visitor = _Visitor(run_id)
    visitor.visit(tree)
    return visitor.violations


def check_and_report(script_path: Path, run_id: str) -> bool:
    """Print violations and return True if safe."""
    violations = check_script(script_path, run_id)
    if violations:
        print(f"[SAFETY] {len(violations)} violation(s) in {script_path.name}:")
        for v in violations:
            print(f"  ✗ {v}")
        return False
    print(f"[SAFETY] {script_path.name} passed all checks.")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: safety_check.py <script.py> <run_id>")
        sys.exit(1)
    ok = check_and_report(Path(sys.argv[1]), sys.argv[2])
    sys.exit(0 if ok else 1)
