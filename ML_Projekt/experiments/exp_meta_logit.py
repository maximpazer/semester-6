"""Meta-learner variant: logit-transform OOF inputs + LogReg(C=1.0)."""
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(ROOT))
from automation.eval_core import load_baseline, evaluate_meta_variant

ID         = "meta_logit_transform"
HYPOTHESIS = "Logit-Transform der S6-OOF-Inputs vor LogReg-Meta-Learner"
TIMEOUT_S  = 30


def logit_transform(S):
    S_c = np.clip(S, 1e-6, 1 - 1e-6)
    return np.log(S_c / (1 - S_c))


def make_meta():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
    ])


def run(run_dir: Path):
    result = evaluate_meta_variant(make_meta, run_dir, transform_fn=logit_transform)
    # Expose for run_experiment.py compatibility
    run._result = result
    # Return a sentinel — this experiment doesn't produce new_oof.npy
    raise _StackVariantResult(result)


class _StackVariantResult(Exception):
    def __init__(self, result): self.result = result
