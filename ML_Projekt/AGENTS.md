# OpenCode project instructions

Read `AGENT.md` completely before taking action and treat it as mandatory project
context. Also read `runs/LEADERBOARD.md`, `automation/state.json`, and
`automation/queue.json` before starting or repeating an experiment.

The historical baseline is paired repeated-CV F1 `0.6938`. The current best
experimental candidate is `lgbm_native_nan`; its confirmatory result is stored in
`runs/20260629_162030_lgbm_native_nan/conf_metrics.json` (F1 `0.6979`, paired
delta `+0.0057`, SE `0.0008`). It is not yet the committed final submission.

Continue the score-maximization loop autonomously. Do not repeat completed
experiments. Preserve `predictions.csv`, `artifacts/*.npz`, and
`artifacts/tuned_params.json`; write new outputs below `runs/`. Never use exam
labels or tune choices on the exam set. Prefer genuine paired/nested validation
over improvements obtained by repeatedly optimizing the same OOF scores.

The datasets are local-only and expected at:

- `data/data/toxic_data_01111.csv`
- `data/data/toxic_exam_01111.csv`

Use `.venv/bin/python` for all Python commands.
