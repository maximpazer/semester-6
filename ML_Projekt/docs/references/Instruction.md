You are helping me with a binary classification project. The grading metric is the
F1-score of the POSITIVE class on a held-out exam set. I need both (a) the best
possible predictions and (b) a clear explanation of every choice, because I will
be quizzed orally on the code.

## Files
- toxic_data_01111.csv  — 8000 rows, 16 features (f00–f15), column "label" (0/1)
- toxic_exam_01111.csv  — 2000 rows, same 16 features, NO label. I must predict these.

## What I already know about the data
- Class imbalance: ~90% negative, ~10% positive (800 positives).
- Missing values only in f11 (~15.5%) and f15 (~15.4%), in both train and exam.
- Feature scales differ by orders of magnitude:
    * f00–f08, f12 look ~N(0,1)
    * f09–f11, f13, f14 have std ~80–200
    * f15 has std ~14,000 and range ±63,000 — likely heavy-tailed / outliers
- Same column structure in train and exam, so the missingness must be handled,
  not dropped.

## What I need you to do, in this order

1. **EDA** — load both files, print shapes, dtypes, missingness, class balance,
   per-feature describe(). Plot per-feature distributions split by label
   (especially for the high-variance features). Compute correlations and
   mutual information vs. the label. Tell me which features look predictive
   and which look like noise.

2. **Preparation**
   - Use a stratified train/validation split (e.g. 80/20, random_state fixed).
   - Decide on imputation for f11/f15. Try at least two strategies (median;
     model-based or "missing-indicator + median") and tell me which generalizes
     better — DO NOT just pick one silently.
   - Add a binary "is_missing" indicator column for f11 and f15 — missingness
     itself may be informative.
   - Treat extreme outliers in f15 explicitly (winsorize, log-sign transform
     like sign(x)*log1p(|x|), or robust scaling). Compare options.
   - Scale features appropriately for each model family (StandardScaler /
     RobustScaler for linear/MLP, none for tree models).

3. **Modeling** — train and tune several models, all evaluated with the SAME
   stratified k-fold CV (k=5) and the SAME metric (F1 of positive class):
   - Logistic Regression (baseline, with class_weight='balanced')
   - Random Forest (this is what I need to beat for half credit)
   - Gradient boosting: XGBoost OR LightGBM with scale_pos_weight ≈ 9
   - MLP (sklearn MLPClassifier or a small PyTorch net) — this is the upper
     benchmark
   Tune key hyperparameters with a small grid or Optuna. For each model report:
   CV mean F1 ± std, precision, recall, confusion matrix at the chosen threshold.

4. **Threshold tuning** — for the best model, sweep the decision threshold on
   the validation set and pick the one that maximizes F1 of the positive class.
   Show me the precision/recall curve and the chosen operating point. Do NOT
   leave the threshold at the default 0.5.

5. **Sanity checks against overfitting**
   - Compare train vs. CV F1 — flag any large gap.
   - Refit the final model on ALL training data (train+val) with the
     tuned hyperparameters and threshold before predicting on the exam set.

6. **Interpretation**
   - Feature importance (tree importance + permutation importance for the
     winning model).
   - Tell me which features actually drove the predictions and which I could
     have dropped.

7. **Output**
   - A CSV of predicted labels for toxic_exam_01111.csv, same row order,
     single column "label" with 0/1 values.
   - A short markdown "decision log" I can print and bring to the oral exam,
     covering: the metric, what I saw in EDA, why I imputed/scaled the way I
     did, which models I tried and which won, the threshold choice, and the
     CV-vs-validation F1 numbers.

## Rules
- Set and report random_state everywhere. Results must be reproducible.
- Never touch the exam file during training or threshold selection — it is
  ONLY for the final prediction.
- If you are unsure between two reasonable choices, run both and compare on
  CV F1 instead of guessing.
- Explain every non-obvious line of code in a comment, so I can defend it
  verbally.