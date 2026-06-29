"""
Toxic ML Project — Full Modeling Pipeline
Based on reverse-engineering findings for dataset 01111.

KEY INSIGHTS APPLIED:
- f11_missing and f15_missing are top predictors (MNAR injection)
- f12 is the strongest "clean" feature
- f09, f15 values are noise (heavy-tail destroyed signal)
- f00-f07 are noise (no MI with label)
- f10, f13, f14 are scaled — may retain weak signal after normalization
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                             confusion_matrix, precision_recall_curve,
                             make_scorer)
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Project paths (this script lives in src/, data in data/, outputs at project root)
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

df_train = pd.read_csv(DATA / 'toxic_data_01111.csv')
df_exam = pd.read_csv(DATA / 'toxic_exam_01111.csv')
print(f"Train: {df_train.shape}, Exam: {df_exam.shape}")

# =============================================================================
# 2. FEATURE ENGINEERING (based on reverse-engineering findings)
# =============================================================================
print("\n" + "=" * 70)
print("2. FEATURE ENGINEERING")
print("=" * 70)

def engineer_features(df):
    """Apply feature engineering based on our findings."""
    X = pd.DataFrame(index=df.index)
    
    # --- MISSING INDICATORS (our strongest signal!) ---
    X['f11_missing'] = df['f11'].isnull().astype(int)
    X['f15_missing'] = df['f15'].isnull().astype(int)
    # Combined: both missing at once?
    X['both_missing'] = (X['f11_missing'] & X['f15_missing']).astype(int)
    
    # --- CLEAN SIGNAL FEATURES ---
    X['f12'] = df['f12']  # Strongest clean predictor (MI=0.035)
    X['f08'] = df['f08']  # Weak clean signal (MI=0.008)
    
    # --- SCALED FEATURES (after normalization, may have residual signal) ---
    X['f10'] = df['f10']
    X['f13'] = df['f13']
    X['f14'] = df['f14']
    X['f11_value'] = df['f11']  # The value itself (after imputation)
    
    # --- HEAVY-TAIL: use sign-log transform to compress extremes ---
    # f15 has both MNAR + heavy tail. The value is mostly noise, but transform anyway.
    X['f15_signlog'] = np.sign(df['f15']) * np.log1p(np.abs(df['f15']))
    # f09 is pure heavy-tail noise, but include transformed version just in case
    X['f09_signlog'] = np.sign(df['f09']) * np.log1p(np.abs(df['f09']))
    
    # --- NOISE FEATURES: include a few "clean but useless" features ---
    # These have ~0 MI, but RF/XGB might find weak interactions
    # Keep only a couple to avoid overfitting on noise
    X['f04'] = df['f04']  # Always clean across all datasets
    X['f03'] = df['f03']  # Clean in our dataset
    
    return X

X_train = engineer_features(df_train)
y_train = df_train['label']
X_exam = engineer_features(df_exam)

print(f"Engineered features: {list(X_train.columns)}")
print(f"Shape: {X_train.shape}")
print(f"\nMissing values in engineered features:")
print(X_train.isnull().sum()[X_train.isnull().sum() > 0])

# =============================================================================
# 3. IMPUTATION
# =============================================================================
print("\n" + "=" * 70)
print("3. IMPUTATION (median for remaining NaN values)")
print("=" * 70)

# Impute remaining NaN values (from f11_value, f15_signlog, f09_signlog)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train), 
    columns=X_train.columns, index=X_train.index
)
X_exam_imputed = pd.DataFrame(
    imputer.transform(X_exam),
    columns=X_exam.columns, index=X_exam.index
)
print(f"After imputation — NaN remaining: {X_train_imputed.isnull().sum().sum()}")

# =============================================================================
# 4. MODEL TRAINING WITH STRATIFIED 5-FOLD CV
# =============================================================================
print("\n" + "=" * 70)
print("4. MODEL COMPARISON (Stratified 5-Fold CV, metric=F1 positive class)")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
f1_pos_scorer = make_scorer(f1_score, pos_label=1)

# Scale features for linear models & MLP
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_imputed),
    columns=X_train_imputed.columns, index=X_train_imputed.index
)
X_exam_scaled = pd.DataFrame(
    scaler.transform(X_exam_imputed),
    columns=X_exam_imputed.columns, index=X_exam_imputed.index
)

models = {
    'LogisticRegression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE, C=1.0
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        min_samples_leaf=10, subsample=0.8, random_state=RANDOM_STATE
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True,
        validation_fraction=0.15, random_state=RANDOM_STATE, alpha=0.001,
        learning_rate='adaptive'
    ),
}

results = {}
print(f"\n{'Model':<25} {'CV F1 (pos)':<15} {'Precision':<12} {'Recall':<12}")
print("-" * 65)

for name, model in models.items():
    # Use scaled data for LogReg and MLP, raw for tree-based
    if name in ['LogisticRegression', 'MLP']:
        X_cv = X_train_scaled
    else:
        X_cv = X_train_imputed
    
    cv_results = cross_validate(
        model, X_cv, y_train, cv=cv, 
        scoring={'f1': f1_pos_scorer, 
                 'precision': make_scorer(precision_score, pos_label=1),
                 'recall': make_scorer(recall_score, pos_label=1)},
        return_train_score=True
    )
    
    f1_mean = cv_results['test_f1'].mean()
    f1_std = cv_results['test_f1'].std()
    prec_mean = cv_results['test_precision'].mean()
    rec_mean = cv_results['test_recall'].mean()
    train_f1 = cv_results['train_f1'].mean()
    
    results[name] = {
        'f1_mean': f1_mean, 'f1_std': f1_std,
        'precision': prec_mean, 'recall': rec_mean,
        'train_f1': train_f1, 'overfit_gap': train_f1 - f1_mean
    }
    
    print(f"{name:<25} {f1_mean:.4f}±{f1_std:.4f}  {prec_mean:.4f}      {rec_mean:.4f}   "
          f"(train: {train_f1:.4f}, gap: {train_f1-f1_mean:.4f})")

# =============================================================================
# 5. THRESHOLD TUNING (for best model)
# =============================================================================
print("\n" + "=" * 70)
print("5. THRESHOLD TUNING FOR BEST MODEL")
print("=" * 70)

# Find best model
best_model_name = max(results, key=lambda k: results[k]['f1_mean'])
print(f"Best model: {best_model_name} (CV F1 = {results[best_model_name]['f1_mean']:.4f})")

# Refit best model and do threshold sweep on CV folds
best_model = models[best_model_name]
if best_model_name in ['LogisticRegression', 'MLP']:
    X_final = X_train_scaled
    X_exam_final = X_exam_scaled
else:
    X_final = X_train_imputed
    X_exam_final = X_exam_imputed

# Collect out-of-fold predictions for threshold tuning
oof_proba = np.zeros(len(y_train))
for train_idx, val_idx in cv.split(X_final, y_train):
    X_tr, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    
    best_model.fit(X_tr, y_tr)
    oof_proba[val_idx] = best_model.predict_proba(X_val)[:, 1]

# Sweep thresholds
thresholds = np.arange(0.05, 0.95, 0.01)
f1_scores = []
for t in thresholds:
    preds = (oof_proba >= t).astype(int)
    f1_scores.append(f1_score(y_train, preds))

best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)
print(f"\nOptimal threshold: {best_threshold:.2f} (F1 = {best_f1:.4f})")
print(f"Default threshold 0.5 would give F1 = {f1_score(y_train, (oof_proba >= 0.5).astype(int)):.4f}")

# Precision/Recall at the chosen operating point (printed instead of plotted)
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_train, oof_proba)

# Confusion matrix at optimal threshold
final_preds_oof = (oof_proba >= best_threshold).astype(int)
cm = confusion_matrix(y_train, final_preds_oof)
print(f"\nConfusion Matrix (OOF, threshold={best_threshold:.2f}):")
print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
print(f"\n  Precision: {precision_score(y_train, final_preds_oof):.4f}")
print(f"  Recall:    {recall_score(y_train, final_preds_oof):.4f}")
print(f"  F1:        {f1_score(y_train, final_preds_oof):.4f}")

# =============================================================================
# 6. TRAIN FINAL MODEL ON ALL DATA & PREDICT EXAM
# =============================================================================
print("\n" + "=" * 70)
print("6. FINAL MODEL — TRAIN ON ALL DATA & PREDICT EXAM")
print("=" * 70)

# Refit on ALL training data
best_model.fit(X_final, y_train)
exam_proba = best_model.predict_proba(X_exam_final)[:, 1]
exam_preds = (exam_proba >= best_threshold).astype(int)

print(f"Exam predictions: {len(exam_preds)} rows")
print(f"Predicted positive: {exam_preds.sum()} ({exam_preds.mean()*100:.1f}%)")
print(f"Predicted negative: {(1-exam_preds).sum()} ({(1-exam_preds).mean()*100:.1f}%)")

# Sanity check: predicted positive rate should be ~10%
if 0.05 < exam_preds.mean() < 0.20:
    print("✓ Positive rate is in expected range (5-20%)")
else:
    print("⚠ WARNING: Positive rate is outside expected range!")

# Save predictions
output = pd.DataFrame({'label': exam_preds})
output.to_csv(ROOT / 'predictions.csv', index=False)
print(f"\nSaved predictions to: predictions.csv")

# =============================================================================
# 7. FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 70)
print("7. FEATURE IMPORTANCE")
print("=" * 70)

# Get feature importance from the best model
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X_train_imputed.columns)
elif hasattr(best_model, 'coef_'):
    importances = pd.Series(np.abs(best_model.coef_[0]), index=X_train_scaled.columns)
else:
    importances = None

if importances is not None:
    importances = importances.sort_values(ascending=False)
    print("\nFeature Importances:")
    for feat, imp in importances.items():
        bar = '█' * int(imp / importances.max() * 30)
        print(f"  {feat:<15} {imp:.4f} {bar}")

# =============================================================================
# 8. ALSO TRY XGBOOST (if available)
# =============================================================================
print("\n" + "=" * 70)
print("8. TRYING XGBOOST (if available)")
print("=" * 70)

try:
    from xgboost import XGBClassifier
    
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        scale_pos_weight=9,  # Handle class imbalance
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, gamma=0.1,
        random_state=RANDOM_STATE, eval_metric='logloss',
        use_label_encoder=False
    )
    
    # CV evaluation
    cv_results_xgb = cross_validate(
        xgb_model, X_train_imputed, y_train, cv=cv,
        scoring={'f1': f1_pos_scorer,
                 'precision': make_scorer(precision_score, pos_label=1),
                 'recall': make_scorer(recall_score, pos_label=1)},
        return_train_score=True
    )
    
    xgb_f1 = cv_results_xgb['test_f1'].mean()
    xgb_std = cv_results_xgb['test_f1'].std()
    xgb_train = cv_results_xgb['train_f1'].mean()
    
    print(f"XGBoost CV F1: {xgb_f1:.4f}±{xgb_std:.4f} (train: {xgb_train:.4f})")
    
    # If XGBoost is better, redo threshold tuning and predictions
    if xgb_f1 > results[best_model_name]['f1_mean']:
        print(f"\n*** XGBoost beats {best_model_name}! Redoing threshold tuning... ***")
        
        oof_proba_xgb = np.zeros(len(y_train))
        for train_idx, val_idx in cv.split(X_train_imputed, y_train):
            X_tr, X_val = X_train_imputed.iloc[train_idx], X_train_imputed.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            xgb_model.fit(X_tr, y_tr)
            oof_proba_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        
        f1_scores_xgb = [f1_score(y_train, (oof_proba_xgb >= t).astype(int)) for t in thresholds]
        best_threshold_xgb = thresholds[np.argmax(f1_scores_xgb)]
        best_f1_xgb = max(f1_scores_xgb)
        
        print(f"XGBoost optimal threshold: {best_threshold_xgb:.2f} (F1 = {best_f1_xgb:.4f})")
        
        # Final XGBoost predictions
        xgb_model.fit(X_train_imputed, y_train)
        exam_proba_xgb = xgb_model.predict_proba(X_exam_imputed)[:, 1]
        exam_preds_xgb = (exam_proba_xgb >= best_threshold_xgb).astype(int)
        
        print(f"XGBoost exam predictions: positive={exam_preds_xgb.sum()} ({exam_preds_xgb.mean()*100:.1f}%)")
        
        # Overwrite predictions with XGBoost
        output_xgb = pd.DataFrame({'label': exam_preds_xgb})
        output_xgb.to_csv(ROOT / 'predictions.csv', index=False)
        print(f"Updated predictions.csv with XGBoost results")
        
        # Feature importance
        xgb_imp = pd.Series(xgb_model.feature_importances_, index=X_train_imputed.columns)
        xgb_imp = xgb_imp.sort_values(ascending=False)
        print(f"\nXGBoost Feature Importances:")
        for feat, imp in xgb_imp.items():
            bar = '█' * int(imp / xgb_imp.max() * 30)
            print(f"  {feat:<15} {imp:.4f} {bar}")
    else:
        print(f"XGBoost ({xgb_f1:.4f}) did NOT beat {best_model_name} ({results[best_model_name]['f1_mean']:.4f})")

except ImportError:
    print("XGBoost not installed. Install with: python -m pip install xgboost")
    print("Skipping XGBoost...")

# =============================================================================
# 9. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("9. FINAL SUMMARY")
print("=" * 70)

print(f"""
Model Comparison:
{'─' * 60}""")
for name, r in sorted(results.items(), key=lambda x: x[1]['f1_mean'], reverse=True):
    print(f"  {name:<25} F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}  "
          f"P={r['precision']:.3f} R={r['recall']:.3f}  gap={r['overfit_gap']:.3f}")

print(f"""
Final Configuration:
{'─' * 60}
  Best model: {best_model_name}
  Threshold: {best_threshold:.2f}
  OOF F1: {best_f1:.4f}
  Predictions saved: predictions.csv
  
  Key features used:
    1. f11_missing (MNAR indicator)
    2. f15_missing (MNAR indicator)  
    3. f12 (clean signal)
    4. f08 (weak clean signal)
    5. f10, f13, f14 (scaled, residual signal)
""")

print("DONE!")
