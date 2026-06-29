"""
Toxic ML Project - MAXIMUM PERFORMANCE DEEP DIVE
Goal: Squeeze every last point out of the F1 score.

Strategy:
1. Deep-dive: try to decode the EXACT label generation rule
2. Extended feature engineering (interactions, bins, etc.)
3. Optimized MLP (sklearn + grid)
4. Optimized XGBoost (grid)
5. LightGBM
6. Stacking ensemble
7. Aggressive threshold optimization
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Project paths (this script lives in src/, data in data/, outputs at project root)
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'

# =============================================================================
# 1. DEEP DIVE: UNDERSTANDING THE LABEL GENERATION RULE
# =============================================================================
print("=" * 70)
print("1. DEEP DIVE: UNDERSTANDING THE LABEL GENERATION RULE")
print("=" * 70)

df_train = pd.read_csv(DATA / 'toxic_data_01111.csv')
df_exam = pd.read_csv(DATA / 'toxic_exam_01111.csv')
features = [c for c in df_train.columns if c != 'label']
y_train = df_train['label']

pos = df_train[df_train['label'] == 1]
neg = df_train[df_train['label'] == 0]

print("\nFeature means for pos vs neg (clean features):")
for feat in features:
    col_pos = pos[feat].dropna()
    col_neg = neg[feat].dropna()
    if col_pos.std() < 5:
        diff = col_pos.mean() - col_neg.mean()
        if abs(diff) > 0.05:
            print(f"  {feat}: pos={col_pos.mean():.4f}, neg={col_neg.mean():.4f}, "
                  f"diff={diff:.4f}, effect={diff/col_neg.std():.3f}")

both_missing = df_train[df_train['f11'].isnull() & df_train['f15'].isnull()]
one_missing = df_train[df_train['f11'].isnull() ^ df_train['f15'].isnull()]
none_missing = df_train[df_train['f11'].notna() & df_train['f15'].notna()]

print(f"\n--- Missingness combinations ---")
print(f"  Both missing:    n={len(both_missing)}, pos_rate={both_missing['label'].mean():.4f}")
print(f"  Only one missing: n={len(one_missing)}, pos_rate={one_missing['label'].mean():.4f}")
print(f"  Neither missing:  n={len(none_missing)}, pos_rate={none_missing['label'].mean():.4f}")

print(f"\n--- Among neither-missing (n={len(none_missing)}): what predicts label? ---")
corrs = none_missing[features + ['label']].corr()['label'].drop('label').abs().sort_values(ascending=False)
for feat, c in corrs.head(8).items():
    print(f"    {feat}: |r|={c:.4f}")

print(f"\n--- Among both-missing (n={len(both_missing)}): what predicts label? ---")
if len(both_missing) > 50:
    corrs_bm = both_missing[features + ['label']].corr()['label'].drop('label').abs().sort_values(ascending=False)
    for feat, c in corrs_bm.head(8).items():
        print(f"    {feat}: |r|={c:.4f}")

# =============================================================================
# 2. EXTENDED FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("2. EXTENDED FEATURE ENGINEERING")
print("=" * 70)

def engineer_features_v2(df):
    """Maximum feature engineering based on all findings."""
    X = pd.DataFrame(index=df.index)
    
    # TIER 1: MNAR INDICATORS (strongest signal)
    X['f11_miss'] = df['f11'].isnull().astype(int)
    X['f15_miss'] = df['f15'].isnull().astype(int)
    X['both_miss'] = (X['f11_miss'] & X['f15_miss']).astype(int)
    X['any_miss'] = (X['f11_miss'] | X['f15_miss']).astype(int)
    X['miss_count'] = X['f11_miss'] + X['f15_miss']
    
    # TIER 2: CLEAN SIGNAL FEATURES
    X['f12'] = df['f12']
    X['f08'] = df['f08']
    
    # TIER 3: SCALED FEATURES
    for feat in ['f10', 'f13', 'f14']:
        X[feat] = df[feat]
    X['f11_val'] = df['f11']
    
    # TIER 4: INTERACTIONS
    X['f12_x_miss'] = df['f12'] * X['any_miss']
    X['f12_x_nomiss'] = df['f12'] * (1 - X['any_miss'])
    X['f08_x_miss'] = df['f08'] * X['any_miss']
    X['f12_sq'] = df['f12'] ** 2
    X['f12_x_f08'] = df['f12'] * df['f08']
    
    # TIER 5: HEAVY-TAIL (compressed)
    X['f15_slog'] = np.sign(df['f15']) * np.log1p(np.abs(df['f15']))
    X['f09_slog'] = np.sign(df['f09']) * np.log1p(np.abs(df['f09']))
    
    # TIER 6: BINNED
    X['f12_high'] = (df['f12'] > 0.5).astype(int)
    X['f12_vhigh'] = (df['f12'] > 1.0).astype(int)
    
    # TIER 7: Other clean features
    X['f04'] = df['f04']
    X['f03'] = df['f03']
    X['f00'] = df['f00']
    X['f01'] = df['f01']
    
    return X

X_train_v2 = engineer_features_v2(df_train)
X_exam_v2 = engineer_features_v2(df_exam)

print(f"V2 features: {X_train_v2.shape[1]} features")
print(f"Features: {list(X_train_v2.columns)}")

# Impute
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_v2), 
                           columns=X_train_v2.columns, index=X_train_v2.index)
X_exam_imp = pd.DataFrame(imputer.transform(X_exam_v2),
                          columns=X_exam_v2.columns, index=X_exam_v2.index)

# Scale for MLP
scaler = RobustScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_imp),
                          columns=X_train_imp.columns, index=X_train_imp.index)
X_exam_sc = pd.DataFrame(scaler.transform(X_exam_imp),
                         columns=X_exam_imp.columns, index=X_exam_imp.index)

# =============================================================================
# 3. HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 70)
print("3. HYPERPARAMETER TUNING")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
thresholds = np.arange(0.05, 0.95, 0.005)

def get_oof_proba(model, X, y, cv_obj):
    """Get out-of-fold probability predictions."""
    oof = np.zeros(len(y))
    for tr_idx, val_idx in cv_obj.split(X, y):
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return oof

def find_best_threshold(oof_proba, y_true):
    """Find threshold that maximizes F1."""
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (oof_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

# --- XGBoost Tuning ---
print("\n--- XGBoost Tuning ---")
xgb_configs = [
    {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'scale_pos_weight': 9,
     'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3, 'gamma': 0.1},
    {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03, 'scale_pos_weight': 9,
     'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 5, 'gamma': 0.2},
    {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.05, 'scale_pos_weight': 8,
     'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1, 'gamma': 0},
    {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.1, 'scale_pos_weight': 10,
     'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 5, 'gamma': 0.1},
    {'n_estimators': 600, 'max_depth': 4, 'learning_rate': 0.02, 'scale_pos_weight': 9,
     'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 3, 'gamma': 0.05},
    {'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.05, 'scale_pos_weight': 7,
     'subsample': 0.85, 'colsample_bytree': 0.85, 'min_child_weight': 2, 'gamma': 0.1},
]

best_xgb_f1 = 0
best_xgb_config = None
for i, config in enumerate(xgb_configs):
    model = XGBClassifier(**config, random_state=RANDOM_STATE, eval_metric='logloss',
                          use_label_encoder=False)
    oof = get_oof_proba(model, X_train_imp, y_train, cv)
    _, f1_val = find_best_threshold(oof, y_train)
    print(f"  Config {i+1}: F1={f1_val:.4f} (depth={config['max_depth']}, lr={config['learning_rate']})")
    if f1_val > best_xgb_f1:
        best_xgb_f1 = f1_val
        best_xgb_config = config
        oof_xgb = oof

print(f"\n  Best XGBoost: F1={best_xgb_f1:.4f}")

# --- MLP Tuning ---
print("\n--- MLP Tuning ---")
mlp_configs = [
    {'hidden_layer_sizes': (128, 64), 'alpha': 0.001, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (256, 128, 64), 'alpha': 0.0005, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (64, 32, 16), 'alpha': 0.01, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (128, 64, 32), 'alpha': 0.001, 'learning_rate_init': 0.0005},
    {'hidden_layer_sizes': (512, 256, 128), 'alpha': 0.0001, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (64, 64), 'alpha': 0.005, 'learning_rate_init': 0.002},
    {'hidden_layer_sizes': (128,), 'alpha': 0.01, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (256, 128), 'alpha': 0.001, 'learning_rate_init': 0.0003},
]

best_mlp_f1 = 0
best_mlp_config = None
for i, config in enumerate(mlp_configs):
    model = MLPClassifier(**config, max_iter=1000, early_stopping=True,
                          validation_fraction=0.15, random_state=RANDOM_STATE,
                          learning_rate='adaptive', batch_size=128)
    oof = get_oof_proba(model, X_train_sc, y_train, cv)
    _, f1_val = find_best_threshold(oof, y_train)
    print(f"  Config {i+1}: F1={f1_val:.4f} layers={config['hidden_layer_sizes']}")
    if f1_val > best_mlp_f1:
        best_mlp_f1 = f1_val
        best_mlp_config = config
        oof_mlp = oof

print(f"\n  Best MLP: F1={best_mlp_f1:.4f}")

# --- Random Forest ---
print("\n--- Random Forest ---")
rf_best = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3,
                                 class_weight='balanced_subsample',
                                 random_state=RANDOM_STATE, n_jobs=-1)
oof_rf = get_oof_proba(rf_best, X_train_imp, y_train, cv)
_, f1_rf = find_best_threshold(oof_rf, y_train)
print(f"  RF: F1={f1_rf:.4f}")

# --- LightGBM ---
print("\n--- LightGBM ---")
try:
    from lightgbm import LGBMClassifier
    lgbm_configs = [
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'scale_pos_weight': 9,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 10, 'num_leaves': 31},
        {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03, 'scale_pos_weight': 9,
         'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_samples': 20, 'num_leaves': 50},
        {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.05, 'scale_pos_weight': 8,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 5, 'num_leaves': 63},
    ]
    best_lgbm_f1 = 0
    for i, config in enumerate(lgbm_configs):
        model = LGBMClassifier(**config, random_state=RANDOM_STATE, verbose=-1)
        oof = get_oof_proba(model, X_train_imp, y_train, cv)
        _, f1_val = find_best_threshold(oof, y_train)
        print(f"  Config {i+1}: F1={f1_val:.4f}")
        if f1_val > best_lgbm_f1:
            best_lgbm_f1 = f1_val
            best_lgbm_config = config
            oof_lgbm = oof
    print(f"\n  Best LightGBM: F1={best_lgbm_f1:.4f}")
    has_lgbm = True
except ImportError:
    print("  LightGBM not available")
    has_lgbm = False
    best_lgbm_f1 = 0

# =============================================================================
# 4. ENSEMBLE: WEIGHTED BLEND
# =============================================================================
print("\n" + "=" * 70)
print("4. ENSEMBLE: WEIGHTED BLEND")
print("=" * 70)

from itertools import product
models_oof = [oof_xgb, oof_mlp, oof_rf]
model_names = ['XGB', 'MLP', 'RF']
if has_lgbm:
    models_oof.append(oof_lgbm)
    model_names.append('LGBM')

# Equal blend
oof_equal = np.mean(models_oof, axis=0)
_, f1_equal = find_best_threshold(oof_equal, y_train)
print(f"  Equal blend: F1={f1_equal:.4f}")

# Weighted blend optimization (coarse grid then refine)
best_wf1 = 0
best_weights = None
weight_range = np.arange(0, 1.05, 0.25)

for weights in product(weight_range, repeat=len(models_oof)):
    if sum(weights) < 0.01:
        continue
    w = np.array(weights) / sum(weights)
    blend = sum(w_i * o_i for w_i, o_i in zip(w, models_oof))
    _, f1_w = find_best_threshold(blend, y_train)
    if f1_w > best_wf1:
        best_wf1 = f1_w
        best_weights = w

# Fine-tune around best weights
fine_range = np.linspace(-0.15, 0.15, 7)
for deltas in product(fine_range, repeat=len(models_oof)):
    w_try = best_weights + np.array(deltas)
    w_try = np.clip(w_try, 0, None)
    if w_try.sum() < 0.01:
        continue
    w_try = w_try / w_try.sum()
    blend = sum(w_i * o_i for w_i, o_i in zip(w_try, models_oof))
    _, f1_w = find_best_threshold(blend, y_train)
    if f1_w > best_wf1:
        best_wf1 = f1_w
        best_weights = w_try

print(f"  Weighted blend: F1={best_wf1:.4f}")
print(f"  Weights: {dict(zip(model_names, [f'{w:.2f}' for w in best_weights]))}")

oof_best_blend = sum(w_i * o_i for w_i, o_i in zip(best_weights, models_oof))
t_final, f1_final = find_best_threshold(oof_best_blend, y_train)
print(f"  Final threshold: {t_final:.3f}")

# =============================================================================
# 5. COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("5. FINAL COMPARISON")
print("=" * 70)

all_results = {
    'XGBoost': best_xgb_f1,
    'MLP': best_mlp_f1,
    'RandomForest': f1_rf,
    'Blend (equal)': f1_equal,
    'Blend (weighted)': best_wf1,
}
if has_lgbm:
    all_results['LightGBM'] = best_lgbm_f1

for name, f1 in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
    marker = " <-- BEST" if f1 == max(all_results.values()) else ""
    print(f"  {name:<25} F1={f1:.4f}{marker}")

# =============================================================================
# 6. FINAL PREDICTIONS
# =============================================================================
print("\n" + "=" * 70)
print("6. FINAL PREDICTIONS")
print("=" * 70)

# Train all on full data
xgb_final = XGBClassifier(**best_xgb_config, random_state=RANDOM_STATE,
                           eval_metric='logloss', use_label_encoder=False)
xgb_final.fit(X_train_imp, y_train)
exam_xgb = xgb_final.predict_proba(X_exam_imp)[:, 1]

mlp_final = MLPClassifier(**best_mlp_config, max_iter=1000, early_stopping=True,
                          validation_fraction=0.15, random_state=RANDOM_STATE,
                          learning_rate='adaptive', batch_size=128)
mlp_final.fit(X_train_sc, y_train)
exam_mlp = mlp_final.predict_proba(X_exam_sc)[:, 1]

rf_final = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3,
                                  class_weight='balanced_subsample',
                                  random_state=RANDOM_STATE, n_jobs=-1)
rf_final.fit(X_train_imp, y_train)
exam_rf = rf_final.predict_proba(X_exam_imp)[:, 1]

exam_models = [exam_xgb, exam_mlp, exam_rf]
if has_lgbm:
    lgbm_final = LGBMClassifier(**best_lgbm_config, random_state=RANDOM_STATE, verbose=-1)
    lgbm_final.fit(X_train_imp, y_train)
    exam_lgbm = lgbm_final.predict_proba(X_exam_imp)[:, 1]
    exam_models.append(exam_lgbm)

# Weighted blend
exam_blend = sum(w_i * p_i for w_i, p_i in zip(best_weights, exam_models))
exam_preds = (exam_blend >= t_final).astype(int)

print(f"Predictions: {exam_preds.sum()} positive ({exam_preds.mean()*100:.1f}%), "
      f"{(1-exam_preds).sum()} negative")

# Save
output = pd.DataFrame({'label': exam_preds})
output.to_csv(ROOT / 'predictions.csv', index=False)
print(f"Saved: predictions.csv")

# Sanity checks
exam_miss_11 = df_exam['f11'].isnull()
exam_miss_15 = df_exam['f15'].isnull()
print(f"\n  Pred pos where f11 missing: {exam_preds[exam_miss_11.values].mean()*100:.1f}%")
print(f"  Pred pos where f11 present: {exam_preds[~exam_miss_11.values].mean()*100:.1f}%")
print(f"  Pred pos where f15 missing: {exam_preds[exam_miss_15.values].mean()*100:.1f}%")
print(f"  Pred pos where f15 present: {exam_preds[~exam_miss_15.values].mean()*100:.1f}%")

# Feature importance from XGBoost
imp = pd.Series(xgb_final.feature_importances_, index=X_train_imp.columns).sort_values(ascending=False)
print(f"\nXGBoost Feature Importances:")
for feat, v in imp.head(10).items():
    bar = '#' * int(v / imp.max() * 30)
    print(f"  {feat:<15} {v:.4f} {bar}")

print(f"\n{'='*70}")
print(f"DONE! Best OOF F1 = {max(all_results.values()):.4f}")
print(f"{'='*70}")
