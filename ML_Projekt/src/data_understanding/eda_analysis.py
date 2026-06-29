"""
Toxic ML Project — Exploratory Data Analysis
Goal: Understand the data, detect the professor's "toxic" injections, identify predictive features.
Metric: F1-score of the positive class.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

# Settings
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Project paths (this script lives in src/, data in data/)
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'

# =============================================================================
# 1. Load Data & Basic Overview
# =============================================================================
print("=" * 70)
print("1. LOAD DATA & BASIC OVERVIEW")
print("=" * 70)

df_train = pd.read_csv(DATA / 'toxic_data_01111.csv')
df_exam = pd.read_csv(DATA / 'toxic_exam_01111.csv')

print(f"Training set: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
print(f"Exam set:     {df_exam.shape[0]} rows, {df_exam.shape[1]} columns")
print(f"\nTraining columns: {list(df_train.columns)}")
print(f"Exam columns:     {list(df_exam.columns)}")

features = [c for c in df_train.columns if c != 'label']

print(f"\n--- Training describe() ---")
print(df_train.describe().T.to_string())

# =============================================================================
# 2. Class Balance
# =============================================================================
print("\n" + "=" * 70)
print("2. CLASS BALANCE")
print("=" * 70)

class_counts = df_train['label'].value_counts()
print(f"Class 0 (Negative): {class_counts[0]}")
print(f"Class 1 (Positive): {class_counts[1]}")
print(f"Positive rate: {class_counts[1] / len(df_train) * 100:.2f}%")
print(f"Imbalance ratio (neg:pos): {class_counts[0] / class_counts[1]:.2f}:1")

# =============================================================================
# 3. Missingness Analysis
# =============================================================================
print("\n" + "=" * 70)
print("3. MISSINGNESS ANALYSIS")
print("=" * 70)

print("\n--- Missing values in TRAINING set ---")
missing_train = df_train.isnull().sum()
missing_train_pct = (missing_train / len(df_train) * 100).round(2)
missing_info = pd.DataFrame({'count': missing_train, 'pct': missing_train_pct})
print(missing_info[missing_info['count'] > 0].to_string())

print("\n--- Missing values in EXAM set ---")
missing_exam = df_exam.isnull().sum()
missing_exam_pct = (missing_exam / len(df_exam) * 100).round(2)
missing_info_exam = pd.DataFrame({'count': missing_exam, 'pct': missing_exam_pct})
print(missing_info_exam[missing_info_exam['count'] > 0].to_string())

# Is missingness correlated with label?
print("\n--- Missingness vs. Label (Chi-square test) ---")
for col in ['f11', 'f15']:
    is_missing = df_train[col].isnull()
    pos_rate_missing = df_train.loc[is_missing, 'label'].mean()
    pos_rate_present = df_train.loc[~is_missing, 'label'].mean()
    
    contingency = pd.crosstab(is_missing, df_train['label'])
    chi2, p_val, _, _ = stats.chi2_contingency(contingency)
    
    print(f"\n  {col}:")
    print(f"    Positive rate when MISSING:  {pos_rate_missing:.4f} ({is_missing.sum()} rows)")
    print(f"    Positive rate when PRESENT:  {pos_rate_present:.4f} ({(~is_missing).sum()} rows)")
    print(f"    Chi-square: {chi2:.2f}, p-value: {p_val:.2e}")
    if p_val < 0.01:
        print(f"    *** MISSINGNESS IS LABEL-DEPENDENT (p < 0.01) ***")
    else:
        print(f"    Missingness appears random w.r.t. label")

# =============================================================================
# 4. Outlier Detection
# =============================================================================
print("\n" + "=" * 70)
print("4. OUTLIER DETECTION (IQR method)")
print("=" * 70)

outlier_summary = []
for feat in features:
    col = df_train[feat].dropna()
    Q1, Q3 = col.quantile(0.25), col.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    
    outlier_mask = (df_train[feat] < lower) | (df_train[feat] > upper)
    n_outliers = outlier_mask.sum()
    
    if n_outliers > 0:
        pos_among_outliers = df_train.loc[outlier_mask, 'label'].mean()
    else:
        pos_among_outliers = 0.0
    
    outlier_summary.append({
        'feature': feat,
        'n_outliers': n_outliers,
        'pct_outliers': round(n_outliers / len(df_train) * 100, 2),
        'pos_rate_outliers': round(pos_among_outliers, 4),
        'overall_pos_rate': round(df_train['label'].mean(), 4)
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values('n_outliers', ascending=False)
outlier_df['enrichment'] = (outlier_df['pos_rate_outliers'] / outlier_df['overall_pos_rate']).round(2)
print(outlier_df.to_string(index=False))

print(f"\n*** Features where outliers are enriched for positive class (>2x): ***")
enriched = outlier_df[outlier_df['enrichment'] > 2]
if len(enriched) > 0:
    print(enriched[['feature', 'n_outliers', 'pct_outliers', 'enrichment']].to_string(index=False))
else:
    print("  None found with >2x enrichment")

# =============================================================================
# 5. Skewness & Kurtosis
# =============================================================================
print("\n" + "=" * 70)
print("5. SKEWNESS & KURTOSIS")
print("=" * 70)

skew_kurt = pd.DataFrame({
    'skewness': df_train[features].skew().round(4),
    'kurtosis': df_train[features].kurtosis().round(4),
}).sort_values('skewness', key=abs, ascending=False)

print(skew_kurt.to_string())
print(f"\n*** Features with |skewness| > 1: ***")
skewed = skew_kurt[skew_kurt['skewness'].abs() > 1]
print(skewed.to_string() if len(skewed) > 0 else "  None")

print(f"\n*** Features with kurtosis > 5 (heavy-tailed): ***")
heavy = skew_kurt[skew_kurt['kurtosis'] > 5]
print(heavy.to_string() if len(heavy) > 0 else "  None")

# =============================================================================
# 6. Correlations with Label
# =============================================================================
print("\n" + "=" * 70)
print("6. CORRELATIONS WITH LABEL")
print("=" * 70)

spearman_with_label = df_train[features].corrwith(df_train['label'], method='spearman').abs().sort_values(ascending=False)
pearson_with_label = df_train[features].corrwith(df_train['label']).abs().sort_values(ascending=False)

print("\n--- |Spearman correlation| with label ---")
print(spearman_with_label.round(4).to_string())

print("\n--- |Pearson correlation| with label ---")
print(pearson_with_label.round(4).to_string())

# =============================================================================
# 7. Mutual Information with Label
# =============================================================================
print("\n" + "=" * 70)
print("7. MUTUAL INFORMATION WITH LABEL")
print("=" * 70)

X_for_mi = SimpleImputer(strategy='median').fit_transform(df_train[features])
mi_scores = mutual_info_classif(X_for_mi, df_train['label'], random_state=RANDOM_STATE)
mi_series = pd.Series(mi_scores, index=features).sort_values(ascending=False)

print(mi_series.round(4).to_string())

# =============================================================================
# 8. Feature-Feature Correlations
# =============================================================================
print("\n" + "=" * 70)
print("8. HIGH FEATURE-FEATURE CORRELATIONS (|r| > 0.3)")
print("=" * 70)

corr_matrix = df_train[features].corr()
pairs = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.3:
            pairs.append((features[i], features[j], round(r, 4)))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)

if pairs:
    for f1, f2, r in pairs:
        print(f"  {f1} <-> {f2}: r = {r}")
else:
    print("  No pairs with |r| > 0.3")

# =============================================================================
# 9. SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("9. EDA SUMMARY")
print("=" * 70)

print(f"\n  Dataset: {df_train.shape[0]} rows, {len(features)} features, {df_exam.shape[0]} exam rows")
print(f"  Class balance: {class_counts[1]} positive ({class_counts[1]/len(df_train)*100:.1f}%), {class_counts[0]/class_counts[1]:.1f}:1 ratio")
print(f"\n  Missingness: f11 ({df_train['f11'].isnull().sum()} = {df_train['f11'].isnull().mean()*100:.1f}%), f15 ({df_train['f15'].isnull().sum()} = {df_train['f15'].isnull().mean()*100:.1f}%)")
print(f"\n  Top features by MI: {', '.join(mi_series.head(5).index.tolist())}")
print(f"  Likely noise (low MI): {', '.join(mi_series.tail(4).index.tolist())}")
print(f"\n  f15 extremes: range [{df_train['f15'].min():.0f}, {df_train['f15'].max():.0f}], skew={df_train['f15'].skew():.2f}")

print("\nDONE!")
