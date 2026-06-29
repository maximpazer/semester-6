"""
Toxic ML Project — Cross-Dataset Comparison
Goal: Compare our dataset (01111) with peers' datasets to reverse-engineer
the professor's common "toxic injection" pattern.
"Same, same but different" — there's a shared manipulation across all datasets.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

RANDOM_STATE = 42

# Project paths (this script lives in src/, data in data/)
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'

# =============================================================================
# 1. Load ALL Datasets
# =============================================================================
print("=" * 70)
print("1. LOADING ALL DATASETS")
print("=" * 70)

# Our dataset
ours = pd.read_csv(DATA / 'toxic_data_01111.csv')
ours_exam = pd.read_csv(DATA / 'toxic_exam_01111.csv')

# Peer datasets
peer_ids = ['00000', '00001', '00010', '00011', '00100']
peers = {}
peers_exam = {}

for pid in peer_ids:
    peers[pid] = pd.read_csv(DATA / f'toxic_data_{pid}.csv')
    peers_exam[pid] = pd.read_csv(DATA / f'toxic_exam_{pid}.csv')
    print(f"  Dataset {pid}: {peers[pid].shape[0]} rows, {peers[pid].shape[1]} cols, "
          f"pos_rate={peers[pid]['label'].mean():.3f}")

print(f"\n  Our dataset (01111): {ours.shape[0]} rows, {ours.shape[1]} cols, "
      f"pos_rate={ours['label'].mean():.3f}")

# Combine all for comparison
all_datasets = {'01111': ours, **peers}
features = [c for c in ours.columns if c != 'label']

# =============================================================================
# 2. Compare Basic Stats Across Datasets
# =============================================================================
print("\n" + "=" * 70)
print("2. COMPARING BASIC STATS ACROSS DATASETS")
print("=" * 70)

# Class balance comparison
print("\n--- Class Balance ---")
for did, df in all_datasets.items():
    n_pos = df['label'].sum()
    print(f"  {did}: {n_pos:.0f} positive ({df['label'].mean()*100:.1f}%), "
          f"total={len(df)}")

# =============================================================================
# 3. MISSINGNESS COMPARISON — Is it the same pattern everywhere?
# =============================================================================
print("\n" + "=" * 70)
print("3. MISSINGNESS COMPARISON")
print("=" * 70)

print("\n--- Which features have missing values in each dataset? ---")
missing_summary = {}
for did, df in all_datasets.items():
    missing_cols = {}
    for feat in features:
        pct = df[feat].isnull().mean() * 100
        if pct > 0:
            missing_cols[feat] = pct
    missing_summary[did] = missing_cols
    print(f"\n  Dataset {did}:")
    if missing_cols:
        for col, pct in sorted(missing_cols.items()):
            print(f"    {col}: {pct:.1f}% missing")
    else:
        print(f"    No missing values!")

# --- Is missingness label-dependent in ALL datasets? ---
print("\n--- Missingness vs. Label (across all datasets) ---")
miss_label_results = []
for did, df in all_datasets.items():
    for feat in features:
        n_missing = df[feat].isnull().sum()
        if n_missing > 10:  # Only check if meaningful missingness
            is_missing = df[feat].isnull()
            pos_rate_missing = df.loc[is_missing, 'label'].mean()
            pos_rate_present = df.loc[~is_missing, 'label'].mean()
            contingency = pd.crosstab(is_missing, df['label'])
            chi2, p_val, _, _ = stats.chi2_contingency(contingency)
            miss_label_results.append({
                'dataset': did,
                'feature': feat,
                'n_missing': n_missing,
                'pct_missing': n_missing / len(df) * 100,
                'pos_rate_missing': pos_rate_missing,
                'pos_rate_present': pos_rate_present,
                'ratio': pos_rate_missing / max(pos_rate_present, 0.001),
                'chi2': chi2,
                'p_value': p_val
            })

miss_label_df = pd.DataFrame(miss_label_results)
print(miss_label_df.sort_values(['feature', 'dataset']).to_string(index=False))

# =============================================================================
# 4. FEATURE DISTRIBUTION COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("4. FEATURE DISTRIBUTION COMPARISON (mean, std, skew, kurtosis)")
print("=" * 70)

for feat in features:
    print(f"\n  --- {feat} ---")
    for did, df in all_datasets.items():
        col = df[feat].dropna()
        print(f"    {did}: mean={col.mean():10.3f}, std={col.std():10.3f}, "
              f"skew={col.skew():7.3f}, kurt={col.kurtosis():8.3f}, "
              f"min={col.min():10.3f}, max={col.max():10.3f}")

# =============================================================================
# 5. CHECK: Are the "base" distributions the same across datasets?
# =============================================================================
print("\n" + "=" * 70)
print("5. KOLMOGOROV-SMIRNOV TEST: Are features from same distribution?")
print("=" * 70)
print("  (Comparing each peer dataset to ours for each feature)")

ks_results = []
for feat in features:
    ours_col = ours[feat].dropna()
    for pid in peer_ids:
        peer_col = peers[pid][feat].dropna()
        ks_stat, p_val = stats.ks_2samp(ours_col, peer_col)
        ks_results.append({
            'feature': feat,
            'peer': pid,
            'ks_stat': ks_stat,
            'p_value': p_val,
            'different': p_val < 0.01
        })

ks_df = pd.DataFrame(ks_results)
# Summarize: which features differ significantly across datasets?
print("\n--- Features that DIFFER between our dataset and peers (KS p < 0.01) ---")
diff_features = ks_df[ks_df['different']].groupby('feature').size().sort_values(ascending=False)
if len(diff_features) > 0:
    print(diff_features.to_string())
else:
    print("  All features appear to come from the same distribution!")

print("\n--- Features that are SAME across all datasets ---")
same_features = set(features) - set(diff_features.index)
print(f"  {sorted(same_features)}")

# =============================================================================
# 6. THE KEY QUESTION: What's the common "toxic" pattern?
# =============================================================================
print("\n" + "=" * 70)
print("6. IDENTIFYING THE COMMON TOXIC PATTERN")
print("=" * 70)

# Check if the dataset IDs (binary) encode something about which features are modified
print("\n--- Dataset IDs are binary numbers. Do they encode which features are 'toxic'? ---")
print("  Dataset IDs: 00000, 00001, 00010, 00011, 00100, 01111 (ours)")
print("  Could each bit position indicate a specific type of injection?")

# Let's check which features have different stats per dataset
print("\n--- Per-dataset anomaly check ---")
for did, df in all_datasets.items():
    bits = [int(b) for b in did]
    print(f"\n  Dataset {did} (bits: {bits}):")
    
    # Check missingness
    missing_feats = [f for f in features if df[f].isnull().sum() > 10]
    print(f"    Missing features: {missing_feats}")
    
    # Check heavy tails (kurtosis > 5)
    heavy_feats = [f for f in features if df[f].dropna().kurtosis() > 5]
    print(f"    Heavy-tailed (kurtosis>5): {heavy_feats}")
    
    # Check features with |skew| > 0.5
    skewed_feats = [f for f in features if abs(df[f].dropna().skew()) > 0.5]
    print(f"    Skewed (|skew|>0.5): {skewed_feats}")
    
    # Check features where mean != 0 by more than 5
    shifted_feats = [f for f in features if abs(df[f].dropna().mean()) > 5]
    print(f"    Shifted mean (|mean|>5): {shifted_feats}")
    
    # Check features with std > 10
    high_var_feats = [f for f in features if df[f].dropna().std() > 10]
    print(f"    High variance (std>10): {high_var_feats}")

# =============================================================================
# 7. Decode the binary ID pattern
# =============================================================================
print("\n" + "=" * 70)
print("7. DECODING THE BINARY ID → FEATURE MANIPULATION MAPPING")
print("=" * 70)

# Build a matrix: for each dataset, which properties differ from a "clean" baseline?
# Use dataset 00000 as the potential "cleanest" baseline
baseline = all_datasets['00000']

print("\n--- Comparing each dataset to 00000 (potential baseline) ---")
for did, df in all_datasets.items():
    if did == '00000':
        continue
    print(f"\n  Dataset {did} vs 00000:")
    for feat in features:
        col_base = baseline[feat].dropna()
        col_this = df[feat].dropna()
        
        # Check if distributions differ
        ks_stat, p_val = stats.ks_2samp(col_base, col_this)
        if p_val < 0.001:
            mean_diff = col_this.mean() - col_base.mean()
            std_diff = col_this.std() - col_base.std()
            print(f"    {feat}: KS={ks_stat:.3f} p={p_val:.2e}, "
                  f"Δmean={mean_diff:.3f}, Δstd={std_diff:.3f}")
    
    # Compare missingness
    for feat in features:
        miss_base = baseline[feat].isnull().mean()
        miss_this = df[feat].isnull().mean()
        if abs(miss_this - miss_base) > 0.01:
            print(f"    {feat}: Δmissing = {(miss_this-miss_base)*100:.1f}pp")

# =============================================================================
# 8. Check if label generation uses the same rule
# =============================================================================
print("\n" + "=" * 70)
print("8. LABEL GENERATION: Is the labeling mechanism the same?")
print("=" * 70)

# For each dataset, check which features correlate with label
print("\n--- Mutual Information with label per dataset ---")
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

mi_all = {}
for did, df in all_datasets.items():
    X = SimpleImputer(strategy='median').fit_transform(df[features])
    mi = mutual_info_classif(X, df['label'], random_state=RANDOM_STATE)
    mi_all[did] = pd.Series(mi, index=features)
    top3 = pd.Series(mi, index=features).nlargest(5)
    print(f"  {did} top-5 MI: {', '.join([f'{f}={v:.4f}' for f, v in top3.items()])}")

# Also check: does the missing-indicator predict label in all datasets?
print("\n--- Missing-indicator as predictor across datasets ---")
for did, df in all_datasets.items():
    predictive_missing = []
    for feat in features:
        if df[feat].isnull().sum() > 10:
            is_miss = df[feat].isnull().astype(int)
            mi_miss = mutual_info_classif(is_miss.values.reshape(-1, 1), df['label'],
                                          discrete_features=True, random_state=RANDOM_STATE)[0]
            predictive_missing.append((feat, mi_miss))
    if predictive_missing:
        print(f"  {did}: {[(f, f'{v:.4f}') for f, v in predictive_missing]}")
    else:
        print(f"  {did}: No missing features")

print("\n\nDONE — Check the output above to decode the injection pattern!")
