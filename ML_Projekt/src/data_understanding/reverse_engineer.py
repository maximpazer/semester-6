"""
Toxic ML Project — REVERSE ENGINEERING THE INJECTION
Clean, structured analysis with dedicated plots to decode what the professor did.

FINDINGS SO FAR:
- Every dataset has EXACTLY 800 positives (10%), same structure
- The binary dataset ID encodes WHICH injections are active
- Three injection types discovered:
  1. MNAR Missingness (always 2 features, ~15%, label-dependent)
  2. Linear Transformation (shift + scale from N(0,1) to wider distribution)
  3. Heavy-Tail Injection (kurtosis ~17, Cauchy-like extreme values)
- Some features are NEVER touched (f04, f08 stay N(0,1) everywhere)
- The label is generated from the CLEAN features
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from pathlib import Path
import os

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10
RANDOM_STATE = 42

# Project paths (this script lives in src/, data in data/, plots in plots/)
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
PLOTS = ROOT / 'plots'

# --- Load all datasets ---
ours = pd.read_csv(DATA / 'toxic_data_01111.csv')
peer_ids = ['00000', '00001', '00010', '00011', '00100']
peers = {pid: pd.read_csv(DATA / f'toxic_data_{pid}.csv') for pid in peer_ids}
all_datasets = {'01111': ours, **peers}
features = [c for c in ours.columns if c != 'label']

# Create output directory for plots (kept out of version control via .gitignore)
os.makedirs(PLOTS, exist_ok=True)

# =============================================================================
# PLOT 1: Feature Classification — Which features are "clean" vs "injected"?
# =============================================================================
print("=" * 70)
print("PLOT 1: Feature State Matrix (clean/missing/scaled/heavy-tailed)")
print("=" * 70)

# Classify each feature in each dataset
states = {}  # {dataset_id: {feature: state}}
for did, df in all_datasets.items():
    states[did] = {}
    for feat in features:
        col = df[feat].dropna()
        pct_missing = df[feat].isnull().mean() * 100
        kurt = col.kurtosis()
        std = col.std()
        mean = col.mean()
        
        if pct_missing > 5:
            states[did][feat] = 'MNAR'  # Missing not at random
        elif kurt > 5:
            states[did][feat] = 'HEAVY_TAIL'  # Heavy-tailed injection
        elif std > 10:
            states[did][feat] = 'SCALED'  # Linear transform (shifted/scaled)
        else:
            states[did][feat] = 'CLEAN'  # Untouched, ~N(0,1)

# Print the state matrix
print("\nFeature State Matrix:")
print(f"{'Feature':<8}", end='')
for did in ['00000', '00001', '00010', '00011', '00100', '01111']:
    print(f"{did:<12}", end='')
print()
print("-" * 80)

for feat in features:
    print(f"{feat:<8}", end='')
    for did in ['00000', '00001', '00010', '00011', '00100', '01111']:
        state = states[did][feat]
        print(f"{state:<12}", end='')
    print()

# Create heatmap visualization
state_map = {'CLEAN': 0, 'SCALED': 1, 'HEAVY_TAIL': 2, 'MNAR': 3}
state_matrix = np.zeros((len(features), len(all_datasets)))
dataset_order = ['00000', '00001', '00010', '00011', '00100', '01111']

for j, did in enumerate(dataset_order):
    for i, feat in enumerate(features):
        state_matrix[i, j] = state_map[states[did][feat]]

fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.cm.colors.ListedColormap(['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

im = ax.imshow(state_matrix, cmap=cmap, norm=norm, aspect='auto')
ax.set_xticks(range(len(dataset_order)))
ax.set_xticklabels(dataset_order, fontsize=11)
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontsize=11)
ax.set_xlabel('Dataset ID', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('Feature Injection State Matrix\n(What the professor did to each feature)', fontsize=13)

# Add text labels in cells
for i in range(len(features)):
    for j in range(len(dataset_order)):
        state = states[dataset_order[j]][features[i]]
        ax.text(j, i, state[:5], ha='center', va='center', fontsize=8,
                color='white' if state_map[state] > 0 else 'black', fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='CLEAN (~N(0,1))'),
    Patch(facecolor='#3498db', label='SCALED (linear transform)'),
    Patch(facecolor='#e74c3c', label='HEAVY_TAIL (kurtosis>5)'),
    Patch(facecolor='#9b59b6', label='MNAR (label-dependent missingness)')
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08),
          ncol=2, fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS / '01_injection_state_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/01_injection_state_matrix.png")

# =============================================================================
# PLOT 2: Missingness is ALWAYS label-dependent (the universal pattern)
# =============================================================================
print("\n" + "=" * 70)
print("PLOT 2: MNAR Pattern — Missingness is ALWAYS correlated with label")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, (did, df) in enumerate([(k, all_datasets[k]) for k in dataset_order]):
    ax = axes[idx]
    missing_feats = [f for f in features if df[f].isnull().mean() > 0.05]
    
    if not missing_feats:
        ax.text(0.5, 0.5, 'No missing features', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Dataset {did}')
        continue
    
    # Bar chart: positive rate when missing vs present
    x = np.arange(len(missing_feats))
    width = 0.35
    
    rates_missing = [df.loc[df[f].isnull(), 'label'].mean() for f in missing_feats]
    rates_present = [df.loc[df[f].notna(), 'label'].mean() for f in missing_feats]
    
    bars1 = ax.bar(x - width/2, rates_missing, width, label='When MISSING', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, rates_present, width, label='When PRESENT', color='#2ecc71', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(missing_feats, fontsize=11)
    ax.set_ylabel('Positive Rate')
    ax.set_title(f'Dataset {did}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.35)
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='overall 10%')
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.0%}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.0%}', ha='center', va='bottom', fontsize=9)

plt.suptitle('UNIVERSAL PATTERN: Missingness is ALWAYS Label-Dependent (MNAR)\n'
             'Positive rate ~27% when missing vs ~7% when present — across ALL datasets',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS / '02_mnar_pattern_universal.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/02_mnar_pattern_universal.png")

# =============================================================================
# PLOT 3: Heavy-Tail Injection Pattern
# =============================================================================
print("\n" + "=" * 70)
print("PLOT 3: Heavy-Tail Injection (kurtosis ~17)")
print("=" * 70)

# Find all heavy-tailed features across datasets
heavy_tail_examples = []
for did in dataset_order:
    df = all_datasets[did]
    for feat in features:
        col = df[feat].dropna()
        if col.kurtosis() > 5:
            heavy_tail_examples.append((did, feat, col.kurtosis(), col.std(), col.min(), col.max()))

print("Heavy-tailed features found:")
for did, feat, kurt, std, mn, mx in heavy_tail_examples:
    print(f"  {did}/{feat}: kurtosis={kurt:.1f}, std={std:.0f}, range=[{mn:.0f}, {mx:.0f}]")

# Plot examples of heavy-tail injection
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

# Pick diverse examples
examples = [
    ('01111', 'f09', 'Our f09'),
    ('01111', 'f15', 'Our f15'),
    ('00000', 'f01', '00000/f01'),
    ('00000', 'f02', '00000/f02'),
    ('00010', 'f07', '00010/f07'),
    ('00100', 'f02', '00100/f02'),
]

for idx, (did, feat, title) in enumerate(examples):
    ax = axes[idx]
    col = all_datasets[did][feat].dropna()
    
    # Histogram
    ax.hist(col, bins=100, color='#e74c3c', alpha=0.7, density=True)
    ax.set_title(f'{title}\nkurt={col.kurtosis():.1f}, std={col.std():.0f}', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel(feat)
    
    # Add normal overlay for comparison
    x_range = np.linspace(col.min(), col.max(), 200)
    normal_pdf = stats.norm.pdf(x_range, col.mean(), col.std())
    ax.plot(x_range, normal_pdf, 'k--', alpha=0.5, label='Normal fit')
    ax.legend(fontsize=8)

plt.suptitle('HEAVY-TAIL INJECTION: Features with kurtosis ~17\n'
             '(Much heavier tails than normal — likely t-distribution or contaminated)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS / '03_heavy_tail_injection.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/03_heavy_tail_injection.png")

# =============================================================================
# PLOT 4: Linear Scaling Pattern
# =============================================================================
print("\n" + "=" * 70)
print("PLOT 4: Linear Scaling Injection")
print("=" * 70)

# Identify features that are just linearly scaled from N(0,1)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

# Compare same feature across datasets: clean in one, scaled in another
scale_examples = [
    ('01111', '00000', 'f00', 'f00: clean in ours, scaled in 00000'),
    ('01111', '00000', 'f05', 'f05: clean in ours, scaled in 00000'),
    ('01111', '00000', 'f06', 'f06: clean in ours, scaled in 00000'),
    ('00000', '01111', 'f10', 'f10: clean in 00000, scaled in ours'),
    ('00000', '01111', 'f13', 'f13: clean in 00000, scaled in ours'),
    ('00000', '01111', 'f14', 'f14: clean in 00000, scaled in ours'),
]

for idx, (clean_did, scaled_did, feat, title) in enumerate(scale_examples):
    ax = axes[idx]
    clean_col = all_datasets[clean_did][feat].dropna()
    scaled_col = all_datasets[scaled_did][feat].dropna()
    
    ax.hist(clean_col, bins=60, alpha=0.6, color='#2ecc71', density=True, 
            label=f'{clean_did} (std={clean_col.std():.1f})')
    
    # For scaled, use a twin axis
    ax2 = ax.twinx()
    ax2.hist(scaled_col, bins=60, alpha=0.4, color='#3498db', density=True,
             label=f'{scaled_did} (std={scaled_col.std():.0f})')
    ax2.set_ylabel('')
    ax2.set_yticks([])
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

plt.suptitle('LINEAR SCALING INJECTION: Same feature, different datasets\n'
             'Green = clean N(0,1), Blue = linearly transformed (shifted + scaled)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS / '04_linear_scaling_injection.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/04_linear_scaling_injection.png")

# =============================================================================
# PLOT 5: Which features actually predict the label? (MI comparison)
# =============================================================================
print("\n" + "=" * 70)
print("PLOT 5: Mutual Information with Label — Cross-Dataset Comparison")
print("=" * 70)

# Compute MI for all datasets (including missing indicators)
mi_results = {}
for did, df in all_datasets.items():
    # Features + missing indicators
    X = df[features].copy()
    for feat in features:
        if df[feat].isnull().sum() > 10:
            X[f'{feat}_missing'] = df[feat].isnull().astype(int)
    
    X_imputed = SimpleImputer(strategy='median').fit_transform(X)
    mi = mutual_info_classif(X_imputed, df['label'], random_state=RANDOM_STATE)
    mi_results[did] = pd.Series(mi, index=X.columns)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, did in enumerate(dataset_order):
    ax = axes[idx]
    mi = mi_results[did].sort_values(ascending=True)
    colors = []
    for feat_name in mi.index:
        if '_missing' in feat_name:
            colors.append('#9b59b6')  # Purple for missing indicators
        elif states[did].get(feat_name, 'CLEAN') == 'CLEAN':
            colors.append('#2ecc71')  # Green for clean
        elif states[did].get(feat_name, '') == 'SCALED':
            colors.append('#3498db')  # Blue for scaled
        elif states[did].get(feat_name, '') == 'HEAVY_TAIL':
            colors.append('#e74c3c')  # Red for heavy-tail
        else:
            colors.append('#95a5a6')  # Gray
    
    ax.barh(range(len(mi)), mi.values, color=colors)
    ax.set_yticks(range(len(mi)))
    ax.set_yticklabels(mi.index, fontsize=8)
    ax.set_title(f'Dataset {did}', fontsize=11, fontweight='bold')
    ax.set_xlabel('MI Score')

plt.suptitle('MUTUAL INFORMATION WITH LABEL — Per Dataset\n'
             'Green=clean, Blue=scaled, Red=heavy-tail, Purple=missing indicator',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS / '05_mutual_information_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/05_mutual_information_comparison.png")

# =============================================================================
# PLOT 6: "Untouched" features — f04, f08 are ALWAYS clean
# =============================================================================
print("\n" + "=" * 70)
print("PLOT 6: Features that are NEVER injected (always clean)")
print("=" * 70)

# Find features that are CLEAN in ALL datasets
always_clean = []
for feat in features:
    all_clean = all(states[did][feat] == 'CLEAN' for did in dataset_order)
    if all_clean:
        always_clean.append(feat)
print(f"  Features that are CLEAN in ALL datasets: {always_clean}")

# Also check: features that are clean in MOST datasets
mostly_clean = []
for feat in features:
    n_clean = sum(1 for did in dataset_order if states[did][feat] == 'CLEAN')
    if n_clean >= 4:
        mostly_clean.append((feat, n_clean))
print(f"  Features clean in ≥4/6 datasets: {mostly_clean}")

fig, axes = plt.subplots(2, len(always_clean), figsize=(5*len(always_clean), 8))
if len(always_clean) == 1:
    axes = axes.reshape(-1, 1)

for col_idx, feat in enumerate(always_clean):
    # Top row: distribution across all datasets (should overlap)
    ax = axes[0, col_idx] if len(always_clean) > 1 else axes[0]
    for did in dataset_order:
        col = all_datasets[did][feat].dropna()
        col.plot(kind='kde', ax=ax, alpha=0.6, label=did)
    ax.set_title(f'{feat} — Distribution (all datasets overlap)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    
    # Bottom row: MI with label across datasets
    ax = axes[1, col_idx] if len(always_clean) > 1 else axes[1]
    mi_vals = [mi_results[did].get(feat, 0) for did in dataset_order]
    ax.bar(dataset_order, mi_vals, color='#2ecc71', alpha=0.8)
    ax.set_title(f'{feat} — MI with label per dataset', fontsize=11, fontweight='bold')
    ax.set_ylabel('MI Score')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('FEATURES THAT ARE NEVER INJECTED (always ~N(0,1))\n'
             'These are candidate "true signal" features for label generation',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS / '06_always_clean_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/06_always_clean_features.png")

# =============================================================================
# PLOT 7: Decode the binary ID — what does each bit position mean?
# =============================================================================
print("\n" + "=" * 70)
print("PLOT 7: DECODING THE BINARY ID")
print("=" * 70)

print("\nDataset IDs (5-bit binary):")
print("  00000 = 0")
print("  00001 = 1")
print("  00010 = 2")
print("  00011 = 3")
print("  00100 = 4")
print("  01111 = 15 (ours)")

print("\n--- Checking which features CHANGE when a single bit flips ---")
# Compare 00000 vs 00001 (bit 0 flips)
print("\n  BIT 0 (00000 → 00001): What changes?")
for feat in features:
    s0 = states['00000'][feat]
    s1 = states['00001'][feat]
    if s0 != s1:
        print(f"    {feat}: {s0} → {s1}")

# Compare 00000 vs 00010 (bit 1 flips)
print("\n  BIT 1 (00000 → 00010): What changes?")
for feat in features:
    s0 = states['00000'][feat]
    s1 = states['00010'][feat]
    if s0 != s1:
        print(f"    {feat}: {s0} → {s1}")

# Compare 00001 vs 00011 (bit 1 flips, with bit 0 already set)
print("\n  BIT 1 (00001 → 00011): What changes?")
for feat in features:
    s0 = states['00001'][feat]
    s1 = states['00011'][feat]
    if s0 != s1:
        print(f"    {feat}: {s0} → {s1}")

# Compare 00000 vs 00100 (bit 2 flips)
print("\n  BIT 2 (00000 → 00100): What changes?")
for feat in features:
    s0 = states['00000'][feat]
    s1 = states['00100'][feat]
    if s0 != s1:
        print(f"    {feat}: {s0} → {s1}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY: PROFESSOR'S INJECTION METHOD")
print("=" * 70)

print("""
REVERSE-ENGINEERED INJECTION METHOD:
=====================================

1. BASE DATA: All 16 features start as independent N(0,1) draws (8000 samples)

2. LABEL GENERATION: The label (0/1, 10% positive) is generated as a function
   of some subset of CLEAN features. The specific features used likely vary by
   dataset — in each dataset, 2-3 "clean" features have non-zero MI with label.

3. THREE TYPES OF "TOXIC" INJECTION (controlled by the binary dataset ID):

   a) MNAR MISSINGNESS (always applied to exactly 2 features):
      - ~15% of values are set to NaN
      - Missingness is NOT random — it's correlated with the label
      - Positive class has ~27% missingness vs ~7% for negatives
      - This is the STRONGEST signal (MI ~0.02-0.04 from missing indicator alone)

   b) LINEAR SCALING (applied to several features):
      - Feature is transformed: new = a * original + b
      - Creates features with std >> 1 (typically 50-200)
      - The feature itself retains its correlation with label (if any)
      - But the scale change is a red herring / distraction

   c) HEAVY-TAIL INJECTION (applied to 1-2 features):
      - Feature values are replaced/contaminated with heavy-tailed distribution
      - Results in kurtosis ~17 (vs 0 for normal)
      - Creates extreme outliers (e.g., ±63,000 for f15 in our dataset)
      - Obscures any original signal in the feature

4. FEATURES NEVER TOUCHED: f04, f08 appear clean across ALL datasets.
   f12 is clean in most. These are likely "control" features.

5. FOR OUR DATASET (01111):
   - MNAR features: f11, f15
   - HEAVY-TAIL features: f09, f15 (f15 has BOTH missing AND heavy tails!)
   - SCALED features: f10, f11, f13, f14 (and f15 after heavy-tail)
   - CLEAN features: f00, f01, f02, f03, f04, f05, f06, f07, f08, f12
   
   Top predictors for label: 
   - f11 (via missing indicator), f12 (clean signal), f15 (via missing indicator)
   - f08, f13 have weak signal
""")

print("\nAll plots saved in plots/ directory.")
print("DONE!")
