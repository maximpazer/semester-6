import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import warnings

# Set random seed for reproducibility
np.random.seed(42)

# Define root path
ROOT = Path(__file__).resolve().parent.parent.parent

# Find training data path
for _p in [ROOT / "data" / "data" / "toxic_data_01111.csv", ROOT / "data" / "toxic_data_01111.csv"]:
    if _p.exists():
        data_path = _p
        break
else:
    raise FileNotFoundError("Training data not found")

# Load training data
train_df = pd.read_csv(data_path)

# Prepare features and labels
X = train_df[[f'f{i:02d}' for i in range(16)]].values
y = train_df['label'].values

# Initialize OOF array
oof = np.zeros(len(y))

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Train and predict with ExtraTreesClassifier
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    oof[val_idx] = model.predict_proba(X_val)[:, 1]

# Save OOF predictions
np.save(Path("new_oof.npy"), oof.astype(np.float64))