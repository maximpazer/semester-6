import numpy
import pandas
import os
import pathlib
import warnings
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Find training data path
for _p in [ROOT / "data" / "data" / "toxic_data_01111.csv", ROOT / "data" / "toxic_data_01111.csv"]:
    if _p.exists():
        data_path = _p
        break
else:
    raise FileNotFoundError("Training data not found")

# Load training data
data = pandas.read_csv(data_path)
X = data[[f'f{i:02d}' for i in range(16)]].values
y = data['label'].values

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Initialize OOF array
oof = numpy.zeros(len(y), dtype=numpy.float64)

# Define CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train and predict
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    oof[val_idx] = model.predict_proba(X_val)[:, 1]

# Save OOF predictions
numpy.save(pathlib.Path("new_oof.npy"), oof.astype(numpy.float64))