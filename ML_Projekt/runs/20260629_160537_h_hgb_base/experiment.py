import numpy
import pandas
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import warnings

# Root directory
ROOT = Path(__file__).resolve().parent.parent.parent

# Find data file
for _p in [ROOT / "data" / "data" / "toxic_data_01111.csv", ROOT / "data" / "toxic_data_01111.csv"]:
    if _p.exists():
        data_path = _p
        break
else:
    raise FileNotFoundError("Training data not found")

# Load data
data = pandas.read_csv(data_path)
X = data[[f'f{i:02d}' for i in range(16)]]
y = data['label']

# Initialize OOF array
oof = numpy.zeros(len(y))

# Initialize CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train and predict
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Initialize model
    model = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=100,
        learning_rate=0.1
    )

    # Train model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    # Predict probabilities
    oof[val_idx] = model.predict_proba(X_val)[:, 1]

# Save OOF predictions
numpy.save(Path("new_oof.npy"), oof.astype(numpy.float64))