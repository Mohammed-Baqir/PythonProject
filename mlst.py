"""
=====================================================================
 END-TO-END ML PIPELINE — FROM RAW DATA TO A TRAINED SUPERVISED MODEL
=====================================================================
Scenario: You received a dataset where only a SMALL FRACTION of rows
have labels (this is the normal situation in real projects — labeling
everything by hand is slow and expensive).

Pipeline stages (read the numbered comments top to bottom):
  1. Load data
  2. Exploratory Data Analysis (EDA)
  3. Preprocessing (cleaning, scaling, encoding)
  4. Simulate the "mostly unlabeled" situation
  5. LABEL the unlabeled rows (self-training / pseudo-labeling)
  6. Train the final supervised model on the now-fully-labeled data
  7. Evaluate
  8. Predict on brand-new, never-seen data

Run with:  python .py
Needs:     pip install packages
=====================================================================
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

RANDOM_STATE = 42

# ---------------------------------------------------------------
# STAGE 1 — LOAD DATA
# ---------------------------------------------------------------
# In a real job this is: pd.read_csv(), a SQL query, an API pull, etc.
# Here we use a built-in dataset (wine chemistry -> wine cultivar)
# so the script runs with zero setup.
raw = load_wine()
X = pd.DataFrame(raw.data, columns=raw.feature_names)
y_true = pd.Series(raw.target, name="cultivar")   # the REAL labels (we normally wouldn't have most of these)

print("=" * 70)
print("STAGE 1 — LOAD DATA")
print("=" * 70)
print(f"Rows: {X.shape[0]}   Columns: {X.shape[1]}")
print(X.head(3))

# ---------------------------------------------------------------
# STAGE 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("STAGE 2 — EDA")
print("=" * 70)
print(X.describe().T[["mean", "std", "min", "max"]])
print("\nMissing values per column:\n", X.isna().sum().sum(), "total missing cells")
print("Class balance (of the TRUE labels, just to understand the data):")
print(y_true.value_counts())

# ---------------------------------------------------------------
# STAGE 3 — PREPROCESSING
# ---------------------------------------------------------------
# Scaling matters a lot for distance-based / SVM models.
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("\n" + "=" * 70)
print("STAGE 3 — PREPROCESSING DONE (StandardScaler applied)")
print("=" * 70)

# ---------------------------------------------------------------
# STAGE 4 — SIMULATE A MOSTLY-UNLABELED DATASET
# ---------------------------------------------------------------
# In scikit-learn's semi-supervised API, an unlabeled row is marked
# with the label -1. We'll keep only 10% of the real labels and
# throw away ("hide") the other 90%, exactly like a real unlabeled
# batch you'd receive from a client.
rng = np.random.RandomState(RANDOM_STATE)
mask_labeled = rng.rand(len(y_true)) < 0.10          # 10% stay labeled
y_partial = y_true.copy()
y_partial[~mask_labeled] = -1                         # -1 = "no label yet"

print("\n" + "=" * 70)
print("STAGE 4 — SIMULATED UNLABELED DATA")
print("=" * 70)
print(f"Labeled rows:   {(y_partial != -1).sum()}")
print(f"Unlabeled rows: {(y_partial == -1).sum()}  (marked as -1)")

# ---------------------------------------------------------------
# STAGE 5 — LABEL THE UNLABELED ROWS (the key step you asked about)
# ---------------------------------------------------------------
# Technique used here: SELF-TRAINING.
#   1) Train a classifier on only the labeled rows.
#   2) Use it to predict the unlabeled rows.
#   3) Keep only the predictions the model is MOST confident about,
#      add them to the labeled pool as "pseudo-labels".
#   4) Retrain. Repeat until no more confident predictions are left.
#
# This is exactly what SelfTrainingClassifier automates for us.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
base_classifier = SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE)
self_trainer = SelfTrainingClassifier(base_classifier, threshold=0.85, verbose=False)
self_trainer.fit(X_scaled, y_partial)

y_pseudo_labeled = self_trainer.transduction_   # the full column, now labeled

n_newly_labeled = ((y_partial == -1) & (y_pseudo_labeled != -1)).sum()
print("\n" + "=" * 70)
print("STAGE 5 — PSEUDO-LABELING WITH SELF-TRAINING")
print("=" * 70)
print(f"Rows newly labeled by the model: {n_newly_labeled}")
print(f"Rows still unlabeled (model wasn't confident enough): {(y_pseudo_labeled == -1).sum()}")

# Sanity check ONLY possible here because this is a demo dataset where
# we secretly know the ground truth. In a real unlabeled project you
# would instead spot-check a sample of pseudo-labels by hand.
known_rows = y_pseudo_labeled != -1
pseudo_quality = accuracy_score(y_true[known_rows], y_pseudo_labeled[known_rows])
print(f"(Demo-only check) Pseudo-label accuracy vs real ground truth: {pseudo_quality:.3f}")

# ---------------------------------------------------------------
# STAGE 6 — TRAIN THE FINAL SUPERVISED MODEL
# ---------------------------------------------------------------
# Now that (almost) every row has a label, train a normal supervised
# classifier on it, same as you would with a fully hand-labeled dataset.
keep = y_pseudo_labeled != -1
X_final = X_scaled[keep]
y_final = y_pseudo_labeled[keep]

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.25, random_state=RANDOM_STATE, stratify=y_final
)

model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

print("\n" + "=" * 70)
print("STAGE 6 — FINAL SUPERVISED MODEL TRAINED (RandomForest)")
print("=" * 70)
print(f"Training rows: {len(X_train)}   Test rows: {len(X_test)}")

# ---------------------------------------------------------------
# STAGE 7 — EVALUATE
# ---------------------------------------------------------------
y_pred = model.predict(X_test)
print("\n" + "=" * 70)
print("STAGE 7 — EVALUATION")
print("=" * 70)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nFull report:\n", classification_report(y_test, y_pred))

# Feature importance — which columns actually drove the predictions
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 5 most important features:\n", importances.head(5))

# ---------------------------------------------------------------
# STAGE 8 — PREDICT ON BRAND-NEW DATA
# ---------------------------------------------------------------
# This is what "deployment" looks like at the smallest scale:
# one new sample comes in, you scale it the SAME way, and predict.
new_sample = X.iloc[[0]]                      # pretend this just arrived
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)

print("\n" + "=" * 70)
print("STAGE 8 — PREDICTION ON A NEW SAMPLE")
print("=" * 70)
print(f"Predicted class: {prediction[0]}")