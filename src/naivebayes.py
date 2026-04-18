import pandas as pd
import numpy as np
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# =========================
# Paths
# =========================
TRAIN_PATH  = "../data/preprocessed/preprocessed_mineral.csv"
TEST_PATH   = "../data/preprocessed/validation_preprocessed.csv"
RESULTS_DIR = "../experiments/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# 1. Load datasets
# =========================
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape :", test_df.shape)

# =========================
# 2. Keep only shared minerals
# =========================
train_classes  = set(train_df["mineral_name"])
test_classes   = set(test_df["mineral_name"])
common_classes = train_classes.intersection(test_classes)

print("Train classes :", len(train_classes))
print("Test classes  :", len(test_classes))
print("Common classes:", len(common_classes))

train_df = train_df[train_df["mineral_name"].isin(common_classes)].copy()
test_df  = test_df[test_df["mineral_name"].isin(common_classes)].copy()

print("Filtered training shape:", train_df.shape)
print("Filtered testing shape :", test_df.shape)

# =========================
# 3. Define features/target
# =========================
target_col = "mineral_name"

X_train = train_df.drop(columns=["mineral_name", "label"], errors="ignore")
y_train = train_df[target_col]

X_test = test_df.drop(columns=["mineral_name", "label"], errors="ignore")
y_test = test_df[target_col]

# =========================
# 4. Match feature columns
# =========================
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("Train features:", X_train.shape)
print("Test features :", X_test.shape)

# =========================
# 5. Train Gaussian Naive Bayes
# =========================
model = GaussianNB()
print("Training Gaussian Naive Bayes...")
model.fit(X_train, y_train)

# =========================
# 6. Predict
# =========================
print("Predicting...")
y_pred = model.predict(X_test)

# =========================
# 7. Evaluate
# =========================
acc    = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print("\nExternal Test Accuracy:", acc)
print("\nClassification Report:")
print(report)

# =========================
# 8. Optional probability example
# =========================
probs = model.predict_proba(X_test.iloc[:5])
print("\nPrediction probabilities for first 5 samples:")
print(probs)

# =========================
# 9. Save results
# =========================
report_path = os.path.join(RESULTS_DIR, "naivebayes_classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"External Test Accuracy: {acc}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"\nResults saved:")
print(f"  → {report_path}")
