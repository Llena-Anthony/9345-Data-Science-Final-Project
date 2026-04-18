import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =========================
# Paths
# =========================
TRAIN_PATH   = "../data/preprocessed/preprocessed_mineral.csv"
TEST_PATH    = "../data/preprocessed/validation_preprocessed.csv"
RESULTS_DIR  = "../experiments/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# 1. Load training data
# =========================
train_df = pd.read_csv(TRAIN_PATH)
print("Training dataset shape:", train_df.shape)

# =========================
# 2. Load external test data
# =========================
test_df = pd.read_csv(TEST_PATH)
print("Testing dataset shape:", test_df.shape)

# =========================
# Keep only shared minerals
# =========================
train_classes = set(train_df["mineral_name"])
test_classes  = set(test_df["mineral_name"])
common_classes = train_classes.intersection(test_classes)

print("Train classes :", len(train_classes))
print("Test classes  :", len(test_classes))
print("Common classes:", len(common_classes))

train_df = train_df[train_df["mineral_name"].isin(common_classes)].copy()
test_df  = test_df[test_df["mineral_name"].isin(common_classes)].copy()

print("Filtered training shape:", train_df.shape)
print("Filtered testing shape :", test_df.shape)

# =========================
# 3. Define features and target
# =========================
target_col = "mineral_name"

X_train = train_df.drop(columns=["mineral_name", "label"], errors="ignore")
y_train = train_df[target_col]

X_test = test_df.drop(columns=["mineral_name", "label"], errors="ignore")
y_test = test_df[target_col]

print("Train features:", X_train.shape)
print("Test features :", X_test.shape)

# =========================
# 4. Ensure matching columns
# =========================
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# =========================
# 5. Optional sampling for huge dataset
# =========================
sample_size = 200000

if len(train_df) > sample_size:
    sampled = train_df.sample(n=sample_size, random_state=42)
    X_train = sampled.drop(columns=["mineral_name", "label"], errors="ignore")
    y_train = sampled[target_col]
    print(f"Using subset of {sample_size} rows for training")

# =========================
# 6. Train Random Forest
# =========================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

# =========================
# 7. Predict on measured data
# =========================
print("Predicting on measured dataset...")
y_pred = rf_model.predict(X_test)

# =========================
# 8. Evaluate
# =========================
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print("\nExternal Test Accuracy:", acc)
print("\nClassification Report:")
print(report)

# =========================
# 9. Feature importance
# =========================
feature_importance = pd.DataFrame({
    "feature":    X_train.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop 20 important features:")
print(feature_importance.head(20))

# =========================
# 10. Save results
# =========================
report_path = os.path.join(RESULTS_DIR, "randomforest_classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"External Test Accuracy: {acc}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

features_path = os.path.join(RESULTS_DIR, "randomforest_top20_features.txt")
with open(features_path, "w") as f:
    f.write("Top 20 Important Features:\n\n")
    f.write(feature_importance.head(20).to_string(index=False))

print(f"\nResults saved:")
print(f"  → {report_path}")
print(f"  → {features_path}")
