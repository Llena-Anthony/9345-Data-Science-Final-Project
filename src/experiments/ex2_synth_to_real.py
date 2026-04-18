"""
Programmer: Nathaniel Dela Rosa

Description:
This script runs Experiment 2: Synthetic-to-Real mineral classification
using a Random Forest model.

Remark:
This file does not define the Random Forest model directly. Instead, it imports
and calls the function `build_random_forest()` from `src.models.randomforest`.

The model file (randomforest.py) acts as a blueprint that defines how the model
is constructed, while this experiment file is responsible for:
- loading and preparing the datasets
- calling the model builder
- training the model using `.fit()`
- evaluating and saving the results

This separation ensures a clean architecture where model definitions are reusable
across multiple experiments, and experiment scripts handle the execution logic.
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from src.models.randomforest import build_random_forest

# =================================
# Paths
# =================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed_mineral.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "validation_preprocessed.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =================================
# 1. Load Data
# =================================
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

# =================================
# 2. Keep only shared minerals
# =================================
target_col = "mineral_name"

train_classes = set(train_df[target_col])
test_classes = set(test_df[target_col])
common_classes = train_classes.intersection(test_classes)

train_df = train_df[train_df[target_col].isin(common_classes)].copy()
test_df = test_df[test_df[target_col].isin(common_classes)].copy()

print("Filtered training shape:", train_df.shape)
print("Filtered testing shape:", test_df.shape)

# =================================
# 3. Define features and target
# =================================
X_train = train_df.drop(columns=["mineral_name", "label"], errors="ignore")
y_train = train_df[target_col]

X_test = test_df.drop(columns=["mineral_name", "label"], errors="ignore")
y_test = test_df[target_col]

print("Train features:", X_train.shape)
print("Test features :", X_test.shape)

# =================================
# 4. Ensure matching columns
# =================================
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# =================================
# 5. Optional sampling
# =================================
sample_size = 200000

if len(train_df) > sample_size:
    sampled = train_df.sample(n=sample_size, random_state=42)
    X_train = sampled.drop(columns=["mineral_name", "label"], errors="ignore")
    y_train = sampled[target_col]
    print(f"Using subset of {sample_size} rows for training")

# =================================
# 6. Build and train model
# =================================
rf_model = build_random_forest()

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

# =================================
# 7. Predict
# =================================
print("Predicting on validation dataset...")
y_pred = rf_model.predict(X_test)

# =================================
# 8. Evaluate
# =================================
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print("\nValidation Accuracy:", acc)
print("\nClassification Report:")
print(report)

# =================================
# 9. Feature importance
# =================================
feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop 20 important features:")
print(feature_importance.head(20))

# =================================
# 10. Save results
# =================================
report_path = RESULTS_DIR / "random_forest_classification_report.txt"
features_path = RESULTS_DIR / "random_forest_top20_features.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Validation Accuracy: {acc}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

with open(features_path, "w", encoding="utf-8") as f:
    f.write("Top 20 Important Features:\n\n")
    f.write(feature_importance.head(20).to_string(index=False))

print("\nResults saved:")
print(f"  -> {report_path}")
print(f"  -> {features_path}")