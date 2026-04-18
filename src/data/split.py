"""
Programmer: Anthony Llena

Split Script:
-------------
This script splits the filtered measured datasets into training and testing sets.

Key Rules:
- Only measured (real) data is split
- Synthetic data is NEVER split
- 80/20 train-test ratio is used
- Stratified split is applied when possible to preserve class distribution
- Output is saved per threshold for experimental consistency

Stratified Split:
-----------------
A stratified split ensures that the proportion of each mineral class
in the training and testing sets remains similar to the original dataset.

This is important for imbalanced datasets, where some mineral classes
have very few samples. Without stratification, rare classes might not
appear in either the training or testing set, leading to unreliable
model training and evaluation.

Stratified splitting is applied using the target variable (mineral_name),
ensuring fair and consistent representation of all classes.
"""

import os  # Used for directory creation
from pathlib import Path  # Used for handling file paths cleanly
import pandas as pd  # Data manipulation
from sklearn.model_selection import train_test_split  # Splitting function

# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Gets the root directory of the project

INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "filtered"
# Folder containing filtered datasets (measured_thrXX.csv)

OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
# Folder where train/test splits will be saved

os.makedirs(OUTPUT_DIR, exist_ok=True)
# Ensures the output directory exists

# =========================
# Configuration
# =========================
TARGET_COL = "mineral_name"
# Column representing the mineral class

THRESHOLDS = ["thr10", "thr20", "thr50"]
# Different imbalance levels to process

TEST_SIZE = 0.20
# 80/20 split

RANDOM_STATE = 42
# Ensures reproducibility

# =========================
# Main Processing Loop
# =========================
for thr in THRESHOLDS:
    print(f"\nProcessing threshold: {thr}")

    # =========================
    # Load dataset
    # =========================
    input_file = INPUT_DIR / f"measured_{thr}.csv"

    if not input_file.exists():
        print(f"[WARNING] File not found: {input_file}")
        continue

    df = pd.read_csv(input_file)
    print("Dataset shape:", df.shape)

    # =========================
    # Prepare features and target
    # =========================
    X = df.drop(columns=[TARGET_COL])
    # All feature columns

    y = df[TARGET_COL]
    # Target labels (mineral classes)

    # =========================
    # Check if stratified split is possible
    # =========================
    min_class_count = y.value_counts().min()
    # Minimum number of samples in any class

    if min_class_count >= 2:
        stratify_target = y
        print("Using stratified split")
    else:
        stratify_target = None
        print("Stratified split not possible (some classes too small)")

    # =========================
    # Perform train-test split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_target
    )

    # =========================
    # Reconstruct datasets
    # =========================
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test

    # =========================
    # Save outputs
    # =========================
    output_path = OUTPUT_DIR / thr
    os.makedirs(output_path, exist_ok=True)

    train_df.to_csv(output_path / "train.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    # =========================
    # Logging
    # =========================
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("Train classes:", train_df[TARGET_COL].nunique())
    print("Test classes:", test_df[TARGET_COL].nunique())

print("\nSplitting complete.")