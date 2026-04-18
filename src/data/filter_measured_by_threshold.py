"""
Programmer: Anthony Llena

Description:
------------
This script performs threshold-based class filtering on the preprocessed
measured mineral dataset. It is part of the controlled data preparation phase
of the project pipeline.

Purpose:
--------
The goal is to handle class imbalance by removing mineral classes with low
sample counts. This introduces a controlled experimental variable that allows
evaluation of model performance under different levels of class imbalance.

Input:
------
- data/processed/measured_preprocessed.csv
  Preprocessed measured dataset containing:
  - mineral_name (target variable)
  - numeric mineral property features

Processing Steps:
-----------------
1. Load the measured_preprocessed.csv dataset
2. Compute the number of samples per mineral class using value_counts()
3. For each defined threshold:
   - Identify classes with sample count >= threshold
   - Filter the dataset to keep only those classes
4. Generate filtered datasets for each threshold

Thresholds:
-----------
- 10 (low filtering, more classes retained)
- 20 (main experimental threshold)
- 50 (strict filtering, fewer but more stable classes)

Output:
-------
The script generates the following files:
- data/processed/measured_thr10.csv
- data/processed/measured_thr20.csv
- data/processed/measured_thr50.csv

Each file contains only the classes that meet the corresponding threshold.

Notes:
------
- This script only modifies the measured dataset
- Synthetic data is NOT filtered in this step
- This is NOT a class balancing technique (no oversampling/undersampling)
- This step must be executed BEFORE the train-test split
- The resulting datasets will be used to create D_real_train and D_real_test

Pipeline Alignment:
-------------------
This script implements Phase 3.1: Class Filtering (Controlled Imbalance Handling)
from the official project pipeline.

Next Step:
----------
After running this script, proceed to:
- src/data/split.py
to perform stratified train-test splitting on each filtered dataset.
"""

import os  # Used for interacting with the operating system (e.g., creating folders)
from pathlib import Path  # Used for handling file paths in a clean, cross-platform way
import pandas as pd  # Main library for data manipulation and analysis

# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Gets the root directory of the project by going 2 levels up from this file

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "measured_preprocessed.csv"
# Full path to the input dataset (preprocessed measured data)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "filtered"
# Directory where filtered datasets will be saved

os.makedirs(OUTPUT_DIR, exist_ok=True)
# Ensures the output directory exists; creates it if it does not

# =========================
# Configuration
# =========================
TARGET_COL = "mineral_name"
# Column name representing the target variable (mineral class)

THRESHOLDS = [10, 20, 50]
# List of thresholds for filtering classes based on minimum sample count


def log_step(message: str) -> None:
    """
    Print a simple progress message to make script execution visible in terminal.
    """
    print(f"\n[INFO] {message}")


def filter_by_threshold(df: pd.DataFrame, target_col: str, threshold: int) -> pd.DataFrame:
    """
    Keep only classes whose frequency is greater than or equal to the threshold.
    """
    log_step(f"Counting class frequencies for threshold {threshold}...")

    class_counts = df[target_col].value_counts()
    # Count how many samples each mineral class has

    log_step(f"Total classes found: {len(class_counts)}")

    valid_classes = class_counts[class_counts >= threshold].index
    # Select only classes that meet or exceed the threshold

    log_step(f"Classes meeting threshold ({threshold}): {len(valid_classes)}")

    filtered_df = df[df[target_col].isin(valid_classes)].copy()
    # Keep only rows where the mineral class belongs to the valid class list

    return filtered_df
    # Return the filtered dataset


def save_filtered_dataset(df: pd.DataFrame, threshold: int, output_dir: Path) -> None:
    """
    Save filtered dataset using the required naming convention.
    """
    output_path = output_dir / f"measured_thr{threshold}.csv"
    # Create filename based on threshold (e.g., measured_thr20.csv)

    df.to_csv(output_path, index=False)
    # Save DataFrame to CSV file without row indices

    print(f"Saved: {output_path}")
    # Print confirmation message


def print_summary(
    original_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    threshold: int,
    target_col: str
) -> None:
    """
    Print before/after summary for transparency.
    """
    original_classes = original_df[target_col].nunique()
    # Count number of unique classes before filtering

    filtered_classes = filtered_df[target_col].nunique()
    # Count number of unique classes after filtering

    print(f"\nThreshold: {threshold}")
    # Display current threshold being applied

    print(f"Original rows: {len(original_df)}")
    # Total number of rows before filtering

    print(f"Filtered rows: {len(filtered_df)}")
    # Total number of rows after filtering

    print(f"Original classes: {original_classes}")
    # Number of classes before filtering

    print(f"Retained classes: {filtered_classes}")
    # Number of classes that remain after filtering

    print(f"Removed classes: {original_classes - filtered_classes}")
    # Number of classes removed


def main() -> None:
    # -------------------------
    # 1. Load measured dataset
    # -------------------------
    log_step("Loading measured dataset...")
    df = pd.read_csv(INPUT_PATH)
    # Load the preprocessed measured dataset into a DataFrame

    log_step("Dataset loaded successfully.")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {INPUT_PATH}")
        # Ensure that the target column exists; stop execution if not

    print(f"Loaded measured dataset: {INPUT_PATH}")
    # Confirm dataset loading

    print(f"Shape: {df.shape}")
    # Show number of rows and columns

    print(f"Total classes: {df[TARGET_COL].nunique()}")
    # Show number of unique mineral classes

    print("\nFULL CLASS DISTRIBUTION (TOP 20):")
    print(df[TARGET_COL].value_counts().head(20))
    # Show the top 20 most frequent mineral classes before filtering

    # -------------------------
    # 2. Apply each threshold
    # -------------------------
    for i, threshold in enumerate(THRESHOLDS, start=1):
        print("\n" + "=" * 60)
        print(f"[PROCESS] ({i}/{len(THRESHOLDS)}) Applying threshold = {threshold}")
        print("=" * 60)

        filtered_df = filter_by_threshold(df, TARGET_COL, threshold)
        # Apply filtering based on current threshold

        print_summary(df, filtered_df, threshold, TARGET_COL)
        # Display summary of filtering results

        log_step("Saving filtered dataset...")
        save_filtered_dataset(filtered_df, threshold, OUTPUT_DIR)
        # Save the filtered dataset

        class_counts = filtered_df[TARGET_COL].value_counts()
        # Compute class counts for the filtered dataset

        counts_path = OUTPUT_DIR / f"class_counts_thr{threshold}.csv"
        # Create output path for class-count summary

        class_counts.to_csv(counts_path)
        # Save filtered class counts to CSV

        print(f"Saved class counts: {counts_path}")
        # Confirm class-count file save

        log_step(f"Finished processing threshold {threshold}")

    log_step("All thresholds processed successfully.")


if __name__ == "__main__":
    main()
    # Run the main function only when the script is executed directly
