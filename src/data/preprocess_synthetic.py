"""
Programmer: Nathaniel Dela Rosa

Preprocessing Steps Performed:
----------------------------
1. Data Loading and Merging
   - Loads all synthetic dataset parts (MineralTDSyntheticPart*.csv)
   - Combines them into a single DataFrame

2. Metadata Removal
   - Drops non-essential columns such as location, source, and identifiers
   - Ensures only relevant mineral composition features remain

3. Missing Value Handling
   - Identifies missing values in feature columns
   - Replaces all missing values with 0 for consistency

4. Data Cleaning (Target Variable)
   - Standardizes 'mineral_name' by trimming whitespace and converting to lowercase

5. Label Encoding
   - Encodes 'mineral_name' into numerical labels using LabelEncoder
   - Adds a new column 'label' for model training

6. Output Generation
   - Saves the cleaned and processed dataset to a CSV file
   - Ensures directory structure exists before saving

7. Sanity Checks
   - Verifies total rows and number of classes
   - Confirms no remaining missing values

Note:
- This script prepares synthetic data for training purposes.
- It does not perform class filtering, imbalance handling, or data splitting.
"""
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_PATTERN = str(PROJECT_ROOT / "data" / "raw" / "MineralTDSyntheticPart*.csv")
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "synthetic_preprocessed.csv"

# =========================
# 1. Load and combine synthetic files
# =========================
print("\n[1/7] Searching for synthetic files...")
files = sorted(glob.glob(SYNTHETIC_PATTERN))

if not files:
    raise FileNotFoundError(
        f"No files matched pattern: {SYNTHETIC_PATTERN}\n"
        "Check the file path or naming convention."
    )

print(f"Found {len(files)} file(s):")
for f in files:
    print(f"  - {os.path.basename(f)}")

print("\nLoading files...")
dataframes = []

for file in tqdm(files, desc="Reading CSV files", unit="file"):
    temp_df = pd.read_csv(file, encoding="ISO-8859-1")
    dataframes.append(temp_df)

df = pd.concat(dataframes, ignore_index=True)

print(f"✔ Combined shape: {df.shape}")

# =========================
# 2. Drop metadata columns
# =========================
print("\n[2/7] Dropping metadata columns...")
metadata_cols = [
    "mineral_frequency",
    "sample_label",
    "rock_name",
    "classification",
    "latitude",
    "longitude",
    "doi/ref",
    "igsn",
    "analytical_method",
    "data_source"
]

cols_to_drop = [col for col in metadata_cols if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)

print(f"✔ Dropped {len(cols_to_drop)} metadata column(s)")
if cols_to_drop:
    print("  Removed:", cols_to_drop)

# =========================
# 3. Fill missing values
# =========================
print("\n[3/7] Handling missing values...")

feature_cols = [col for col in df.columns if col != "mineral_name"]
missing_before = df[feature_cols].isnull().sum().sum()

df[feature_cols] = df[feature_cols].fillna(0)

print(f"✔ Filled {missing_before} missing value(s) with 0")

# =========================
# 4. Clean target variable
# =========================
print("\n[4/7] Cleaning mineral names...")

if "mineral_name" not in df.columns:
    raise KeyError("Column 'mineral_name' was not found in the dataset.")

missing_names_before = df["mineral_name"].isnull().sum()

df["mineral_name"] = (
    df["mineral_name"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("✔ Standardized mineral_name values")
print(f"  Missing mineral_name entries before standardization: {missing_names_before}")

# =========================
# 5. Label encoding
# =========================
print("\n[5/7] Encoding class labels...")

le = LabelEncoder()
df["label"] = le.fit_transform(df["mineral_name"])

num_classes = df["mineral_name"].nunique()
print(f"✔ Encoded {num_classes} mineral class(es)")

example_mapping = {cls: int(i) for i, cls in enumerate(le.classes_[:5])}
print(f"  Example mapping: {example_mapping} ...")

# =========================
# 6. Save processed dataset
# =========================
print("\n[6/7] Saving processed dataset...")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"✔ Saved file to: {OUTPUT_FILE}")
print(f"  Output shape: {df.shape}")

# =========================
# 7. Sanity check
# =========================
print("\n[7/7] Running sanity checks...")

remaining_nan = df[feature_cols].isnull().sum().sum()
print("── Sanity Check ─────────────────────────────")
print(f"Total rows:        {len(df)}")
print(f"Total columns:     {df.shape[1]}")
print(f"Total classes:     {df['mineral_name'].nunique()}")
print(f"Remaining NaNs:    {remaining_nan}")
print(f"Unique labels:     {df['label'].nunique()}")

if remaining_nan == 0:
    print("✔ No missing values remain in feature columns")
else:
    print("⚠ Warning: Some missing values still remain")

print("\n🚀 Synthetic preprocessing complete.")