"""
Programmer: Nathaniel Dela Rosa

Preprocessing Steps Performed:
----------------------------
1. Data Loading
   - Loads the measured dataset (MineralTDMeasured.csv)

2. Metadata Removal
   - Drops non-essential columns such as location, source, and identifiers
   - Ensures only relevant mineral composition features remain

3. Column Name Cleaning
   - Fixes inconsistent or duplicated column names using a rename mapping
   - Aligns feature names with the training dataset

4. Feature Alignment
   - Defines expected training feature columns
   - Ensures all required columns are present for model compatibility

5. Missing Column Handling
   - Adds any missing columns with default value 0
   - Ensures consistency with the training feature space

6. Column Reordering
   - Reorders columns to match the expected training schema

7. Missing Value Handling
   - Replaces all remaining missing values with 0

8. Output Generation
   - Saves the cleaned and standardized dataset to a CSV file
   - Ensures directory structure exists before saving

Note:
- This script prepares the measured dataset for evaluation.
- It does not perform class filtering, imbalance handling, or train-test splitting.
- Threshold-based filtering and splitting are handled in separate steps of the pipeline.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

print("\n[1/8] Loading dataset...")
df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "MineralTDMeasured.csv")
print(f"✔ Loaded dataset with shape: {df.shape}")

# =========================
# 2. Metadata columns to remove
# =========================
print("\n[2/8] Dropping metadata columns...")
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

before_cols = len(df.columns)
df = df.drop(columns=metadata_cols, errors="ignore")
after_cols = len(df.columns)
print(f"✔ Dropped {before_cols - after_cols} columns")

# =========================
# 3. Fix column names
# =========================
print("\n[3/8] Cleaning column names...")
rename_map = {
    "(SO3)2-": "SO3",
    "ThO2.1": "V2O5",
    "Tb2O3.1": "Li",
    "ThO2 ": "PbO2",
    "Tb2O3 ": "TeO2"
}

df = df.rename(columns=rename_map)
print("✔ Column names standardized")

# =========================
# 4. Expected training columns
# =========================
print("\n[4/8] Preparing expected feature columns...")
expected_columns = [
    "mineral_name",
    "SiO2","TiO2","Al2O3","FeO","MnO","MgO","Cr2O3","Fe2O3","CaO",
    "Na2O","K2O","P2O5","NiO","BaO","CO2","SO3","SO2","PbO","SrO",
    "ZrO2","Nb2O5","B2O3","WO3","As2O5","ZnO","MoO3","CuO","CdO",
    "Mn2O3","Cu2O","SnO","BeO","SnO2","H2O","F","Cl",
    "Si","Ti","Al","Fe","S","C","Cu","Pb","Zn","Co","Ni","As","Ag",
    "Sb","Hg","Bi","Te","Mo","Mn","Mg","Ca","Na","K","Cr","Sr","Ba",
    "Y2O3","Sc2O3","La2O3","Ce2O3","Pr2O3","Nd2O3","Sm2O3","Gd2O3",
    "Dy2O3","ThO2","UO2","Tb2O3","V2O5","Li","PbO2","TeO2","V2O3",
    "MnO2","Li2O","Cs2O","GeO2","Rb2O","NH42O","Ti2O3"
]

# =========================
# 5. Add missing columns (with progress bar)
# =========================
print("\n[5/8] Adding missing columns...")
missing_count = 0

for col in tqdm(expected_columns, desc="Checking columns"):
    if col not in df.columns:
        df[col] = 0
        missing_count += 1

print(f"✔ Added {missing_count} missing columns")

# =========================
# 6. Keep only expected columns
# =========================
print("\n[6/8] Aligning column order...")
df = df[expected_columns]
print("✔ Columns aligned to training schema")

# =========================
# 7. Fill missing values
# =========================
print("\n[7/8] Filling missing values...")
missing_before = df.isnull().sum().sum()
df = df.fillna(0)
print(f"✔ Filled {missing_before} missing values")

# =========================
# 8. Save file
# =========================
print("\n[8/8] Saving processed dataset...")
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "measured_preprocessed.csv"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\n🚀 Preprocessing complete!")
print(f"Final shape: {df.shape}")
print(f"Saved → {OUTPUT_FILE}")