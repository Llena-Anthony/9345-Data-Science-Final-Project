"""
preprocess_training.py
---------------------
Preprocessing pipeline for MineralTDSynthetic dataset.
Loads all 4 parts, cleans, encodes, and saves to a single CSV.
Output:
    - ../data/preprocessed/preprocessed_mineral.csv
"""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import LabelEncoder

# ── CONFIG ────────────────────────────────────────────────────────────────────
SYNTHETIC_PATTERN = "../data/MineralTDSyntheticPart*.csv"
OUTPUT_FILE       = "../data/preprocessed/preprocessed_mineral.csv"
# ─────────────────────────────────────────────────────────────────────────────

# ── STEP 1: Load and combine all 4 parts ─────────────────────────────────────
print("Loading synthetic files...")
files = sorted(glob.glob(SYNTHETIC_PATTERN))

if not files:
    raise FileNotFoundError(
        f"No files matched pattern '{SYNTHETIC_PATTERN}'. "
        "Check the file path or naming convention."
    )

df = pd.concat(
    [pd.read_csv(f, encoding="ISO-8859-1") for f in files],
    ignore_index=True
)
print(f"  Combined shape: {df.shape}")
print(f"  Files loaded:   {[os.path.basename(f) for f in files]}")

# ── STEP 2: Drop metadata columns ────────────────────────────────────────────
metadata_cols = [
    "mineral_frequency", "sample_label", "rock_name", "classification",
    "latitude", "longitude", "doi/ref", "igsn",
    "analytical_method", "data_source"
]
cols_to_drop = [c for c in metadata_cols if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
print(f"\nDropped metadata columns: {cols_to_drop}")

# ── STEP 3: Fill missing values with 0 ───────────────────────────────────────
feature_cols = [c for c in df.columns if c != "mineral_name"]
missing_before = df[feature_cols].isnull().sum().sum()
df[feature_cols] = df[feature_cols].fillna(0)
print(f"Filled {missing_before} missing values with 0")

# ── STEP 4: Encode mineral_name ───────────────────────────────────────────────
df["mineral_name"] = df["mineral_name"].str.strip().str.lower()
le = LabelEncoder()
df["label"] = le.fit_transform(df["mineral_name"])
print(f"\nEncoded {df['mineral_name'].nunique()} mineral classes")
print(f"  Example: { {cls: i for i, cls in enumerate(le.classes_[:5])} } ...")

# ── STEP 5: Save ──────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved → {OUTPUT_FILE} ({df.shape[0]} rows, {df.shape[1]} columns)")

# ── Sanity check ──────────────────────────────────────────────────────────────
print("\n── Sanity Check ──────────────────────────────────────────────────────")
print(f"Total rows:     {len(df)}")
print(f"Total classes:  {df['mineral_name'].nunique()}")
print(f"Any NaN left:   {df[feature_cols].isnull().any().any()}")
print("\nPreprocessing complete.")
