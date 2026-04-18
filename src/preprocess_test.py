import pandas as pd
import numpy as np
import os

# =========================
# 1. Load measured validation data
# =========================
df = pd.read_csv("../data/MineralTDMeasuredPre-processed.csv")

print("Original shape:", df.shape)

# =========================
# 2. Metadata columns to remove
# =========================
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

df = df.drop(columns=metadata_cols, errors="ignore")

# =========================
# 3. Fix column names
# =========================
rename_map = {
    "(SO3)2-": "SO3",
    "ThO2.1": "V2O5",
    "Tb2O3.1": "Li",
    "ThO2 ": "PbO2",
    "Tb2O3 ": "TeO2"
}

df = df.rename(columns=rename_map)

# =========================
# 4. Expected training columns
# =========================
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
# 5. Add missing columns
# =========================
for col in expected_columns:
    if col not in df.columns:
        df[col] = 0

# =========================
# 6. Keep only expected columns
# =========================
df = df[expected_columns]

# =========================
# 7. Fill missing values
# =========================
df = df.fillna(0)

# =========================
# 8. Save validation-ready file
# =========================
OUTPUT_FILE = "../data/preprocessed/validation_preprocessed.csv"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nValidation preprocessing complete.")
print("Final shape:", df.shape)
print(f"Saved → {OUTPUT_FILE}")
