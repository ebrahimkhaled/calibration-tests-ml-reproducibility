"""Clean the four public medical data sets into one numeric format.

Each output file has the outcome in column `y` (1 = disease) and numeric predictors only.
Run from anywhere:  python 01_prepare_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

ROOT = Path(__file__).resolve().parents[1]
RAW, CLEAN = ROOT / "data" / "raw", ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)


def save(df, name):
    df = df.reset_index(drop=True)
    df.to_csv(CLEAN / f"{name}.csv.gz", index=False)
    print(f"{name:14s} n = {len(df):7d}  predictors = {df.shape[1] - 1:3d}  events = {df.y.mean():.3f}")


# Pima Indians diabetes (OpenML 37). Zeros in these columns mean "not measured".
raw, _ = arff.loadarff(RAW / "pima_diabetes.arff")
pima = pd.DataFrame(raw)
pima["y"] = (pima.pop("class").str.decode("utf-8") == "tested_positive").astype(int)
for col in ["plas", "pres", "skin", "insu", "mass"]:
    pima[col] = pima[col].replace(0, np.nan)
    pima[col] = pima[col].fillna(pima[col].median())
save(pima, "pima")

# Wisconsin diagnostic breast cancer (UCI 17): malignant = 1.
bc = pd.read_csv(RAW / "breast_cancer.csv").drop(columns="ID")
bc["y"] = (bc.pop("Diagnosis") == "M").astype(int)
save(bc, "breast_cancer")

# Cleveland heart disease (UCI 45): any narrowing (num > 0) = 1; 6 rows with missing ca/thal dropped.
hd = pd.read_csv(RAW / "heart_disease.csv").dropna()
hd["y"] = (hd.pop("num") > 0).astype(int)
hd = pd.get_dummies(hd, columns=["cp", "restecg", "slope", "thal"], drop_first=True, dtype=float)
save(hd, "heart")

# CDC Diabetes Health Indicators, BRFSS 2015 (UCI 891): diabetes or prediabetes = 1.
cdc = pd.read_csv(RAW / "cdc_diabetes.csv").drop(columns="ID")
cdc["y"] = cdc.pop("Diabetes_binary").astype(int)
save(cdc, "cdc")
