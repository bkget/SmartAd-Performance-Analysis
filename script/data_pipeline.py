# -*- coding: utf-8 -*-
"""
data_pipeline.py
================
Reproducible data loading, cleaning, and feature engineering pipeline.

This script handles all data preparation steps:
  1. Load raw CSV data.
  2. Remove non-respondents (yes=0, no=0).
  3. Engineer the target variable `aware` (yes=1 → aware, no=1 → not aware).
  4. Encode categorical features (device_make, browser, platform_os).
  5. Save clean data to data/clean_data.csv.
  6. Track all parameters and outputs with MLflow.

Usage
-----
    python script/data_pipeline.py

    # Optionally override data path:
    python script/data_pipeline.py --data-path data/AdSmart_AB_testing_data.csv

Author: SmartAd Performance Analysis Team
"""

import os
import sys
import argparse
import warnings

import pandas as pd
import numpy as np
import mlflow

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(REPO_ROOT, "data", "AdSmart_AB_testing_data.csv")
CLEAN_DATA_PATH = os.path.join(REPO_ROOT, "data", "clean_data.csv")


# ---------------------------------------------------------------------------
# Loading & Cleaning
# ---------------------------------------------------------------------------

def load_raw(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw AdSmart A/B dataset from CSV."""
    df = pd.read_csv(path)
    print(f"  Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features from the raw dataset.

    Steps:
      1. Remove rows where yes=0 AND no=0 (no survey response).
      2. Create binary target: aware = yes (1=brand aware, 0=not aware).
      3. Extract date features: day_of_week, week_number.
      4. Encode high-cardinality device_make into brand groups.
      5. Encode experiment (exposed → 1, control → 0).
      6. Drop original yes/no columns.

    Parameters
    ----------
    df : pd.DataFrame – Raw data with columns:
         auction_id, experiment, date, hour, device_make,
         platform_os, browser, yes, no

    Returns
    -------
    pd.DataFrame – Cleaned dataset ready for modelling.
    """
    # 1. Remove non-respondents
    respondents = df.query("not (yes == 0 and no == 0)").copy()
    n_removed = len(df) - len(respondents)
    print(f"  Removed {n_removed:,} non-respondents -> {len(respondents):,} rows remain")

    # 2. Target variable
    respondents["aware"] = respondents["yes"].astype(int)

    # 3. Date features
    respondents["date"] = pd.to_datetime(respondents["date"])
    respondents["day_of_week"] = respondents["date"].dt.dayofweek   # 0=Monday
    respondents["day_name"] = respondents["date"].dt.day_name()
    respondents["week"] = respondents["date"].dt.isocalendar().week.astype(int)

    # 4. Device brand grouping
    def extract_brand(device: str) -> str:
        device_lower = str(device).lower()
        if "samsung" in device_lower:
            return "Samsung"
        elif "iphone" in device_lower:
            return "Apple"
        elif "pixel" in device_lower:
            return "Google"
        elif "huawei" in device_lower or "ele-" in device_lower or "clt-" in device_lower \
                or "vog-" in device_lower or "lya-" in device_lower or "ant-" in device_lower:
            return "Huawei"
        elif "oneplus" in device_lower:
            return "OnePlus"
        elif "xiaomi" in device_lower or "redmi" in device_lower:
            return "Xiaomi"
        elif "htc" in device_lower:
            return "HTC"
        elif "nokia" in device_lower:
            return "Nokia"
        elif "lg" in device_lower:
            return "LG"
        elif "motorola" in device_lower or "moto" in device_lower or "xt" in device_lower:
            return "Motorola"
        elif "nexus" in device_lower:
            return "Google"
        elif "generic" in device_lower:
            return "Generic"
        else:
            return "Other"

    respondents["device_brand"] = respondents["device_make"].apply(extract_brand)

    # 5. Experiment encoding
    respondents["is_exposed"] = (respondents["experiment"] == "exposed").astype(int)

    # 6. Platform OS mapping
    respondents["os_name"] = respondents["platform_os"].map({5: "iOS", 6: "Android"}).fillna("Other")

    # 7. Browser simplification
    def simplify_browser(b: str) -> str:
        b = str(b).lower()
        if "facebook" in b:
            return "Facebook"
        elif "samsung" in b:
            return "Samsung Internet"
        elif "chrome mobile webview" in b:
            return "Chrome WebView"
        elif "chrome mobile ios" in b:
            return "Chrome iOS"
        elif "chrome mobile" in b:
            return "Chrome Mobile"
        elif "mobile safari" in b:
            return "Mobile Safari"
        else:
            return "Other"

    respondents["browser_group"] = respondents["browser"].apply(simplify_browser)

    # 8. Drop originals no longer needed
    drop_cols = ["yes", "no", "auction_id"]
    respondents = respondents.drop(columns=[c for c in drop_cols if c in respondents.columns])

    print(f"  Final clean dataset: {respondents.shape[0]:,} rows × {respondents.shape[1]} columns")
    print(f"  Aware rate: {respondents['aware'].mean():.2%}")
    return respondents


def encode_for_ml(df: pd.DataFrame) -> tuple:
    """
    One-hot encode categorical features and return X, y arrays.

    Parameters
    ----------
    df : pd.DataFrame – Output of clean().

    Returns
    -------
    (X, y) : tuple(pd.DataFrame, pd.Series)
    """
    feature_cols = ["is_exposed", "hour", "day_of_week", "device_brand",
                    "os_name", "browser_group", "week"]
    cat_cols = ["device_brand", "os_name", "browser_group"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    cat_encoded = pd.get_dummies(df[cat_cols], drop_first=True)
    X = pd.concat([df[num_cols].reset_index(drop=True), cat_encoded.reset_index(drop=True)], axis=1)
    y = df["aware"].reset_index(drop=True)
    print(f"  ML feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    return X, y


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(data_path: str = RAW_DATA_PATH, save_path: str = CLEAN_DATA_PATH):
    """Execute the full data preparation pipeline with MLflow logging."""
    with mlflow.start_run(run_name="data_pipeline"):
        print("\n[data_pipeline] Starting data preparation pipeline...")
        mlflow.log_param("raw_data_path", data_path)

        df_raw = load_raw(data_path)
        mlflow.log_param("raw_rows", df_raw.shape[0])
        mlflow.log_param("raw_cols", df_raw.shape[1])

        df_clean = clean(df_raw)
        mlflow.log_param("clean_rows", df_clean.shape[0])
        mlflow.log_metric("awareness_rate", float(df_clean["aware"].mean()))
        mlflow.log_metric(
            "exposed_rate",
            float((df_clean["experiment"] == "exposed").mean())
            if "experiment" in df_clean.columns else 0.0
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_clean.to_csv(save_path, index=False)
        mlflow.log_artifact(save_path)
        print(f"[data_pipeline] Clean data saved -> {save_path}")

    return df_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartAd data preparation pipeline.")
    parser.add_argument("--data-path", default=RAW_DATA_PATH, help="Path to raw CSV.")
    parser.add_argument("--save-path", default=CLEAN_DATA_PATH, help="Output clean CSV path.")
    args = parser.parse_args()
    run_pipeline(args.data_path, args.save_path)
