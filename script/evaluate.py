# -*- coding: utf-8 -*-
"""
evaluate.py
===========
DVC 'evaluate' stage — loads the saved model and clean data,
computes final metrics, and writes metrics.json.

Usage
-----
    python script/evaluate.py
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, classification_report
)

# ---------------------------------------------------------------------------
# Feature engineering (mirrors train_pipeline.py)
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame):
    cat_cols = []
    feature_cols = []
    for c in ["is_exposed", "hour", "day_of_week"]:
        if c in df.columns:
            feature_cols.append(c)
    for col in ["device_brand", "os_name", "browser_group"]:
        if col in df.columns:
            feature_cols.append(col)
            cat_cols.append(col)

    df_ml = df[feature_cols + ["aware"]].dropna()
    num_cols = [c for c in feature_cols if c not in cat_cols]
    X = pd.concat([
        df_ml[num_cols].reset_index(drop=True),
        pd.get_dummies(df_ml[cat_cols], drop_first=True).reset_index(drop=True)
    ], axis=1)
    y = df_ml["aware"].reset_index(drop=True)
    return X, y


def run_evaluation():
    clean_path = os.path.join(REPO_ROOT, "data", "clean_data.csv")
    model_path = os.path.join(REPO_ROOT, "models", "model.pkl")
    metrics_path = os.path.join(REPO_ROOT, "metrics.json")

    print("[evaluate] Loading data and model...")
    df = pd.read_csv(clean_path)
    X, y = prepare_features(df)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    clf = joblib.load(model_path)
    print(f"[evaluate] Loaded model: {type(clf).__name__}")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.0,
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[evaluate] Evaluation complete!")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    mlflow.set_experiment("SmartAd_Classification")
    with mlflow.start_run(run_name="evaluate"):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(metrics_path)


if __name__ == "__main__":
    run_evaluation()
