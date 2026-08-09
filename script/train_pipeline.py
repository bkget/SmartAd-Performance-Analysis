# -*- coding: utf-8 -*-
"""
train_pipeline.py
=================
Standalone MLOps training script — called by DVC 'train' stage.

Reads clean_data.csv, trains the configured classifier,
logs everything to MLflow, and saves the model to models/model.pkl.

Usage
-----
    python script/train_pipeline.py

    # Override classifier via CLI:
    python script/train_pipeline.py --classifier xgboost --folds 5
"""

import os
import sys
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
import joblib

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from train_classifiers import TrainingClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Load params
# ---------------------------------------------------------------------------

def load_params() -> dict:
    params_path = os.path.join(REPO_ROOT, "params.yaml")
    with open(params_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame):
    cat_cols = []
    feature_cols = []

    if "is_exposed" in df.columns:
        feature_cols.append("is_exposed")
    if "hour" in df.columns:
        feature_cols.append("hour")
    if "day_of_week" in df.columns:
        feature_cols.append("day_of_week")
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


# ---------------------------------------------------------------------------
# Main training entry
# ---------------------------------------------------------------------------

def run_training(classifier: str, n_folds: int, test_size: float, random_state: int):
    clean_path = os.path.join(REPO_ROOT, "data", "clean_data.csv")
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Clean data not found: {clean_path}. Run data_pipeline.py first.")

    df = pd.read_csv(clean_path)
    print(f"[train_pipeline] Loaded clean data: {df.shape}")

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"[train_pipeline] Train: {X_train.shape}, Test: {X_test.shape}")

    mlflow.set_experiment("SmartAd_Classification")

    with mlflow.start_run(run_name=f"train_{classifier}"):
        mlflow.log_param("classifier", classifier)
        mlflow.log_param("n_folds", n_folds)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        trainer = TrainingClassifier(X_train, X_test, y_train, y_test, classifier)

        print(f"\n[train_pipeline] Training {classifier} with {n_folds}-fold CV...")
        train_res = trainer.train(folds=n_folds, track_mlflow=True)

        print("[train_pipeline] Evaluating on test set...")
        test_res = trainer.test(track_mlflow=True)

        # Save model
        model_path = os.path.join(REPO_ROOT, "models", "model.pkl")
        trainer.save_model(model_path, track_mlflow=True)

        # Save metrics JSON for DVC
        metrics = {
            "test_accuracy": test_res["accuracy"],
            "test_f1": test_res["f1"],
            "test_auc": float(test_res["auc"]) if not np.isnan(test_res["auc"]) else 0.0,
            "test_rmse": test_res["rmse"],
            "cv_mean_accuracy": float(np.mean(train_res["fold_accuracies"])),
            "cv_mean_f1": float(np.mean(train_res["fold_f1s"])),
            "cv_mean_auc": float(np.nanmean(train_res["fold_aucs"])),
        }
        metrics_path = os.path.join(REPO_ROOT, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)

        print("\n[train_pipeline] Training complete!")
        print(f"  Test Accuracy : {metrics['test_accuracy']:.4f}")
        print(f"  Test F1       : {metrics['test_f1']:.4f}")
        print(f"  Test AUC      : {metrics['test_auc']:.4f}")


if __name__ == "__main__":
    params = load_params()
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default=params["train"]["classifier"])
    parser.add_argument("--folds", type=int, default=params["train"]["n_folds"])
    parser.add_argument("--test-size", type=float, default=params["train"]["test_size"])
    parser.add_argument("--random-state", type=int, default=params["train"]["random_state"])
    args = parser.parse_args()

    run_training(args.classifier, args.folds, args.test_size, args.random_state)
