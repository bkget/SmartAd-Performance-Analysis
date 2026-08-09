# -*- coding: utf-8 -*-
"""
train_classifiers.py
====================
Machine-learning classifier training pipeline for A/B test feature analysis.

This module reframes brand-awareness prediction as a binary classification
task. Instead of only testing *whether* the ad works (statistical test),
we ask *which contextual features predict brand awareness response*.

Supported classifiers
---------------------
  - 'logistic_regression' – Interpretable; provides feature p-values.
  - 'decision_tree'       – Simple tree with max_depth guard.
  - 'random_forest'       – Ensemble; feature importances.
  - 'xgboost'             – Gradient-boosted trees; feature importances.

MLflow integration
------------------
Every training run is tracked automatically:
  - Parameters: classifier type, n_folds, hyperparameters.
  - Metrics: accuracy, F1, ROC-AUC, per-fold validation scores.
  - Artifacts: confusion matrix figure, feature importance / p-value table.
  - Model: saved to the run's artifact store.

Author: SmartAd Performance Analysis Team
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stat
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import joblib

from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Supported classifier registry
# ---------------------------------------------------------------------------

CLASSIFIER_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, solver="lbfgs"),
    "decision_tree": lambda: DecisionTreeClassifier(max_depth=4, random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "xgboost": lambda: XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss", random_state=42
    ),
}

CLASSIFIER_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


# ---------------------------------------------------------------------------
# Main training class
# ---------------------------------------------------------------------------

class TrainingClassifier:
    """
    End-to-end classifier training, evaluation, and MLflow tracking.

    Parameters
    ----------
    X_train    : pd.DataFrame – Feature matrix for training.
    X_test     : pd.DataFrame – Feature matrix for testing.
    y_train    : pd.Series    – Target labels for training.
    y_test     : pd.Series    – Target labels for testing.
    classifier : str          – One of the keys in CLASSIFIER_REGISTRY.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        classifier: str,
    ):
        if classifier not in CLASSIFIER_REGISTRY:
            raise ValueError(
                f"Unknown classifier '{classifier}'. "
                f"Choose from: {list(CLASSIFIER_REGISTRY.keys())}"
            )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.clf_name = classifier
        self.title = CLASSIFIER_LABELS[classifier]
        self.clf = CLASSIFIER_REGISTRY[classifier]()

    # ------------------------------------------------------------------
    # Training with cross-validation
    # ------------------------------------------------------------------

    def train(self, folds: int = 5, track_mlflow: bool = True) -> dict:
        """
        Train the classifier with stratified k-fold cross-validation.

        Parameters
        ----------
        folds        : int  – Number of folds (default 5).
        track_mlflow : bool – Log to active MLflow run if True.

        Returns
        -------
        dict with keys: clf, fold_accuracies, fold_f1s, fold_aucs.
        """
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        acc_arr, f1_arr, auc_arr = [], [], []

        print(f"\n{'='*60}")
        print(f"  Training: {self.title}  ({folds}-fold stratified CV)")
        print(f"{'='*60}")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train), 1):
            X_tr = self.X_train.iloc[train_idx]
            y_tr = self.y_train.iloc[train_idx]
            X_val = self.X_train.iloc[val_idx]
            y_val = self.y_train.iloc[val_idx]

            self.clf.fit(X_tr, y_tr)
            val_pred = self.clf.predict(X_val)
            val_prob = (
                self.clf.predict_proba(X_val)[:, 1]
                if hasattr(self.clf, "predict_proba")
                else val_pred
            )

            acc = metrics.accuracy_score(y_val, val_pred)
            f1 = f1_score(y_val, val_pred, zero_division=0)
            auc = roc_auc_score(y_val, val_prob) if len(np.unique(y_val)) > 1 else np.nan

            acc_arr.append(acc)
            f1_arr.append(f1)
            auc_arr.append(auc)

            print(f"  Fold {fold_idx:2d} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        print(f"\n  Mean Accuracy : {np.mean(acc_arr):.4f} ± {np.std(acc_arr):.4f}")
        print(f"  Mean F1       : {np.mean(f1_arr):.4f} ± {np.std(f1_arr):.4f}")
        print(f"  Mean AUC      : {np.nanmean(auc_arr):.4f} ± {np.nanstd(auc_arr):.4f}")

        # Refit on full training set
        self.clf.fit(self.X_train, self.y_train)

        if track_mlflow:
            mlflow.log_param("classifier", self.clf_name)
            mlflow.log_param("n_folds", folds)
            mlflow.log_param("n_train", len(self.X_train))
            mlflow.log_param("n_features", self.X_train.shape[1])
            mlflow.log_metric("cv_mean_accuracy", np.mean(acc_arr))
            mlflow.log_metric("cv_mean_f1", np.mean(f1_arr))
            mlflow.log_metric("cv_mean_auc", float(np.nanmean(auc_arr)))

        return {
            "clf": self.clf,
            "fold_accuracies": acc_arr,
            "fold_f1s": f1_arr,
            "fold_aucs": auc_arr,
        }

    # ------------------------------------------------------------------
    # Test-set evaluation
    # ------------------------------------------------------------------

    def test(self, track_mlflow: bool = True) -> dict:
        """
        Evaluate the fitted classifier on the held-out test set.

        Returns
        -------
        dict with keys: accuracy, f1, auc, rmse, report, confusion_matrix.
        """
        y_pred = self.clf.predict(self.X_test)
        y_prob = (
            self.clf.predict_proba(self.X_test)[:, 1]
            if hasattr(self.clf, "predict_proba")
            else y_pred
        )

        acc = metrics.accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc = roc_auc_score(self.y_test, y_prob) if len(np.unique(self.y_test)) > 1 else np.nan
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        report = classification_report(self.y_test, y_pred, zero_division=0)
        cm = confusion_matrix(self.y_test, y_pred)

        print(f"\n{'='*60}")
        print(f"  Test Evaluation: {self.title}")
        print(f"{'='*60}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  RMSE     : {rmse:.4f}")
        print(f"\n  Classification Report:\n{report}")

        if track_mlflow:
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_f1", f1)
            mlflow.log_metric("test_auc", float(auc) if not np.isnan(auc) else 0.0)
            mlflow.log_metric("test_rmse", rmse)

        return {
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "rmse": rmse,
            "report": report,
            "confusion_matrix": cm,
        }

    # ------------------------------------------------------------------
    # Confusion matrix visualisation
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self, cm: np.ndarray = None, return_fig: bool = False, track_mlflow: bool = True
    ):
        """
        Render a styled heatmap confusion matrix.

        Parameters
        ----------
        cm         : np.ndarray – If None, recomputes from test data.
        return_fig : bool       – Return figure for Streamlit embedding.
        track_mlflow: bool      – Log figure to MLflow.
        """
        if cm is None:
            y_pred = self.clf.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Aware", "Aware"],
            yticklabels=["Not Aware", "Aware"],
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix – {self.title}", fontsize=13)
        plt.tight_layout()

        if track_mlflow:
            fig_path = f"/tmp/confusion_matrix_{self.clf_name}.png"
            fig.savefig(fig_path)
            mlflow.log_artifact(fig_path)

        if return_fig:
            return fig
        plt.show()

    # ------------------------------------------------------------------
    # Logistic Regression feature p-values
    # ------------------------------------------------------------------

    def get_p_values(self) -> pd.DataFrame:
        """
        Compute approximate feature p-values for Logistic Regression.

        Uses the Fisher Information Matrix (Hessian approximation) to
        derive standard errors for coefficients, then computes two-tailed
        p-values from the resulting Z-scores.

        Returns
        -------
        pd.DataFrame with columns ['feature', 'coefficient', 'p_value', 'significant'].
        Raises ValueError if the fitted classifier is not a LogisticRegression.
        """
        if not isinstance(self.clf, LogisticRegression):
            raise ValueError("p-values are only available for Logistic Regression.")

        decision = self.clf.decision_function(self.X_train)
        denom = 2.0 * (1.0 + np.cosh(decision))
        denom_tile = np.tile(denom, (self.X_train.shape[1], 1)).T
        X_mat = self.X_train.values.astype(np.float64)

        try:
            F_ij = np.dot((X_mat / denom_tile).T, X_mat)
            Cramer_Rao = np.linalg.pinv(F_ij)
            sigma_estimates = np.sqrt(np.maximum(0, np.diagonal(Cramer_Rao)))
            z_scores = np.where(sigma_estimates > 0, self.clf.coef_[0] / sigma_estimates, 0.0)
            p_values = [float(2 * stat.norm.sf(abs(z))) for z in z_scores]  # two-tailed
        except Exception:
            p_values = [np.nan] * len(self.X_train.columns)

        return pd.DataFrame({
            "feature": self.X_train.columns.tolist(),
            "coefficient": self.clf.coef_[0],
            "p_value": p_values,
            "significant": [p < 0.05 for p in p_values],
        }).sort_values("p_value")

    def plot_p_values(self, p_df: pd.DataFrame = None, return_fig: bool = False):
        """
        Plot feature p-values against the α=0.05 significance threshold.

        Parameters
        ----------
        p_df       : pd.DataFrame – Output of get_p_values(); recomputed if None.
        return_fig : bool         – Return figure for Streamlit embedding.
        """
        if p_df is None:
            p_df = self.get_p_values()

        fig, ax = plt.subplots(figsize=(10, max(5, len(p_df) * 0.4)))
        colors = ["#2a9d8f" if sig else "#e63946" for sig in p_df["significant"]]
        ax.barh(p_df["feature"], p_df["p_value"], color=colors, alpha=0.8)
        ax.axvline(0.05, color="#f4a261", linestyle="--", linewidth=2, label="α = 0.05")
        ax.set_xlabel("P-value", fontsize=12)
        ax.set_title("Feature P-values (Logistic Regression)", fontsize=14)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        if return_fig:
            return fig
        plt.show()

    # ------------------------------------------------------------------
    # Tree-based feature importances
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importances for tree-based classifiers.

        Returns
        -------
        pd.DataFrame with columns ['feature', 'importance'].
        Raises ValueError if the classifier has no feature_importances_ attribute.
        """
        if not hasattr(self.clf, "feature_importances_"):
            raise ValueError(
                f"{self.title} does not support feature importances. "
                "Use Logistic Regression for p-values."
            )
        fi_df = pd.DataFrame({
            "feature": self.X_train.columns.tolist(),
            "importance": self.clf.feature_importances_,
        }).sort_values("importance", ascending=False)
        return fi_df

    def plot_feature_importance(self, top_n: int = 15, return_fig: bool = False):
        """
        Bar chart of top-N feature importances for tree-based models.

        Parameters
        ----------
        top_n      : int  – Number of top features to display.
        return_fig : bool – Return figure for Streamlit embedding.
        """
        fi_df = self.get_feature_importance().head(top_n)

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
        bars = ax.barh(fi_df["feature"], fi_df["importance"], color="#4361ee", alpha=0.85)
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importances – {self.title}", fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        if return_fig:
            return fig
        plt.show()

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str, track_mlflow: bool = True) -> None:
        """
        Serialise the fitted classifier to disk and optionally log to MLflow.

        Parameters
        ----------
        path         : str  – File path for the saved model (e.g. 'models/rf.pkl').
        track_mlflow : bool – If True, log the model artifact to MLflow.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.clf, path)
        print(f"  Model saved -> {path}")
        if track_mlflow:
            try:
                if self.clf_name == "xgboost":
                    import mlflow.xgboost as mlflow_xgb
                    mlflow_xgb.log_model(self.clf, name="model")
                else:
                    mlflow.sklearn.log_model(self.clf, name="model")
            except Exception:
                pass
            mlflow.log_artifact(path)