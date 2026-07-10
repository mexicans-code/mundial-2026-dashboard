"""Model trainer — trains XGBoost + Random Forest ensemble for match prediction."""

from __future__ import annotations

import json
import os
import pickle
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from sports_ml.utils.config import Config


TARGET_MAP = {
    "1X2": {"label": "result", "classes": ["home_win", "draw", "away_win"]},
    "over_under": {"label": "over_2_5", "classes": [0, 1]},
    "btts": {"label": "btts", "classes": [0, 1]},
}


class ModelTrainer:
    """Train, evaluate, and save ensemble models for sports prediction."""

    def __init__(self, config: Config):
        self.cfg = config
        self.models = {}
        self.label_encoders = {}
        self.feature_cols = []
        self._is_fitted = False

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        exclude = {
            "fixture_id", "date", "home_team", "away_team",
            "home_goals", "away_goals", "home_win", "away_win", "draw",
            "total_goals", "over_2_5", "btts", "result",
            "home_team_id", "away_team_id",
            "status_short", "league_id", "league_season", "league_round",
            "venue_city", "venue_name", "timestamp",
        }
        return [c for c in df.columns if c not in exclude and df[c].dtype in ("int64", "float64") and df[c].notna().sum() > 0]

    def _prepare_data(self, df: pd.DataFrame, target_key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        target_info = TARGET_MAP[target_key]
        target_col = target_info["label"]

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")

        df_clean = df.dropna(subset=[target_col]).copy()
        self.feature_cols = self._get_feature_cols(df_clean)
        X = df_clean[self.feature_cols].fillna(0).values
        y = df_clean[target_col].values.astype(int)
        return X, y, self.feature_cols

    def train(self, df: pd.DataFrame, target_key: str = "1X2") -> dict:
        """Train ensemble on historical data. Returns metrics."""
        print(f"\n[TRAIN] Target: {target_key}")
        X, y, features = self._prepare_data(df, target_key)

        if len(X) < 50:
            raise ValueError(f"Not enough samples ({len(X)}). Need at least 50.")

        tscv = TimeSeriesSplit(n_splits=self.cfg.model.cv_folds)
        n_classes = len(np.unique(y))

        models = []
        acc_scores = []

        # --- Random Forest ---
        rf = RandomForestClassifier(
            n_estimators=self.cfg.model.n_estimators,
            max_depth=self.cfg.model.max_depth,
            random_state=self.cfg.model.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

        cv_acc = cross_val_score(rf, X, y, cv=tscv, scoring="accuracy")
        rf.fit(X, y)
        models.append(("rf", rf))
        acc_scores.append(cv_acc.mean())
        print(f"  [RF]    CV acc: {cv_acc.mean():.3f} (+-{cv_acc.std():.3f})")

        # --- XGBoost ---
        if _HAS_XGB:
            xgb = XGBClassifier(
                n_estimators=self.cfg.model.n_estimators,
                max_depth=self.cfg.model.max_depth,
                learning_rate=self.cfg.model.learning_rate,
                random_state=self.cfg.model.random_state,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                use_label_encoder=False,
                early_stopping_rounds=self.cfg.model.early_stopping_rounds,
                n_jobs=-1,
            )

            split_idx = int(len(X) * (1 - self.cfg.model.test_size))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            xgb.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            models.append(("xgb", xgb))
            xgb_acc = accuracy_score(y_val, xgb.predict(X_val))
            acc_scores.append(xgb_acc)
            print(f"  [XGB]   Val acc: {xgb_acc:.3f}")

        self.models[target_key] = models
        self._is_fitted = True

        # Evaluate on full data
        y_pred_proba = self.predict_proba(X, target_key)
        y_pred = self.predict(X, target_key)

        metrics = {
            "target": target_key,
            "accuracy": accuracy_score(y, y_pred),
            "cv_mean": float(np.mean(acc_scores)),
            "n_samples": len(X),
            "n_features": len(features),
            "feature_cols": features,
        }

        if len(np.unique(y)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y, y_pred_proba[:, 1]))
            except Exception:
                pass

        print(f"  [ACC]   Overall: {metrics['accuracy']:.3f}")
        return metrics

    def predict_proba(self, X: np.ndarray, target_key: str = "1X2") -> np.ndarray:
        """Ensemble average of predicted probabilities."""
        if target_key not in self.models:
            raise ValueError(f"No model trained for target '{target_key}'.")

        probas = []
        for name, model in self.models[target_key]:
            p = model.predict_proba(X)
            # Handle different class counts across models
            probas.append(p)

        if not probas:
            return np.array([])

        max_classes = max(p.shape[1] for p in probas)
        aligned = []
        for p in probas:
            if p.shape[1] < max_classes:
                pad = np.zeros((p.shape[0], max_classes - p.shape[1]))
                p = np.hstack([p, pad])
            aligned.append(p)

        return np.mean(aligned, axis=0)

    def predict(self, X: np.ndarray, target_key: str = "1X2") -> np.ndarray:
        return self.predict_proba(X, target_key).argmax(axis=1)

    def save(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.cfg.model.model_dir, "ensemble.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "models": self.models,
            "feature_cols": self.feature_cols,
            "config": self.cfg,
            "_is_fitted": self._is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[SAVE]  Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "ModelTrainer":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        instance = cls(payload["config"])
        instance.models = payload["models"]
        instance.feature_cols = payload["feature_cols"]
        instance._is_fitted = payload["_is_fitted"]
        return instance
