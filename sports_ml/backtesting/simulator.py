"""Backtesting — simulates betting history to evaluate model performance."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from sports_ml.features.builder import FeatureBuilder
from sports_ml.models.trainer import ModelTrainer
from sports_ml.utils.config import Config


class Backtester:
    """Walk-forward backtesting to simulate real betting performance."""

    def __init__(self, config: Config, trainer: ModelTrainer):
        self.cfg = config
        self.trainer = trainer
        self.feature_builder = FeatureBuilder(config)

    @staticmethod
    def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if "home_win" not in d.columns:
            d["home_win"] = (d["home_goals"] > d["away_goals"]).astype(int)
            d["away_win"] = (d["away_goals"] > d["home_goals"]).astype(int)
            d["draw"] = (d["home_goals"] == d["away_goals"]).astype(int)
        if "total_goals" not in d.columns:
            d["total_goals"] = d["home_goals"] + d["away_goals"]
        if "over_2_5" not in d.columns:
            d["over_2_5"] = (d["total_goals"] > 2.5).astype(int)
        if "btts" not in d.columns:
            d["btts"] = ((d["home_goals"] > 0) & (d["away_goals"] > 0)).astype(int)
        if "result" not in d.columns:
            d["result"] = d["home_win"] * 0 + d["draw"] * 1 + d["away_win"] * 2
        return d

    def simulate(self, df: pd.DataFrame, initial_bankroll: float = 1000.0) -> dict:
        """Walk-forward: train on past, predict next match, evaluate."""
        df = df.sort_values("date").reset_index(drop=True)
        df = self._add_targets(df)
        bankroll = initial_bankroll
        bet_history = []

        min_train = 100
        step = 10

        print(f"\n[BACKTEST] {len(df)} matches, walk-forward...")

        for i in range(min_train, len(df) - 1, step):
            train_raw = df.iloc[:i]
            test_df = df.iloc[i:i + step]

            if len(test_df) == 0:
                break

            # Build features from raw training data
            train_features = self.feature_builder.build(train_raw)

            # Train on window
            try:
                self.trainer.train(train_features, target_key="1X2")
                self.trainer.train(train_features, target_key="over_under")
                self.trainer.train(train_features, target_key="btts")
            except Exception as e:
                print(f"  [FAIL] Step {i}: {e}")
                continue

            # Predict each match in test window
            for _, match in test_df.iterrows():
                pred_features = self.feature_builder.build_prediction_features(
                    match, train_raw, feature_cols=self.trainer.feature_cols
                )
                X = pred_features.values.reshape(1, -1) if hasattr(pred_features, 'values') else np.array(pred_features).reshape(1, -1)

                if "1X2" in self.trainer.models:
                    probs = self.trainer.predict_proba(X, "1X2")
                    if probs.size <= 1:
                        continue
                    probs = np.atleast_1d(probs.squeeze())
                    pred_class = int(np.argmax(probs))
                    confidence = float(probs[pred_class])

                    # Actual result
                    if match["home_win"]:
                        actual = 0
                    elif match["draw"]:
                        actual = 1
                    else:
                        actual = 2

                    correct = pred_class == actual

                    # Simulate bet if confidence above threshold
                    placed = False
                    if confidence >= self.cfg.betting.confidence_threshold and pred_class != 1:
                        placed = True
                        odds = 1.85  # Simplified odds for simulation
                        if correct:
                            profit = (odds - 1) * self.cfg.betting.stake_per_bet
                        else:
                            profit = -self.cfg.betting.stake_per_bet
                        bankroll += profit * 100  # Scale to bankroll units
                    else:
                        profit = 0

                    bet_history.append({
                        "index": match.name,
                        "home": match["home_team"],
                        "away": match["away_team"],
                        "predicted": pred_class,
                        "actual": actual,
                        "confidence": confidence,
                        "correct": int(correct),
                        "placed": int(placed),
                        "profit": profit,
                        "bankroll": bankroll,
                    })

        summary = self._summarize(bet_history, initial_bankroll, df)
        return summary

    @staticmethod
    def _summarize(bets: list[dict], initial: float, df: pd.DataFrame) -> dict:
        if not bets:
            return {"error": "No bets placed during backtest."}

        bet_df = pd.DataFrame(bets)
        placed = bet_df[bet_df["placed"] == 1]

        summary = {
            "total_matches": len(bets),
            "bets_placed": len(placed),
            "correct": int(placed["correct"].sum()) if not placed.empty else 0,
            "accuracy": float(placed["correct"].mean()) if not placed.empty else 0,
            "final_bankroll": float(bet_df["bankroll"].iloc[-1]) if not bet_df.empty else initial,
            "roi": float((bet_df["bankroll"].iloc[-1] - initial) / initial * 100) if not bet_df.empty else 0,
        }

        if not placed.empty:
            roi_series = placed["profit"].cumsum() / initial * 100
            summary["max_drawdown"] = float(
                (roi_series.cummax() - roi_series).max()
            )
            summary["total_profit"] = float(placed["profit"].sum())

        return summary
