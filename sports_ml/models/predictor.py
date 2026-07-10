"""Predictor — generates match predictions with confidence scores and value analysis."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from sports_ml.features.builder import FeatureBuilder
from sports_ml.models.trainer import ModelTrainer
from sports_ml.utils.config import Config

TARGET_NAMES_1X2 = ["Local", "Empate", "Visitante"]


class MatchPredictor:
    """Predict outcomes for upcoming matches and identify value bets."""

    def __init__(self, trainer: ModelTrainer, config: Config):
        self.trainer = trainer
        self.cfg = config
        self.feature_builder = FeatureBuilder(config)

    def predict_match(
        self, match: pd.Series, historical: pd.DataFrame, neutral_venue: bool = False
    ) -> dict:
        """Predict a single match. Returns probabilities and recommendation."""
        features = self.feature_builder.build_prediction_features(
            match, historical, feature_cols=self.trainer.feature_cols,
            neutral_venue=neutral_venue,
        )
        X = features.values.reshape(1, -1) if hasattr(features, 'values') else np.array(features).reshape(1, -1)

        results = {}
        for target in ["1X2", "over_under", "btts"]:
            if target in self.trainer.models:
                probs = self.trainer.predict_proba(X, target)
                if probs.size > 0:
                    results[target] = probs[0]

        output = {
            "fixture_id": match.get("fixture_id"),
            "home_team": match.get("home_team"),
            "away_team": match.get("away_team"),
            "date": str(match.get("date", "")),
        }

        # Parse 1X2
        if "1X2" in results:
            probs = results["1X2"]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])

            output["predicted"] = TARGET_NAMES_1X2[pred_class] if pred_class < 3 else str(pred_class)
            output["confidence"] = round(confidence, 4)
            output["prob_home"] = round(float(probs[0]), 4) if len(probs) > 0 else 0
            output["prob_draw"] = round(float(probs[1]), 4) if len(probs) > 1 else 0
            output["prob_away"] = round(float(probs[2]), 4) if len(probs) > 2 else 0

        # Parse Over/Under
        if "over_under" in results:
            probs = results["over_under"]
            if len(probs) >= 2:
                output["prob_over_2_5"] = round(float(probs[1]), 4)
                output["prob_under_2_5"] = round(float(probs[0]), 4)

        # Parse BTTS
        if "btts" in results:
            probs = results["btts"]
            if len(probs) >= 2:
                output["prob_btts_yes"] = round(float(probs[1]), 4)
                output["prob_btts_no"] = round(float(probs[0]), 4)

        # Exact score prediction using Poisson (weighted by 1X2)
        output["exact_scores"] = self.predict_exact_score(features, results.get("1X2"))

        # Store lambda (expected goals) for computing O/U lines
        home_gf = features.get("home_avg_gf_last5")
        home_ga = features.get("home_avg_ga_last5")
        away_gf = features.get("away_avg_gf_last5")
        away_ga = features.get("away_avg_ga_last5")
        if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in [home_gf, home_ga, away_gf, away_ga]):
            lam_h = (home_gf + away_ga) / 2
            lam_a = (away_gf + home_ga) / 2
            if lam_h <= 0: lam_h = home_gf
            if lam_a <= 0: lam_a = away_gf
            if results.get("1X2") is not None and len(results["1X2"]) >= 3:
                probs = results["1X2"]
                total_g = lam_h + lam_a
                h_str = probs[0] / (probs[0] + probs[2]) if (probs[0] + probs[2]) > 0 else 0.5
                lam_h = total_g * h_str
                lam_a = total_g * (1 - h_str)
            output["lambda_home"] = round(lam_h, 4)
            output["lambda_away"] = round(lam_a, 4)

        return output

    def predict_match_from_features(
        self, features: pd.Series, match: pd.Series, synthetic: bool = False
    ) -> dict:
        """Predict a match using pre-built features (avoids rebuilding features)."""
        X = features.values.reshape(1, -1) if hasattr(features, 'values') else np.array(features).reshape(1, -1)

        results = {}
        for target in ["1X2", "over_under", "btts"]:
            if target in self.trainer.models:
                probs = self.trainer.predict_proba(X, target)
                if probs.size > 0:
                    results[target] = probs[0]

        output = {
            "fixture_id": match.get("fixture_id"),
            "home_team": match.get("home_team"),
            "away_team": match.get("away_team"),
            "date": str(match.get("date", "")),
        }

        if "1X2" in results:
            probs = results["1X2"]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            output["predicted"] = ["Local", "Empate", "Visitante"][pred_class] if pred_class < 3 else str(pred_class)
            output["confidence"] = round(confidence, 4)
            output["prob_home"] = round(float(probs[0]), 4) if len(probs) > 0 else 0
            output["prob_draw"] = round(float(probs[1]), 4) if len(probs) > 1 else 0
            output["prob_away"] = round(float(probs[2]), 4) if len(probs) > 2 else 0

        if "over_under" in results:
            probs = results["over_under"]
            if len(probs) >= 2:
                output["prob_over_2_5"] = round(float(probs[1]), 4)
                output["prob_under_2_5"] = round(float(probs[0]), 4)

        if "btts" in results:
            probs = results["btts"]
            if len(probs) >= 2:
                output["prob_btts_yes"] = round(float(probs[1]), 4)
                output["prob_btts_no"] = round(float(probs[0]), 4)

        output["exact_scores"] = self.predict_exact_score(features, results.get("1X2"))

        # Store lambda for O/U lines
        home_gf = features.get("home_avg_gf_last5")
        home_ga = features.get("home_avg_ga_last5")
        away_gf = features.get("away_avg_gf_last5")
        away_ga = features.get("away_avg_ga_last5")
        if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in [home_gf, home_ga, away_gf, away_ga]):
            lam_h = (home_gf + away_ga) / 2
            lam_a = (away_gf + home_ga) / 2
            if lam_h <= 0: lam_h = home_gf
            if lam_a <= 0: lam_a = away_gf
            if results.get("1X2") is not None and len(results["1X2"]) >= 3:
                probs = results["1X2"]
                total_g = lam_h + lam_a
                h_str = probs[0] / (probs[0] + probs[2]) if (probs[0] + probs[2]) > 0 else 0.5
                lam_h = total_g * h_str
                lam_a = total_g * (1 - h_str)
            output["lambda_home"] = round(lam_h, 4)
            output["lambda_away"] = round(lam_a, 4)

        return output

    @staticmethod
    def predict_exact_score(features: pd.Series, probs_1x2: np.ndarray | None = None, max_goals: int = 5) -> list[dict]:
        """Predict top exact scores using Poisson weighted by 1X2 probabilities."""
        home_gf = features.get("home_avg_gf_last5")
        home_ga = features.get("home_avg_ga_last5")
        away_gf = features.get("away_avg_gf_last5")
        away_ga = features.get("away_avg_ga_last5")

        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [home_gf, home_ga, away_gf, away_ga]):
            return [{"score": "N/A", "prob": 0}]

        lambda_home = (home_gf + away_ga) / 2
        lambda_away = (away_gf + home_ga) / 2

        if lambda_home <= 0:
            lambda_home = home_gf
        if lambda_away <= 0:
            lambda_away = away_gf

        # Redistribute expected goals by 1X2 probabilities so scores match predicted outcome
        if probs_1x2 is not None and len(probs_1x2) >= 3:
            total_goals = lambda_home + lambda_away
            home_strength = probs_1x2[0] / (probs_1x2[0] + probs_1x2[2]) if (probs_1x2[0] + probs_1x2[2]) > 0 else 0.5
            lambda_home = total_goals * home_strength
            lambda_away = total_goals * (1 - home_strength)

        scores = []
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = (math.exp(-lambda_home) * (lambda_home ** h) / math.factorial(h)) * \
                       (math.exp(-lambda_away) * (lambda_away ** a) / math.factorial(a))
                if prob > 0.001:
                    scores.append({"score": f"{h}-{a}", "prob": round(prob, 4)})

        scores.sort(key=lambda x: x["prob"], reverse=True)
        total = sum(s["prob"] for s in scores)
        if total > 0:
            for s in scores:
                s["prob"] = round(s["prob"] / total, 4)
        return scores[:5]

    def find_value_bets(
        self, predictions: list[dict], market_odds: dict
    ) -> list[dict]:
        """Compare model probabilities vs market odds to find value."""
        value_bets = []
        for pred in predictions:
            home = pred["home_team"]
            away = pred["away_team"]
            key = f"{home} vs {away}"

            odds_row = market_odds.get(key, {})
            if not odds_row:
                continue

            # Value = (model_prob * decimal_odds) - 1
            for outcome, prob_key, odds_key in [
                ("Local", "prob_home", "home"),
                ("Empate", "prob_draw", "draw"),
                ("Visitante", "prob_away", "away"),
            ]:
                model_prob = pred.get(prob_key, 0)
                market_od = odds_row.get(odds_key, 0)

                if model_prob > 0 and market_od > 0:
                    expected_value = (model_prob * market_od) - 1
                    if expected_value > 0:
                        value_bets.append({
                            "match": key,
                            "prediction": outcome,
                            "model_prob": round(model_prob, 3),
                            "market_odds": market_od,
                            "expected_value": round(expected_value, 3),
                            "kelly_stake": round(
                                self._kelly_fraction(model_prob, market_od),
                                4,
                            ),
                        })
        return sorted(value_bets, key=lambda x: x["expected_value"], reverse=True)

    @staticmethod
    def _kelly_fraction(prob: float, odds: float) -> float:
        """Kelly Criterion: f = (bp - q) / b  where b = odds - 1."""
        b = odds - 1
        q = 1 - prob
        if b <= 0:
            return 0
        return (b * prob - q) / b

    def predict_batch(
        self, upcoming: pd.DataFrame, historical: pd.DataFrame, neutral_venue: bool = False
    ) -> list[dict]:
        """Predict all upcoming matches."""
        predictions = []
        for _, match in upcoming.iterrows():
            try:
                pred = self.predict_match(match, historical, neutral_venue=neutral_venue)
                predictions.append(pred)
            except Exception as e:
                print(f"  [SKIP] {match.get('home_team')} vs {match.get('away_team')}: {e}")
        return predictions

    def format_predictions(self, predictions: list[dict]) -> str:
        """Pretty-print predictions."""
        lines = []
        for p in predictions:
            home = p["home_team"]
            away = p["away_team"]
            pred = p.get("predicted", "?")
            conf = p.get("confidence", 0)

            line = f"  {home:20s} vs {away:20s} → {pred:10s} (conf: {conf:.1%})"

            if "prob_home" in p:
                line += f"  [{p['prob_home']:.0%} / {p['prob_draw']:.0%} / {p['prob_away']:.0%}]"
            if "prob_over_2_5" in p:
                line += f"  O2.5: {p['prob_over_2_5']:.0%}"
            if "prob_btts_yes" in p:
                line += f"  BTTS: {p['prob_btts_yes']:.0%}"

            lines.append(line)

        return "\n".join(lines)
