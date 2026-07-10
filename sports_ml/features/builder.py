"""Feature engineering — transforms raw match data into ML-ready features."""

from __future__ import annotations

import os
import math
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from sports_ml.utils.config import Config


FIFA_NAME_MAP = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde Islands": "Cabo Verde",
    "Côte d'Ivoire": "Côte d'Ivoire",
    "Korea Republic": "Korea Republic",
    "Korea DPR": "Korea DPR",
    "United States": "USA",
    "DR Congo": "Congo DR",
    "Czech Republic": "Czechia",
    "IR Iran": "IR Iran",
    "China PR": "China PR",
    "Hong Kong": "Hong Kong China",
    "St. Kitts and Nevis": "St Kitts and Nevis",
    "St. Lucia": "St Lucia",
    "St. Vincent and the Grenadines": "St Vincent and the Grenadines",
    "Cape Verde": "Cabo Verde",
    "Curacao": "Curaçao",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "North Korea": "Korea DPR",
    "South Korea": "Korea Republic",
    "Iran": "IR Iran",
    "Turkey": "Türkiye",
    "Turkiye": "Türkiye",
    "The Gambia": "The Gambia",
}


def _normalize_team_name(name: str) -> str:
    """Map common alternate team names to FIFA ranking CSV names."""
    name = name.strip()
    if name in FIFA_NAME_MAP:
        return FIFA_NAME_MAP[name]
    parts = name.split()
    if len(parts) > 2 and parts[-2].upper() in ("U", "U-") or (parts[-1].upper().startswith("U") and parts[-1][1:].isdigit()):
        base = " ".join(parts[:-1])
        return _normalize_team_name(base)
    return name


class FeatureBuilder:
    """Build features from historical match data for model training."""

    def __init__(self, config: Config):
        self.cfg = config
        self.roll = config.features.rolling_windows
        self._fifa_rankings: dict[str, int] | None = None

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Full feature pipeline from raw matches DataFrame."""
        df = matches.copy()
        if df.empty:
            return df

        for col in ["home_team", "away_team"]:
            df[col] = df[col].astype(str)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

        df = df.sort_values(["home_team", "date"]).reset_index(drop=True)

        df = self._add_form_features(df)
        df = self._add_streak_features(df)
        df = self._add_rest_days(df)
        df = self._add_head_to_head_features(df)
        df = self._add_historical_strength(df)
        df = self._add_league_context(df)
        df = self._add_fifa_rankings(df)

        df = self._encode_teams(df)

        return df

    def _load_fifa_rankings(self) -> dict[str, int]:
        """Load FIFA rankings CSV into {team_name: rank} dict."""
        if self._fifa_rankings is not None:
            return self._fifa_rankings
        path = self.cfg.features.fifa_rankings_path
        if not os.path.exists(path):
            print(f"  [WARN] FIFA rankings file not found: {path}")
            self._fifa_rankings = {}
            return self._fifa_rankings
        try:
            df = pd.read_csv(path)
            ranks = {}
            for _, row in df.iterrows():
                name = str(row.get("Team", "")).strip()
                rank = int(row.get("Ranking", 0))
                if name:
                    ranks[name] = rank
            self._fifa_rankings = ranks
            return ranks
        except Exception as e:
            print(f"  [WARN] Failed to load FIFA rankings: {e}")
            self._fifa_rankings = {}
            return self._fifa_rankings

    def _get_fifa_rank(self, team_name: str) -> int:
        """Look up a team's FIFA rank, with name normalization and fallback."""
        ranks = self._load_fifa_rankings()
        if not ranks:
            return self.cfg.features.fifa_rank_default
        name = _normalize_team_name(team_name)
        return ranks.get(name, self.cfg.features.fifa_rank_default)

    def _add_fifa_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add FIFA ranking difference as a feature."""
        if not self.cfg.features.include_fifa_rankings:
            df["home_fifa_rank"] = self.cfg.features.fifa_rank_default
            df["away_fifa_rank"] = self.cfg.features.fifa_rank_default
            df["fifa_rank_diff"] = 0
            return df
        df["home_fifa_rank"] = df["home_team"].apply(self._get_fifa_rank)
        df["away_fifa_rank"] = df["away_team"].apply(self._get_fifa_rank)
        df["fifa_rank_diff"] = df["away_fifa_rank"] - df["home_fifa_rank"]
        # Log ratio captures that rank 1 vs 10 is bigger gap than 101 vs 110
        df["log_rank_ratio"] = np.log(
            (df["away_fifa_rank"] + 1) / (df["home_fifa_rank"] + 1)
        )
        return df

    def _ewm_mean(self, series: pd.Series, halflife: int) -> pd.Series:
        """Exponentially weighted moving average (more weight to recent)."""
        return series.ewm(halflife=halflife, min_periods=1, adjust=False).mean()

    def _add_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling averages of goals over recent matches with recency weighting."""
        # Match-level goal stats (for O/U and BTTS features)
        has_goals = df["home_goals"].notna()
        df["_total_goals_match"] = 0.0
        df["_over_2_5_match"] = 0.0
        df["_btts_match"] = 0.0
        df.loc[has_goals, "_total_goals_match"] = df.loc[has_goals, "home_goals"] + df.loc[has_goals, "away_goals"]
        df.loc[has_goals, "_over_2_5_match"] = (df.loc[has_goals, "_total_goals_match"] > 2.5).astype(float)
        df.loc[has_goals, "_btts_match"] = (
            (df.loc[has_goals, "home_goals"] > 0) & (df.loc[has_goals, "away_goals"] > 0)
        ).astype(float)

        for team_col, goals_for, goals_against in [
            ("home_team", "home_goals", "away_goals"),
            ("away_team", "away_goals", "home_goals"),
        ]:
            window = self.cfg.features.rolling_windows[-1]
            halflife = max(1, window // 2)  # decay halflife = half the window
            gb = df.groupby(team_col)
            gf = gb[goals_for].transform(
                lambda x: x.ewm(halflife=halflife, min_periods=1, adjust=False).mean()
            ).shift(1)
            ga = gb[goals_against].transform(
                lambda x: x.ewm(halflife=halflife, min_periods=1, adjust=False).mean()
            ).shift(1)

            prefix = "home" if "home" in team_col else "away"
            df[f"{prefix}_avg_gf_last5"] = gf
            df[f"{prefix}_avg_ga_last5"] = ga
            df[f"{prefix}_gd_last5"] = gf - ga

            # Goal-specific rolling features: total goals avg, O/U rate, BTTS rate
            total_avg = gb["_total_goals_match"].transform(
                lambda x: x.ewm(halflife=halflife, min_periods=1, adjust=False).mean()
            ).shift(1)
            over_rate = gb["_over_2_5_match"].transform(
                lambda x: x.ewm(halflife=halflife, min_periods=1, adjust=False).mean()
            ).shift(1)
            btts_rate = gb["_btts_match"].transform(
                lambda x: x.ewm(halflife=halflife, min_periods=1, adjust=False).mean()
            ).shift(1)

            df[f"{prefix}_total_goals_avg_last5"] = total_avg
            df[f"{prefix}_over_rate_last5"] = over_rate
            df[f"{prefix}_btts_rate_last5"] = btts_rate

        df.drop(
            columns=["_total_goals_match", "_over_2_5_match", "_btts_match"],
            inplace=True, errors="ignore",
        )
        return df

    def _add_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count consecutive wins prior to current match (shift 1 to avoid leakage)."""
        for team_col, result_col in [
            ("home_team", "home_win"),
            ("away_team", "away_win"),
        ]:
            prefix = "home" if "home" in team_col else "away"
            streaks = []
            for team in df[team_col].unique():
                mask = df[team_col] == team
                team_df = df[mask].copy()
                if result_col not in team_df.columns:
                    continue
                team_df = team_df.sort_values("date")
                streak = team_df[result_col].groupby(
                    (team_df[result_col] != team_df[result_col].shift()).cumsum()
                ).cumcount()
                streaks.append(streak)

            if streaks:
                combined = pd.concat(streaks)
                idx = combined.index
                df.loc[idx, f"{prefix}_win_streak"] = combined

        df.fillna({c: 0 for c in df.columns if "streak" in c.lower()}, inplace=True)
        return df

    def _add_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.features.include_rest_days:
            return df

        for team_col in ["home_team", "away_team"]:
            prefix = "home" if "home" in team_col else "away"
            df[f"{prefix}_rest_days"] = df.groupby(team_col)["date"].diff().dt.days
        df.fillna({c: 7 for c in df.columns if "rest_days" in c.lower()}, inplace=True)
        return df

    def _add_head_to_head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.features.include_h2h:
            return df

        # Drop existing H2H columns to avoid duplication on re-build
        h2h_cols = ["h2h_home_wins", "h2h_draws", "h2h_away_wins"]
        df = df.drop(columns=[c for c in h2h_cols if c in df.columns], errors="ignore")

        h2h_rows = []
        for _, match in df.iterrows():
            home = match["home_team"]
            away = match["away_team"]
            date = match["date"]

            past = df[
                ((df["home_team"] == home) & (df["away_team"] == away) |
                 (df["home_team"] == away) & (df["away_team"] == home))
                & (df["date"] < date)
            ].tail(5)

            if past.empty:
                h2h_rows.append({"h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0})
            else:
                h2h_home = past[(past["home_team"] == home) & (past["home_goals"] > past["away_goals"])].shape[0]
                h2h_home += past[(past["away_team"] == home) & (past["away_goals"] > past["home_goals"])].shape[0]
                h2h_away = past[(past["home_team"] == away) & (past["home_goals"] > past["away_goals"])].shape[0]
                h2h_away += past[(past["away_team"] == away) & (past["away_goals"] > past["home_goals"])].shape[0]
                h2h_draws = past[past["home_goals"] == past["away_goals"]].shape[0]
                h2h_rows.append({"h2h_home_wins": h2h_home, "h2h_draws": h2h_draws, "h2h_away_wins": h2h_away})

        h2h_df = pd.DataFrame(h2h_rows, index=df.index)
        return pd.concat([df, h2h_df], axis=1)

    def _add_historical_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add baseline team quality features from ALL historical data.
        
        For each team, computes:
        - Historical win rate (proportion of matches won)
        - Average goal difference adjusted for opponent FIFA ranking
        - Combined strength_diff = win_rate_diff * 0.5 + rank_adjusted_gd_diff * 0.5
        
        Using win rate avoids the cross-confederation inflation issue
        (a team beating weak opponents vs a team facing strong opponents).
        """
        historical = df[df["home_goals"].notna()].copy()
        if historical.empty:
            df["home_strength_gd"] = 0.0
            df["away_strength_gd"] = 0.0
            df["strength_diff"] = 0.0
            df["home_win_rate"] = 0.5
            df["away_win_rate"] = 0.5
            df["win_rate_diff"] = 0.0
            return df

        # Compute per-team stats across all history
        all_teams = set(historical["home_team"]) | set(historical["away_team"])
        
        team_wins = {}
        team_matches = {}
        team_gf = {}
        team_ga = {}
        team_games = {}

        for team in all_teams:
            as_home = historical[historical["home_team"] == team]
            as_away = historical[historical["away_team"] == team]
            
            home_wins = (as_home["home_goals"] > as_home["away_goals"]).sum()
            away_wins = (as_away["away_goals"] > as_away["home_goals"]).sum()
            home_games = len(as_home)
            away_games = len(as_away)
            total = home_games + away_games
            
            team_wins[team] = home_wins + away_wins
            team_matches[team] = total
            team_games[team] = total

            if total > 0:
                h_gf = as_home["home_goals"].sum() if home_games > 0 else 0
                h_ga = as_home["away_goals"].sum() if home_games > 0 else 0
                a_gf = as_away["away_goals"].sum() if away_games > 0 else 0
                a_ga = as_away["home_goals"].sum() if away_games > 0 else 0
                team_gf[team] = (h_gf + a_gf) / total
                team_ga[team] = (h_ga + a_ga) / total
            else:
                team_gf[team] = 1.2
                team_ga[team] = 1.2

        # Win rate and goal difference features
        def _win_rate(team):
            n = team_matches.get(team, 0)
            if n >= 5:
                return team_wins.get(team, 0) / n
            # Bootstrap: shrink toward 0.35 (slightly below average team)
            w = team_wins.get(team, 0)
            return (w + 2) / (n + 5) if n > 0 else 0.35

        def _gd(team):
            return team_gf.get(team, 1.2) - team_ga.get(team, 1.2)

        df["home_win_rate"] = df["home_team"].map(_win_rate)
        df["away_win_rate"] = df["away_team"].map(_win_rate)
        df["win_rate_diff"] = df["home_win_rate"] - df["away_win_rate"]

        df["home_strength_gd"] = df["home_team"].map(lambda t: _gd(t))
        df["away_strength_gd"] = df["away_team"].map(lambda t: _gd(t))
        # Blend: win rate is more robust across confederations
        df["strength_diff"] = df["win_rate_diff"] * 3.0 + (df["home_strength_gd"] - df["away_strength_gd"]) * 0.5

        # Attack/defense ratios relative to global average (useful for O/U and BTTS)
        global_avg_gf = np.mean(list(team_gf.values())) if team_gf else 1.2
        global_avg_ga = np.mean(list(team_ga.values())) if team_ga else 1.2

        def _attack_ratio(team):
            t_gf = team_gf.get(team, global_avg_gf)
            return t_gf / global_avg_gf if global_avg_gf > 0 else 1.0

        def _defense_ratio(team):
            t_ga = team_ga.get(team, global_avg_ga)
            return t_ga / global_avg_ga if global_avg_ga > 0 else 1.0

        df["home_attack_ratio"] = df["home_team"].map(_attack_ratio)
        df["away_attack_ratio"] = df["away_team"].map(_attack_ratio)
        df["home_defense_ratio"] = df["home_team"].map(_defense_ratio)
        df["away_defense_ratio"] = df["away_team"].map(_defense_ratio)

        return df

    def _add_league_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add league-level features like home advantage."""
        if not self.cfg.features.use_home_advantage:
            return df

        historical = df[df["home_goals"].notna()]
        if historical.empty:
            return df

        home_win_rate = (historical["home_goals"] > historical["away_goals"]).mean()
        avg_home_goals = historical["home_goals"].mean()
        avg_away_goals = historical["away_goals"].mean()

        df["league_home_win_rate"] = home_win_rate
        df["league_avg_home_goals"] = avg_home_goals
        df["league_avg_away_goals"] = avg_away_goals

        return df

    @staticmethod
    def _encode_teams(df: pd.DataFrame) -> pd.DataFrame:
        """Frequency-encode team names to avoid sparse one-hot issues."""
        if df.empty:
            return df

        all_teams = pd.concat([df["home_team"], df["away_team"]])
        team_freq = all_teams.value_counts().to_dict()

        df["home_team_freq"] = df["home_team"].map(team_freq)
        df["away_team_freq"] = df["away_team"].map(team_freq)

        return df

    def build_prediction_features(
        self, match: pd.Series, historical: pd.DataFrame, feature_cols: list[str] | None = None,
        neutral_venue: bool = False,
    ) -> pd.Series:
        """Build features for a single upcoming match using historical data.
        
        If neutral_venue=True, home/away advantage features are averaged so neither
        team benefits from being arbitrarily designated as 'home'.
        """
        temp = historical.copy()
        row = match.to_dict()

        # Append the match to history with null results
        for col in ["home_goals", "away_goals", "home_win", "away_win", "draw", "total_goals", "over_2_5", "btts"]:
            row[col] = None

        row["fixture_id"] = -1  # marker for synthetic match

        # Ensure string dtypes to avoid categorical issues
        for col in ["home_team", "away_team"]:
            temp[col] = temp[col].astype(str)
            row[col] = str(row.get(col, ""))

        temp = pd.concat([temp, pd.DataFrame([row])], ignore_index=True)
        featured = self.build(temp)

        # Find synthetic match row by fixture_id marker
        match_idx = featured.index[featured["fixture_id"] == -1]
        if len(match_idx) == 0:
            raise ValueError("Synthetic match row not found after feature building.")
        result = featured.loc[match_idx[0]]

        # Neutralise home advantage for neutral venues
        # Only remove home advantage baseline; team-specific form stays intact
        if neutral_venue:
            if "league_home_win_rate" in result:
                result["league_home_win_rate"] = 0.5

        # Align to training feature columns if specified
        if feature_cols is not None:
            result = result[feature_cols].fillna(0)

        return result.copy()
