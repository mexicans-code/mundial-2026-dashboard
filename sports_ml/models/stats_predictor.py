from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from sports_ml.utils.config import Config

STATS_KEYS = [
    "Shots on Goal",
    "Shots off Goal",
    "Total Shots",
    "Fouls",
    "Corner Kicks",
    "Yellow Cards",
    "Ball Possession",
    "Offsides",
    "Goalkeeper Saves",
]

STATS_LABELS = {
    "Shots on Goal": "Remates a puerta",
    "Shots off Goal": "Remates fuera",
    "Total Shots": "Tiros totales",
    "Fouls": "Faltas",
    "Corner Kicks": "Tiros de esquina",
    "Yellow Cards": "Tarjetas amarillas",
    "Ball Possession": "Posesion",
    "Offsides": "Fuera de juego",
    "Goalkeeper Saves": "Atajadas",
}

# Fallback heuristic values based on team region/style when API data unavailable.
# Key: region/type -> {stat: avg_value}
# Derived from general football knowledge for national team matches.
TEAM_PROFILES = {
    "african": {
        "Shots on Goal": 3.5,
        "Shots off Goal": 4.0,
        "Total Shots": 8.0,
        "Fouls": 14.0,
        "Corner Kicks": 3.5,
        "Yellow Cards": 2.5,
        "Ball Possession": 44.0,
        "Offsides": 1.5,
        "Goalkeeper Saves": 3.5,
    },
    "european": {
        "Shots on Goal": 5.0,
        "Shots off Goal": 5.5,
        "Total Shots": 12.0,
        "Fouls": 11.0,
        "Corner Kicks": 5.5,
        "Yellow Cards": 1.5,
        "Ball Possession": 54.0,
        "Offsides": 1.5,
        "Goalkeeper Saves": 2.5,
    },
    "south_american": {
        "Shots on Goal": 4.0,
        "Shots off Goal": 4.5,
        "Total Shots": 10.0,
        "Fouls": 15.0,
        "Corner Kicks": 4.0,
        "Yellow Cards": 3.0,
        "Ball Possession": 48.0,
        "Offsides": 2.0,
        "Goalkeeper Saves": 3.0,
    },
    "concacaf": {
        "Shots on Goal": 3.5,
        "Shots off Goal": 4.0,
        "Total Shots": 8.5,
        "Fouls": 13.0,
        "Corner Kicks": 3.5,
        "Yellow Cards": 2.5,
        "Ball Possession": 45.0,
        "Offsides": 1.5,
        "Goalkeeper Saves": 3.5,
    },
    "asian": {
        "Shots on Goal": 3.0,
        "Shots off Goal": 3.5,
        "Total Shots": 7.5,
        "Fouls": 12.0,
        "Corner Kicks": 3.0,
        "Yellow Cards": 2.0,
        "Ball Possession": 46.0,
        "Offsides": 1.0,
        "Goalkeeper Saves": 4.0,
    },
}

TEAM_REGION = {
    "Ivory Coast": "african",
    "Norway": "european",
    "Brazil": "south_american",
    "Japan": "asian",
    "Germany": "european",
    "Paraguay": "south_american",
    "Netherlands": "european",
    "Morocco": "african",
    "France": "european",
    "Sweden": "european",
    "Mexico": "concacaf",
    "Ecuador": "south_american",
    "Argentina": "south_american",
    "England": "european",
    "Spain": "european",
    "Portugal": "european",
    "Uruguay": "south_american",
    "Cape Verde": "african",
    "Belgium": "european",
    "Senegal": "african",
    "USA": "concacaf",
    "Bosnia & Herzegovina": "european",
    "Austria": "european",
    "Croatia": "european",
    "Switzerland": "european",
    "Algeria": "african",
    "Australia": "asian",
    "Egypt": "african",
    "Colombia": "south_american",
    "Ghana": "african",
    "Congo DR": "african",
}


def get_team_region(team: str) -> str:
    """Map team name to region profile. Defaults to european."""
    return TEAM_REGION.get(team, "european")


def get_heuristic_stats(team: str) -> dict[str, float]:
    """Return heuristic stat averages for a team based on region profile."""
    region = get_team_region(team)
    return dict(TEAM_PROFILES.get(region, TEAM_PROFILES["european"]))


class MatchStatsPredictor:
    """Predict per-team match statistics using API data or heuristics."""

    def __init__(self, config: Config, max_fixtures: int = 3):
        self.cfg = config
        self.api_key = config.api.football_api_key
        self.base = config.api.football_api_base
        self.max_fixtures = max_fixtures
        # Cache for API stats to avoid repeated calls
        self._stats_cache: dict[int, dict[str, dict[str, int]]] = {}

    def _fetch_fixture_stats(self, fixture_id: int) -> dict[str, dict[str, int]]:
        """Fetch statistics for a single fixture. Returns {team_name: {stat_type: value}}."""
        if not self.api_key:
            return {}
        if fixture_id in self._stats_cache:
            return self._stats_cache[fixture_id]

        url = f"{self.base}/fixtures?id={fixture_id}"
        headers = {"x-apisports-key": self.api_key}
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except HTTPError as e:
            if e.code == 429:
                print("  [API] Rate limit hit. Using heuristics for remaining stats.")
            return {}
        except Exception:
            return {}

        raw = data.get("response", [])
        if not raw:
            return {}

        result: dict[str, dict[str, int]] = {}
        for team_block in raw[0].get("statistics", []):
            team_name = team_block.get("team", {}).get("name", "")
            if not team_name:
                continue
            stats_dict: dict[str, int] = {}
            for stat in team_block.get("statistics", []):
                stype = stat.get("type", "")
                sval = stat.get("value")
                if stype in STATS_KEYS and sval is not None:
                    try:
                        # Strip % suffix
                        cleaned = str(sval).replace("%", "").strip()
                        stats_dict[stype] = int(float(cleaned))
                    except (ValueError, TypeError):
                        pass
            if stats_dict:
                result[team_name] = stats_dict

        self._stats_cache[fixture_id] = result
        return result

    def get_team_stats_from_csv(
        self, team: str, historical: pd.DataFrame, max_fixtures: int | None = None
    ) -> list[dict[str, float]]:
        """Fetch stats for recent completed fixtures of a team from the API."""
        n = max_fixtures or self.max_fixtures
        team_matches = historical[
            ((historical["home_team"] == team) | (historical["away_team"] == team))
            & (historical["home_goals"].notna())
        ].sort_values("date", ascending=False)

        fixture_ids = team_matches["fixture_id"].dropna().unique().astype(int)[:n]

        results: list[dict[str, float]] = []
        for fid in fixture_ids:
            stats = self._fetch_fixture_stats(fid)
            if team in stats and stats[team]:
                results.append(stats[team])
        return results

    @staticmethod
    def compute_averages(
        stats_list: list[dict[str, float]]
    ) -> dict[str, float]:
        """Average multiple stat dicts into per-stat means."""
        if not stats_list:
            return {}
        all_keys = set()
        for s in stats_list:
            all_keys.update(s.keys())
        averages: dict[str, float] = {}
        for key in all_keys:
            values = [s[key] for s in stats_list if key in s and s[key] is not None]
            if values:
                averages[key] = sum(values) / len(values)
        return averages

    def predict_stats(
        self, home_team: str, away_team: str, historical: pd.DataFrame
    ) -> dict:
        """Return per-team stat projections for the match.
        
        Uses API statistics if available, otherwise falls back to
        region-based heuristics.
        """
        home_stats = self.get_team_stats_from_csv(home_team, historical)
        away_stats = self.get_team_stats_from_csv(away_team, historical)

        home_avg = self.compute_averages(home_stats)
        away_avg = self.compute_averages(away_stats)

        missing_api = False
        if not home_avg:
            home_avg = get_heuristic_stats(home_team)
            missing_api = True
        if not away_avg:
            away_avg = get_heuristic_stats(away_team)
            missing_api = True

        note = "Stats historicos (API)" if not missing_api else \
                "Basado en perfil del equipo (sin datos API disponibles)"

        return {
            "home": {k: round(v, 1) for k, v in home_avg.items()},
            "away": {k: round(v, 1) for k, v in away_avg.items()},
            "note": note,
        }
