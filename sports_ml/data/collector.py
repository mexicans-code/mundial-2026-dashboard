"""Data collector — loads match data from APIs or local files."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from sports_ml.utils.config import Config


class FootballDataCollector:
    """Fetch football match data from API-Football (v3)."""

    def __init__(self, config: Config):
        self.cfg = config
        self.api_key = config.api.football_api_key
        self.base = config.api.football_api_base

    def _request(self, endpoint: str, params: dict) -> list[dict]:
        if not self.api_key:
            return []
        query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        url = f"{self.base}/{endpoint}?{query}"
        headers = {"x-apisports-key": self.api_key}
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            return data.get("response", [])
        except HTTPError as e:
            print(f"  [API Error {e.code}] {e.reason} — {url}")
            return []
        except Exception as e:
            print(f"  [Request Failed] {e}")
            return []

    def get_fixtures_by_league(self, league_id: int, season: int, status: str = "") -> pd.DataFrame:
        """Fetch fixtures for a league. Statuses: NS (not started), FT (finished), etc."""
        raw = self._request("fixtures", {
            "league": league_id, "season": season, "status": status or None
        })
        return self._parse_fixtures(raw) if raw else pd.DataFrame()

    def get_fixtures_between(self, date_from: str, date_to: str, league_id: int, season: int) -> pd.DataFrame:
        raw = self._request("fixtures", {
            "league": league_id, "season": season,
            "from": date_from, "to": date_to
        })
        return self._parse_fixtures(raw) if raw else pd.DataFrame()

    def get_fixtures_by_team(self, team_id: int, seasons: list[int] | None = None) -> pd.DataFrame:
        """Fetch fixtures for a specific team across multiple seasons (free tier: 2022-2024)."""
        seasons = seasons or FREE_TIER_SEASONS
        all_dfs = []
        for season in seasons:
            raw = self._request("fixtures", {
                "team": team_id, "season": season, "status": "FT"
            })
            if raw:
                df = self._parse_fixtures(raw)
                if not df.empty:
                    all_dfs.append(df)
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["fixture_id"])
        return pd.DataFrame()

    def get_team_stats(self, team_id: int, league_id: int, season: int) -> dict:
        raw = self._request("teams/statistics", {
            "team": team_id, "league": league_id, "season": season
        })
        return raw[0] if raw else {}

    def get_head_to_head(self, home_id: int, away_id: int, last: int = 10) -> pd.DataFrame:
        raw = self._request("fixtures/headtohead", {
            "h2h": f"{home_id}-{away_id}", "last": last
        })
        return self._parse_fixtures(raw) if raw else pd.DataFrame()

    @staticmethod
    def _parse_fixtures(raw: list[dict]) -> pd.DataFrame:
        rows = []
        for f in raw:
            fixture = f.get("fixture", {})
            league = f.get("league", {})
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            score = f.get("score", {})

            home_id = teams.get("home", {}).get("id")
            away_id = teams.get("away", {}).get("id")

            rows.append({
                "fixture_id": fixture.get("id"),
                "date": fixture.get("date"),
                "timestamp": fixture.get("timestamp"),
                "status_short": fixture.get("status", {}).get("short"),
                "league_id": league.get("id"),
                "league_season": league.get("season"),
                "league_round": league.get("round"),
                "home_team_id": home_id,
                "home_team": teams.get("home", {}).get("name"),
                "away_team_id": away_id,
                "away_team": teams.get("away", {}).get("name"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "venue_city": fixture.get("venue", {}).get("city"),
                "venue_name": fixture.get("venue", {}).get("name"),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df


class MockDataProvider:
    """Generates synthetic historical match data for testing the pipeline."""

    TEAMS = {
        "football": [
            "Brasil", "Argentina", "Japón", "Alemania", "Francia",
            "Inglaterra", "Países Bajos", "Portugal", "España", "Italia",
            "Uruguay", "Colombia", "Marruecos", "Senegal", "Corea del Sur",
        ],
        "basketball": [
            "Lakers", "Celtics", "Warriors", "Bulls", "Heat",
            "Nuggets", "Bucks", "Suns", "76ers", "Knicks",
        ],
    }

    def __init__(self, sport: str = "football"):
        self.sport = sport
        self.teams = self.TEAMS.get(sport, self.TEAMS["football"])

    def generate_matches(self, n_matches: int = 500) -> pd.DataFrame:
        import random
        random.seed(42)

        records = []
        start_date = datetime(2023, 1, 1)

        for i in range(n_matches):
            date = start_date + timedelta(days=random.randint(0, 3))
            home, away = random.sample(self.teams, 2)
            home_goals = random.choices([0, 1, 2, 3, 4, 5], weights=[12, 25, 28, 20, 10, 5])[0]
            away_goals = random.choices([0, 1, 2, 3, 4, 5], weights=[20, 28, 25, 15, 8, 4])[0]

            records.append({
                "fixture_id": i + 1,
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_team_id": hash(home) % 10000,
                "away_team_id": hash(away) % 10000,
                "status_short": "FT",
            })
            start_date = date

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)
        df["away_win"] = (df["away_goals"] > df["home_goals"]).astype(int)
        df["draw"] = (df["home_goals"] == df["away_goals"]).astype(int)
        df["total_goals"] = df["home_goals"] + df["away_goals"]
        df["over_2_5"] = (df["total_goals"] > 2.5).astype(int)
        df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
        return df.sort_values("date").reset_index(drop=True)


NATIONAL_TEAMS = {
    "brasil": {"id": 6, "name": "Brazil"},
    "japon": {"id": 12, "name": "Japan"},
    "argentina": {"id": 26, "name": "Argentina"},
    "alemania": {"id": 25, "name": "Germany"},
    "paraguay": {"id": 2380, "name": "Paraguay"},
    "francia": {"id": 2, "name": "France"},
    "inglaterra": {"id": 10, "name": "England"},
    "espana": {"id": 9, "name": "Spain"},
    "paises bajos": {"id": 1118, "name": "Netherlands"},
    "portugal": {"id": 27, "name": "Portugal"},
    "marruecos": {"id": 31, "name": "Morocco"},
    "uruguay": {"id": 7, "name": "Uruguay"},
    "costa de marfil": {"id": 1501, "name": "Ivory Coast"},
    "noruega": {"id": 1090, "name": "Norway"},
    "suecia": {"id": 5, "name": "Sweden"},
    "mexico": {"id": 16, "name": "Mexico"},
    "ecuador": {"id": 2382, "name": "Ecuador"},
    "cabo verde": {"id": 1533, "name": "Cape Verde"},
    "belgica": {"id": 1, "name": "Belgium"},
    "senegal": {"id": 13, "name": "Senegal"},
    "estados unidos": {"id": 2384, "name": "USA"},
    "bosnia": {"id": 1113, "name": "Bosnia & Herzegovina"},
    "austria": {"id": 775, "name": "Austria"},
    "croacia": {"id": 3, "name": "Croatia"},
    "suiza": {"id": 15, "name": "Switzerland"},
    "argelia": {"id": 1532, "name": "Algeria"},
    "australia": {"id": 20, "name": "Australia"},
    "egipto": {"id": 32, "name": "Egypt"},
    "colombia": {"id": 8, "name": "Colombia"},
    "ghana": {"id": 1504, "name": "Ghana"},
    "rd congo": {"id": 1508, "name": "Congo DR"},
}

FREE_TIER_SEASONS = [2022, 2023, 2024]


def load_matches_by_teams(
    config: Config, team_ids: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load recent matches for specific teams (national teams). Free tier compatible."""
    collector = FootballDataCollector(config)
    all_matches = []

    for tid in team_ids:
        df = collector.get_fixtures_by_team(tid)
        team_name = next((v["name"] for k, v in NATIONAL_TEAMS.items() if v["id"] == tid), str(tid))
        if not df.empty:
            n = len(df[df["home_goals"].notna()]) if "home_goals" in df.columns else 0
            print(f"     {team_name}: {n} finished matches")
            all_matches.append(df)
        else:
            print(f"     {team_name}: no data")

    if not all_matches:
        return pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(all_matches, ignore_index=True).drop_duplicates(subset=["fixture_id"])

    finished = combined[combined["home_goals"].notna()].copy()
    upcoming = combined[combined["home_goals"].isna()].copy()

    print(f"     Total: {len(finished)} finished, {len(upcoming)} upcoming")
    return finished, upcoming


def load_matches(config: Config, use_mock: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load matches from API or mock data. Returns (finished_matches, upcoming_matches)."""
    collector = FootballDataCollector(config)

    if use_mock or not config.api.football_api_key:
        print("[MOCK] Using mock data (no API key provided or --mock flag set)")
        mock = MockDataProvider(sport=config.sport)
        full = mock.generate_matches(500)
        finished = full.iloc[:-5].copy()
        upcoming = full.iloc[-5:].copy()
        upcoming["home_goals"] = None
        upcoming["away_goals"] = None
        return finished, upcoming

    # If team_ids configured, fetch by team
    if hasattr(config, "team_ids") and config.team_ids:
        print(f"[API]  Fetching last matches for team IDs: {config.team_ids}")
        return load_matches_by_teams(config, config.team_ids)

    print(f"[API]  Fetching data from API-Football (league {config.league_id}, season {config.season})")

    df_finished = collector.get_fixtures_by_league(
        config.league_id, config.season, status="FT"
    )

    today = datetime.now().strftime("%Y-%m-%d")
    future = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
    df_upcoming = collector.get_fixtures_between(
        today, future, config.league_id, config.season
    )

    if df_finished.empty:
        print("  [WARN] No data from API, falling back to mock data.")
        return load_matches(config, use_mock=True)

    for col in ["home_possession", "away_possession", "home_shots", "away_shots"]:
        df_finished[col] = None

    return df_finished, df_upcoming
