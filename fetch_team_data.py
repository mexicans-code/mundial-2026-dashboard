"""Fetch per-team data for teams with limited matches in the dataset."""
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pandas as pd
from datetime import datetime

API_KEY = "fc578f6e6829a158d545c5308ac23946"
BASE = "https://v3.football.api-sports.io"

TEAMS = {
    "Portugal": 1241,
    "Croatia": 799,
}

SEASONS = [2022, 2023, 2024]

def fetch_team_fixtures(team_id: int, season: int) -> list[dict]:
    url = f"{BASE}/fixtures?team={team_id}&season={season}&status=FT"
    headers = {"x-apisports-key": API_KEY}
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        return data.get("response", [])
    except HTTPError as e:
        print(f"  [Error {e.code}] {e.reason}")
        return []
    except Exception as e:
        print(f"  [Error] {e}")
        return []

def parse_fixtures(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for f in raw:
        fixture = f.get("fixture", {})
        league = f.get("league", {})
        teams = f.get("teams", {})
        goals = f.get("goals", {})
        rows.append({
            "fixture_id": fixture.get("id"),
            "date": fixture.get("date"),
            "timestamp": fixture.get("timestamp"),
            "status_short": fixture.get("status", {}).get("short"),
            "league_id": league.get("id"),
            "league_season": league.get("season"),
            "league_round": league.get("round"),
            "home_team_id": teams.get("home", {}).get("id"),
            "home_team": teams.get("home", {}).get("name"),
            "away_team_id": teams.get("away", {}).get("id"),
            "away_team": teams.get("away", {}).get("name"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "venue_city": fixture.get("venue", {}).get("city"),
            "venue_name": fixture.get("venue", {}).get("name"),
            "source_league": f"{league.get('name', 'Unknown')} {season}",
        })
    return pd.DataFrame(rows)

existing = pd.read_csv("national_teams_data.csv")
existing_ids = set(existing["fixture_id"].dropna().astype(int).unique())
print(f"Existing dataset: {len(existing)} matches, {len(existing_ids)} unique fixture IDs")

all_new = []
api_calls = 0

for team_name, team_id in TEAMS.items():
    for season in SEASONS:
        print(f"Fetching {team_name} ({team_id}) season {season}...")
        raw = fetch_team_fixtures(team_id, season)
        api_calls += 1
        if raw:
            df = parse_fixtures(raw)
            before = len(df)
            df = df[~df["fixture_id"].isin(existing_ids)]
            new_count = len(df)
            total_finished = len(df[df["home_goals"].notna()])
            print(f"  -> {len(raw)} fixtures, {new_count} new, {total_finished} finished")
            if not df.empty:
                all_new.append(df)
        else:
            print(f"  -> No data")

if all_new:
    combined = pd.concat(all_new, ignore_index=True).drop_duplicates(subset=["fixture_id"])
    combined["date"] = pd.to_datetime(combined["date"], utc=True, errors="coerce")
    
    # Remove duplicates with existing
    combined = combined[~combined["fixture_id"].isin(existing_ids)]
    
    print(f"\nNew matches to add: {len(combined)}")
    finished_new = combined[combined["home_goals"].notna()]
    print(f"  Finished: {len(finished_new)}")
    print(f"  API calls used: {api_calls}")
    print(f"\nBy team:")
    for t in ["Bosnia & Herzegovina", "Congo DR", "England", "Belgium", "Senegal", "USA", "Ivory Coast", "France", "Sweden", "Mexico", "Ecuador"]:
        sub = combined[(combined.home_team==t)|(combined.away_team==t)]
        if not sub.empty:
            print(f"  {t}: {len(sub)} new matches")
    
    existing["date"] = pd.to_datetime(existing["date"], utc=True, errors="coerce")
    updated = pd.concat([existing, combined], ignore_index=True)
    updated = updated.drop_duplicates(subset=["fixture_id"]).sort_values("date")
    updated.to_csv("national_teams_data.csv", index=False)
    print(f"\nDataset now: {len(updated)} matches (was {len(existing)}, +{len(updated)-len(existing)})")
else:
    print("\nNo new data to add.")
    print(f"API calls used: {api_calls}")
