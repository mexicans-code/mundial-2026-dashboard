"""Fetch national team data from key competitions with minimal API calls, save to CSV."""
import os, sys, warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sports_ml.data.collector import FootballDataCollector, FREE_TIER_SEASONS
from sports_ml.utils.config import Config

warnings.filterwarnings("ignore")

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "national_teams_data.csv")

# Only free-tier-compatible seasons
LEAGUES = {
    1:   {"seasons": [2022],               "name": "World Cup"},
    9:   {"seasons": [2024],               "name": "Copa America"},
    4:   {"seasons": [2024],               "name": "Euro Championship"},
    10:  {"seasons": [2022, 2023, 2024],   "name": "Friendlies"},
    133: {"seasons": [2023],               "name": "AFC Asian Cup"},
    166: {"seasons": [2023],               "name": "Africa Cup of Nations"},
}

def load_cfg():
    import json
    cfg = Config()
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            overrides = json.load(f)
        for section, values in overrides.items():
            if hasattr(cfg, section):
                for k, v in values.items():
                    if hasattr(getattr(cfg, section), k):
                        setattr(getattr(cfg, section), k, v)
                    elif hasattr(cfg, k):
                        setattr(cfg, k, v)
    if not cfg.api.football_api_key:
        cfg.api.football_api_key = os.environ.get("FOOTBALL_API_KEY", "")
    return cfg

def fetch_all():
    collector = FootballDataCollector(load_cfg())

    all_matches = []
    total_calls = 0

    for league_id, info in LEAGUES.items():
        for season in info["seasons"]:
            print(f"  Fetching {info['name']} ({season})... ", end="", flush=True)
            df = collector.get_fixtures_by_league(league_id, season, status="FT")
            total_calls += 1

            if not df.empty:
                df["source_league"] = f"{info['name']} {season}"
                all_matches.append(df)
                print(f"{len(df)} matches")
            else:
                print("0 matches (empty or rate-limited)")

    if not all_matches:
        print("No data fetched. Check API key and plan limits.")
        return False

    combined = pd.concat(all_matches, ignore_index=True).drop_duplicates(subset=["fixture_id"])
    combined.to_csv(CSV_PATH, index=False)
    print(f"\nSaved {len(combined)} total matches to national_teams_data.csv")
    print(f"API calls used: {total_calls}")
    return True

if __name__ == "__main__":
    fetch_all()
