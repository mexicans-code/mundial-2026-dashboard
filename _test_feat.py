import pandas as pd, time, os, sys
sys.path.insert(0, os.path.dirname("."))
from sports_ml.features.builder import FeatureBuilder
from sports_ml.utils.config import Config

BASE = os.path.dirname(".")
df = pd.read_csv(os.path.join(BASE, "national_teams_data.csv"))
print(f"Full CSV: {len(df)} rows")

df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
df_finished = df[df["home_goals"].notna()].copy()
df_finished = df_finished[df_finished["date"] >= "2000-01-01"]
print(f"Finished >= 2000: {len(df_finished)}")

t0 = time.time()
fb = FeatureBuilder(Config())
df_feat = fb.build(df_finished)
t1 = time.time()
print(f"Feature build: {t1-t0:.1f}s")
print(f"Features: {df_feat.shape[1]} cols, {len(df_feat)} rows")

cols = [c for c in df_feat.columns if c not in (
    "fixture_id","date","home_team","away_team",
    "home_goals","away_goals"
)]
print(f"Feature cols ({len(cols)}): {cols}")
