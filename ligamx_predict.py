import sys
sys.stdout.reconfigure(encoding='utf-8')  # Fix lambda char

"""Train and predict Liga MX using the existing pipeline."""
import pandas as pd
import numpy as np
from sports_ml.features.builder import FeatureBuilder
from sports_ml.models.trainer import ModelTrainer
from sports_ml.models.predictor import MatchPredictor
from sports_ml.utils.config import Config
import warnings
warnings.filterwarnings('ignore')

# Load Liga MX data
df = pd.read_csv('ligamx_data.csv')
df = df[df['home_goals'].notna()].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.reset_index(drop=True)

# Drop columns that cause data leakage (match IDs, extra stats, venue IDs)
leak_cols = ['id', 'referee', 'venue_id', 'venue_name', 'venue_city',
             'home_goals_half_time', 'away_goals_half_time',
             'home_goals_fulltime', 'away_goals_fulltime',
             'home_goals_extra_time', 'away_goals_extratime',
             'home_goals_penalty', 'away_goals_penalty',
             'timezone', 'season']
df = df.drop(columns=[c for c in leak_cols if c in df.columns])

# Add required columns the pipeline expects
has_goals = df['home_goals'].notna()
df['home_win'] = 0
df['away_win'] = 0
df['draw'] = 0
df.loc[has_goals, 'home_win'] = (df.loc[has_goals, 'home_goals'] > df.loc[has_goals, 'away_goals']).astype(int)
df.loc[has_goals, 'draw'] = (df.loc[has_goals, 'home_goals'] == df.loc[has_goals, 'away_goals']).astype(int)
df.loc[has_goals, 'away_win'] = (df.loc[has_goals, 'away_goals'] > df.loc[has_goals, 'home_goals']).astype(int)
df['result'] = df['home_win'] * 0 + df['draw'] * 1 + df['away_win'] * 2
df['total_goals'] = df['home_goals'] + df['away_goals']
df['over_2_5'] = (df['total_goals'] > 2.5).astype(int)
df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)

# Disable H2H and FIFA rankings for Liga MX
cfg = Config()
cfg.features.include_h2h = False
cfg.features.include_fifa_rankings = False
cfg.features.fifa_rank_default = 50
cfg.features.use_home_advantage = True

# Build features
fb = FeatureBuilder(cfg)
featured = fb.build(df)

# Train models
trainer = ModelTrainer(cfg)

print("=== ENTRENANDO 1X2 ===")
m1 = trainer.train(featured, "1X2")

print("\n=== ENTRENANDO OVER/UNDER ===")
m2 = trainer.train(featured, "over_under")

print("\n=== ENTRENANDO BTTS ===")
m3 = trainer.train(featured, "btts")

# Save model
trainer.save("models/saved/ligamx_ensemble.pkl")
print("\nModelo guardado en models/saved/ligamx_ensemble.pkl")

# --- Predict upcoming matches (last 10 unfinished) ---
predictor = MatchPredictor(trainer, cfg)

# Find the last matches that might be upcoming (recent no-result entries)
unfinished = df[df['home_goals'].isna()].tail(20)
if len(unfinished) == 0:
    # Use the most recent teams to create hypothetical matchups
    last_matches = df.dropna(subset=['home_goals']).tail(10)
    teams = []
    for _, r in last_matches.iterrows():
        teams.append((r['home_team'], r['away_team']))
    # Create unique matchups from recent teams
    recent_teams = list(set([t for pair in teams for t in pair]))
    matchups = []
    for i in range(0, len(recent_teams)-1, 2):
        matchups.append((recent_teams[i], recent_teams[i+1]))
    print(f"\n=== PREDICCIONES (matchups simuladas) ===")
else:
    matchups = [(r['home_team'], r['away_team']) for _, r in unfinished.iterrows()]
    print(f"\n=== PREDICCIONES ({len(unfinished)} partidos pendientes) ===")

for home, away in matchups[:8]:  # Max 8 predictions
    match_series = pd.Series({
        'fixture_id': -1, 'home_team': home, 'away_team': away,
        'date': pd.Timestamp.now(),
        'home_goals': None, 'away_goals': None,
    })
    try:
        pred = predictor.predict_match(match_series, df)
        ph = pred.get('prob_home', 0)
        pd_ = pred.get('prob_draw', 0)
        pa = pred.get('prob_away', 0)
        o25 = pred.get('prob_over_2_5', 0)
        btts = pred.get('prob_btts_yes', 0)
        lam_h = pred.get('lambda_home', 0)
        lam_a = pred.get('lambda_away', 0)
        scores = pred.get('exact_scores', [])
        top_score = scores[0]['score'] if scores and scores[0]['score'] != 'N/A' else '-'
        
        print(f"\n  {home:25s} vs {away:25s}")
        print(f"  1X2: {ph:.0%} / {pd_:.0%} / {pa:.0%}  |  O2.5: {o25:.0%}  |  BTTS: {btts:.0%}")
        print(f"  λ: {lam_h:.2f} - {lam_a:.2f}  |  Score top: {top_score}")
        print(f"  Predicción: {pred.get('predicted', '?')} (conf: {pred.get('confidence', 0):.0%})")
    except Exception as e:
        print(f"\n  {home:25s} vs {away:25s}  -> ERROR: {e}")
