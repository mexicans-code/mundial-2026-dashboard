import pandas as pd, numpy as np
from sports_ml.features.builder import FeatureBuilder
from sports_ml.utils.config import Config
from sports_ml.models.trainer import ModelTrainer

cfg = Config()
cfg.home_team_name = 'Portugal'
cfg.away_team_name = 'Croatia'

df = pd.read_csv('national_teams_data.csv')
df['date'] = pd.to_datetime(df['date'])
df_finished = df[df['home_goals'].notna()].copy()
df_finished['home_team'] = df_finished['home_team'].astype(str)
df_finished['away_team'] = df_finished['away_team'].astype(str)

fb = FeatureBuilder(cfg)
df_feat = fb.build(df_finished)
for col in ["home_win","draw","away_win","total_goals","over_2_5","btts","result"]:
    if col not in df_feat.columns:
        if col == "home_win":
            df_feat[col] = (df_feat["home_goals"] > df_feat["away_goals"]).astype(int)
        elif col == "draw":
            df_feat[col] = (df_feat["home_goals"] == df_feat["away_goals"]).astype(int)
        elif col == "away_win":
            df_feat[col] = (df_feat["home_goals"] < df_feat["away_goals"]).astype(int)
        elif col == "total_goals":
            df_feat[col] = df_feat["home_goals"] + df_feat["away_goals"]
        elif col == "over_2_5":
            df_feat[col] = (df_feat["total_goals"] > 2.5).astype(int)
        elif col == "btts":
            df_feat[col] = ((df_feat["home_goals"] > 0) & (df_feat["away_goals"] > 0)).astype(int)
        elif col == "result":
            df_feat[col] = df_feat["home_win"] * 0 + df_feat["draw"] * 1 + df_feat["away_win"] * 2

trainer = ModelTrainer(cfg)
trainer.train(df_feat, target_key='1X2')

synthetic = pd.DataFrame([{
    'fixture_id': 0, 'date': pd.Timestamp.now(),
    'home_team': 'Portugal', 'away_team': 'Croatia',
    'home_goals': None, 'away_goals': None,
    'home_team_id': 1241, 'away_team_id': 799, 'status_short': 'NS',
}])

# Build features without passing to historical (to avoid H2H duplication bug)
features = fb.build_prediction_features(
    synthetic.iloc[0], df_feat, feature_cols=trainer.feature_cols
)
# Fix: use only first occurrence of each feature (H2H columns get duplicated)
uniq = features[~features.index.duplicated(keep='first')]
X = uniq.values.reshape(1, -1)

print('=== FEATURES ===')
for c in uniq.index:
    print(f'  {c}: {uniq[c]}')

print('\n=== INDIVIDUAL MODEL PREDICTIONS ===')
for target, models_list in trainer.models.items():
    for i, tup in enumerate(models_list):
        name, m = tup
        p = m.predict_proba(X)[0]
        print(f'{name}: H={p[0]:.3f} D={p[1]:.3f} A={p[2]:.3f}')
        if hasattr(m, 'feature_importances_'):
            imp = m.feature_importances_
            cols = list(uniq.index)
            top = sorted(zip(cols, imp), key=lambda x: -x[1])[:5]
            print(f'  Top 5 features:')
            for c, iv in top:
                print(f'    {c}: imp={iv:.4f}  val={uniq[c]}')
