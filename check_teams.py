import pandas as pd
df = pd.read_csv("national_teams_data.csv")
teams = [
    "Brazil", "Japan", "Argentina", "Germany", "Paraguay", "France",
    "England", "Spain", "Netherlands", "Portugal", "Morocco", "Uruguay",
    "Ivory Coast", "Norway", "Sweden", "Mexico", "Ecuador", "Cape Verde",
    "Belgium", "Senegal", "USA", "Croatia", "Switzerland", "Algeria",
    "Australia", "Egypt", "Colombia", "Ghana", "Austria", "DR Congo"
]
for t in teams:
    home = df[df["home_team"].str.contains(t, case=False, na=False)]
    away = df[df["away_team"].str.contains(t, case=False, na=False)]
    if not home.empty:
        print(f"{t}: id={home.iloc[0]['home_team_id']} ({len(home)}H / {len(away)}A matches)")
    elif not away.empty:
        print(f"{t}: id={away.iloc[0]['away_team_id']} ({len(home)}H / {len(away)}A matches)")
    else:
        print(f"{t}: NOT FOUND in dataset")
