import re
import csv
from datetime import datetime

TEAM_MAP = {
    "CF América": "Club America",
    "Gallos Blancos": "Club Queretaro",
    "Deportivo Guadalajara": "Guadalajara Chivas",
    "UANL Tigres": "Tigres UANL",
    "Deportivo Toluca": "Toluca",
    "Pumas UNAM": "U.N.A.M. - Pumas",
    "Mazatlán FC": "Mazatlán",
    "Atlético San Luis": "Atletico San Luis",
    "Club León": "Leon",
    "CF Pachuca": "Pachuca",
    "CF Monterrey": "Monterrey",
    "Cruz Azul": "Cruz Azul",
    "Club Tijuana": "Club Tijuana",
    "Santos Laguna": "Santos Laguna",
    "Club Necaxa": "Necaxa",
    "Puebla FC": "Puebla",
    "FC Juárez": "FC Juarez",
    "Atlas Guadalajara": "Atlas",
}

with open(r"C:\Users\Ricardo Medina\Downloads\2024-25_mx1.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

rows = []
current_tournament = None
current_round = None
current_date = None

for line in lines:
    line = line.rstrip()

    stage_match = re.match(r"^\s*▪\s*(.+)$", line)
    if stage_match:
        raw = stage_match.group(1).strip()
        if raw.startswith("Apertura Playoffs"):
            current_tournament = "Apertura Playoffs"
        elif raw.startswith("Apertura"):
            current_tournament = "Apertura"
        elif raw.startswith("Clausura Playoffs"):
            current_tournament = "Clausura Playoffs"
        elif raw.startswith("Clausura"):
            current_tournament = "Clausura"
        else:
            current_tournament = raw

        md_match = re.search(r"Matchday\s+(\d+)", raw)
        pi_match = re.search(r"Play-in round\s+(\d+)", raw)
        qf_match = re.search(r"Quarterfinals", raw)
        sf_match = re.search(r"Semifinals", raw)
        fin_match = re.search(r"Final", raw)
        if md_match:
            current_round = f"{current_tournament} - {md_match.group(1)}"
        elif pi_match:
            current_round = f"{current_tournament} - Play-in {pi_match.group(1)}"
        elif qf_match:
            current_round = f"{current_tournament} - QF"
        elif sf_match:
            current_round = f"{current_tournament} - SF"
        elif fin_match:
            current_round = f"{current_tournament} - Final"
        else:
            current_round = current_tournament
        continue

    date_match = re.match(
        r"^\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})(?:\s+(\d{4}))?",
        line
    )
    if date_match:
        month_str = date_match.group(1)
        day_str = date_match.group(2)
        year_str = date_match.group(3)
        if year_str is None:
            if current_tournament and "Clausura" in current_tournament:
                year_str = "2025"
            else:
                year_str = "2024"
        current_date = datetime.strptime(f"{month_str} {day_str} {year_str}", "%b %d %Y").strftime("%Y-%m-%d")
        continue

    match_match = re.match(
        r"^\s+(?:\d{1,2}:\d{2}\s+)?(.+?)\s+v\s+(.+?)\s+(\d+)-(\d+)",
        line
    )

    if match_match and current_date:
        home_raw = match_match.group(1).strip()
        away_raw = match_match.group(2).strip()
        home_goals = int(match_match.group(3))
        away_goals = int(match_match.group(4))

        home_team = TEAM_MAP.get(home_raw, home_raw)
        away_team = TEAM_MAP.get(away_raw, away_raw)

        home_ht = away_ht = ""
        ht_match = re.search(r"\((\d+)-(\d+)\)", line)
        if ht_match:
            home_ht = int(ht_match.group(1))
            away_ht = int(ht_match.group(2))

        is_playoff = "Playoffs" in (current_tournament or "")
        season = "2025" if current_tournament and "Clausura" in current_tournament else "2024"

        rows.append({
            "id": "",
            "referee": "",
            "timezone": "UTC",
            "date": current_date + "T00:00:00+00:00",
            "venue_id": "",
            "venue_name": "",
            "venue_city": "",
            "season": season,
            "round": current_round,
            "home_team": home_team,
            "away_team": away_team,
            "home_win": str(home_goals > away_goals),
            "away_win": str(away_goals > home_goals),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_goals_half_time": home_ht,
            "away_goals_half_time": away_ht,
            "home_goals_fulltime": home_goals,
            "away_goals_fulltime": away_goals,
            "home_goals_extra_time": "",
            "away_goals_extratime": "",
            "home_goals_penalty": "",
            "away_goals_penalty": "",
        })

print(f"Parsed {len(rows)} matches")

rounds = {}
for r in rows:
    t = r['round'].split(' - ')[0]
    rounds[t] = rounds.get(t, 0) + 1
for t, c in sorted(rounds.items()):
    print(f"  {t}: {c}")

print("\nSample rows:")
for r in rows[:3]:
    print(f"  {r['date'][:10]} | {r['home_team']} vs {r['away_team']} | {r['home_goals']}-{r['away_goals']} (HT: {r['home_goals_half_time']}-{r['away_goals_half_time']}) | {r['round']}")
for r in [x for x in rows if "Playoffs" in x['round']][:5]:
    print(f"  {r['date'][:10]} | {r['home_team']} vs {r['away_team']} | {r['home_goals']}-{r['away_goals']} (HT: {r['home_goals_half_time']}-{r['away_goals_half_time']}) | {r['round']}")

with open(r"C:\Users\Ricardo Medina\Desktop\Apuestas\parsed_2024_25.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "id", "referee", "timezone", "date", "venue_id", "venue_name", "venue_city",
        "season", "round", "home_team", "away_team", "home_win", "away_win",
        "home_goals", "away_goals", "home_goals_half_time", "away_goals_half_time",
        "home_goals_fulltime", "away_goals_fulltime", "home_goals_extra_time",
        "away_goals_extratime", "home_goals_penalty", "away_goals_penalty"
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved to parsed_2024_25.csv")
