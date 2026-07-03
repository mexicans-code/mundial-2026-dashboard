"""Generate dashboard/index.html from the trained model and CSV data."""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sports_ml.features.builder import FeatureBuilder
from sports_ml.models.predictor import MatchPredictor
from sports_ml.models.stats_predictor import MatchStatsPredictor, STATS_LABELS
from sports_ml.models.trainer import ModelTrainer
from sports_ml.utils.config import Config

BASE = os.path.dirname(os.path.dirname(__file__))
CSV = os.path.join(BASE, "national_teams_data.csv")
MODEL = os.path.join(BASE, "models/saved/ensemble.pkl")
CONFIG = os.path.join(BASE, "config.json")
OUT = os.path.join(os.path.dirname(__file__), "index.html")

MATCHES = [
    ("Portugal", "Croatia", "Toronto", "R32", "Jue 3pm", "jul2"),
    ("Spain", "Austria", "Los Angeles", "R32", "Jue 3pm", "jul2"),
    ("Switzerland", "Algeria", "Vancouver", "R32", "Jue 11pm", "jul2"),
    ("Australia", "Egypt", "Dallas", "R32", "Sab 12pm", "jul3"),
    ("Argentina", "Cape Verde", "Miami", "R32", "Sab 4pm", "jul3"),
    ("Colombia", "Ghana", "Kansas City", "R32", "Sab 7:30pm", "jul3"),
]

NAME_MAP_ES = {
    "Portugal": "Portugal",
    "Croatia": "Croacia",
    "Spain": "España",
    "Austria": "Austria",
    "Switzerland": "Suiza",
    "Algeria": "Argelia",
    "Australia": "Australia",
    "Egypt": "Egipto",
    "Argentina": "Argentina",
    "Cape Verde": "Cabo Verde",
    "Colombia": "Colombia",
    "Ghana": "Ghana",
}

TEAM_KEY_MAP = {
    "Portugal": "Portugal",
    "Croatia": "Croatia",
    "Spain": "Spain",
    "Austria": "Austria",
    "Switzerland": "Switzerland",
    "Algeria": "Algeria",
    "Australia": "Australia",
    "Egypt": "Egypt",
    "Argentina": "Argentina",
    "Cape Verde": "Cape Verde",
    "Colombia": "Colombia",
    "Ghana": "Ghana",
}


def load_config() -> Config:
    cfg = Config()
    if os.path.exists(CONFIG):
        with open(CONFIG) as f:
            overrides = json.load(f)
        for section, values in overrides.items():
            if hasattr(cfg, section):
                for k, v in values.items():
                    if hasattr(getattr(cfg, section), k):
                        setattr(getattr(cfg, section), k, v)
    return cfg


def get_team_form(team: str, df: pd.DataFrame, n: int = 8) -> dict:
    alt_names = {
        "Cape Verde": ["Cape Verde", "Cape Verde Islands", "Cabo Verde", "Cabo Verde Islands"],
    }
    names = alt_names.get(team, [team])
    team_matches = df[
        ((df["home_team"].isin(names)) | (df["away_team"].isin(names)))
        & (df["home_goals"].notna())
    ].sort_values("date", ascending=False).head(n)

    form_str = ""
    o_count = 0
    b_count = 0
    matches = []

    for _, row in team_matches.iterrows():
        is_home = row["home_team"] == team
        home_goals = int(row["home_goals"])
        away_goals = int(row["away_goals"])

        if is_home:
            result = "W" if home_goals > away_goals else ("L" if home_goals < away_goals else "D")
            score = f"{home_goals}-{away_goals}"
            opponent = row["away_team"]
            venue = "V"
        else:
            result = "W" if away_goals > home_goals else ("L" if away_goals < home_goals else "D")
            score = f"{away_goals}-{home_goals}"
            opponent = row["home_team"]
            venue = "L"

        total_goals = home_goals + away_goals
        o_val = "S" if total_goals > 2.5 else "N"
        b_val = "S" if (home_goals > 0 and away_goals > 0) else "N"

        form_str += result
        if o_val == "S":
            o_count += 1
        if b_val == "S":
            b_count += 1

        matches.append({
            "v": venue,
            "vs": opponent,
            "s": score,
            "r": result,
            "o": o_val,
            "b": b_val,
        })

    while len(matches) < n:
        matches.append({"v": "V", "vs": "N/A", "s": "0-0", "r": "D", "o": "N", "b": "N"})

    return {
        "f": form_str,
        "o": f"{o_count}/{n}",
        "b": f"{b_count}/{n}",
        "n": len([m for m in matches if m["vs"] != "N/A"]),
        "m": matches,
    }


def team_form_to_json(teams: list[str], df: pd.DataFrame) -> str:
    data = {}
    for team in teams:
        key = TEAM_KEY_MAP.get(team, team)
        data[key] = get_team_form(team, df)
    return json.dumps(data, ensure_ascii=False)


def format_prob(val: float) -> str:
    return f"{val * 100:.0f}%"


def format_prob_num(val: float) -> str:
    return f"{val * 100:.0f}"


def prob_class(val: float) -> str:
    if val >= 0.55:
        return "green"
    elif val >= 0.50:
        return "amber"
    else:
        return "red"


def pick_1x2(home: float, draw: float, away: float) -> tuple[str, str, str]:
    if home >= draw and home >= away:
        return "Local", "green", f"{format_prob(home)}"
    elif away >= home and away >= draw:
        return "Visitante", "red", f"{format_prob(away)}"
    else:
        return "Empate", "amber", f"{format_prob(draw)}"


def pick_name_1x2(home: float, draw: float, away: float, home_name: str, away_name: str) -> str:
    if home >= draw and home >= away:
        return home_name
    elif away >= home and away >= draw:
        return away_name
    else:
        return "Empate"


def value_bet_text(home: float, draw: float, away: float, home_name: str, away_name: str, pred: dict) -> str:
    probs = [(home, "home", home_name), (draw, "draw", "Empate"), (away, "away", away_name)]
    probs.sort(key=lambda x: x[0], reverse=True)
    best = probs[0]
    h_display = home_name if len(home_name) <= 12 else home_name
    a_display = away_name if len(away_name) <= 12 else away_name
    
    fair_odds = round(1.0 / max(best[0], 0.01), 2)
    market_odds = round(fair_odds * 0.92, 2)
    edge = round((best[0] * market_odds - 1) * 100)
    
    if edge > 5:
        text = f"{h_display if best[1] == 'home' else (a_display if best[1] == 'away' else 'Empate')} @ {market_odds} — Modelo {format_prob(best[0])} vs Mercado ~{format_prob(1.0/market_odds)} (+{edge}% edge)"
        color_class = "val"
    elif best[0] >= 0.60:
        text = f"{h_display if best[1] == 'home' else (a_display if best[1] == 'away' else 'Empate')} gana — Modelo {format_prob(best[0])}, favoritismo {'absoluto' if best[0] >= 0.70 else 'solido'}"
        color_class = "val"
    else:
        text = f"{h_display if best[1] == 'home' else (a_display if best[1] == 'away' else 'Empate')} — Modelo {format_prob(best[0])}"
        if edge > 0:
            text += f", EV+{edge}%"
        color_class = "val amber"
    return text, color_class


def generate_html(predictions: dict, team_data: dict, stats: dict) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Mundial 2026 — Dashboard Predictivo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #0A0E17;
  --surface: #131920;
  --border: #1E293B;
  --title: #F8FAFC;
  --body: #94A3B8;
  --green: #22C55E;
  --red: #EF4444;
  --amber: #F59E0B;
  --font: 'Inter', system-ui, -apple-system, sans-serif;
  --radius: 12px;
  --gap: 16px;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  background: var(--bg);
  color: var(--body);
  font-family: var(--font);
  font-weight: 400;
  font-size: 14px;
  line-height: 1.5;
  padding: 24px;
  -webkit-font-smoothing: antialiased;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}
.header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 20px;
}}
.header-left h1 {{
  font-size: 24px;
  font-weight: 600;
  color: var(--title);
  margin-bottom: 2px;
}}
.header-left .sub {{ font-size: 13px; color: var(--body); }}
.header-right {{
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 13px;
}}
.record-badge {{
  background: var(--green);
  color: #000;
  font-weight: 700;
  font-size: 13px;
  padding: 4px 14px;
  border-radius: 20px;
}}
.record-badge.low {{ background: var(--amber); }}
.record-badge.bad {{ background: var(--red); }}
.disclaimer {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 16px;
  font-size: 12px;
  color: var(--body);
  margin-bottom: 20px;
}}
.nav {{
  display: flex;
  gap: 4px;
  margin-bottom: 20px;
  flex-wrap: wrap;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0;
}}
.nav button {{
  background: none;
  border: none;
  color: var(--body);
  padding: 10px 18px;
  font-size: 13px;
  font-weight: 500;
  font-family: var(--font);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  transition: color 0.15s, border-color 0.15s;
}}
.nav button:hover {{ color: var(--title); }}
.nav button.active {{ color: var(--title); border-bottom-color: var(--title); }}
.section {{ display: none; }}
.section.active {{ display: block; }}
.card-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: var(--gap);
  margin-bottom: 24px;
}}
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
}}
.card.full {{ grid-column: 1 / -1; }}
.card h2 {{
  font-size: 14px;
  font-weight: 600;
  color: var(--body);
  text-transform: uppercase;
  letter-spacing: 0.4px;
  margin-bottom: 14px;
}}
.card h2 span {{ color: var(--green); }}
.match-name {{ font-size: 17px; font-weight: 600; color: var(--title); margin-bottom: 4px; }}
.match-name .vs {{ color: var(--body); font-weight: 400; font-size: 14px; margin: 0 6px; }}
.match-meta {{ font-size: 12px; color: var(--body); margin-bottom: 14px; }}
.match-meta span {{ margin-right: 14px; }}
.match-meta .badge {{
  display: inline-block;
  background: var(--green);
  color: #000;
  font-size: 10px;
  font-weight: 700;
  padding: 2px 10px;
  border-radius: 10px;
}}
.match-meta .badge.amber {{ background: var(--amber); }}
.prob-bar {{
  display: flex;
  height: 28px;
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 12px;
  font-size: 11px;
  font-weight: 600;
}}
.prob-bar div {{
  display: flex;
  align-items: center;
  justify-content: center;
  transition: width 0.3s;
}}
.prob-bar .home {{ background: var(--green); color: #000; }}
.prob-bar .draw {{ background: var(--amber); color: #000; }}
.prob-bar .away {{ background: var(--red); color: #fff; }}
.markets {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-bottom: 12px;
}}
.market-item {{
  background: rgba(255,255,255,0.04);
  border-radius: 8px;
  padding: 10px;
  text-align: center;
}}
.market-item .lbl {{ font-size: 10px; color: var(--body); text-transform: uppercase; letter-spacing: 0.3px; }}
.market-item .val {{ font-size: 18px; font-weight: 700; color: var(--title); }}
.market-item .val.green {{ color: var(--green); }}
.market-item .val.red {{ color: var(--red); }}
.market-item .val.amber {{ color: var(--amber); }}
.details-toggle {{ font-size: 12px; color: var(--body); cursor: pointer; user-select: none; padding: 4px 0; display: inline-block; }}
.details-toggle:hover {{ color: var(--title); }}
.details-content {{ display: none; margin-top: 12px; }}
.details-content.open {{ display: block; }}
.stats-grid {{
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 4px 12px;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}}
.stats-grid .lbl {{ color: var(--body); grid-column: 2; text-align: center; font-size: 11px; }}
.stats-grid .h {{ text-align: right; color: var(--title); }}
.stats-grid .a {{ text-align: left; color: var(--title); }}
.scores-list {{ margin-top: 4px; }}
.score-row {{
  display: flex;
  justify-content: space-between;
  padding: 3px 0;
  font-size: 13px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}}
.score-row:last-child {{ border-bottom: none; }}
.score-row .prob {{ color: var(--green); font-weight: 500; }}
.form-badge {{ font-size: 10px; color: var(--body); margin-top: 4px; }}
.form-badge strong {{ font-weight: 600; }}
.btn-form {{ font-size: 11px; color: #3B82F6; background: none; border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; cursor: pointer; font-family: var(--font); margin-top: 8px; transition: border-color 0.15s; }}
.btn-form:hover {{ border-color: #3B82F6; }}
.modal-overlay {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 1000; align-items: center; justify-content: center; padding: 20px; }}
.modal-overlay.open {{ display: flex; }}
.modal {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); max-width: 700px; width: 100%; max-height: 90vh; overflow-y: auto; padding: 24px; position: relative; }}
.modal-close {{ position: absolute; top: 12px; right: 16px; background: none; border: none; color: var(--body); font-size: 22px; cursor: pointer; font-family: var(--font); line-height: 1; }}
.modal-close:hover {{ color: var(--title); }}
.modal h3 {{ font-size: 16px; font-weight: 600; color: var(--title); margin-bottom: 16px; }}
.modal .team-section {{ margin-bottom: 20px; }}
.modal .team-section:last-child {{ margin-bottom: 0; }}
.modal .team-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
.modal .team-name {{ font-size: 14px; font-weight: 600; color: var(--title); }}
.modal .form-str {{ font-size: 13px; font-weight: 700; letter-spacing: 2px; }}
.modal .form-str .w {{ color: var(--green); }}
.modal .form-str .l {{ color: var(--red); }}
.modal .form-str .d {{ color: var(--amber); }}
.modal .form-stats {{ display: flex; gap: 16px; font-size: 12px; color: var(--body); margin-bottom: 10px; }}
.modal .form-stats strong {{ color: var(--title); }}
.modal table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.modal th {{ text-align: left; padding: 6px 8px; color: var(--body); font-weight: 500; border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; }}
.modal td {{ padding: 6px 8px; border-bottom: 1px solid rgba(255,255,255,0.04); color: var(--body); }}
.modal td .fw {{ font-weight: 600; }}
:root {{ --blue: #3B82F6; }}
.value-bet {{
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
  margin-top: 12px;
}}
.value-bet .lbl {{ font-size: 10px; color: var(--body); text-transform: uppercase; letter-spacing: 0.3px; }}
.value-bet .val {{ font-size: 13px; font-weight: 600; color: var(--green); }}
.value-bet .val.amber {{ color: var(--amber); }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{
  text-align: left;
  padding: 8px 10px;
  color: var(--body);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  border-bottom: 1px solid var(--border);
}}
td {{
  padding: 8px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  color: var(--body);
}}
td strong {{ color: var(--title); }}
td.green {{ color: var(--green); }}
td.red {{ color: var(--red); }}
td.amber {{ color: var(--amber); }}
td .badge {{
  display: inline-block;
  font-size: 10px;
  font-weight: 700;
  padding: 2px 10px;
  border-radius: 10px;
  color: #000;
}}
td .badge.green {{ background: var(--green); }}
td .badge.amber {{ background: var(--amber); }}
td .badge.blue {{ background: #3B82F6; color: #fff; }}
.parlay-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: var(--gap);
}}
.parlay-card .title {{ font-size: 15px; font-weight: 600; color: var(--title); margin-bottom: 2px; }}
.parlay-card .sub {{ font-size: 12px; color: var(--body); margin-bottom: 12px; }}
.pick-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}}
.pick-row:last-child {{ border-bottom: none; }}
.pick-row .left .lbl {{ font-size: 11px; color: var(--body); }}
.pick-row .left .val {{ font-size: 14px; color: var(--title); font-weight: 500; }}
.pick-row .right {{ text-align: right; }}
.pick-row .right .odds {{ font-size: 15px; font-weight: 700; color: var(--green); }}
.pick-row .right .pick {{ font-size: 11px; color: var(--body); }}
.parlay-leg {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}}
.parlay-leg:last-child {{ border-bottom: none; }}
.parlay-leg .num {{
  background: var(--green);
  color: #000;
  width: 24px; height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  flex-shrink: 0;
}}
.parlay-leg .detail {{ flex: 1; }}
.parlay-leg .detail .match {{ font-size: 13px; color: var(--title); font-weight: 500; }}
.parlay-leg .detail .pick {{ font-size: 12px; color: var(--body); }}
.parlay-leg .odds {{ font-size: 15px; font-weight: 700; color: var(--green); text-align: right; }}
.parlay-total {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0 0;
  margin-top: 12px;
  border-top: 1px solid var(--border);
}}
.parlay-total .lbl {{ font-size: 12px; color: var(--body); }}
.parlay-total .odds {{ font-size: 22px; font-weight: 700; color: var(--green); }}
.parlay-total .prob {{ font-size: 13px; color: var(--amber); }}
.record-line {{ font-size: 12px; color: var(--body); margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); }}
.record-line strong {{ color: var(--title); }}
.footer {{ text-align: center; color: rgba(148,163,184,0.5); font-size: 11px; margin-top: 32px; padding: 16px 0; border-top: 1px solid var(--border); }}
.footer a {{ color: var(--body); text-decoration: none; }}
.footer a:hover {{ color: var(--title); }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <div class="header-left">
    <h1>Mundial 2026</h1>
    <div class="sub">Dashboard predictivo &middot; Machine Learning &middot; Solo educativo</div>
  </div>
  <div class="header-right">
    <span class="record-badge" id="recordBadge">--</span>
    <span id="updateDate"></span>
  </div>
</div>

<div class="disclaimer">
  Estas predicciones son generadas por un modelo de Machine Learning entrenado con datos historicos.
  No constituyen asesoramiento financiero. Juega con responsabilidad.
</div>

<div class="nav">
  <button class="active" data-section="jul2">2 Jul</button>
  <button data-section="jul3">3 Jul</button>
  <button data-section="apuestas">Apuestas</button>
  <button data-section="modelo">Modelo</button>
</div>

<!-- ===== 2 JUL ===== -->
<div id="s-jul2" class="section active">
  <div class="card-grid">
{generate_match_cards(predictions, stats, "jul2")}
  </div>
</div>

<!-- ===== 3 JUL ===== -->
<div id="s-jul3" class="section">
  <div class="card-grid">
{generate_match_cards(predictions, stats, "jul3")}
  </div>
</div>

<!-- ===== APUESTAS ===== -->
<div id="s-apuestas" class="section">
{generate_bets_section(predictions)}
</div>

<!-- ===== MODELO ===== -->
<div id="s-modelo" class="section">
{generate_model_section(predictions)}
</div>

<div class="footer">
  <p>Sports ML Pipeline v1.0 &middot; Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
  <p>Solo fines educativos. Juega con responsabilidad.</p>
</div>

</div>

<!-- Modal -->
<div class="modal-overlay" id="formModal" onclick="if(event.target===this)closeForm()">
  <div class="modal">
    <button class="modal-close" onclick="closeForm()">&times;</button>
    <h3>Forma — Ultimos 8 Partidos</h3>
    <div id="modalContent"></div>
  </div>
</div>

<script>
const nameMap = {{
  'Portugal':'Portugal','Croacia':'Croatia','Espana':'Spain','Austria':'Austria',
  'Suiza':'Switzerland','Argelia':'Algeria','Australia':'Australia','Egipto':'Egypt',
  'Argentina':'Argentina','Cabo Verde':'Cape Verde','Colombia':'Colombia','Ghana':'Ghana'
}};

const teamData = {json.dumps(team_data, ensure_ascii=False)};

function openForm(team1Display, team2Display) {{
  const t1 = nameMap[team1Display];
  const t2 = nameMap[team2Display];
  const d1 = teamData[t1], d2 = teamData[t2];
  if (!d1 || !d2) return;

  const formStr = (d) => d.f.split('').map(c => `<span class="${{c === 'W' ? 'w' : c === 'L' ? 'l' : 'd'}}">${{c === 'W' ? 'G' : c === 'L' ? 'P' : 'E'}}</span>`).join(' ');

  const table = (d) => {{
    if (!d.n) return '<div style="color:var(--body);font-size:13px">Sin datos disponibles</div>';
    let rows = d.m.map(m => {{
      const cls = m.r === 'W' ? 'w' : m.r === 'L' ? 'l' : 'd';
      const label = m.r === 'W' ? 'G' : m.r === 'L' ? 'P' : 'E';
      return `<tr><td class="fw ${{cls}}">${{label}}</td><td>${{m.v === 'L' ? 'vs' : '@'}}</td><td>${{m.vs}}</td><td class="fw">${{m.s}}</td><td>${{m.o === 'S' ? '\u2713' : '\u2717'}}</td><td>${{m.b === 'S' ? '\u2713' : '\u2717'}}</td></tr>`;
    }}).join('');
    return `<table><thead><tr><th>Res</th><th></th><th>Rival</th><th>Marcador</th><th>O2.5</th><th>BTTS</th></tr></thead><tbody>${{rows}}</tbody></table>`;
  }};

  document.getElementById('modalContent').innerHTML =
    '<div class="team-section"><div class="team-header"><span class="team-name">' + team1Display + '</span><span class="form-str">' + formStr(d1) + '</span></div>' +
    '<div class="form-stats"><span>Ultimos: <strong>' + d1.n + '</strong></span><span>O2.5: <strong>' + d1.o + '</strong></span><span>BTTS: <strong>' + d1.b + '</strong></span></div>' +
    table(d1) + '</div>' +
    '<div class="team-section" style="margin-top:16px;padding-top:16px;border-top:1px solid var(--border)"><div class="team-header"><span class="team-name">' + team2Display + '</span><span class="form-str">' + formStr(d2) + '</span></div>' +
    '<div class="form-stats"><span>Ultimos: <strong>' + d2.n + '</strong></span><span>O2.5: <strong>' + d2.o + '</strong></span><span>BTTS: <strong>' + d2.b + '</strong></span></div>' +
    table(d2) + '</div>';

  document.getElementById('formModal').classList.add('open');
}}

function closeForm() {{
  document.getElementById('formModal').classList.remove('open');
}}

document.querySelectorAll('.nav button').forEach(btn => {{
  btn.addEventListener('click', function() {{
    document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
    this.classList.add('active');
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById('s-' + this.dataset.section).classList.add('active');
  }});
}});
document.getElementById('updateDate').textContent = new Date().toLocaleDateString('es-MX', {{
  year:'numeric', month:'short', day:'numeric',
  hour:'2-digit', minute:'2-digit'
}});
</script>
</body>
</html>"""


def generate_match_cards(predictions: dict, stats: dict, section: str) -> str:
    cards = []
    for home, away, venue, stage, time, sec in MATCHES:
        if sec != section:
            continue
        key = f"{home} vs {away}"
        p = predictions.get(key, {})
        s = stats.get(key, {})

        h_prob = p.get("prob_home", 0.33)
        d_prob = p.get("prob_draw", 0.33)
        a_prob = p.get("prob_away", 0.34)
        o25 = p.get("prob_over_2_5", 0.5)
        btts = p.get("prob_btts_yes", 0.5)

        pick = pick_name_1x2(h_prob, d_prob, a_prob, NAME_MAP_ES[home], NAME_MAP_ES[away])
        h_display = NAME_MAP_ES[home]
        a_display = NAME_MAP_ES[away]

        home_st = s.get("home", {})
        away_st = s.get("away", {})

        pick_color = "green" if pick == h_display else ("red" if pick == a_display else "amber")

        vb_text, vb_class = value_bet_text(h_prob, d_prob, a_prob, h_display, a_display, p)

        exact = p.get("exact_scores", [])
        scores_html = ""
        if exact and exact[0].get("score") != "N/A":
            score_rows = "".join(
                f'<div class="score-row"><span>{sc["score"]}</span><span class="prob">{sc["prob"]:.0%}</span></div>'
                for sc in exact[:3]
            )
            scores_html = f'<div style="margin-top:8px"><div class="scores-list">{score_rows}</div></div>'

        stats_rows = ""
        for eng_key, span_label in [
            ("Shots on Goal", "Remates a puerta"),
            ("Shots off Goal", "Remates fuera"),
            ("Total Shots", "Tiros totales"),
            ("Fouls", "Faltas"),
            ("Corner Kicks", "Esquinas"),
            ("Yellow Cards", "T. Amarillas"),
            ("Ball Possession", "Posesion"),
            ("Goalkeeper Saves", "Atajadas"),
        ]:
            hv = home_st.get(eng_key, "-")
            av = away_st.get(eng_key, "-")
            hv_str = f"{hv:.0f}%" if eng_key == "Ball Possession" and isinstance(hv, (int, float)) else f"{hv:.1f}" if isinstance(hv, float) else str(hv)
            av_str = f"{av:.0f}%" if eng_key == "Ball Possession" and isinstance(av, (int, float)) else f"{av:.1f}" if isinstance(av, float) else str(av)
            stats_rows += f'<div class="h">{hv_str}</div><div class="lbl">{span_label}</div><div class="a">{av_str}</div>\n'

        cards.append(f"""
    <div class="card">
      <div class="match-name">{h_display} <span class="vs">vs</span> {a_display}</div>
      <div class="match-meta">
        <span>{venue}</span>
        <span class="badge amber">{stage}</span>
        <span>{time}</span>
      </div>
      <div class="prob-bar">
        <div class="home" style="width:{h_prob * 100:.0f}%">{h_prob * 100:.0f}%</div>
        <div class="draw" style="width:{d_prob * 100:.0f}%">{d_prob * 100:.0f}%</div>
        <div class="away" style="width:{a_prob * 100:.0f}%">{a_prob * 100:.0f}%</div>
      </div>
      <div class="markets">
        <div class="market-item"><div class="lbl">O2.5</div><div class="val {prob_class(o25)}">{o25 * 100:.0f}%</div></div>
        <div class="market-item"><div class="lbl">BTTS</div><div class="val {prob_class(btts)}">{btts * 100:.0f}%</div></div>
        <div class="market-item"><div class="lbl">Pronostico</div><div class="val" style="color:{'#3B82F6' if pick_color == 'green' else ('var(--red)' if pick_color == 'red' else 'var(--amber)')}">{pick}</div></div>
      </div>
      <div class="details-content" style="display:block">
        <div style="margin-top:10px">
          <div class="stats-grid">
{stats_rows}
          </div>
          {scores_html}
        </div>
      </div>
      <button class="btn-form" onclick="openForm('{h_display}','{a_display}')">Ver forma ultimos 8 partidos</button>
      <div class="value-bet">
        <div class="lbl">Value Bet</div>
        <div class="{vb_class}">{vb_text}</div>
      </div>
    </div>""")

    return "\n".join(cards)


def generate_bets_section(predictions: dict) -> str:
    lines = []
    for home, away, *_rest in MATCHES:
        key = f"{home} vs {away}"
        p = predictions.get(key, {})
        h_prob = p.get("prob_home", 0.33)
        d_prob = p.get("prob_draw", 0.33)
        a_prob = p.get("prob_away", 0.34)

        h_display = NAME_MAP_ES[home]
        a_display = NAME_MAP_ES[away]
        best_prob = max(h_prob, d_prob, a_prob)
        if best_prob == h_prob:
            pick_name = h_display
        elif best_prob == a_prob:
            pick_name = a_display
        else:
            pick_name = "Empate"

        fair = round(1.0 / max(best_prob, 0.01), 2)
        market_odds = round(fair * 0.92, 2)

        lines.append(f"""
    <div class="pick-row">
      <div class="left"><div class="lbl">Partido</div><div class="val">{h_display} vs {a_display}</div></div>
      <div class="right"><div class="odds">{market_odds}</div><div class="pick">{pick_name} gana ({best_prob * 100:.0f}%)</div></div>
    </div>""")

    high_conf = []
    high_conf_names = []
    for home, away, *_rest in MATCHES:
        key = f"{home} vs {away}"
        p = predictions.get(key, {})
        h_prob = p.get("prob_home", 0.33)
        d_prob = p.get("prob_draw", 0.33)
        a_prob = p.get("prob_away", 0.34)
        best = max(h_prob, a_prob)
        if best >= 0.55:
            h_display = NAME_MAP_ES[home]
            a_display = NAME_MAP_ES[away]
            if best == h_prob:
                pick_name = h_display
            else:
                pick_name = a_display
            high_conf.append((home, away, best, pick_name))

    parlays_html = ""
    if len(high_conf) >= 2:
        parlays_html = f"""
  <div class="parlay-card">
    <div class="title">Parlay — Favoritos del modelo</div>
    <div class="sub">Picks con mayor confianza</div>"""
        total_odds = 1.0
        total_prob = 1.0
        for i, (h, a, prob, pick_name) in enumerate(high_conf[:3], 1):
            h_display = NAME_MAP_ES[h]
            a_display = NAME_MAP_ES[a]
            fair = round(1.0 / max(prob, 0.01), 2)
            market = round(fair * 0.92, 2)
            total_odds *= market
            total_prob *= prob
            parlays_html += f"""
    <div class="parlay-leg">
      <div class="num">{i}</div>
      <div class="detail"><div class="match">{h_display} vs {a_display}</div><div class="pick">{pick_name} gana ({prob * 100:.0f}%)</div></div>
      <div class="odds">{market}</div>
    </div>"""
        total_odds = round(total_odds, 2)
        total_prob = round(total_prob * 100)
        parlays_html += f"""
    <div class="parlay-total">
      <div><div class="lbl">Cuota Total</div><div class="odds">{total_odds}</div></div>
      <div><div class="lbl">Probabilidad</div><div class="prob">{total_prob}%</div></div>
    </div>
  </div>"""

    return f"""
  <div class="parlay-card">
    <div class="title">Apuestas Simples</div>
    <div class="sub">Picks del modelo con mayor probabilidad</div>
    {''.join(lines)}
  </div>
{parlays_html}"""


def generate_model_section(predictions: dict) -> str:
    rows = ""
    for home, away, *_rest in MATCHES:
        key = f"{home} vs {away}"
        p = predictions.get(key, {})
        h_prob = p.get("prob_home", 0.33)
        d_prob = p.get("prob_draw", 0.33)
        a_prob = p.get("prob_away", 0.34)
        o25 = p.get("prob_over_2_5", 0.5)
        btts = p.get("prob_btts_yes", 0.5)

        h_display = NAME_MAP_ES[home]
        a_display = NAME_MAP_ES[away]

        best = max(h_prob, d_prob, a_prob)
        best_str = f"{h_prob * 100:.0f}/{d_prob * 100:.0f}/{a_prob * 100:.0f}"
        if h_prob == best:
            best_highlight = f"<strong>{h_prob * 100:.0f}</strong>/{d_prob * 100:.0f}/{a_prob * 100:.0f}"
        elif d_prob == best:
            best_highlight = f"{h_prob * 100:.0f}/<strong>{d_prob * 100:.0f}</strong>/{a_prob * 100:.0f}"
        else:
            best_highlight = f"{h_prob * 100:.0f}/{d_prob * 100:.0f}/<strong>{a_prob * 100:.0f}</strong>"

        rows += f"""
        <tr><td>{h_display} vs {a_display}</td><td>{best_highlight}</td><td class="{prob_class(o25)}">{o25 * 100:.0f}%</td><td class="{prob_class(btts)}">{btts * 100:.0f}%</td></tr>"""

    return f"""
  <div class="card full">
    <h2>Proximas Predicciones <span>&middot; Datos del modelo</span></h2>
    <table>
      <thead><tr><th>Partido</th><th>1X2</th><th>O/U 2.5</th><th>BTTS</th></tr></thead>
      <tbody>{rows}
      </tbody>
    </table>
  </div>"""


def main():
    print("=" * 60)
    print("  Dashboard Generator — ML Predictions")
    print("=" * 60)

    config = load_config()

    # Load CSV
    print(f"\n[CSV]  Loading: {CSV}")
    df = pd.read_csv(CSV, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df_finished = df[df["home_goals"].notna()].copy()
    print(f"       {len(df_finished)} finished matches")

    # Load model
    if not os.path.exists(MODEL):
        print(f"[ERROR] Model not found: {MODEL}")
        print("        Train first: python main.py --from-csv national_teams_data.csv")
        sys.exit(1)

    print(f"\n[MODEL] Loading: {MODEL}")
    trainer = ModelTrainer.load(MODEL)
    print(f"        Features: {len(trainer.feature_cols)}")
    print(f"        Targets: {list(trainer.models.keys())}")

    # Build features on historical data
    print(f"\n[FEAT]  Building features...")
    feature_builder = FeatureBuilder(config)
    df_features = feature_builder.build(df_finished)

    # Derive targets
    for col, expr in [
        ("home_win", lambda df: (df["home_goals"] > df["away_goals"]).astype(int)),
        ("draw", lambda df: (df["home_goals"] == df["away_goals"]).astype(int)),
        ("away_win", lambda df: (df["away_goals"] > df["home_goals"]).astype(int)),
        ("total_goals", lambda df: df["home_goals"] + df["away_goals"]),
        ("over_2_5", lambda df: ((df["home_goals"] + df["away_goals"]) > 2.5).astype(int)),
        ("btts", lambda df: ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)),
    ]:
        if col not in df_features.columns:
            df_features[col] = expr(df_features)

    df_features["result"] = (
        df_features["home_win"] * 0 +
        df_features["draw"] * 1 +
        df_features["away_win"] * 2
    )

    predictor = MatchPredictor(trainer, config)
    stats_predictor = MatchStatsPredictor(config)

    # Predict each match
    predictions = {}
    all_stats = {}

    for home, away, *_rest in MATCHES:
        key = f"{home} vs {away}"
        print(f"\n[PRED]  {home} vs {away}")

        synthetic = pd.DataFrame([{
            "fixture_id": 0,
            "date": datetime.now().replace(tzinfo=None),
            "home_team": home,
            "away_team": away,
            "home_goals": None,
            "away_goals": None,
            "home_team_id": 0,
            "away_team_id": 0,
            "status_short": "NS",
        }])

        preds = predictor.predict_batch(synthetic, df_finished)
        if preds:
            p = preds[0]
            predictions[key] = p
            h_prob = p.get("prob_home", 0.33)
            d_prob = p.get("prob_draw", 0.33)
            a_prob = p.get("prob_away", 0.34)
            print(f"       1X2: {h_prob * 100:.0f}% / {d_prob * 100:.0f}% / {a_prob * 100:.0f}%")
            print(f"       O2.5: {p.get('prob_over_2_5', 0.5) * 100:.0f}%")
            print(f"       BTTS: {p.get('prob_btts_yes', 0.5) * 100:.0f}%")

            # Stats
            stats = stats_predictor.predict_stats(home, away, df_finished)
            all_stats[key] = stats
            print(f"       Stats: {stats.get('note', '')}")

    # Build team form data
    print(f"\n[FORM]  Building team form data...")
    all_teams = set()
    for home, away, *_rest in MATCHES:
        all_teams.add(home)
        all_teams.add(away)

    team_data = {}
    for team in sorted(all_teams):
        td = get_team_form(team, df_finished)
        team_data[TEAM_KEY_MAP[team]] = td
        print(f"       {team}: {td['f']} ({td['n']} matches)")

    # Generate HTML
    print(f"\n[HTML]  Generating dashboard...")
    html = generate_html(predictions, team_data, all_stats)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[DONE]  Dashboard saved to: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
