"""Liga MX Dashboard — Apertura 2026 Jornada 1"""
import sys, os, json, math
from datetime import datetime
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from sports_ml.features.builder import FeatureBuilder
from sports_ml.models.trainer import ModelTrainer
from sports_ml.models.predictor import MatchPredictor
from sports_ml.utils.config import Config

CSV = os.path.join(os.path.dirname(__file__), "ligamx_data.csv")
MODEL = os.path.join(os.path.dirname(__file__), "models/saved/ligamx_ensemble.pkl")
OUT = os.path.join(os.path.dirname(__file__), "dashboard", "ligamx.html")

# Real Jornada 1 Apertura 2026
MATCHES = [
    ("Club Tijuana", "Tigres UANL", "Jue 16 Jul"),
    ("Leon", "Atlas", "Vie 17 Jul"),
    ("Atletico San Luis", "Cruz Azul", "Vie 17 Jul"),
    ("FC Juarez", "Puebla", "Vie 17 Jul"),
    ("U.N.A.M. - Pumas", "Pachuca", "Sab 18 Jul"),
    ("Monterrey", "Santos Laguna", "Sab 18 Jul"),
    ("Guadalajara Chivas", "Toluca", "Sab 18 Jul"),
    ("Club Queretaro", "Club America", "Sab 18 Jul"),
]

# Real results (home goals, away goals) — update as matches finish
# None = pending / no result yet
ACTUAL_RESULTS_J1 = {
    ("Club Tijuana", "Tigres UANL"): "3–1",
    ("Leon", "Atlas"): "2–3",
    ("Atletico San Luis", "Cruz Azul"): "2–3",
    ("FC Juarez", "Puebla"): "0–1",
    ("U.N.A.M. - Pumas", "Pachuca"): "0–3",
    ("Monterrey", "Santos Laguna"): "3–2",
    ("Guadalajara Chivas", "Toluca"): "0–2",
    ("Club Queretaro", "Club America"): "0–1",
}

# Jornada 2 — Apertura 2026
MATCHES_J2 = [
    ("Cruz Azul", "Puebla", "Mar 21 Jul"),
    ("Toluca", "U.N.A.M. - Pumas", "Mar 21 Jul"),
    ("Club Tijuana", "Leon", "Vie 24 Jul"),
    ("Atlante", "Club America", "Vie 24 Jul"),   # Atlante not in training → will skip
    ("Guadalajara Chivas", "FC Juarez", "Sab 25 Jul"),
    ("Santos Laguna", "Atlas", "Sab 25 Jul"),
    ("Tigres UANL", "Atletico San Luis", "Sab 25 Jul"),
    ("Necaxa", "Monterrey", "Dom 26 Jul"),
    ("Pachuca", "Club Queretaro", "Dom 26 Jul"),
]

# Load & clean data
df = pd.read_csv(CSV)
df = df[df['home_goals'].notna()].copy()
df['date'] = pd.to_datetime(df['date'])
has_goals = df['home_goals'].notna()
df['home_win'] = 0; df['away_win'] = 0; df['draw'] = 0
df.loc[has_goals, 'home_win'] = (df.loc[has_goals, 'home_goals'] > df.loc[has_goals, 'away_goals']).astype(int)
df.loc[has_goals, 'draw'] = (df.loc[has_goals, 'home_goals'] == df.loc[has_goals, 'away_goals']).astype(int)
df.loc[has_goals, 'away_win'] = (df.loc[has_goals, 'away_goals'] > df.loc[has_goals, 'home_goals']).astype(int)
df['result'] = df['home_win'] * 0 + df['draw'] * 1 + df['away_win'] * 2
df['total_goals'] = df['home_goals'] + df['away_goals']
df['over_2_5'] = (df['total_goals'] > 2.5).astype(int)
df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)

leak_cols = ['id', 'referee', 'venue_id', 'venue_name', 'venue_city',
             'home_goals_half_time', 'away_goals_half_time',
             'home_goals_fulltime', 'away_goals_fulltime',
             'home_goals_extra_time', 'away_goals_extratime',
             'home_goals_penalty', 'away_goals_penalty', 'timezone', 'season']
df = df.drop(columns=[c for c in leak_cols if c in df.columns])

cfg = Config()
cfg.features.include_h2h = False
cfg.features.include_fifa_rankings = False
cfg.features.use_home_advantage = True

# Team stats for table
all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
team_stats = {}
for t in all_teams:
    h = df[df['home_team'] == t]; a = df[df['away_team'] == t]
    p = len(h) + len(a)
    w = len(h[h['home_goals'] > h['away_goals']]) + len(a[a['away_goals'] > a['home_goals']])
    d = len(h[h['home_goals'] == h['away_goals']]) + len(a[a['away_goals'] == a['home_goals']])
    l = p - w - d
    gf = h['home_goals'].sum() + a['away_goals'].sum()
    ga = h['away_goals'].sum() + a['home_goals'].sum()
    last10 = pd.concat([h[['date','home_goals','away_goals']].rename(columns={'home_goals':'gf','away_goals':'ga'}),
                        a[['date','away_goals','home_goals']].rename(columns={'away_goals':'gf','home_goals':'ga'})]).sort_values('date').tail(10)
    form = ''
    for _, r in last10.iterrows():
        if r['gf'] > r['ga']: form += 'G'
        elif r['gf'] == r['ga']: form += 'E'
        else: form += 'P'
    team_stats[t] = {'p': p, 'w': w, 'd': d, 'l': l, 'gf': gf, 'ga': ga, 'gd': gf-ga, 'pts': w*3+d, 'form': form, 'wr': w/p*100}

sorted_teams = sorted(team_stats.values(), key=lambda x: x['pts'], reverse=True)[:20]

# Build features
fb = FeatureBuilder(cfg)
featured = fb.build(df)

# Load model
if os.path.exists(MODEL):
    trainer = ModelTrainer.load(MODEL)
    print("Model loaded")
else:
    trainer = ModelTrainer(cfg)
    print("\nTraining 1X2..."); trainer.train(featured, "1X2")
    print("Training over_under..."); trainer.train(featured, "over_under")
    print("Training btts..."); trainer.train(featured, "btts")
    trainer.save(MODEL)

predictor = MatchPredictor(trainer, cfg)

# Predict real Jornada 1 matches
pred_cards = []
for home, away, _ in MATCHES:
    match_series = pd.Series({'fixture_id': -1, 'home_team': home, 'away_team': away,
                              'date': pd.Timestamp.now(), 'home_goals': None, 'away_goals': None})
    try:
        p = predictor.predict_match(match_series, df)
        pred_cards.append((home, away, p))
    except Exception as e:
        print(f"  {home} vs {away}: {e}")

# Compute picks and results
card_data = []
correct_count = 0
total_resolved = 0
for home, away, p in pred_cards:
    h = home; a = away
    ph = p.get('prob_home',0)*100; pd_ = p.get('prob_draw',0)*100; pa = p.get('prob_away',0)*100
    o25 = p.get('prob_over_2_5',0)*100; bt = p.get('prob_btts_yes',0)*100
    lh = p.get('lambda_home',0); la = p.get('lambda_away',0)
    if ph >= pd_ and ph >= pa: fave = h; fave_p = ph
    elif pa >= ph and pa >= pd_: fave = a; fave_p = pa
    else: fave = "Empate"; fave_p = pd_
    fair = round(1.0/max(fave_p/100, 0.01), 2)
    market = round(fair * 0.92, 2)

    actual_score = ACTUAL_RESULTS_J1.get((home, away))
    if actual_score:
        parts = actual_score.split('–')
        hg, ag = int(parts[0]), int(parts[1])
        if (fave == h and hg > ag) or (fave == a and ag > hg) or (fave == "Empate" and hg == ag):
            correct = True
            correct_count += 1
        else:
            correct = False
        total_resolved += 1
        badge = '✅' if correct else '❌'
    else:
        hg, ag = None, None
        badge = '⏳'
        correct = None

    card_data.append({
        'home': h, 'away': a, 'ph': ph, 'pd_': pd_, 'pa': pa,
        'o25': o25, 'bt': bt, 'lh': lh, 'la': la,
        'fave': fave, 'fave_p': fave_p, 'fair': fair,
        'actual_score': actual_score, 'badge': badge, 'correct': correct,
        'hg': hg, 'ag': ag,
    })

acc_pct = (correct_count / total_resolved * 100) if total_resolved else 0

# ── Jornada 2 predictions ──
j2_card_data = []
for home, away, _ in MATCHES_J2:
    match_series = pd.Series({'fixture_id': -1, 'home_team': home, 'away_team': away,
                              'date': pd.Timestamp.now(), 'home_goals': None, 'away_goals': None})
    try:
        p = predictor.predict_match(match_series, df)
    except Exception as e:
        print(f"  J2 {home} vs {away}: {e}")
        continue
    ph = p.get('prob_home',0)*100; pd_ = p.get('prob_draw',0)*100; pa = p.get('prob_away',0)*100
    o25 = p.get('prob_over_2_5',0)*100; bt = p.get('prob_btts_yes',0)*100
    lh = p.get('lambda_home',0); la = p.get('lambda_away',0)
    if ph >= pd_ and ph >= pa: fave = home; fave_p = ph
    elif pa >= ph and pa >= pd_: fave = away; fave_p = pa
    else: fave = "Empate"; fave_p = pd_
    j2_card_data.append({
        'home': home, 'away': away, 'ph': ph, 'pd_': pd_, 'pa': pa,
        'o25': o25, 'bt': bt, 'lh': lh, 'la': la,
        'fave': fave, 'fave_p': fave_p,
    })

# Result table rows
result_rows_html = ""
for i, d in enumerate(card_data, 1):
    pick = d['fave']
    prob = f"{d['fave_p']:.0f}%"
    if d['actual_score']:
        status = d['badge']
        result_str = f"{d['home']} {d['actual_score']} {d['away']}"
    else:
        status = "⏳ Pendiente"
        result_str = "—"
    result_rows_html += f"""
    <tr><td>{i}</td><td>{d['home']} vs {d['away']}</td><td><strong>{pick}</strong></td><td>{prob}</td><td>{result_str}</td><td style="font-size:18px;text-align:center">{status}</td></tr>"""

# ── Generate HTML ──
cards_html = ""
for d in card_data:
    h = d['home']; a = d['away']
    ph = d['ph']; pd_ = d['pd_']; pa = d['pa']
    o25 = d['o25']; bt = d['bt']
    lh = d['lh']; la = d['la']
    fave = d['fave']; fave_p = d['fave_p']
    actual_score = d['actual_score']; badge = d['badge']

    lam_t = lh + la
    def poisson_over(lam, thresh):
        prob = 0.0
        for k in range(int(thresh) + 1):
            prob += (lam ** k) * math.exp(-lam) / math.factorial(k)
        return max(0.01, min(0.99, 1.0 - prob))
    o05 = poisson_over(lam_t, 0.5) * 100
    o15 = poisson_over(lam_t, 1.5) * 100
    o35 = poisson_over(lam_t, 3.5) * 100
    o45 = poisson_over(lam_t, 4.5) * 100

    def prob_class(v):
        if v >= 55: return "green"
        elif v >= 50: return "amber"
        return "red"

    ah_05h = ph + pd_ if fave == a else ph
    ah_m05h = pa if fave == a else pa
    ah_1h = ph + pd_ + pa * 0.45
    ah_m1h = pa * 0.55

    avg_yellow = 4.5
    avg_corner = 9.5
    avg_shot = 7.5
    intensity = max(0.7, min(1.3, lam_t / 2.66))
    y_prob = poisson_over(avg_yellow * intensity, 4.5) * 100
    c_prob = poisson_over(avg_corner * intensity, 9.5) * 100
    s_prob = poisson_over(avg_shot * intensity, 7.5) * 100

    bt_o25 = bt/100 * o25/100 * 100
    fave_o25 = fave_p/100 * o25/100 * 100
    fave_bt = fave_p/100 * bt/100 * 100

    result_line = ""
    if actual_score:
        result_line = f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center"><span style="font-size:14px;color:var(--title)"><strong>Resultado:</strong> {h} {actual_score} {a}</span><span style="font-size:22px">{badge}</span></div>'
    else:
        result_line = f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center"><span style="font-size:12px;color:var(--body)">Resultado: Pendiente</span><span style="font-size:16px">{badge}</span></div>'

    cards_html += f"""
    <div class="card">
      <div class="match-name">{h} <span class="vs">vs</span> {a}</div>
      <div class="prob-bar">
        <div class="home" style="width:{ph:.0f}%">{ph:.0f}%</div>
        <div class="draw" style="width:{pd_:.0f}%">{pd_:.0f}%</div>
        <div class="away" style="width:{pa:.0f}%">{pa:.0f}%</div>
      </div>
      <div class="markets">
        <div class="market-item"><div class="lbl">Pronóstico</div><div class="val" style="color:{'var(--green)' if fave==h else ('var(--red)' if fave==a else 'var(--amber)')}">{fave} ({fave_p:.0f}%)</div></div>
        <div class="market-item"><div class="lbl">O2.5</div><div class="val green">{o25:.0f}%</div></div>
        <div class="market-item"><div class="lbl">BTTS</div><div class="val amber">{bt:.0f}%</div></div>
      </div>
      {result_line}
      <div class="mercados-extra">
        <div class="me-header" onclick="this.parentElement.classList.toggle('open')">
          <span>Mercados Adicionales</span>
          <span class="me-arrow">&#9654;</span>
        </div>
        <div class="me-content">
          <div class="me-section">
            <div class="me-title">Líneas de Gol (Poisson λ={lam_t:.2f})</div>
            <div class="markets-grid">
              <div class="market-item"><div class="lbl">O0.5</div><div class="val {prob_class(o05)}">{o05:.0f}%</div></div>
              <div class="market-item"><div class="lbl">O1.5</div><div class="val {prob_class(o15)}">{o15:.0f}%</div></div>
              <div class="market-item"><div class="lbl">O2.5</div><div class="val {prob_class(o25)}">{o25:.0f}%</div></div>
              <div class="market-item"><div class="lbl">O3.5</div><div class="val {prob_class(o35)}">{o35:.0f}%</div></div>
              <div class="market-item"><div class="lbl">O4.5</div><div class="val {prob_class(o45)}">{o45:.0f}%</div></div>
            </div>
          </div>
          <div class="me-section">
            <div class="me-title">Hándicap Asiático</div>
            <div class="markets-grid" style="grid-template-columns:1fr 1fr">
              <div class="market-item"><div class="lbl">{h} AH -0.5</div><div class="val {prob_class(ah_m05h)}">{ah_m05h:.0f}%</div></div>
              <div class="market-item"><div class="lbl">{a} AH -0.5</div><div class="val {prob_class(pa)}">{pa:.0f}%</div></div>
              <div class="market-item"><div class="lbl">{h} AH +0.5</div><div class="val {prob_class(ah_05h)}">{ah_05h:.0f}%</div></div>
              <div class="market-item"><div class="lbl">{a} AH +0.5</div><div class="val {prob_class(ph)}">{ph:.0f}%</div></div>
            </div>
          </div>
          <div class="me-section">
            <div class="me-title">Estadísticas (Estimación Liga MX)</div>
            <div class="markets-grid">
              <div class="market-item"><div class="lbl">T. Amarillas O4.5</div><div class="val {prob_class(y_prob)}">{y_prob:.0f}%</div></div>
              <div class="market-item"><div class="lbl">Córners O9.5</div><div class="val {prob_class(c_prob)}">{c_prob:.0f}%</div></div>
              <div class="market-item"><div class="lbl">Remates P. O7.5</div><div class="val {prob_class(s_prob)}">{s_prob:.0f}%</div></div>
            </div>
          </div>
          <div class="me-section">
            <div class="me-title">Combinadas</div>
            <div class="combo-grid">
              <div class="m-line"><div class="m-name">BTTS + O2.5</div><div class="m-prob">{bt_o25:.0f}%</div><div class="m-odds">{1/max(bt_o25/100,0.01):.2f}</div></div>
              <div class="m-line"><div class="m-name">{fave} Gana + O2.5</div><div class="m-prob">{fave_o25:.0f}%</div><div class="m-odds">{1/max(fave_o25/100,0.01):.2f}</div></div>
              <div class="m-line"><div class="m-name">{fave} Gana + BTTS</div><div class="m-prob">{fave_bt:.0f}%</div><div class="m-odds">{1/max(fave_bt/100,0.01):.2f}</div></div>
            </div>
          </div>
        </div>
      </div>
    </div>"""

# ── J2 cards ──
j2_cards_html = ""
for d in j2_card_data:
    h = d['home']; a = d['away']
    ph = d['ph']; pd_ = d['pd_']; pa = d['pa']
    o25 = d['o25']; bt = d['bt']
    lh = d['lh']; la = d['la']
    fave = d['fave']; fave_p = d['fave_p']

    lam_t = lh + la
    def poisson_over(lam, thresh):
        prob = 0.0
        for k in range(int(thresh) + 1):
            prob += (lam ** k) * math.exp(-lam) / math.factorial(k)
        return max(0.01, min(0.99, 1.0 - prob))
    o05 = poisson_over(lam_t, 0.5) * 100
    o15 = poisson_over(lam_t, 1.5) * 100
    o35 = poisson_over(lam_t, 3.5) * 100
    o45 = poisson_over(lam_t, 4.5) * 100

    def prob_class(v):
        if v >= 55: return "green"
        elif v >= 50: return "amber"
        return "red"

    ah_05h = ph + pd_ if fave == a else ph
    ah_m05h = pa if fave == a else pa

    avg_yellow = 4.5
    avg_corner = 9.5
    avg_shot = 7.5
    intensity = max(0.7, min(1.3, lam_t / 2.66))
    y_prob = poisson_over(avg_yellow * intensity, 4.5) * 100
    c_prob = poisson_over(avg_corner * intensity, 9.5) * 100
    s_prob = poisson_over(avg_shot * intensity, 7.5) * 100

    bt_o25 = bt/100 * o25/100 * 100
    fave_o25 = fave_p/100 * o25/100 * 100
    fave_bt = fave_p/100 * bt/100 * 100

    j2_cards_html += f"""
    <div class="card">
      <div class="match-name">{h} <span class="vs">vs</span> {a}</div>
      <div class="prob-bar">
        <div class="home" style="width:{ph:.0f}%">{ph:.0f}%</div>
        <div class="draw" style="width:{pd_:.0f}%">{pd_:.0f}%</div>
        <div class="away" style="width:{pa:.0f}%">{pa:.0f}%</div>
      </div>
      <div class="markets">
        <div class="market-item"><div class="lbl">Pronóstico</div><div class="val" style="color:{'var(--green)' if fave==h else ('var(--red)' if fave==a else 'var(--amber)')}">{fave} ({fave_p:.0f}%)</div></div>
        <div class="market-item"><div class="lbl">O2.5</div><div class="val green">{o25:.0f}%</div></div>
        <div class="market-item"><div class="lbl">BTTS</div><div class="val amber">{bt:.0f}%</div></div>
      </div>
      <div class="mercados-extra">
        <div class="me-header" onclick="this.parentElement.classList.toggle('open')">
          <span>Mercados Adicionales</span>
          <span class="me-arrow">&#9654;</span>
        </div>
        <div class="me-content">
          <div class="me-section">
            <div class="me-title">Líneas de Gol (Poisson λ={lam_t:.2f})</div>
            <div class="markets-grid">
              <div class="market-item"><div class="lbl">O0.5</div><div class="val {prob_class(o05)}">{o05:.0f}%</div></div>
              <div class="market-item"><div class="lbl">O1.5</div><div class="val {prob_class(o15)}">{o15:.0f}%</div></div>
              <div class="market-item"><div class="lbl">O2.5</div><div class="val {prob_class(o25)}">{o25:.0f}%</div></div>
              <div class="market-item"><div class="lbl">O3.5</div><div class="val {prob_class(o35)}">{o35:.0f}%</div></div>
              <div class="market-item"><div class="lbl">O4.5</div><div class="val {prob_class(o45)}">{o45:.0f}%</div></div>
            </div>
          </div>
          <div class="me-section">
            <div class="me-title">Hándicap Asiático</div>
            <div class="markets-grid" style="grid-template-columns:1fr 1fr">
              <div class="market-item"><div class="lbl">{h} AH -0.5</div><div class="val {prob_class(ah_m05h)}">{ah_m05h:.0f}%</div></div>
              <div class="market-item"><div class="lbl">{a} AH -0.5</div><div class="val {prob_class(pa)}">{pa:.0f}%</div></div>
              <div class="market-item"><div class="lbl">{h} AH +0.5</div><div class="val {prob_class(ah_05h)}">{ah_05h:.0f}%</div></div>
              <div class="market-item"><div class="lbl">{a} AH +0.5</div><div class="val {prob_class(ph)}">{ph:.0f}%</div></div>
            </div>
          </div>
          <div class="me-section">
            <div class="me-title">Estadísticas (Estimación Liga MX)</div>
            <div class="markets-grid">
              <div class="market-item"><div class="lbl">T. Amarillas O4.5</div><div class="val {prob_class(y_prob)}">{y_prob:.0f}%</div></div>
              <div class="market-item"><div class="lbl">Córners O9.5</div><div class="val {prob_class(c_prob)}">{c_prob:.0f}%</div></div>
              <div class="market-item"><div class="lbl">Remates P. O7.5</div><div class="val {prob_class(s_prob)}">{s_prob:.0f}%</div></div>
            </div>
          </div>
          <div class="me-section">
            <div class="me-title">Combinadas</div>
            <div class="combo-grid">
              <div class="m-line"><div class="m-name">BTTS + O2.5</div><div class="m-prob">{bt_o25:.0f}%</div><div class="m-odds">{1/max(bt_o25/100,0.01):.2f}</div></div>
              <div class="m-line"><div class="m-name">{fave} Gana + O2.5</div><div class="m-prob">{fave_o25:.0f}%</div><div class="m-odds">{1/max(fave_o25/100,0.01):.2f}</div></div>
              <div class="m-line"><div class="m-name">{fave} Gana + BTTS</div><div class="m-prob">{fave_bt:.0f}%</div><div class="m-odds">{1/max(fave_bt/100,0.01):.2f}</div></div>
            </div>
          </div>
        </div>
      </div>
    </div>"""

# Table
team_name_by_stats = {}
for name, st in team_stats.items():
    key = (st['p'], st['w'], st['pts'], st['gf'])
    team_name_by_stats[key] = name

table_rows = ""
for i, ts in enumerate(sorted_teams, 1):
    name = team_name_by_stats[(ts['p'], ts['w'], ts['pts'], ts['gf'])]
    form_html = ''.join(f'<span class="fw-{c}">{c}</span>' for c in ts['form'][-8:]) if ts['form'] else '-'
    table_rows += f"""
    <tr><td>{i}</td><td><strong>{name}</strong></td><td>{int(ts['p'])}</td><td class="green">{int(ts['w'])}</td><td>{int(ts['d'])}</td><td class="red">{int(ts['l'])}</td><td>{int(ts['gf'])}</td><td>{int(ts['ga'])}</td><td class="{'green' if ts['gd']>0 else 'red'}">{ts['gd']:+}</td><td class="green">{ts['pts']}</td><td>{ts['wr']:.0f}%</td><td>{form_html}</td></tr>"""

tg = df['home_goals'] + df['away_goals']
avg_g = tg.mean()
o25_pct = (tg > 2.5).mean() * 100
btts_pct = ((df['home_goals']>0)&(df['away_goals']>0)).mean() * 100
h_win = (df['home_goals'] > df['away_goals']).mean() * 100

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Liga MX — Apertura 2026 J1</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{ --bg: #0A0E17; --surface: #131920; --border: #1E293B; --title: #F8FAFC; --body: #94A3B8; --green: #22C55E; --red: #EF4444; --amber: #F59E0B; --font: 'Inter', system-ui, sans-serif; --radius: 12px; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--body); font-family:var(--font); font-size:14px; padding:24px; line-height:1.5; }}
.container {{ max-width:1200px; margin:0 auto; }}
.header {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:20px; }}
.header-left h1 {{ font-size:24px; font-weight:600; color:var(--title); margin-bottom:2px; }}
.header-left .sub {{ font-size:13px; color:var(--body); }}
.header-right {{ font-size:12px; color:var(--body); }}
.disclaimer {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:12px 16px; font-size:12px; color:var(--body); margin-bottom:20px; }}
.nav {{ display:flex; gap:4px; margin-bottom:20px; flex-wrap:wrap; border-bottom:1px solid var(--border); }}
.nav button {{ background:none; border:none; color:var(--body); padding:10px 18px; font-size:13px; font-weight:500; font-family:var(--font); cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-1px; transition:color 0.15s; }}
.nav button:hover {{ color:var(--title); }}
.nav button.active {{ color:var(--title); border-bottom-color:var(--title); }}
.section {{ display:none; }}
.section.active {{ display:block; }}
.card-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:16px; margin-bottom:24px; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:20px; }}
.card.full {{ grid-column:1/-1; }}
.card h2 {{ font-size:14px; font-weight:600; color:var(--body); text-transform:uppercase; letter-spacing:0.4px; margin-bottom:14px; }}
.match-name {{ font-size:17px; font-weight:600; color:var(--title); margin-bottom:4px; }}
.match-name .vs {{ color:var(--body); font-weight:400; font-size:14px; margin:0 6px; }}
.prob-bar {{ display:flex; height:28px; border-radius:6px; overflow:hidden; margin-bottom:12px; font-size:11px; font-weight:600; }}
.prob-bar div {{ display:flex; align-items:center; justify-content:center; }}
.prob-bar .home {{ background:var(--green); color:#000; }}
.prob-bar .draw {{ background:var(--amber); color:#000; }}
.prob-bar .away {{ background:var(--red); color:#fff; }}
.markets {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:12px; }}
.market-item {{ background:rgba(255,255,255,0.04); border-radius:8px; padding:10px; text-align:center; }}
.market-item .lbl {{ font-size:10px; color:var(--body); text-transform:uppercase; letter-spacing:0.3px; }}
.market-item .val {{ font-size:16px; font-weight:700; color:var(--title); }}
.market-item .val.green {{ color:var(--green); }}
.market-item .val.red {{ color:var(--red); }}
.market-item .val.amber {{ color:var(--amber); }}
.value-bet {{ background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:8px; padding:10px 14px; margin-top:8px; }}
.value-bet .lbl {{ font-size:10px; color:var(--body); text-transform:uppercase; letter-spacing:0.3px; }}
.value-bet .val {{ font-size:13px; font-weight:600; color:var(--green); }}
.stat-box {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(180px,1fr)); gap:12px; margin-bottom:20px; }}
.stat-item {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:16px; text-align:center; }}
.stat-item .num {{ font-size:28px; font-weight:700; color:var(--title); }}
.stat-item .lbl {{ font-size:12px; color:var(--body); margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }}
th {{ text-align:left; padding:8px 10px; color:var(--body); font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.3px; border-bottom:1px solid var(--border); }}
td {{ padding:8px 10px; border-bottom:1px solid rgba(255,255,255,0.04); color:var(--body); white-space:nowrap; }}
td strong {{ color:var(--title); }}
td.green {{ color:var(--green); }}
td.red {{ color:var(--red); }}
td.amber {{ color:var(--amber); }}
.fw-G {{ color:var(--green); font-weight:700; }} .fw-E {{ color:var(--amber); font-weight:700; }} .fw-P {{ color:var(--red); font-weight:700; }}
.mercados-extra {{ margin-top:12px; border-top:1px solid var(--border); padding-top:8px; }}
.me-header {{ display:flex; justify-content:space-between; align-items:center; cursor:pointer; font-size:12px; font-weight:600; color:var(--body); text-transform:uppercase; letter-spacing:0.3px; padding:4px 0; user-select:none; }}
.me-arrow {{ transition:transform 0.2s; font-size:10px; }}
.mercados-extra.open .me-arrow {{ transform:rotate(90deg); }}
.me-content {{ display:none; margin-top:8px; }}
.mercados-extra.open .me-content {{ display:block; }}
.me-section {{ margin-bottom:10px; }}
.me-title {{ font-size:11px; font-weight:600; color:var(--body); margin-bottom:4px; letter-spacing:0.2px; }}
.markets-grid {{ display:flex; flex-wrap:wrap; gap:4px; }}
.markets-grid .market-item {{ flex:0 0 auto; min-width:65px; background:var(--bg); border:1px solid var(--border); border-radius:6px; padding:4px 8px; display:flex; justify-content:space-between; align-items:center; gap:6px; }}
.markets-grid .market-item .lbl {{ font-size:10px; color:var(--body); }}
.markets-grid .market-item .val {{ font-size:11px; font-weight:600; }}
.combo-grid {{ display:grid; gap:4px; }}
.m-line {{ display:flex; align-items:center; padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.04); gap:8px; }}
.m-line:last-child {{ border-bottom:none; }}
.m-name {{ flex:1; font-size:12px; color:var(--title); }}
.m-prob {{ font-size:12px; font-weight:600; color:var(--amber); min-width:36px; text-align:right; }}
.m-odds {{ font-size:13px; font-weight:700; color:var(--green); min-width:42px; text-align:right; }}
.footer {{ text-align:center; color:rgba(148,163,184,0.4); font-size:11px; margin-top:32px; padding:16px 0; border-top:1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <div class="header-left">
    <h1>Liga MX — Apertura 2026</h1>
    <div class="sub">{len(MATCHES)} J1 + {len(j2_card_data)} J2 &middot; {datetime.now().strftime('%d/%m/%Y')}</div>
  </div>
  <div class="header-right">{datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
</div>

<div class="disclaimer">
  Predicciones generadas por modelo ML entrenado con datos 2016-2024 (Random Forest + XGBoost). Solo fines educativos. Juega con responsabilidad.
</div>

<div class="nav">
  <button class="active" data-section="jornada1">Jornada 1</button>
  <button data-section="jornada2">Jornada 2</button>
  <button data-section="resultados">Resultados J1</button>
  <button data-section="estadisticas">Estadísticas</button>
  <button data-section="tabla">Tabla Histórica</button>
</div>

<!-- ===== JORNADA 1 ===== -->
<div id="s-jornada1" class="section active">
  <div class="card full" style="margin-bottom:16px">
    <h2>Rendimiento del Modelo — Jornada 1</h2>
    <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:center">
      <div style="font-size:20px;font-weight:700;color:var(--title)">{correct_count}/{total_resolved}</div>
      <div style="font-size:14px;color:var(--body)">aciertos en partidos resueltos</div>
      <div style="flex:1;min-width:120px;height:8px;background:var(--border);border-radius:4px;overflow:hidden">
        <div style="height:100%;width:{acc_pct:.0f}%;background:{'var(--green)' if acc_pct>=50 else 'var(--red)'};border-radius:4px"></div>
      </div>
      <div style="font-size:22px;font-weight:700;color:{'var(--green)' if acc_pct>=50 else 'var(--red)'}">{acc_pct:.0f}%</div>
    </div>
  </div>
  <div class="card-grid">
{cards_html}
  </div>
</div>

<!-- ===== JORNADA 2 ===== -->
<div id="s-jornada2" class="section">
  <div class="card full" style="margin-bottom:16px">
    <h2>Predicciones — Jornada 2</h2>
    <div style="font-size:13px;color:var(--body)">Mar 21 Jul — Dom 26 Jul &middot; {len(j2_card_data)} de 9 partidos</div>
    <div style="margin-top:8px;font-size:12px;color:var(--body);background:rgba(255,255,255,0.03);padding:8px 12px;border-radius:6px;border:1px solid var(--border)">
      El partido Atlante vs América se excluye porque Atlante no está en los datos de entrenamiento (2016-2024, Atlante recién ascendió).
    </div>
  </div>
  <div class="card-grid">
{j2_cards_html}
  </div>
</div>

<!-- ===== RESULTADOS ===== -->
<div id="s-resultados" class="section">
  <div class="card full">
    <h2>Picks del Modelo vs Resultados Reales — J1</h2>
    <table>
      <thead><tr><th>#</th><th>Partido</th><th>Pronóstico</th><th>Prob.</th><th>Resultado</th><th>Estado</th></tr></thead>
      <tbody>
{result_rows_html}
      </tbody>
    </table>
  </div>
</div>

<!-- ===== ESTADISTICAS ===== -->
<div id="s-estadisticas" class="section">
  <div class="stat-box">
    <div class="stat-item"><div class="num">{avg_g:.2f}</div><div class="lbl">Goles x Partido</div></div>
    <div class="stat-item"><div class="num">{o25_pct:.0f}%</div><div class="lbl">Over 2.5</div></div>
    <div class="stat-item"><div class="num">{btts_pct:.0f}%</div><div class="lbl">BTTS</div></div>
    <div class="stat-item"><div class="num">{h_win:.0f}%</div><div class="lbl">Local Gana</div></div>
  </div>
  <div class="card full">
    <h2>Rendimiento del Modelo (CV)</h2>
    <div style="font-size:13px;color:var(--body)">
      1X2: <strong style="color:var(--title)">61%</strong> &middot;
      O/U 2.5: <strong style="color:var(--title)">51%</strong> &middot;
      BTTS: <strong style="color:var(--title)">53%</strong>
    </div>
    <div style="margin-top:8px;font-size:12px;color:var(--body)">
      Random Forest (100 árboles, max_depth=5) + XGBoost (learning_rate=0.05) &middot; 33 features
    </div>
  </div>
</div>

<!-- ===== TABLA ===== -->
<div id="s-tabla" class="section">
  <div class="card full">
    <h2>Tabla Histórica 2016-2024</h2>
    <div style="font-size:12px;color:var(--body);margin-bottom:8px">{len(df)} partidos · {len(all_teams)} equipos</div>
    <table>
      <thead><tr><th>#</th><th>Equipo</th><th>PJ</th><th>G</th><th>E</th><th>P</th><th>GF</th><th>GA</th><th>DG</th><th>Pts</th><th>%</th><th>Forma</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>
</div>

<div class="footer">
  <p>Sports ML Pipeline v1.0 &middot; Apertura 2026 J1 + J2</p>
</div>

</div>

<script>
document.querySelectorAll('.nav button').forEach(btn => {{
  btn.addEventListener('click', function() {{
    document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
    this.classList.add('active');
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById('s-' + this.dataset.section).classList.add('active');
  }});
}});
</script>
</body>
</html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\n✅ Dashboard: {OUT}")
