import subprocess, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sports_ml.utils.config import load_config
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))
CSV = os.path.join(BASE, 'national_teams_data.csv')
OUT = os.path.join(os.path.dirname(__file__), 'index.html')

def get_prediction(match_name):
    r = subprocess.run(
        ['python', 'main.py', '--from-csv', CSV, '--predict', '--match', match_name],
        capture_output=True, text=True, timeout=300, cwd=BASE
    )
    out = r.stdout
    lines = out.split('\n')
    data = {}
    for line in lines:
        if 'LOCAL' in line and 'VISITANTE' in line:
            continue
        if '---' in line and len(line) > 70:
            continue
        parts = line.strip().split()
        if len(parts) >= 7:
            try:
                data['home'] = parts[0]
                data['away'] = parts[1]
                data['pred'] = parts[2]
                data['conf'] = parts[3]
                data['probs'] = parts[4]
                data['o25'] = parts[5]
                data['btts'] = parts[6]
            except:
                pass
        if 'MARC. EXACTO' in line:
            data['scores'] = []
            scores_part = line.split('MARC. EXACTO')[1].strip()
            for sc in scores_part.split('|')[:5]:
                sc = sc.strip()
                if '(' in sc:
                    score, prob = sc.rsplit('(', 1)
                    prob = prob.replace(')', '').strip()
                    data['scores'].append({'score': score.strip(), 'prob': prob})
    return data

def load_csv_data():
    df = pd.read_csv(CSV)
    teams = {
        'USA': len(df[(df['home_team']=='USA')|(df['away_team']=='USA')]),
        'England': len(df[(df['home_team']=='England')|(df['away_team']=='England')]),
        'Belgium': len(df[(df['home_team']=='Belgium')|(df['away_team']=='Belgium')]),
        'Senegal': len(df[(df['home_team']=='Senegal')|(df['away_team']=='Senegal')]),
        'Bosnia & Herzegovina': len(df[(df['home_team'].str.contains('Bosnia'))|(df['away_team'].str.contains('Bosnia'))]),
        'Congo DR': len(df[(df['home_team']=='Congo DR')|(df['away_team']=='Congo DR')]),
    }
    return len(df), teams

if __name__ == '__main__':
    match_count, team_counts = load_csv_data()
    print(f"Dashboard generado con {match_count} partidos")
