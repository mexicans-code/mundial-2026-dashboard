import subprocess, json, webbrowser, threading, os
from flask import Flask, render_template_string

app = Flask(__name__)

INDEX_HTML = open(os.path.join(os.path.dirname(__file__), 'index.html'), encoding='utf-8').read()

@app.route('/')
def home():
    return INDEX_HTML

@app.route('/predict/<home>/<away>')
def predict(home, away):
    try:
        r = subprocess.run(
            ['python', 'main.py', '--from-csv', 'national_teams_data.csv', '--predict', '--match', f'{home} vs {away}'],
            capture_output=True, text=True, timeout=300, cwd=os.path.dirname(os.path.dirname(__file__))
        )
        return f'<pre>{r.stdout[:3000]}</pre>'
    except Exception as e:
        return f'<pre>Error: {e}</pre>'

if __name__ == '__main__':
    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5000')).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
