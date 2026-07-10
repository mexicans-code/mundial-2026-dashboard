"""GUI visual para mostrar predicciones del modelo ML."""

import os, sys, json, math
from tkinter import ttk, Tk, Frame, Label, Canvas, Button, Scrollbar, Entry
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sports_ml.features.builder import FeatureBuilder
from sports_ml.models.trainer import ModelTrainer
from sports_ml.models.predictor import MatchPredictor
from sports_ml.models.stats_predictor import MatchStatsPredictor, STATS_LABELS
from sports_ml.data.collector import NATIONAL_TEAMS
from sports_ml.utils.config import Config

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "national_teams_data.csv")

TEAM_FLAGS = {
    "Brazil": "\U0001F1E7\U0001F1F7",
    "Japan": "\U0001F1EF\U0001F1F5",
    "Germany": "\U0001F1E9\U0001F1EA",
    "Paraguay": "\U0001F1F5\U0001F1FE",
    "Netherlands": "\U0001F1F3\U0001F1F1",
    "Morocco": "\U0001F1F2\U0001F1E6",
    "France": "\U0001F1EB\U0001F1F7",
    "Sweden": "\U0001F1F8\U0001F1EA",
    "Mexico": "\U0001F1F2\U0001F1FD",
    "Ecuador": "\U0001F1EA\U0001F1E8",
    "England": "\U0001F1EC\U0001F1E7",
    "Spain": "\U0001F1EA\U0001F1F8",
    "Portugal": "\U0001F1F5\U0001F1F9",
    "Argentina": "\U0001F1E6\U0001F1F7",
    "Uruguay": "\U0001F1FA\U0001F1FE",
    "Ivory Coast": "\U0001F1E8\U0001F1EE",
    "Norway": "\U0001F1F3\U0001F1F4",
}


class PredictionApp:
    def __init__(self):
        self.root = Tk()
        self.root.title("ML Betting Predictor — World Cup 2026")
        self.root.geometry("900x650")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        self.predictions = []
        self.status_var = ""
        self.setup_ui()
        self.run_pipeline()

    def setup_ui(self):
        title = Label(
            self.root, text="\U0001F3C6 ML BETTING PREDICTOR",
            font=("Segoe UI", 20, "bold"), fg="#e94560", bg="#1a1a2e"
        )
        title.pack(pady=(15, 0))

        subtitle = Label(
            self.root, text="World Cup 2026 — Todos los partidos",
            font=("Segoe UI", 11), fg="#888", bg="#1a1a2e"
        )
        subtitle.pack(pady=(0, 5))

        # Search bar
        search_frame = Frame(self.root, bg="#1a1a2e")
        search_frame.pack(fill="x", padx=20, pady=(0, 5))
        Label(search_frame, text="\U0001F50D", font=("Segoe UI", 11),
              fg="#888", bg="#1a1a2e").pack(side="left", padx=(0, 5))
        self.search_entry = Entry(search_frame, font=("Segoe UI", 10),
                                  bg="#16213e", fg="white", bd=0,
                                  insertbackground="white")
        self.search_entry.pack(side="left", fill="x", expand=True, ipady=3)
        self.search_entry.bind("<KeyRelease>", lambda e: self.filter_cards())

        self.status = Label(
            self.root, text="", font=("Segoe UI", 10),
            fg="#aaa", bg="#1a1a2e"
        )
        self.status.pack()

        self.canvas = Canvas(self.root, bg="#1a1a2e", highlightthickness=0)
        self.scrollbar = Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = Frame(self.canvas, bg="#1a1a2e")
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=5)
        self.scrollbar.pack(side="right", fill="y", pady=5)

        # Bind mousewheel for scrolling
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(pady=5)
        self.progress.start()

        self.root.update()

    def run_pipeline(self):
        self.status.config(text="\U0001F504 Cargando datos...")
        self.root.update()

        try:
            cfg = Config()
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    for s, vals in json.load(f).items():
                        if hasattr(cfg, s):
                            for k, v in vals.items():
                                if hasattr(getattr(cfg, s), k):
                                    setattr(getattr(cfg, s), k, v)

            df = pd.read_csv(CSV_PATH)
            df["date"] = pd.to_datetime(df["date"])
            finished = df[df["home_goals"].notna()].copy()
            self.finished = finished

            # Build match list from all finished matches sorted by date
            self.raw_matches = finished.sort_values("date", ascending=False)

            # Initialize predictions list with raw match data
            self.all_predictions = []
            for _, row in self.raw_matches.iterrows():
                self.all_predictions.append({
                    "fixture_id": row.get("fixture_id"),
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "home_goals": row.get("home_goals"),
                    "away_goals": row.get("away_goals"),
                    "date": str(row.get("date", ""))[:10],
                    "source_league": row.get("source_league", ""),
                    "venue_city": row.get("venue_city", ""),
                })

            # Train model and predict for recent matches in background
            self.status.config(text="\U0001F504 Entrenando modelo (no cierres)...")
            self.root.update()

            trainer = ModelTrainer(cfg)
            builder = FeatureBuilder(cfg)

            featured = builder.build(finished)

            if "home_win" not in featured.columns:
                featured["home_win"] = (featured["home_goals"] > featured["away_goals"]).astype(int)
            if "draw" not in featured.columns:
                featured["draw"] = (featured["home_goals"] == featured["away_goals"]).astype(int)
            if "away_win" not in featured.columns:
                featured["away_win"] = (featured["away_goals"] > featured["home_goals"]).astype(int)
            if "total_goals" not in featured.columns:
                featured["total_goals"] = featured["home_goals"] + featured["away_goals"]
            if "over_2_5" not in featured.columns:
                featured["over_2_5"] = (featured["total_goals"] > 2.5).astype(int)
            if "btts" not in featured.columns:
                featured["btts"] = ((featured["home_goals"] > 0) & (featured["away_goals"] > 0)).astype(int)
            featured["result"] = (
                featured["home_win"] * 0 + featured["draw"] * 1 + featured["away_win"] * 2
            )

            for target in ["1X2", "over_under", "btts"]:
                try:
                    trainer.train(featured, target)
                except Exception as e:
                    print(f"  [SKIP] {target}: {e}")

            predictor = MatchPredictor(trainer, cfg)
            trainer.save()

            # Predict for unique team pairings (cache predictions by matchup)
            self.match_cache = {}
            self.match_stats = {}
            stats_pred = MatchStatsPredictor(cfg)
            total = len(self.all_predictions)
            self.status.config(text=f"\U0001F504 Prediciendo partidos...")

            for i, p in enumerate(self.all_predictions):
                key = f"{p['home_team']} vs {p['away_team']}"
                if key not in self.match_cache:
                    match_series = pd.Series({
                        "fixture_id": -1, "home_team": p["home_team"], "away_team": p["away_team"],
                        "date": datetime.now().replace(tzinfo=None),
                        "home_goals": None, "away_goals": None,
                        "home_team_id": 0, "away_team_id": 0, "status_short": "NS",
                    })
                    try:
                        pred = predictor.predict_match(match_series, finished)
                        self.match_cache[key] = pred
                    except Exception as e:
                        self.match_cache[key] = {"predicted": "?", "confidence": 0}

                    # Stats
                    stats = stats_pred.predict_stats(p["home_team"], p["away_team"], finished)
                    self.match_stats[key] = stats

                # Merge cached prediction into this match entry
                cached = self.match_cache[key]
                p.update({k: v for k, v in cached.items() if k not in ("fixture_id", "home_team", "away_team", "date")})

                if i % 200 == 0:
                    pct = int(i / total * 100)
                    self.status.config(text=f"\U0001F504 Prediciendo... {pct}% ({i}/{total})")
                    self.root.update()

            self.search_term = ""
            self.progress.stop()
            self.progress.pack_forget()
            self.status.config(text=f"\u2705 {len(self.all_predictions)} partidos cargados — usa el buscador arriba")
            self.render_cards()

        except Exception as e:
            self.progress.stop()
            self.progress.pack_forget()
            self.status.config(text=f"\u274c Error: {e}", fg="#e94560")

    def filter_cards(self):
        self.search_term = self.search_entry.get().strip().lower()
        self.render_cards()

    def render_cards(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        # Determine which predictions to show
        show = self.all_predictions[:500]  # max 500 for performance

        if self.search_term:
            show = [p for p in self.all_predictions
                    if self.search_term in p["home_team"].lower()
                    or self.search_term in p["away_team"].lower()]

        if not show:
            Label(self.scroll_frame, text="No se encontraron partidos",
                  font=("Segoe UI", 12), fg="#888", bg="#1a1a2e").pack(pady=30)
            self.root.update()
            return

        self.status.config(text=f"\u2705 {len(show)} partidos (de {len(self.all_predictions)} totales)")

        for i, p in enumerate(show):
            card = Frame(self.scroll_frame, bg="#16213e", bd=0, highlightbackground="#0f3460", highlightthickness=1)
            card.pack(fill="x", pady=4, padx=10)

            home_flag = TEAM_FLAGS.get(p["home_team"], "")
            away_flag = TEAM_FLAGS.get(p["away_team"], "")

            # Header row
            header = Frame(card, bg="#16213e")
            header.pack(fill="x", padx=15, pady=(8, 3))

            Label(header, text=f"{home_flag} {p['home_team']}", font=("Segoe UI", 12, "bold"),
                  fg="white", bg="#16213e").pack(side="left")
            Label(header, text=" vs ", font=("Segoe UI", 12), fg="#555", bg="#16213e").pack(side="left", padx=5)
            Label(header, text=f"{away_flag} {p['away_team']}", font=("Segoe UI", 12, "bold"),
                  fg="white", bg="#16213e").pack(side="left")

            # Actual result if available
            home_goals = p.get("home_goals")
            away_goals = p.get("away_goals")
            if home_goals is not None and away_goals is not None:
                hg = int(home_goals) if pd.notna(home_goals) else None
                ag = int(away_goals) if pd.notna(away_goals) else None
                if hg is not None and ag is not None:
                    result_color = "#00b894" if hg > ag else "#fdcb6e" if hg == ag else "#e94560"
                    result_label = Label(header, text=f"{hg}-{ag}", font=("Segoe UI", 14, "bold"),
                                         fg=result_color, bg="#16213e")
                    result_label.pack(side="right", padx=(0, 10))

            # Prediction badge
            pred_text = p.get("predicted", "?")
            conf = p.get("confidence", 0)
            badge_color = "#00b894" if conf > 0.6 else "#fdcb6e" if conf > 0.4 else "#e94560"
            badge = Label(header, text=f"{pred_text}  {conf:.0%}",
                          font=("Segoe UI", 10, "bold"), fg="white", bg=badge_color,
                          padx=10, pady=2)
            badge.pack(side="right", padx=(0, 5))

            # Date and league
            date_str = str(p.get("date", ""))[:10]
            league_str = str(p.get("source_league", ""))
            info_text = date_str
            if league_str and league_str != "nan":
                info_text += f" | {league_str}"
            if info_text.strip(" |"):
                Label(header, text=info_text, font=("Segoe UI", 7),
                      fg="#555", bg="#16213e").pack(side="right", padx=(0, 10))

            # Body
            body = Frame(card, bg="#16213e")
            body.pack(fill="x", padx=15, pady=(0, 8))

            # 1X2 bar
            probs = [
                ("Local", p.get("prob_home", 0), "#00b894"),
                ("Empate", p.get("prob_draw", 0), "#fdcb6e"),
                ("Visitante", p.get("prob_away", 0), "#e94560"),
            ]
            bar_frame = Frame(body, bg="#16213e")
            bar_frame.pack(fill="x", pady=3)
            for label, prob, color in probs:
                w = max(prob * 250, 0)
                if w > 0:
                    bar = Frame(bar_frame, bg=color, width=int(w), height=16)
                    bar.pack(side="left", padx=(0, 2))
                    Label(bar, text=f"{label} {prob:.0%}", font=("Segoe UI", 7, "bold"),
                          fg="white", bg=color).place(relx=0.5, rely=0.5, anchor="center")

            # Stats row
            stats = Frame(body, bg="#16213e")
            stats.pack(fill="x", pady=2)

            over = p.get("prob_over_2_5", 0)
            under = p.get("prob_under_2_5", 0)
            btts = p.get("prob_btts_yes", 0)
            no_btts = p.get("prob_btts_no", 0)

            stats_cols = Frame(stats, bg="#16213e")
            stats_cols.pack()
            for text, val in [
                (f"O2.5  {over:.0%}", over > 0.5),
                (f"BTTS  {btts:.0%}", btts > 0.5),
                (f"Under {under:.0%}", under > 0.5),
                (f"No BTTS {no_btts:.0%}", no_btts > 0.5),
            ]:
                c = "#00b894" if val else "#636e72"
                lbl = Label(stats_cols, text=text, font=("Segoe UI", 8, "bold"),
                            fg=c, bg="#16213e", padx=6)
                lbl.pack(side="left")

            # Match stats
            stats_key = f"{p['home_team']} vs {p['away_team']}"
            if hasattr(self, 'match_stats') and stats_key in self.match_stats:
                stats_data = self.match_stats[stats_key]
                if stats_data.get("home") or stats_data.get("away"):
                    stats_frame = Frame(body, bg="#0f3460")
                    stats_frame.pack(fill="x", pady=(3, 0), padx=5)
                    grid = Frame(stats_frame, bg="#0f3460")
                    grid.pack(padx=5, pady=2)
                    for j, (eng_key, span_label) in enumerate(STATS_LABELS.items()):
                        hv = stats_data["home"].get(eng_key, "-")
                        av = stats_data["away"].get(eng_key, "-")
                        if eng_key == "Ball Possession":
                            hv_str = f"{hv:.0f}%" if isinstance(hv, float) else str(hv)
                            av_str = f"{av:.0f}%" if isinstance(av, float) else str(av)
                        else:
                            hv_str = f"{hv:.1f}" if isinstance(hv, float) else str(hv)
                            av_str = f"{av:.1f}" if isinstance(av, float) else str(av)
                        Label(grid, text=span_label, font=("Segoe UI", 7),
                              fg="#ccc", bg="#0f3460", width=14, anchor="w").grid(row=j, column=0)
                        Label(grid, text=hv_str, font=("Segoe UI", 7, "bold"),
                              fg="#eee", bg="#0f3460", width=10, anchor="e").grid(row=j, column=1)
                        Label(grid, text=av_str, font=("Segoe UI", 7, "bold"),
                              fg="#eee", bg="#0f3460", width=10, anchor="e").grid(row=j, column=2)

            # Exact scores
            scores = p.get("exact_scores", [])
            if scores and scores[0].get("score") != "N/A":
                sc_frame = Frame(body, bg="#1a1a2e")
                sc_frame.pack(fill="x", pady=(2, 0))
                Label(sc_frame, text="\U0001F3AF Marcadores:",
                      font=("Segoe UI", 8), fg="#888", bg="#1a1a2e").pack(side="left", padx=(0, 5))
                for s in scores[:3]:
                    prob_pct = s["prob"] * 100
                    lbl_sc = Label(sc_frame,
                                   text=f"{s['score']}  ({prob_pct:.0f}%)",
                                   font=("Segoe UI", 8, "bold"), fg="#e94560", bg="#1a1a2e", padx=4)
                    lbl_sc.pack(side="left")

            i += 1
            if i % 50 == 0:
                self.root.update()

        self.root.update()


def main():
    app = PredictionApp()
    app.root.mainloop()


if __name__ == "__main__":
    main()
