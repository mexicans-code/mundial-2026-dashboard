"""Main entry point — run the full sports betting ML pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

from sports_ml.backtesting.simulator import Backtester
from sports_ml.data.collector import load_matches, NATIONAL_TEAMS
from sports_ml.features.builder import FeatureBuilder
from sports_ml.models.predictor import MatchPredictor
from sports_ml.models.stats_predictor import MatchStatsPredictor, STATS_LABELS
from sports_ml.models.trainer import ModelTrainer
from sports_ml.utils.config import Config


def resolve_match(match_arg: str) -> tuple[list[int], str, str]:
    """Parse 'Brasil vs Japon' into team IDs."""
    parts = match_arg.lower().split(" vs ")
    if len(parts) != 2:
        print(f"[ERROR] Formato invalido. Use: 'Brasil vs Japon'")
        sys.exit(1)
    t1, t2 = parts[0].strip(), parts[1].strip()
    if t1 not in NATIONAL_TEAMS or t2 not in NATIONAL_TEAMS:
        available = ", ".join(NATIONAL_TEAMS.keys())
        print(f"[ERROR] Equipo no encontrado. Opciones: {available}")
        sys.exit(1)
    return (
        [NATIONAL_TEAMS[t1]["id"], NATIONAL_TEAMS[t2]["id"]],
        NATIONAL_TEAMS[t1]["name"],
        NATIONAL_TEAMS[t2]["name"],
    )


def print_banner():
    print(r"""
   +----------------------------------------------+
   |      Sports Betting ML Pipeline v1.0         |
   |  Predict | Analyze | Win with Data           |
   +----------------------------------------------+
    """)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sports Betting ML Pipeline — predict match outcomes with machine learning."
    )
    parser.add_argument("--sport", choices=["football", "basketball"], default="football")
    parser.add_argument("--league-id", type=int, default=71, help="API-Football league ID")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--match", type=str, default="", help="Ej: 'Brasil vs Japon'")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of API")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward backtest")
    parser.add_argument("--predict", action="store_true", help="Predict upcoming matches")
    parser.add_argument("--from-csv", type=str, help="Load historical data from CSV file")
    parser.add_argument("--api-key", type=str, help="API-Football API key")
    parser.add_argument("--config", type=str, default="config.json", help="Path to JSON config file")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.sport = args.sport
    cfg.league_id = args.league_id
    cfg.season = args.season

    if args.api_key:
        cfg.api.football_api_key = args.api_key

    # Auto-load config.json if exists
    config_path = args.config or "config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            overrides = json.load(f)
        for section, values in overrides.items():
            if hasattr(cfg, section):
                for k, v in values.items():
                    if hasattr(getattr(cfg, section), k):
                        setattr(getattr(cfg, section), k, v)

    return cfg


def print_summary(metrics: dict, title: str = "Training Summary"):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:25s}: {v:.4f}")
        elif isinstance(v, list):
            print(f"    {k:25s}: {v[:3]}... ({len(v)} total)")
        else:
            print(f"    {k:25s}: {v}")
    print(f"{'=' * 60}\n")


def get_match_analysis(home: str, away: str) -> list[str]:
    """Return human analysis lines specific to the match being predicted."""
    match_key = f"{home.lower()} vs {away.lower()}"

    analysis_map = {
        "france vs sweden": [
            "Francia: mejor ataque del torneo (10 goles en 3 partidos)",
            "Mbappe (4) + Dembele (4) en racha",
            "Suecia: defensa fragil (7 goles encajados en 3 partidos)",
            "KO factor: Danny Makkelie arbitra, pocas tarjetas (~3/partido)",
            "Modelo (3846 partidos): Local 75% — confianza alta en Francia",
            "Mi pick: Francia gana | Over 2.5 | France -1.5",
        ],
        "mexico vs ecuador": [
            "Ecuador: solido defensivamente, 2 goles encajados en 3 partidos",
            "Mexico: irregular, 4 goles en 3 partidos, dependen de Vega",
            "Modelo (3846 partidos): Empate 55% con alto valor (EV +91%)",
            "Probable partido tactico, pocos goles",
            "Mi pick: Empate o Ecuador DC | Under 2.5 | BTTS No",
        ],
        "england vs congo dr": [
            "Inglaterra favorita 52% pero Congo DR tiene 48% combinado",
            "Arbitro: Adham Makhadmeh (Jordania) — ~3.5 tarjetas, moderado",
            "Congo DR: defensa africana solida (14 faltas/partido)",
            "Partido KO: estudio primeros 45', 1-0 o empate",
            "Mi pick: DC Inglaterra | Under 2.5 | BTTS No | U4.5 tarjetas",
        ],
        "belgium vs senegal": [
            "Belgica 54% favorita vs Senegal fisico",
            "Arbitro: Said Martinez (Honduras) — CONCACAF, ~3 tarjetas",
            "Senegal: 14 faltas/partido, Belgica 6.7 — total ~21",
            "Belgica sin generacion dorada pero con De Bruyne + Lukaku",
            "Mi pick: DC Belgica | Over 2.5 | BTTS Yes | O3.5 tarjetas",
        ],
        "usa vs bosnia & herzegovina": [
            "Modelo: USA 39/29/32 Bosnia — MUY parejo, value enorme",
            "Arbitro: Raphael Claus (Brasil) — sudamericano ~4 tarjetas",
            "USA en casa (Levi's), Bosnia sin miedo — debut en KO",
            "Bosnia viene de ganarle a Qatar, eliminar a Italia en repechaje",
            "Mi pick: DC Bosnia | BTTS Yes | Over 2.5 | O4.5 tarjetas",
        ],
    }

    default = [
        "Analisis humano pendiente para este partido",
        f"Modelo: {home} vs {away} — revisar cuotas antes de apostar",
    ]

    return analysis_map.get(match_key, default)


def main():
    print_banner()
    args = parse_args()
    config = load_config(args)

    use_mock = args.mock or not config.api.football_api_key

    print(f"[SPORT]  {config.sport.upper()}")
    print(f"[SEASON] {config.season}")
    print(f"[API]    {'OK' if config.api.football_api_key else 'MOCK (no key)'}")
    print()

    # If --match is set, resolve teams and fetch their data
    if args.match:
        team_ids, home_name, away_name = resolve_match(args.match)
        print(f"[MATCH] {home_name} vs {away_name}")
        config.team_ids = team_ids
        config.home_team_name = home_name
        config.away_team_name = away_name

    # --- Load Data ---
    if args.from_csv and os.path.exists(args.from_csv):
        print(f"[CSV]    Loading: {args.from_csv}")
        full = pd.read_csv(args.from_csv)
        full["date"] = pd.to_datetime(full["date"])
        df_finished = full[full["home_goals"].notna()].copy()
        df_upcoming = full[full["home_goals"].isna()].copy()
        print(f"     Finished: {len(df_finished)} | Upcoming: {len(df_upcoming)}")
    else:
        df_finished, df_upcoming = load_matches(config, use_mock=use_mock)
        if not df_finished.empty:
            print(f"     Finished: {len(df_finished)} | Upcoming: {len(df_upcoming)}")

    if df_finished.empty:
        print("[ERROR] No data available. Provide --api-key, --mock, or --from-csv.")
        sys.exit(1)

    # --- Feature Engineering ---
    print("\n[BUILD] Building features...")
    feature_builder = FeatureBuilder(config)
    df_features = feature_builder.build(df_finished)
    print(f"     Features: {df_features.shape[1]} columns, {len(df_features)} rows")

    # Derive target columns if needed
    if "home_win" not in df_features.columns:
        df_features["home_win"] = (
            df_features["home_goals"] > df_features["away_goals"]
        ).astype(int)
    if "draw" not in df_features.columns:
        df_features["draw"] = (
            df_features["home_goals"] == df_features["away_goals"]
        ).astype(int)
    if "away_win" not in df_features.columns:
        df_features["away_win"] = (
            df_features["away_goals"] > df_features["home_goals"]
        ).astype(int)
    if "total_goals" not in df_features.columns:
        df_features["total_goals"] = df_features["home_goals"] + df_features["away_goals"]
    if "over_2_5" not in df_features.columns:
        df_features["over_2_5"] = (df_features["total_goals"] > 2.5).astype(int)
    if "btts" not in df_features.columns:
        df_features["btts"] = (
            (df_features["home_goals"] > 0) & (df_features["away_goals"] > 0)
        ).astype(int)

    # Combined 1X2 target
    df_features["result"] = (
        df_features["home_win"] * 0 +
        df_features["draw"] * 1 +
        df_features["away_win"] * 2
    )

    # --- Train Models ---
    trainer = ModelTrainer(config)

    for target in ["1X2", "over_under", "btts"]:
        try:
            metrics = trainer.train(df_features, target_key=target)
            print_summary(metrics, title=f"Model: {target}")
        except Exception as e:
            print(f"  [SKIP] {target}: {e}")

    # Save trained models
    model_path = trainer.save()

    # --- Backtest ---
    if args.backtest:
        print("\n[TEST] Running backtest...")
        backtester = Backtester(config, trainer)
        bt_results = backtester.simulate(df_finished)
        print_summary(bt_results, title="Backtest Results")

    # --- Predict ---
    if args.predict:
        print("\n[PREDICT] Match prediction...")
        predictor = MatchPredictor(trainer, config)

        # If --match is set, create a synthetic matchup if no upcoming
        if args.match:
            match_home = config.home_team_name
            match_away = config.away_team_name

            if not df_upcoming.empty:
                target = df_upcoming[
                    (df_upcoming["home_team"].str.contains(match_home, case=False, na=False) &
                     df_upcoming["away_team"].str.contains(match_away, case=False, na=False)) |
                    (df_upcoming["home_team"].str.contains(match_away, case=False, na=False) &
                     df_upcoming["away_team"].str.contains(match_home, case=False, na=False))
                ]
                if not target.empty:
                    df_upcoming = target

            if df_upcoming.empty:
                print(f"  [INFO] Creando prediccion sintetica: {match_home} vs {match_away}")
                synthetic = pd.DataFrame([{
                    "fixture_id": 0,
                    "date": datetime.now().replace(tzinfo=None),
                    "home_team": match_home,
                    "away_team": match_away,
                    "home_goals": None,
                    "away_goals": None,
                    "home_team_id": NATIONAL_TEAMS.get(match_home.lower(), {}).get("id", 0),
                    "away_team_id": NATIONAL_TEAMS.get(match_away.lower(), {}).get("id", 0),
                    "status_short": "NS",
                }])
                df_upcoming = synthetic

        if df_upcoming.empty:
            print("  [SKIP] No hay partidos para predecir.")
        else:
            predictions = predictor.predict_batch(df_upcoming, df_finished)

            if predictions:
                print()
                sep = "-" * 80
                print(sep)
                print(f"  {'LOCAL':20s} {'VISITANTE':20s} {'PREDICCION':15s} {'CONF':8s} {'1X2':20s} {'O2.5':8s} {'BTTS':8s}")
                print(sep)
                for p in predictions:
                    home = p["home_team"] or ""
                    away = p["away_team"] or ""
                    pred = p.get("predicted", "?")
                    conf = f'{p.get("confidence", 0):.0%}'
                    probs = f'{p.get("prob_home", 0):.0%}/{p.get("prob_draw", 0):.0%}/{p.get("prob_away", 0):.0%}'
                    o25 = f'{p.get("prob_over_2_5", 0):.0%}'
                    btts = f'{p.get("prob_btts_yes", 0):.0%}'
                    print(f"  {home:20s} {away:20s} {pred:15s} {conf:8s} {probs:20s} {o25:8s} {btts:8s}")

                    # Exact scores
                    scores = p.get("exact_scores", [])
                    if scores and scores[0].get("score") != "N/A":
                        scores_str = " | ".join(f'{s["score"]} ({s["prob"]:.0%})' for s in scores[:3])
                        print(f"  {'':>20s} {'':>20s} {'MARC. EXACTO':<15s} {scores_str}")

                    # Match stats prediction
                    if args.match:
                        stats_predictor = MatchStatsPredictor(config)
                        stats = stats_predictor.predict_stats(home, away, df_finished)
                        if stats.get("home") or stats.get("away"):
                            print(f"  {'-ESTADISTICAS':->80s}")
                            print(f"  {'':20s} {'':20s} {'':10s} {home:20s} {away:20s}")
                            print(f"  {'':20s} {'':20s} {'':10s} {'---':>20s} {'---':>20s}")
                            for eng_key, span_label in STATS_LABELS.items():
                                hv = stats["home"].get(eng_key, "-")
                                av = stats["away"].get(eng_key, "-")
                                if eng_key == "Ball Possession":
                                    hv_str = f"{hv:.0f}%" if isinstance(hv, float) else str(hv)
                                    av_str = f"{av:.0f}%" if isinstance(av, float) else str(av)
                                else:
                                    hv_str = f"{hv:.1f}" if isinstance(hv, float) else str(hv)
                                    av_str = f"{av:.1f}" if isinstance(av, float) else str(av)
                                print(f"  {'':20s} {'':20s} {span_label:<15s} {hv_str:>20s} {av_str:>20s}")
                            note = stats.get("note", "")
                            if note:
                                print(f"  {'':>20s} {'':>20s} {'':10s} ({note})")

                    # Human analysis
                    if args.match:
                        analysis_lines = get_match_analysis(home, away)
                        print(f"  {'-MI ANALISIS':->80s}")
                        for line in analysis_lines:
                            print(f"  {'':20s} {'':20s} {line:<55s}")
                print(sep)

                mock_odds = {}
                for p in predictions:
                    key = f"{p['home_team']} vs {p['away_team']}"
                    probs = [p.get("prob_home", 0.33), p.get("prob_draw", 0.33), p.get("prob_away", 0.33)]
                    fair_odds = [1 / max(pr, 0.01) for pr in probs]
                    mock_odds[key] = {"home": round(fair_odds[0] * 0.95, 2),
                                       "draw": round(fair_odds[1] * 0.95, 2),
                                       "away": round(fair_odds[2] * 0.95, 2)}

                value_bets = predictor.find_value_bets(predictions, mock_odds)
                if value_bets:
                    print(f"\n[VALUE BETS] ({len(value_bets)} found):")
                    for vb in value_bets:
                        print(f"     {vb['match']:40s} -> {vb['prediction']:10s} "
                              f"Prob: {vb['model_prob']:.0%} | Odds: {vb['market_odds']:.2f} | "
                              f"EV: {vb['expected_value']:+.1%} | Kelly: {vb['kelly_stake']:.1%}")

    print(f"\n[DONE] Model saved at: {model_path}")
    print("[DISCLAIMER] Solo fines educativos. Juega con responsabilidad.\n")


if __name__ == "__main__":
    main()
