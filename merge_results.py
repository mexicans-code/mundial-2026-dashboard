"""Merge results.csv into national_teams_data.csv format."""
import csv
import hashlib
import math
import os
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(__file__))
OUR_CSV = os.path.join(BASE, "national_teams_data.csv")
NEW_CSV = r"C:\Users\Ricardo Medina\Downloads\results.csv"

TEAM_MAP = {
    "United States": "USA",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "DR Congo": "Congo DR",
    "Turkey": "Turkiye",
    "Czech Republic": "Czech Rep",
    "Cape Verde Islands": "Cape Verde",
}

TOURNAMENT_LEAGUE_MAP = {
    "FIFA World Cup": 1,
    "Friendly": 10,
    "African Cup of Nations": 9,
    "Africa Cup of Nations": 9,
    "AFC Asian Cup": 26,
    "AFC Challenge Cup": 26,
    "AFC Solidarity Cup": 26,
    "African Nations Championship": 9,
    "Algarve Cup": 10,
    "America Cup": 2,
    "Arab Cup": 10,
    "AFF Championship": 26,
    "Armenia Cup": 10,
    "Baltic Cup": 10,
    "Bangabandhu Cup": 10,
    "Caribbean Cup": 31,
    "Caribbean Shield": 31,
    "CECAFA Cup": 10,
    "CFU Caribbean Cup": 31,
    "China Cup": 10,
    "CIS Cup": 10,
    "CONCACAF Championship": 31,
    "CONCACAF Gold Cup": 31,
    "CONCACAF Nations League": 136,
    "Confederations Cup": 7,
    "Copa America": 2,
    "Copa del Pacifico": 10,
    "Copa America": 2,
    "COSAFA Cup": 10,
    "Cyprus Cup": 10,
    "EAFF E-1 Football Championship": 26,
    "Eco-Asia Cup": 10,
    "Euro 2008": 4,
    "Euro 2012": 4,
    "Euro 2016": 4,
    "Euro 2020": 4,
    "Euro 2024": 4,
    "Euro 2028": 4,
    "European Championship": 4,
    "FIFA World Cup qualification": 29,
    "FIFA World Cup qualifier": 29,
    "FIFA World Cup Qualifier": 29,
    "Friendly": 10,
    "Gold Cup": 31,
    "Gulf Cup": 10,
    "Hero Cup": 10,
    "Indian Ocean Games": 10,
    "Intercontinental Cup": 10,
    "International Friendly": 10,
    "King's Cup": 10,
    "Kirin Cup": 10,
    "Malaysia Cup": 10,
    "Marianas Cup": 10,
    "Mauritius Cup": 10,
    "MENA Cup": 10,
    "Merlion Cup": 10,
    "Millennium Cup": 10,
    "MLS All-Star": 10,
    "Nations Cup": 10,
    "NBA Cup": 10,
    "Nehru Cup": 10,
    "OFC Nations Cup": 29,
    "Olympic Games": 10,
    "Pan American Games": 10,
    "Philippines Cup": 10,
    "SAFF Championship": 10,
    "Saudi Cup": 10,
    "Shanghai Cup": 10,
    "SheBelieves Cup": 10,
    "Simba Cup": 10,
    "Sri Lanka Cup": 10,
    "Sudan Cup": 10,
    "Super Cup": 10,
    "Tahiti Cup": 10,
    "Thomas Cup": 10,
    "Torneo de Verano": 10,
    "Tournament of Nations": 10,
    "Tri-Nation Series": 10,
    "Trinidad Cup": 10,
    "Tunisia Cup": 10,
    "Turkmenistan Cup": 10,
    "UAE Cup": 10,
    "UEFA Nations League": 5,
    "UEFA Euro": 4,
    "UEFA Euro qualifying": 32,
    "UEFA European Championship": 4,
    "Western Asian Games": 10,
    "WAFU Cup": 10,
    "World Cup": 1,
    "World Cup Qualifier": 29,
    "World Cup qualification": 29,
}

SOURCE_LEAGUE_MAP = {
    "FIFA World Cup": "World Cup %s",
    "FIFA World Cup qualification": "World Cup - Qualification %s",
    "Friendly": "Friendlies %s",
    "International Friendly": "Friendlies %s",
    "African Cup of Nations": "Africa Cup of Nations %s",
    "African Cup of Nations qualification": "Africa Cup of Nations - Qualification %s",
    "CONCACAF Gold Cup": "CONCACAF Gold Cup %s",
    "CONCACAF Gold Cup qualification": "CONCACAF Gold Cup - Qualification %s",
    "CONCACAF Nations League": "CONCACAF Nations League %s",
    "CONCACAF Nations League qualification": "CONCACAF Nations League - Qualification %s",
    "Copa América": "Copa America %s",
    "UEFA Euro": "Euro %s",
    "UEFA European Championship": "Euro %s",
    "UEFA Euro qualification": "Euro - Qualification %s",
    "UEFA Nations League": "UEFA Nations League %s",
    "AFC Asian Cup": "AFC Asian Cup %s",
    "African Nations Championship": "African Nations Championship %s",
    "CONCACAF Championship": "CONCACAF Championship %s",
    "CONCACAF Championship qualification": "CONCACAF Championship - Qualification %s",
    "CFU Caribbean Cup": "CFU Caribbean Cup %s",
    "CFU Caribbean Cup qualification": "CFU Caribbean Cup - Qualification %s",
    "Copa América qualification": "Copa America - Qualification %s",
    "Confederations Cup": "Confederations Cup %s",
    "Gold Cup": "Gold Cup %s",
    "Gold Cup qualification": "Gold Cup - Qualification %s",
    "Oceania Nations Cup": "Oceania Nations Cup %s",
    "Oceania Nations Cup qualification": "Oceania Nations Cup - Qualification %s",
    "Olympic Games": "Olympic Games %s",
    "Pan American Games": "Pan American Games %s",
    "FIFA Series": "FIFA Series %s",
    "Asian Games": "Asian Games %s",
    "AFF Championship": "AFF Championship %s",
    "AFF Championship qualification": "AFF Championship - Qualification %s",
    "ASEAN Championship": "ASEAN Championship %s",
    "EAFF Championship": "EAFF Championship %s",
    "SAFF Cup": "SAFF Cup %s",
    "WAFF Championship": "WAFF Championship %s",
    "CECAFA Cup": "CECAFA Cup %s",
    "COSAFA Cup": "COSAFA Cup %s",
    "Gulf Cup": "Gulf Cup %s",
    "Arab Cup": "Arab Cup %s",
    "CAFA Nations Cup": "CAFA Nations Cup %s",
    "UNCAF Cup": "UNCAF Cup %s",
    "Pacific Games": "Pacific Games %s",
    "Pacific Mini Games": "Pacific Mini Games %s",
    "Bolivarian Games": "Bolivarian Games %s",
    "Central American and Caribbean Games": "Central American and Caribbean Games %s",
    "Far Eastern Championship Games": "Far Eastern Championship Games %s",
    "South Asian Games": "South Asian Games %s",
    "Southeast Asian Games": "Southeast Asian Games %s",
    "Caribbean Cup": "Caribbean Cup %s",
    "Caribbean Shield": "Caribbean Shield %s",
    "Copa del Pacífico": "Copa del Pacifico %s",
    "Superclásico de las Américas": "Superclasico de las Americas %s",
    "Copa Artigas": "Copa Artigas %s",
    "Copa Bernardo O'Higgins": "Copa Bernardo O'Higgins %s",
    "Copa Carlos Dittborn": "Copa Carlos Dittborn %s",
    "Copa Chevallier Boutell": "Copa Chevallier Boutell %s",
    "Copa Newton": "Copa Newton %s",
    "Copa Paz del Chaco": "Copa Paz del Chaco %s",
    "Copa Rio Branco": "Copa Rio Branco %s",
    "Copa Roca": "Copa Roca %s",
    "Copa Lipton": "Copa Lipton %s",
    "Copa Oswaldo Cruz": "Copa Oswaldo Cruz %s",
    "Copa Premio Honor Argentino": "Copa Premio Honor Argentino %s",
    "Copa Premio Honor Uruguayo": "Copa Premio Honor Uruguayo %s",
    "Copa Ramón Castilla": "Copa Ramon Castilla %s",
    "Copa Juan Pinto Durán": "Copa Juan Pinto Duran %s",
    "Copa Félix Bogado": "Copa Felix Bogado %s",
    "Copa Confraternidad": "Copa Confraternidad %s",
    "King's Cup": "Kings Cup %s",
    "Kirin Cup": "Kirin Cup %s",
    "Kirin Challenge Cup": "Kirin Challenge Cup %s",
    "Merdeka Tournament": "Merdeka Tournament %s",
    "Merlion Cup": "Merlion Cup %s",
    "Nehru Cup": "Nehru Cup %s",
    "Peace Cup": "Peace Cup %s",
    "China Cup": "China Cup %s",
    "Cyprus International Tournament": "Cyprus International Tournament %s",
    "Malta International Tournament": "Malta International Tournament %s",
    "Tournoi de France": "Tournoi de France %s",
    "Mundialito": "Mundialito %s",
    "British Home Championship": "British Home Championship %s",
    "Nordic Championship": "Nordic Championship %s",
    "Baltic Cup": "Baltic Cup %s",
    "Balkan Cup": "Balkan Cup %s",
}


def make_fixture_id(date_str, home, away):
    raw = f"{date_str}_{home}_{away}"
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    return int(h, 16) % (2**31 - 1) * -1


def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def map_team(name):
    name = name.strip()
    return TEAM_MAP.get(name, name)


def get_league_id(tournament):
    t = tournament.strip()
    return TOURNAMENT_LEAGUE_MAP.get(t, 10)


def get_source_league(tournament, date_str):
    t = tournament.strip()
    year = date_str[:4]
    tmpl = SOURCE_LEAGUE_MAP.get(t, None)
    if tmpl:
        return tmpl % year
    # For unmapped tournaments, use name + year
    name = t[:30] if len(t) > 30 else t
    return f"{name} {year}"


def get_season(tournament, date_str):
    year = date_str[:4]
    return year


def get_round(tournament, date_str):
    return tournament.strip()


def main():
    rows = []
    total = 0
    skipped = 0

    with open(NEW_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            date_str = row["date"].strip()
            tournament = row["tournament"].strip()
            home = map_team(row["home_team"])
            away = map_team(row["away_team"])
            home_score = row["home_score"].strip()
            away_score = row["away_score"].strip()
            city = row["city"].strip()
            country = row["country"].strip()
            neutral = row.get("neutral", "FALSE").strip().upper() == "TRUE"

            if home_score in ("NA", "") or away_score in ("NA", ""):
                status = "NS"
            else:
                status = "FT"
                try:
                    int(home_score)
                except ValueError:
                    continue

            try:
                dt = parse_date(date_str)
                ts = dt.timestamp()
            except Exception:
                ts = 0

            fixture_id = make_fixture_id(date_str, home, away)
            league_id = get_league_id(tournament)
            season = date_str[:4]

            rows.append({
                "fixture_id": fixture_id,
                "date": f"{date_str} 00:00:00+00:00",
                "timestamp": ts,
                "status_short": status,
                "league_id": league_id,
                "league_season": season,
                "league_round": get_round(tournament, date_str),
                "home_team_id": 0,
                "home_team": home,
                "away_team_id": 0,
                "away_team": away,
                "home_goals": home_score if status == "FT" else "",
                "away_goals": away_score if status == "FT" else "",
                "venue_city": city,
                "venue_name": f"{city} Stadium" if city else "",
                "source_league": get_source_league(tournament, date_str),
            })

    # Read existing CSV
    existing_dates = set()
    existing_rows = []
    if os.path.exists(OUR_CSV):
        with open(OUR_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                dt = row["date"][:10] if row["date"] else ""
                key = (dt, row["home_team"].strip(), row["away_team"].strip())
                existing_dates.add(key)
                existing_rows.append(row)
    else:
        fieldnames = [
            "fixture_id", "date", "timestamp", "status_short", "league_id",
            "league_season", "league_round", "home_team_id", "home_team",
            "away_team_id", "away_team", "home_goals", "away_goals",
            "venue_city", "venue_name", "source_league",
        ]

    # Merge: add new rows not in existing
    added = 0
    for row in rows:
        dt = row["date"][:10]
        key = (dt, row["home_team"], row["away_team"])
        if key not in existing_dates:
            existing_rows.append(row)
            added += 1
            existing_dates.add(key)

    with open(OUR_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)

    print(f"New results.csv rows: {total}")
    print(f"Skipped (no score): {skipped}")
    print(f"Added to our CSV: {added}")
    print(f"Total in our CSV now: {len(existing_rows)}")


if __name__ == "__main__":
    main()
