"""
pregame_ingest.py — Ingesta EXCLUSIVA de datos reales.

Fuentes:
  - pybaseball (Baseball Savant / FanGraphs / MLB): abridores, bullpens, lineups,
    estadísticas de pitcheo/ofensiva, Statcast por lanzamiento.
  - The Odds API (requests): cuotas prepartido, líneas de totales y moneylines.

Nada de este módulo genera datos sintéticos. Cada función devuelve DataFrames
reales con timestamp para garantizar la congelación prepartido (data freeze).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

import pybaseball as pb
from pybaseball import (cache, pitching_stats, batting_stats, team_batting,
                        team_pitching, statcast, standings)

from config import (DATA_DIR, FREEZE_DATA, MINUTES_BEFORE_FIRST_PITCH,
                    ODD_API_PLACEHOLDER, ODD_API_PLACEHOLDER)  # noqa

CACHE_DIR = DATA_DIR / "cache"; CACHE_DIR.mkdir(exist_ok=True)
cache.enable()


# ============================================================
# SECCIÓN 1 — INGESTA DE ESTADÍSTICAS ACUMULADAS (pybaseball)
# ============================================================

def fetch_pitcher_season_stats(season: int) -> pd.DataFrame:
    """Estadísticas de pitcheo acumuladas de la temporada (FanGraphs vía pybaseball).

    Devuelve FIP, SIERA, xERA, K-BB% por lanzador — columnas usadas por el módulo
    de pitcheo (pregame_features.agg_pitching_features).
    """
    return pitching_stats(season, qual=1)


def fetch_batter_season_stats(season: int) -> pd.DataFrame:
    """wOBA, ISO, wRC+ acumulados de temporada por bateador."""
    return batting_stats(season, qual=1)


def fetch_statcast_window(start: str, end: str) -> pd.DataFrame:
    """Statcast por lanzamiento en una ventana [start, end] ('YYYY-MM-DD').

    Se usa para: velocidad promedio (release_speed) y spin rate (release_spin_rate)
    recientes del abridor, y workload de bullpen (pitches thrown).
    """
    return statcast(start_dt=start, end_dt=end, verbose=False)


def fetch_team_records(season: int) -> pd.DataFrame:
    """Standings de la temporada (para contexto y features de equipo)."""
    return standings(season)


# ============================================================
# SECCIÓN 2 — ABRIDOR CONFIRMADO Y LINEUP (API MLB vía pybaseball)
# ============================================================

def fetch_probable_pitchers(game_date: str) -> pd.DataFrame:
    """Abridores probables/confirmados por juego del día.

    game_date: 'YYYY-MM-DD'. Fuente: endpoint schedule de MLB
    (https://statsapi.mlb.com/api/v1/schedule) — API pública oficial de MLB.
    """
    url = ("https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&date={game_date}&hydrate=probablePitcher,team")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    games = []
    for date_block in resp.json().get("dates", []):
        for g in date_block.get("games", []):
            games.append({
                "game_id": g["gamePk"],
                "game_date": game_date,
                "home_team": g["teams"]["home"]["team"]["name"],
                "away_team": g["teams"]["away"]["team"]["name"],
                "home_team_id": g["teams"]["home"]["team"]["id"],
                "away_team_id": g["teams"]["away"]["team"]["id"],
                "home_pitcher_id": (g["teams"]["home"]
                                    .get("probablePitcher", {}).get("id")),
                "away_pitcher_id": (g["teams"]["away"]
                                    .get("probablePitcher", {}).get("id")),
                "home_pitcher": (g["teams"]["home"]
                                 .get("probablePitcher", {}).get("fullName")),
                "away_pitcher": (g["teams"]["away"]
                                 .get("probablePitcher", {}).get("fullName")),
                "game_time_utc": g.get("gameDate"),
            })
    return pd.DataFrame(games)


def fetch_lineup_confirmed(game_id: int) -> pd.DataFrame:
    """Lineup titular confirmado de un juego (boxscore oficial de MLB).

    Devuelve orden al bate, mano (L/R/S) y player_id por equipo. Se filtra por
    'battingOrder' no nulo = lineup titular real confirmado.
    """
    url = f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    box = resp.json()
    rows = []
    for side in ("home", "away"):
        team = box["teams"][side]
        team_name = team["team"]["name"]
        for pid, player in team["players"].items():
            if player.get("battingOrder") is None:
                continue
            rows.append({
                "game_id": game_id,
                "team": team_name,
                "side": side,
                "player_id": int(pid.split("ID")[-1]),
                "player_name": player["person"]["fullName"],
                "batting_order": int(player["battingOrder"]),
                "bat_side": player.get("batSide", {}).get("code"),
                "position": player.get("position", {}).get("abbreviation"),
            })
    return pd.DataFrame(rows).sort_values(["team", "batting_order"])


# ============================================================
# SECCIÓN 3 — THE ODDS API (cuotas reales de cierre prepartido)
# ============================================================

def fetch_odds(sport_key: str, api_key: str, regions: str = "us,us2",
               markets: str = "h2h,spreads,totals") -> pd.DataFrame:
    """Cuotas prepartido reales desde The Odds API.

    Mercados: h2h (moneyline), totals (over/under de carreras).
    Incluye el timestamp de extracción para auditoría del cierre de línea.
    """
    url = (f"{ODDS_API_BASE}/sports/{sport_key}/odds/"
           f"?apiKey={api_key}&regions={regions}&markets={markets}"
           "&oddsFormat=american&dateFormat=iso")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    fetched_at = datetime.now(timezone.utc).isoformat()

    rows = []
    for event in resp.json():
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        rows.append({
                            "fetched_at": fetched_at,
                            "event_id": event["id"],
                            "home_team": event["home_team"],
                            "away_team": event["away_team"],
                            "commence_time": event["commence_time"],
                            "book": bookmaker["title"],
                            "market": "total",
                            "label": outcome["name"],
                            "line": outcome.get("point"),
                            "price": outcome["price"],
                        })
                elif market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        rows.append({
                            "fetched_at": fetched_at,
                            "event_id": event["id"],
                            "home_team": event["home_team"],
                            "away_team": event["away_team"],
                            "commence_time": event["commence_time"],
                            "book": bookmaker["title"],
                            "market": "moneyline",
                            "label": outcome["name"],
                            "line": None,
                            "price": outcome["price"],
                        })
    return pd.DataFrame(rows)


def consensus_line(odds_df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Consenso de línea: mediana de punto y precio entre casas de apuestas."""
    df = odds_df[odds_df["market"] == market]
    grouped = df.groupby(["event_id", "label"], as_index=False).agg(
        median_line=("line", "median"),
        median_price=("price", "median"),
        n_books=("book", "nunique"),
    )
    return grouped


# ============================================================
# SECCIÓN 4 — FREEZE PREPARTIDO Y ORQUESTACIÓN DIARIA
# ============================================================

def game_freeze_time(commence_time_utc: str) -> datetime:
    """Instante en que el pipeline debe ejecutarse y congelar datos:
    N minutos antes del primer lanzamiento."""
    dt = datetime.fromisoformat(commence_time_utc.replace("Z", "+00:00"))
    return dt - timedelta(minutes=MINUTES_BEFORE_FIRST_PITCH)


def run_daily_ingest(game_date: str, odds_api_key: str) -> dict[str, Path]:
    """Orquesta la ingesta completa del día y la CONGELA en disco (snapshot).

    Retorna rutas a los parquets congelados. Una vez ejecutado, los datos no
    cambian aunque se re-corra (FREEZE_DATA=True evita sobreescritura).
    """
    snap = DATA_DIR / f"snapshot_{game_date}"
    snap.mkdir(exist_ok=True)
    paths: dict[str, Path] = {}

    def _save(df: pd.DataFrame, name: str) -> None:
        p = snap / f"{name}.parquet"
        if FREEZE_DATA and p.exists():
            return  # congelado: no se toca
        df.to_parquet(p, index=False)
        paths[name] = p

    season = int(game_date[:4])

    # 1. Abridores probables/confirmados
    games = fetch_probable_pitchers(game_date)
    _save(games, "games")

    # 2. Lineups confirmados (solo si el juego está dentro de la ventana de freeze)
    if not games.empty:
        lineups = pd.concat(
            [fetch_lineup_confirmed(gid) for gid in games["game_id"]],
            ignore_index=True)
        _save(lineups, "lineups")

    # 3. Estadísticas de temporada
    _save(fetch_pitcher_season_stats(season), "pitcher_season")
    _save(fetch_batter_season_stats(season), "batter_season")

    # 4. Statcast de los últimos 14 días (velocidad/spin reciente + bullpen)
    start = (datetime.fromisoformat(game_date) - timedelta(days=14)).strftime("%Y-%m-%d")
    _save(fetch_statcast_window(start, game_date), "statcast_14d")

    # 5. Bullpen de los últimos 5 días (carga de trabajo)
    bp_start = (datetime.fromisoformat(game_date) - timedelta(days=5)).strftime("%Y-%m-%d")
    _save(fetch_statcast_window(bp_start, game_date), "statcast_bullpen")

    # 6. Cuotas reales (The Odds API)
    if odds_api_key:
        odds = fetch_odds(SPORT_KEY, odds_api_key)
        _save(odds, "odds")
        _save(consensus_line(odds, "totals"), "consensus_totals")
        _save(consensus_line(odds, "moneyline"), "consensus_ml")

    print(f"[ingest] Snapshot congelado en {snap}")
    return paths
