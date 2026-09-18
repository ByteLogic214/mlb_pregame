"""
pregame_historical_odds.py — Construye el dataset de entrenamiento de cuotas
desde el endpoint /historical de The Odds API (líneas de cierre REALES,
timestamp real por extracción).

IMPORTANTE — cuota de API: el endpoint /historical consume 10 créditos por
request (1 por día). Una temporada (~190 días) ≈ 1,900 créditos. Con el plan
gratuito (500/mes) NO alcanza para 4 temporadas de una vez; estrategia:
  a) Entrenar en lotes mensuales (reentrenamiento diario va acumulando).
  b) Alternativa gratuita: las líneas de cierre se persisten DÍA A DÍA en
     run_daily_ingest() — con 1 temporada de operación acumulas el dataset
     real sin gastar créditos históricos.

Este módulo existe para el backfill inicial; el reentrenamiento diario usa
la opción (b).
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import requests

from config import DATA_DIR, ODDS_API_BASE, SPORT_KEY

HIST_DIR = DATA_DIR / "historical_odds"; HIST_DIR.mkdir(parents=True, exist_ok=True)


def fetch_historical_odds_day(day: date, api_key: str,
                              markets: str = "h2h,totals") -> pd.DataFrame:
    """Cuotas de cierre de un día histórico REAL (The Odds API /historical).

    Retorna líneas con timestamp del snapshot usado por la casa (pre-cierre).
    """
    url = (f"{ODDS_API_BASE}/historical/sports/{SPORT_KEY}/odds/"
           f"?apiKey={api_key}&date={day.isoformat()}&markets={markets}"
           "&regions=us,us2&oddsFormat=american&dateFormat=iso")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    rows = []
    for event in payload.get("data", []):
        for book in event.get("bookmakers", []):
            for mkt in book.get("markets", []):
                for out in mkt.get("outcomes", []):
                    rows.append({
                        "snapshot_date": day.isoformat(),
                        "commence_time": event.get("commence_time"),
                        "home_team": event["home_team"],
                        "away_team": event["away_team"],
                        "book": book["title"],
                        "market": mkt["key"],
                        "label": out["name"],
                        "line": out.get("point"),
                        "price": out["price"],
                    })
    return pd.DataFrame(rows)


def backfill_historical(start: str, end: str, api_key: str,
                        sleep_s: int = 1) -> pd.DataFrame:
    """Backfill por rango de fechas. Guarda parquet por mes (idempotente)."""
    days = pd.date_range(start, end, freq="D")
    out = []
    for d in days:
        p = HIST_DIR / f"{d.strftime('%Y-%m')}.parquet"
        month_done = p.exists() and d.strftime("%Y-%m") in str(p)
        if p.exists() and d == days[-1]:
            continue
        df = fetch_historical_odds_day(d.date(), api_key)
        out.append(df)
        time.sleep(sleep_s)  # respeta rate limit
        print(f"[hist] {d.date()} — {len(df)} registros")
    if not out:
        return pd.DataFrame()
    full = pd.concat(out, ignore_index=True)
    for month, chunk in full.groupby(full["snapshot_date"].str[:7]):
        mp = HIST_DIR / f"{month}.parquet"
        if mp.exists():
            chunk = pd.concat([pd.read_parquet(mp), chunk], ignore_index=True)
        chunk.to_parquet(mp, index=False)
    return full


def closing_consensus_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Consenso de cierre por juego/día: último precio mediano antes del inicio.

    Merge clave con el dataset de features: por fecha + equipos.
    """
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["commence_time"]).dt.date.astype(str)
    g = df.groupby(["game_date", "home_team", "away_team",
                    "market", "label"], as_index=False).agg(
        close_line=("line", "median"),
        close_price=("price", "median"),
        n_books_close=("book", "nunique"))
    return g
