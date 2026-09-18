"""
predict_pregame.py — Ejecutar N minutos antes del primer lanzamiento.

    python predict_pregame.py --date 2026-09-18

Congela datos, calcula features, carga modelos entrenados y emite tabla
de proyecciones con Edge sobre líneas de consenso reales (The Odds API).
"""
import argparse
import time
from datetime import datetime, timezone

import pandas as pd

from pregame_ingest import run_daily_ingest, fetch_lineup_confirmed
from pregame_features import build_game_vector
from pregame_model import PregamePipeline
from config import ODDS_API_KEY, DATA_DIR

EDGE_MIN = 0.03  # umbral mínimo de valor para señalar apuesta


def wait_until_freeze(game_date: str):
    """Espera hasta N minutos antes del primer juego del día."""
    games = pd.read_parquet(DATA_DIR / f"snapshot_{game_date}" / "games.parquet")
    first = min(games["game_time_utc"])
    from pregame_ingest import game_freeze_time
    freeze_at = game_freeze_time(first)
    now = datetime.now(timezone.utc)
    if freeze_at > now:
        print(f"[predict] Esperando hasta freeze: {freeze_at.isoformat()}")
        time.sleep((freeze_at - now).total_seconds())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    # 1. Ingesta y CONGELACIÓN del snapshot
    run_daily_ingest(args.date, ODDS_API_KEY)
    snap = DATA_DIR / f"snapshot_{args.date}"
    ctx = {p.stem: pd.read_parquet(p) for p in snap.glob("*.parquet")}

    # 2. Features por juego
    vectors = [build_game_vector(g, ctx)
               for _, g in ctx["games"].iterrows()]
    X = pd.DataFrame(vectors)

    # 3. Modelos entrenados
    pipe = PregamePipeline.load()

    # 4. Proyecciones con Edge sobre líneas reales
    edges = pipe.predict_pregame(X, None, ctx["consensus_totals"])
    edges["signal"] = edges["edge"].abs() > EDGE_MIN

    out = edges.sort_values("edge", ascending=False)
    out.to_csv(f"projections_{args.date}.csv", index=False)
    print(out[out["signal"]][["event_id", "market", "line", "model_prob",
                              "implied_prob", "edge", "recommended"]]
          .to_string(index=False))
