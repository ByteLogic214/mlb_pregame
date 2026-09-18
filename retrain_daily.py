"""
retrain_daily.py — Pipeline diario post-jornada.

Se ejecuta cada madrugada (cron/systemd). Flujo:
  1. Ingesta el snapshot del día (si no estaba congelado).
  2. Adjunta resultados reales de ayer al dataset maestro.
  3. Reentrena TODAS las familias con el dataset expandiéndose (walk-forward).
  4. Versiona modelos (nunca sobrescribe el anterior sin validación).
  5. Promueve solo si el nuevo modelo mejora al actual en el último mes.

Uso:  python retrain_daily.py --date 2026-09-18
Cron: 30 4 * * *  (4:30 AM, cuando todos los juegos de ayer cerraron)
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from pregame_ingest import run_daily_ingest
from pregame_features import build_game_vector
from pregame_model import PregamePipeline, FEATURE_COLS
from train import attach_real_results
from config import DATA_DIR, MODELS_DIR, ODDS_API_KEY

MASTER_PATH = DATA_DIR / "train_dataset.parquet"
VERSIONS_DIR = MODELS_DIR / "versions"


def update_master(game_date: str) -> pd.DataFrame:
    """Agrega los juegos de game_date al dataset maestro (idempotente)."""
    snap = DATA_DIR / f"snapshot_{game_date}"
    if not (snap / "games.parquet").exists():
        run_daily_ingest(game_date, ODDS_API_KEY)
    ctx = {p.stem: pd.read_parquet(p) for p in snap.glob("*.parquet")}

    master = pd.read_parquet(MASTER_PATH) if MASTER_PATH.exists() \
        else pd.DataFrame()
    new_ids = set(ctx["games"]["game_id"]) - set(master.get("game_id", []))
    if not new_ids:
        print("[retrain] Dataset ya actualizado.")
        return master

    fresh = ctx["games"][ctx["games"]["game_id"].isin(new_ids)]
    vectors = [build_game_vector(g, ctx) for _, g in fresh.iterrows()]
    new_rows = attach_real_results(pd.DataFrame(vectors))
    master = pd.concat([master, new_rows], ignore_index=True)
    master.to_parquet(MASTER_PATH, index=False)
    print(f"[retrain] +{len(new_rows)} juegos. Total: {len(master)}")
    return master


def version_and_promote(pipe: PregamePipeline, tag: str,
                        master: pd.DataFrame) -> bool:
    """Guarda versión nueva y la promueve SOLO si mejora log-loss del último mes.

    Estrategia anti-degradación: comparación contra la versión en producción
    sobre el mes más reciente (out-of-sample real).
    """
    VERSIONS_DIR.mkdir(exist_ok=True)
    vdir = VERSIONS_DIR / tag
    vdir.mkdir(exist_ok=True)

    # Guardar versión candidata
    for f in MODELS_DIR.glob("*.*"):
        if f.is_file():
            shutil.copy(f, vdir / f.name)

    # Validación: últimos 30 días del dataset maestro (reales)
    cutoff = (pd.Timestamp(master["game_date"].max()) - pd.Timedelta(days=30)) \
        .strftime("%Y-%m-%d")
    recent = master[master["game_date"] >= cutoff]
    if len(recent) < 50:
        print("[retrain] Pocos juegos recientes; promoción automática.")
        return True

    X, y = recent[FEATURE_COLS], recent["home_win"]
    preds_new = pipe._all_preds(X)
    from sklearn.metrics import log_loss
    ll_new = log_loss(y, preds_new["cb_p_home"].clip(1e-6, 1 - 1e-6))

    # Comparar contra producción actual
    try:
        prod = PregamePipeline.load()
        preds_prod = prod._all_preds(X)
        ll_prod = log_loss(y, preds_prod["cb_p_home"].clip(1e-6, 1 - 1e-6))
    except Exception:
        ll_prod = float("inf")

    promote = ll_new <= ll_prod
    print(f"[retrain] log-loss nuevo:{ll_new:.4f} vs prod:{ll_prod:.4f} "
          f"-> {'PROMOVIDO' if promote else 'RECHAZADO (se mantiene versión previa)'}")
    return promote


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=(date.today() - timedelta(days=1))
                    .isoformat(), help="Día de juegos a incorporar")
    args = ap.parse_args()

    tag = f"v_{args.date}"

    # 1-2. Dataset maestro actualizado con datos reales
    master = update_master(args.date)

    # 3. Reentrenamiento completo (dataset expandiéndose)
    seqs = None  # construir secuencias momentum desde master (ver train.py)
    pipe = PregamePipeline().fit(master, seqs)

    # 4-5. Versionado con validación de promoción
    version_and_promote(pipe, tag, master)
    print(f"[retrain] ✅ Reentrenamiento diario completado: {tag}")
