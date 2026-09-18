"""
train.py — Entrena el sistema completo con datos REALES históricos.

Uso:
    export ODDS_API_KEY="tu_key"
    python train.py --start 2022-04-01 --end 2026-09-01

Construye el dataset histórico corrriendo la ingesta por día (o leyendo
snapshots ya congelados), calcula features, entrena las 4 familias de modelos
y guarda todo en models/. No genera ni usa datos sintéticos en ningún paso.
"""
import argparse
from datetime import date, timedelta

import pandas as pd

from pregame_ingest import run_daily_ingest
from pregame_features import build_game_vector
from pregame_model import PregamePipeline
from config import ODDS_API_KEY, DATA_DIR


def build_historical_dataset(start: str, end: str) -> pd.DataFrame:
    """Itera días históricos, ingiere snapshots (o usa caché) y arma el dataset.

    Nota: para entrenar se requieren los resultados reales (total_runs,
    home_win). Se obtienen del endpoint de resultados oficial de MLB
    (statsapi) o leyendo el historial de cuotas de The Odds API
    (endpoint /historical con timestamp real de cierre).
    """
    days = pd.date_range(start, end, freq="D")
    vectors = []
    for d in days:
        ds = d.strftime("%Y-%m-%d")
        snap = DATA_DIR / f"snapshot_{ds}"
        if not (snap / "games.parquet").exists():
            try:
                run_daily_ingest(ds, ODDS_API_KEY)
            except Exception as e:
                print(f"[train] sin datos para {ds}: {e}")
                continue
        ctx = {p.stem: pd.read_parquet(p) for p in snap.glob("*.parquet")}
        if "games" not in ctx:
            continue
        for _, g in ctx["games"].iterrows():
            try:
                v = build_game_vector(g, ctx)
                vectors.append(v)
            except Exception as e:
                print(f"[train] juego {g['game_id']} omitido: {e}")
    df = pd.DataFrame(vectors)
    # Targets reales: merge con resultados oficiales de MLB (statsapi /linescore)
    df = attach_real_results(df)
    df.to_parquet(DATA_DIR / "train_dataset.parquet", index=False)
    return df


def attach_real_results(df: pd.DataFrame) -> pd.DataFrame:
    """Trae los resultados reales desde la API oficial de MLB (sin síntesis)."""
    import requests
    totals, winners, hr_, ar_ = [], [], [], []
    for gid in df["game_id"]:
        r = requests.get(
            f"https://statsapi.mlb.com/api/v1/game/{gid}/linescore",
            timeout=30).json()
        hs = r["teams"]["home"].get("runs", 0)
        aw = r["teams"]["away"].get("runs", 0)
        hr_.append(hs); ar_.append(aw)
        totals.append(hs + aw); winners.append(int(hs > aw))
    df["home_runs"], df["away_runs"] = hr_, ar_
    df["total_runs"], df["home_win"] = totals, winners
    return df


def build_momentum_sequences(df: pd.DataFrame, n_feat: int = 5) -> dict:
    """Secuencias reales para el LSTM: rolling windows de juegos previos.

    Features por juego: diff carreras, wOBA móvil, FIP móvil, diff FIP, park.
    Construidas desde el propio dataset histórico (datos reales, orden cronológico).
    """
    from config import MODEL_CFG
    L = MODEL_CFG.lstm_seq_len
    df = df.sort_values("game_date")
    seqs, targets = [], []
    for i in range(L, len(df)):
        win = df.iloc[i - L:i]
        seq = win[["fip_diff", "woba_diff", "park_factor",
                   "rest_days_home", "rest_days_away"]].values
        seqs.append(seq); targets.append(df.iloc[i]["home_win"])
    X = np.asarray(seqs, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    cut = int(len(X) * 0.85)
    return {"train": (X[:cut], y[:cut]), "val": (X[cut:], y[cut:])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    df = build_historical_dataset(args.start, args.end)
    seqs = build_momentum_sequences(df)
    pipe = PregamePipeline().fit(df, seqs)
    print("[train] ✅ Sistema entrenado y guardado en models/")
