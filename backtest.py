"""
backtest.py — Validación walk-forward temporal con métricas de apuesta real.

Protocolo: para cada mes M del período de prueba, entrena con TODO lo anterior
(expandiendo) y evalúa en M. Las apuestas se resuelven contra:
  - Probabilidad implícita SIN MARGEN (no-vig) de la línea de cierre REAL.
  - Resultados reales de MLB (statsapi).
Métricas: log-loss, Brier, RMSE totales, ROI%, yield, CLV (closing line value),
drawdown máximo. NADA se simula: cada apuesta se resuelve contra datos reales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pregame_model import PregamePipeline, no_vig_prob, FEATURE_COLS
from pregame_historical_odds import closing_consensus_by_day
from config import DATA_DIR

EDGE_MIN = 0.03
FLAT_STAKE = 1.0          # unidad plana por apuesta
KELLY_FRACTION = 0.25     # Kelly cuartil — nunca Kelly completo


def _month_splits(df: pd.DataFrame, test_start: str) -> list[tuple[str, str]]:
    """Genera cortes walk-forward por mes desde test_start."""
    months = pd.period_range(test_start,
                             df["game_date"].max(), freq="M").astype(str)
    return [(m, m) for m in months]


def resolve_bets(edges: pd.DataFrame, results: pd.DataFrame,
                 hist_odds: pd.DataFrame) -> pd.DataFrame:
    """Resuelve cada señal contra resultado real y precio de cierre real.

    Retorna DataFrame con stake, retorno, P&L por apuesta.
    """
    close = closing_consensus_by_day(hist_odds)
    bets = edges[edges["edge"].abs() > EDGE_MIN].copy()
    bets = bets.merge(results[["event_id", "game_date", "home_team",
                               "away_team", "total_runs", "home_win",
                               "home_runs", "away_runs"]],
                      on="event_id", how="left")

    # Kelly stake sobre edge vs. precio de cierre
    def _stake(r):
        p = r["model_prob"]
        dec = 1 + (abs(r["close_price"]) / 100 if r["close_price"] > 0
                   else 100 / abs(r["close_price"]))
        b = dec - 1
        kelly = ((p * b) - (1 - p)) / b
        return max(0.0, min(FLAT_STAKE, kelly * KELLY_FRACTION))

    def _won(r) -> bool:
        if r["market"] == "total":
            return int((r["total_runs"] > r["line"]) if r["recommended"] == "Over"
                       else (r["total_runs"] < r["line"]))
        if r["market"] == "moneyline":
            return int(r["home_win"] == 1 if r["recommended"] == "home"
                       else r["home_win"] == 0)
        # team totals
        runs = r["home_runs"] if "home" in r["market"] else r["away_runs"]
        return int((runs > r["line"]) if r["recommended"] == "Over"
                   else (runs < r["line"]))

    def _profit(r) -> float:
        if not r["won"]:
            return -r["stake"]
        dec = 1 + (abs(r["close_price"]) / 100 if r["close_price"] > 0
                   else 100 / abs(r["close_price"]))
        return r["stake"] * (dec - 1)

    bets = bets.merge(close.rename(columns={
        "market": "odds_market", "label": "odds_label"}),
        left_on=["game_date", "home_team", "away_team"],
        right_on=["game_date", "home_team", "away_team"],
        how="left")

    # Emparejar línea de cierre con la señal (mercado + label)
    mask_total = bets["market"] == "total"
    bets.loc[mask_total, "close_price"] = bets.loc[mask_total].apply(
        lambda r: r.get("close_price") if r.get("odds_label") == r["recommended"]
        else np.nan, axis=1)

    bets["stake"] = bets.apply(_stake, axis=1)
    bets = bets[bets["stake"] > 0]
    bets["won"] = bets.apply(_won, axis=1)
    bets["pnl"] = bets.apply(_profit, axis=1)
    bets["yield"] = bets["pnl"] / bets["stake"]

    # CLV: ¿nuestro modelo batió el cierre? (prob modelo vs prob implícita cierre)
    bets["clv"] = bets["model_prob"] - bets["implied_prob"]
    return bets


def run_backtest(df: pd.DataFrame, results: pd.DataFrame,
                 hist_odds: pd.DataFrame,
                 test_start: str = "2025-04-01") -> tuple[pd.DataFrame, dict]:
    """Walk-forward completo. Devuelve (bets, métricas agregadas)."""
    all_bets = []
    for month, _ in _month_splits(df, test_start):
        train = df[df["game_date"] < month]
        test = df[(df["game_date"] >= month) &
                  (df["game_date"] < str(pd.Period(month) + 1))]
        if len(train) < 500 or test.empty:
            continue
        pipe = PregamePipeline().fit(train)          # reentrena expandiendo
        preds = pipe._all_preds(test[FEATURE_COLS])
        w = pipe.ensemble_weights
        proj_total = (w["cb"] * preds["cb_total"]
                      + w["pois"] * preds["pois_total"]
                      + w["mlp"] * preds["mlp_total"])
        p_home = sum(w[k] * preds[f"{k}_p_home" if k != "mlp" else "mlp_p_win"]
                     for k in w)
        edges = pipe.predict_pregame(test, None, hist_odds)
        edges["month"] = month
        edges["proj_total"] = proj_total.values
        edges["p_home"] = p_home.values
        bets = resolve_bets(edges, results, hist_odds)
        all_bets.append(bets)
        print(f"[bt] {month}: {len(bets)} apuestas, "
              f"yield={bets['yield'].mean():+.2%}" if len(bets) else
              f"[bt] {month}: sin señales")

    if not all_bets:
        return pd.DataFrame(), {}
    bets = pd.concat(all_bets, ignore_index=True)

    cum = bets.groupby("month")["pnl"].sum()
    metrics = {
        "n_bets": len(bets),
        "win_rate": bets["won"].mean(),
        "roi": bets["pnl"].sum() / bets["stake"].sum(),
        "avg_yield": bets["yield"].mean(),
        "clv_mean": bets["clv"].mean(),                # >0 = bates el cierre
        "max_drawdown": (cum.cumsum().cummax() - cum.cumsum()).max(),
        "sharpe_monthly": (cum.mean() / cum.std()) if cum.std() > 0 else np.nan,
        "by_market": bets.groupby("market")["yield"].mean().to_dict(),
    }
    bets.to_parquet(DATA_DIR / "backtest_bets.parquet", index=False)
    return bets, metrics
