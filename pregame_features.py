"""
pregame_features.py — Cálculo de features prepartido.

Métricas implementadas con fórmulas estándar del sabermetrics:
  - FIP, xFIP, SIERA (FanGraphs), K-BB%
  - wOBA, ISO, wRC+
  - BaseRuns (David Smyth) para detección de regresión
  - Park factors, matchup por mano del pitcher, workload de bullpen
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (PARK_FACTORS, OFFENSE_HOT_DAYS, ROLLING_BULLPEN_DAYS)

# --- Pesos clásicos de eventos para wOBA (temporada 2024, FanGraphs) ---
W_WOBA = {"BB": 0.69, "HBP": 0.72, "1B": 0.88, "2B": 1.25, "3B": 1.59, "HR": 2.08}
WOBASCALE = 1.21  # wOBAScale para convertir wOBA -> wRC+


# ============================================================
# MÓDULO 1 — PITCHEO ABRIDOR Y BULLPEN
# ============================================================

def starter_features(pitcher_season: pd.DataFrame, pitcher_name: str,
                     statcast_14d: pd.DataFrame, pitcher_id: int) -> pd.Series:
    """Features del abridor confirmado: FIP, SIERA, xERA, K-BB%, velo/spin reciente.

    - Temporada acumulada desde pitching_stats (FanGraphs real).
    - Velocidad y spin de los últimos 14 días desde Statcast (pitch-by-pitch real).
    """
    row = pitcher_season[pitcher_season["Name"] == pitcher_name]
    if row.empty:
        raise ValueError(f"Pitcher no encontrado en temporada: {pitcher_name}")
    r = row.iloc[0]

    # Statcast reciente del lanzador (filtrar por pitcher ID)
    sc = statcast_14d[statcast_14d["pitcher"] == pitcher_id]
    recent_velo = sc["release_speed"].mean() if not sc.empty else np.nan
    recent_spin = sc["release_spin_rate"].mean() if not sc.empty else np.nan

    return pd.Series({
        "pitcher": pitcher_name,
        "fip": r.get("FIP"),
        "xfip": r.get("xFIP"),
        "siera": r.get("SIERA"),
        "xera": r.get("xERA"),
        "k_bb_pct": r.get("K-BB%"),
        "era": r.get("ERA"),
        "ip_season": r.get("IP"),
        "recent_velo": recent_velo,
        "recent_spin": recent_spin,
    })


def bullpen_features(statcast_bullpen: pd.DataFrame,
                     starter_ids: set[int]) -> pd.DataFrame:
    """Bullpen de los últimos 3-7 días: pitches thrown, innings, carga de trabajo.

    Excluye a los abridores del día (starter_ids). Todo desde Statcast real:
    cada fila es un lanzamiento real registrado por Baseball Savant.
    """
    bp = statcast_bullpen[~statcast_bullpen["pitcher"].isin(starter_ids)].copy()
    if bp.empty:
        return pd.DataFrame()

    g = bp.groupby("pitcher").agg(
        pitches_thrown=("pitch_type", "count"),
        batters_faced=("batter", "nunique"),
        avg_velo=("release_speed", "mean"),
        total_pitches_days=( "game_date", "nunique"),
    ).reset_index()

    # Innings aproximados: outs registrados / 3
    outs = bp[bp["events"].isin([
        "strikeout", "field_out", "force_out", "grounded_into_double_play",
        "sac_fly", "sac_bunt", "double_play", "triple_play",
    ])].groupby("pitcher").size().rename("outs")
    g = g.merge(outs, on="pitcher", how="left").fillna({"outs": 0})
    g["ip_bullpen"] = g["outs"] / 3.0
    g["pitches_per_ip"] = g["pitches_thrown"] / g["ip_bullpen"].replace(0, np.nan)
    g["workload_flag"] = g["pitches_thrown"] > 60  # bullpen muy cargado
    return g


# ============================================================
# MÓDULO 2 — OFENSIVA (LINEUP vs RHP/LHP)
# ============================================================

def _woba_from_events(df: pd.DataFrame) -> float:
    """wOBA calculado desde eventos reales de Statcast."""
    pa = df["events"].notna().sum()
    if pa == 0:
        return np.nan
    num = 0.0
    for ev, w in W_WOBA.items():
        if ev == "BB":
            mask = df["events"] == "walk"
        elif ev == "HBP":
            mask = df["events"] == "hit_by_pitch"
        elif ev == "1B":
            mask = df["events"] == "single"
        elif ev == "2B":
            mask = df["events"] == "double"
        elif ev == "3B":
            mask = df["events"] == "triple"
        elif ev == "HR":
            mask = df["events"] == "home_run"
        num += w * mask.sum()
    return num / pa


def lineup_offense_features(lineup: pd.DataFrame,
                            batter_season: pd.DataFrame,
                            statcast_hot: pd.DataFrame,
                            pitcher_hand: str) -> pd.Series:
    """Agrega wOBA, ISO y wRC+ de la alineación titular.

    - Temporada acumulada: merge con batting_stats (FanGraphs real) por nombre.
    - Últimos 14 días: cálculo directo de wOBA/ISO desde eventos Statcast.
    - Matchup: se repondera por lado de bateo vs mano del abridor rival
      (hand_split disponible si Statcast lo trae vía 'stand' vs 'p_throws').
    """
    lineup = lineup[lineup["batting_order"] <= 900].copy()  # titulares
    merged = lineup.merge(
        batter_season[["Name", "wOBA", "ISO", "wRC+", "PA"]],
        left_on="player_name", right_on="Name", how="left")

    # 14 días calientes por bateador desde Statcast real
    hot = statcast_hot[statcast_hot["batter"].isin(lineup["player_id"])]
    hot_stats = hot.groupby("batter").apply(
        lambda d: pd.Series({
            "woba_14d": _woba_from_events(d),
            "iso_14d": (
                (d["events"].isin(["double"]).sum()
                 + 2 * d["events"].isin(["triple"]).sum()
                 + 3 * d["events"].isin(["home_run"]).sum())
                / max(d["events"].notna().sum(), 1)),
        })).reset_index()
    merged = merged.merge(hot_stats, left_on="player_id",
                          right_on="batter", how="left")

    w_pa = merged["PA"].fillna(100)  # ponderar por exposición real
    out = {
        "lineup_woba_season": np.average(merged["wOBA"].fillna(
            merged["wOBA"].mean()), weights=w_pa),
        "lineup_iso_season": np.average(merged["ISO"].fillna(
            merged["ISO"].mean()), weights=w_pa),
        "lineup_wrc_season": np.average(merged["wRC+"].fillna(100), weights=w_pa),
        "lineup_woba_14d": merged["woba_14d"].mean(),
        "lineup_iso_14d": merged["iso_14d"].mean(),
        "n_missing_stats": merged["wOBA"].isna().sum(),
        "vs_pitcher_hand": pitcher_hand,
    }
    return pd.Series(out)


# ============================================================
# MÓDULO 3 — ENTORNO Y BASERUNS
# ============================================================

def park_factor(home_team_abbr: str) -> float:
    """Park factor oficial del estadio local (config, actualizado anualmente)."""
    return PARK_FACTORS.get(home_team_abbr, 100.0) / 100.0


def baseruns(batting_line: dict) -> float:
    """BaseRuns de David Smyth — estimación de carreras esperadas.

    batting_line: dict con AB, H, 2B, 3B, HR, BB, HBP, SB, CS, SO, PA, TB.
    Se compara contra carreras reales del equipo para detectar regresión
    (equipos sobre/sobrerrendiendo su producción real de bateo).
    """
    ab, h = batting_line["AB"], batting_line["H"]
    tb = batting_line["TB"]
    bb, hbp = batting_line["BB"], batting_line["HBP"]
    sb, cs = batting_line["SB"], batting_line.get("CS", 0)
    singles = h - (batting_line["2B"] + batting_line["3B"] + batting_line["HR"])

    # Componentes de scoring
    a = h - batting_line["2B"] - batting_line["3B"] - batting_line["HR"] + bb + hbp - (0.5 * 0)
    b = (0.883 * singles + 2.376 * batting_line["2B"] + 3.893 * batting_line["3B"]
         + 2.069 * batting_line["HR"] + 0.397 * (bb + hbp) - 0.823 * 0)
    c = ab - h + 0.92 * (bb + hbp) + 1.02 * 0
    d = batting_line["HR"]
    league_scoring = 0.117  # factor de liga ~4.5 runs/juego/9 entradas
    bs = (a * b / (b + c) + d) * (league_scoring / 0.117)
    return bs


def baseruns_regression(team_stats: pd.DataFrame) -> pd.Series:
    """Diferencia BaseRuns - Carreras reales por equipo (regresión a la media).

    Valor positivo => el equipo ha tenido mala suerte/CLV bajo, se espera mejora.
    """
    team_stats["baseruns_est"] = team_stats.apply(
        lambda r: baseruns(r.to_dict()), axis=1)
    team_stats["br_diff"] = team_stats["baseruns_est"] - team_stats["R"]
    return team_stats[["team", "baseruns_est", "br_diff"]]


# ============================================================
# CONSTRUCCIÓN DEL VECTOR FINAL POR JUEGO
# ============================================================

def build_game_vector(game_row: pd.Series, ctx: dict) -> pd.Series:
    """Ensambla todas las features prepartido de UN juego.

    ctx: dict con los DataFrames congelados del snapshot (ver run_daily_ingest).
    """
    home, away = game_row["home_team"], game_row["away_team"]

    # Pitcheo
    away_sp = starter_features(ctx["pitcher_season"], game_row["away_pitcher"],
                               ctx["statcast_14d"], game_row["away_pitcher_id"])
    home_sp = starter_features(ctx["pitcher_season"], game_row["home_pitcher"],
                               ctx["statcast_14d"], game_row["home_pitcher_id"])
    # Bullpens
    starters = {game_row["away_pitcher_id"], game_row["home_pitcher_id"]}
    bp = bullpen_features(ctx["statcast_bullpen"], starters)

    # Ofensiva (lineup vs mano del abridor rival)
    lineups = ctx["lineups"]
    h_lu = lineups[(lineups["game_id"] == game_row["game_id"]) & (lineups["side"] == "home")]
    a_lu = lineups[(lineups["game_id"] == game_row["game_id"]) & (lineups["side"] == "away")]
    # Mano del abridor desde Statcast (pitch_type no nulo -> p_throws)
    hand_map = ctx["statcast_14d"].groupby("pitcher")["p_throws"].last()
    away_hand = hand_map.get(game_row["away_pitcher_id"], "R")
    home_hand = hand_map.get(game_row["home_pitcher_id"], "R")
    h_off = lineup_offense_features(h_lu, ctx["batter_season"],
                                    ctx["statcast_14d"], away_hand)
    a_off = lineup_offense_features(a_lu, ctx["batter_season"],
                                    ctx["statcast_14d"], home_hand)

    pf = park_factor(game_row["home_team_abbr"])

    vec = {
        "game_id": game_row["game_id"],
        "game_date": game_row["game_date"],
        # --- Abridores ---
        "away_fip": away_sp["fip"], "away_siera": away_sp["siera"],
        "away_xera": away_sp["xera"], "away_kbb": away_sp["k_bb_pct"],
        "away_velo": away_sp["recent_velo"], "away_spin": away_sp["recent_spin"],
        "home_fip": home_sp["fip"], "home_siera": home_sp["siera"],
        "home_xera": home_sp["xera"], "home_kbb": home_sp["k_bb_pct"],
        "home_velo": home_sp["recent_velo"], "home_spin": home_sp["recent_spin"],
        # --- Diferenciales abridor (clave del modelo) ---
        "fip_diff": home_sp["fip"] - away_sp["fip"],
        "siera_diff": home_sp["siera"] - away_sp["siera"],
        # --- Ofensivas ---
        "home_woba": h_off["lineup_woba_season"],
        "home_woba_14d": h_off["lineup_woba_14d"],
        "home_iso": h_off["lineup_iso_season"],
        "home_wrc": h_off["lineup_wrc_season"],
        "away_woba": a_off["lineup_woba_season"],
        "away_woba_14d": a_off["lineup_woba_14d"],
        "away_iso": a_off["lineup_iso_season"],
        "away_wrc": a_off["lineup_wrc_season"],
        "woba_diff": h_off["lineup_woba_season"] - a_off["lineup_woba_season"],
        # --- Entorno ---
        "park_factor": pf,
        "rest_days_home": game_row.get("rest_home", 0),
        "rest_days_away": game_row.get("rest_away", 0),
    }

    # Bullpen agregado (equipo)
    if not bp.empty:
        vec["home_bp_pitches"] = bp[bp["team"] == home]["pitches_thrown"].sum()
        vec["away_bp_pitches"] = bp[bp["team"] == away]["pitches_thrown"].sum()
    return pd.Series(vec)
