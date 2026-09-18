"""
config.py — Configuración central del sistema prepartido MLB.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

# --- API Keys (leer desde variables de entorno, NUNCA hardcodear) ---
ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
SPORT_KEY: str = "baseball_mlb"          # The Odds API: MLB
ODDS_API_BASE: str = "https://api.the-odds-api.com/v4"

# --- Congelación prepartido ---
MINUTES_BEFORE_FIRST_PITCH: int = 30     # N minutos antes del primer lanzamiento
FREEZE_DATA: bool = True                 # snapshot inmutable una vez corrido

# --- Ventanas de features ---
ROLLING_BULLPEN_DAYS: int = 5            # carga de bullpen últimos 3-7 días
OFFENSE_HOT_DAYS: int = 14               # métricas ofensivas últimos 14 días
TRAIN_SEASONS: int = 4                   # temporadas históricas para entrenar

# --- Rutas ---
DATA_DIR = Path("data"); MODELS_DIR = Path("models"); DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

@dataclass
class ModelCfg:
    catboost_iterations: int = 2000
    catboost_lr: float = 0.03
    lstm_hidden: int = 64
    lstm_layers: int = 2
    lstm_seq_len: int = 10          # secuencia de momentum (juegos)
    mlp_hidden: list = field(default_factory=lambda: [128, 64, 32])
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    val_frac: float = 0.15          # validación temporal (último 15% cronológico)
    random_state: int = 42

MODEL_CFG = ModelCfg()

# Park Factors oficiales 2024 (Baseball Savant / FanGraphs, factor de carreras).
# Actualizar anualmente. 100 = neutral.
PARK_FACTORS: dict[str, float] = {
    "COL": 113.0,  # Coors Field
    "CIN": 104.0,  # Great American
    "BOS": 103.0,  # Fenway
    "TEX": 103.0,  # Globe Life
    "PHI": 102.0,  "TOR": 101.0,  "LAA": 101.0,  "ARI": 101.0,
    "NYY": 100.0,  "CHC": 100.0, "ATL": 100.0,  "BAL": 100.0,
    "MIL": 99.0,   "HOU": 99.0,   "CLE": 99.0,   "CHW": 99.0,
    "KC":  98.0,   "NYM": 98.0,   "MIN": 97.0,   "PIT": 97.0,
    "STL": 96.0,   "WSH": 96.0,   "DET": 96.0,   "TB":  95.0,
    "SEA": 94.0,   "MIA": 94.0,   "SF":  94.0,   "SD":  93.0,  # Petco
    "LAD": 96.0,   "OAK": 97.0,
}
