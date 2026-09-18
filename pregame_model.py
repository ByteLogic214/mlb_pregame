"""
pregame_model.py — Entrenamiento y proyección prepartido.

Modelos:
  1. CatBoost Regressor/Classifier   -> totales y moneyline
  2. Regresión Poisson (GLM)         -> distribución de carreras por equipo
  3. LSTM de momentum                -> secuencias de forma reciente (10 juegos)
  4. Red Neuronal MLP (PyTorch)      -> interacciones no lineales

Validación: temporal (walk-forward). El ensemble pondera por log-loss/RMSE
de validación. Las proyecciones se comparan contra la línea de consenso real
(The Odds API) para calcular Edge y probabilidad implícita sin margen (no-vig).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.api import GLM, families

from config import MODEL_CFG, MODELS_DIR

FEATURE_COLS = [
    "away_fip", "away_siera", "away_xera", "away_kbb", "away_velo", "away_spin",
    "home_fip", "home_siera", "home_xera", "home_kbb", "home_velo", "home_spin",
    "fip_diff", "siera_diff",
    "home_woba", "home_woba_14d", "home_iso", "home_wrc",
    "away_woba", "away_woba_14d", "away_iso", "away_wrc", "woba_diff",
    "park_factor", "rest_days_home", "rest_days_away",
    "home_bp_pitches", "away_bp_pitches",
]


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------

def temporal_split(df: pd.DataFrame, val_frac: float):
    """Split cronológico estricto: validación = último val_frac del tiempo."""
    df = df.sort_values("game_date").reset_index(drop=True)
    cut = int(len(df) * (1 - val_frac))
    return df.iloc[:cut], df.iloc[cut:]


def no_vig_prob(odds_american: float) -> float:
    """Convierte cuota americana a probabilidad sin margen de la casa."""
    p = odds_american / (odds_american + 100) if odds_american > 0 \
        else 100 / (abs(odds_american) + 100)
    return p


def american_to_decimal(a: float) -> float:
    return 1 + (a / 100 if a > 0 else 100 / abs(a))


# ------------------------------------------------------------
# 1. CATBOOST
# ------------------------------------------------------------

class CatBoostModels:
    """CatBoost para totales (regresión) y moneyline (clasificación binaria)."""
    def __init__(self):
        self.total_model = CatBoostRegressor(
            iterations=MODEL_CFG.catboost_iterations,
            learning_rate=MODEL_CFG.catboost_lr,
            depth=6, loss_function="RMSE", random_seed=MODEL_CFG.random_state,
            verbose=0)
        self.ml_model = CatBoostClassifier(
            iterations=MODEL_CFG.catboost_iterations,
            learning_rate=MODEL_CFG.catboost_lr,
            depth=6, loss_function="Logloss", random_seed=MODEL_CFG.random_state,
            verbose=0)

    def fit(self, X_tr, y_total_tr, y_win_tr, X_va=None, y_win_va=None):
        self.total_model.fit(X_tr, y_total_tr)
        fit_params = {}
        if X_va is not None and y_win_va is not None:
            fit_params = {"eval_set": (X_va, y_win_va), "early_stopping_rounds": 100}
        self.ml_model.fit(X_tr, y_win_tr, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "cb_total": self.total_model.predict(X[FEATURE_COLS]),
            "cb_p_home": self.ml_model.predict_proba(X[FEATURE_COLS])[:, 1],
        }, index=X.index)


# ------------------------------------------------------------
# 2. POISSON GLM
# ------------------------------------------------------------

class PoissonModel:
    """Regresión Poisson por equipo con offset de parque.

    Modela carreras_esperadas de cada equipo; el total = suma de lambdas,
    y la probabilidad de ganar = P(home > away) por convolución de Poisson.
    """
    def __init__(self):
        self.models: dict[str, GLM] = {}
        self.scaler = StandardScaler()

    @staticmethod
    def _design(X: pd.DataFrame, side: str) -> pd.DataFrame:
        cols = {
            "home": ["home_fip", "home_siera", "home_woba", "home_iso",
                     "away_woba", "park_factor", "home_bp_pitches"],
            "away": ["away_fip", "away_siera", "away_woba", "away_iso",
                     "home_woba", "park_factor", "away_bp_pitches"],
        }[side]
        return X[cols].astype(float)

    def fit(self, X_tr, y_home, y_away):
        for side, y in (("home", y_home), ("away", y_away)):
            Z = self._design(X_tr, side)
            Z = self.scaler.fit_transform(Z)
            # Offset log del park factor para escala multiplicativa
            off = np.log(X_tr["park_factor"].clip(0.8, 1.25))
            self.models[side] = GLM(y, Z, family=families.Poisson(),
                                    offset=off).fit()
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        lam_h, lam_a = [], []
        for side in ("home", "away"):
            Z = self.scaler.transform(self._design(X, side))
            off = np.log(X["park_factor"].clip(0.8, 1.25))
            lam = self.models[side].predict(Z, offset=off)
            (lam_h if side == "home" else lam_a).append(np.asarray(lam))
        lam_h, lam_a = np.asarray(lam_h[0]), np.asarray(lam_a[0])

        # Total esperado y P(home win) por convolución truncada (máx 25 carreras)
        total = lam_h + lam_a
        p_home = np.zeros(len(X))
        max_g = 25
        from scipy.stats import poisson
        pmf_h = np.array([poisson.pmf(k, lam_h) for k in range(max_g)]).T
        pmf_a = np.array([poisson.pmf(k, lam_a) for k in range(max_g)]).T
        for i in range(len(X)):
            grid = np.tril(np.ones((max_g, max_g)), -1)  # home > away
            p_home[i] = (pmf_h[i][:, None] * pmf_a[i][None, :] * grid).sum()
        return pd.DataFrame({"pois_total": total, "pois_p_home": p_home},
                            index=X.index)


# ------------------------------------------------------------
# 3. LSTM DE MOMENTUM
# ------------------------------------------------------------

class LSTMModel(nn.Module):
    """LSTM sobre secuencias de momentum (diferenciales de forma reciente).

    Secuencia = últimos L juegos del equipo: diferencial de carreras,
    wOBA móvil, FIP móvil. Entrada construida con datos reales de game logs.
    """
    def __init__(self, n_features: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(n_features, MODEL_CFG.lstm_hidden,
                            num_layers=MODEL_CFG.lstm_layers,
                            batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(MODEL_CFG.lstm_hidden, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class LSTMTrainer:
    def __init__(self, n_features: int):
        self.seq_len = MODEL_CFG.lstm_seq_len
        self.model = LSTMModel(n_features, self.seq_len)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=MODEL_CFG.lr)

    def fit(self, seqs_tr: np.ndarray, y_tr: np.ndarray,
            seqs_va: np.ndarray | None = None, y_va: np.ndarray | None = None):
        X = torch.tensor(seqs_tr, dtype=torch.float32)
        y = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
        ds = torch.utils.data.TensorDataset(X, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=MODEL_CFG.batch_size,
                                         shuffle=False)  # temporal: no shuffle
        best_loss, patience = np.inf, 8
        for epoch in range(MODEL_CFG.epochs):
            self.model.train()
            for xb, yb in dl:
                self.opt.zero_grad()
                loss = nn.functional.binary_cross_entropy(self.model(xb), yb)
                loss.backward(); self.opt.step()
            if seqs_va is not None:
                self.model.eval()
                with torch.no_grad():
                    pv = self.model(torch.tensor(seqs_va, dtype=torch.float32))
                    vl = nn.functional.binary_cross_entropy(
                        pv, torch.tensor(y_va, dtype=torch.float32).view(-1, 1))
                if vl < best_loss:
                    best_loss, patience = vl, 8
                    torch.save(self.model.state_dict(), MODELS_DIR / "lstm_best.pt")
                else:
                    patience -= 1
                    if patience <= 0:
                        break
        return self

    def predict_proba(self, seqs: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(seqs, dtype=torch.float32)).numpy().ravel()


# ------------------------------------------------------------
# 4. RED NEURONAL MLP
# ------------------------------------------------------------

class MLPModel(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        layers, prev = [], n_in
        for h in MODEL_CFG.mlp_hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(0.25)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_total = nn.Linear(prev, 1)
        self.out_win = nn.Linear(prev, 1)

    def forward(self, x):
        z = self.net(x)
        return self.out_total(z).squeeze(-1), torch.sigmoid(self.out_win(z).squeeze(-1))


class MLPTrainer:
    def __init__(self, n_in: int):
        self.model = MLPModel(n_in)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=MODEL_CFG.lr)
        self.scaler = StandardScaler()

    def fit(self, X_tr, y_total_tr, y_win_tr, X_va=None, y_total_va=None):
        X = torch.tensor(self.scaler.fit_transform(X_tr), dtype=torch.float32)
        yt = torch.tensor(y_total_tr, dtype=torch.float32)
        yw = torch.tensor(y_win_tr, dtype=torch.float32)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, yt, yw),
            batch_size=MODEL_CFG.batch_size, shuffle=False)
        best = np.inf; patience = 8
        for _ in range(MODEL_CFG.epochs):
            self.model.train()
            for xb, ytb, ywb in dl:
                self.opt.zero_grad()
                t_hat, w_hat = self.model(xb)
                loss = (nn.functional.mse_loss(t_hat, ytb)
                        + nn.functional.binary_cross_entropy(w_hat, ywb))
                loss.backward(); self.opt.step()
            if X_va is not None and y_total_va is not None:
                self.model.eval()
                with torch.no_grad():
                    tv, _ = self.model(torch.tensor(
                        self.scaler.transform(X_va), dtype=torch.float32))
                    vl = nn.functional.mse_loss(
                        tv, torch.tensor(y_total_va, dtype=torch.float32))
                if vl < best:
                    best, patience = vl, 8
                    torch.save(self.model.state_dict(), MODELS_DIR / "mlp_best.pt")
                else:
                    patience -= 1
                    if patience <= 0:
                        break
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            t, w = self.model(torch.tensor(self.scaler.transform(X),
                                           dtype=torch.float32))
        return t.numpy(), w.numpy()


# ------------------------------------------------------------
# PIPELINE MAESTRO
# ------------------------------------------------------------

class PregamePipeline:
    """Orquesta entrenamiento de las 4 familias + ensemble ponderado."""

    def __init__(self):
        self.cb = CatBoostModels()
        self.pois = PoissonModel()
        self.lstm: LSTMTrainer | None = None
        self.mlp: MLPTrainer | None = None
        self.ensemble_weights: dict = {}

    def fit(self, df: pd.DataFrame,
            seqs: dict[str, np.ndarray] | None = None):
        """df: dataset histórico de juegos con FEATURE_COLS + targets.
        Targets: 'total_runs', 'home_win' (1/0), 'home_runs', 'away_runs'.
        seqs: {'train': (X_seq, y), 'val': (X_seq, y)} para el LSTM.
        """
        tr, va = temporal_split(df, MODEL_CFG.val_frac)
        X_tr, X_va = tr[FEATURE_COLS], va[FEATURE_COLS]

        # 1. CatBoost
        self.cb.fit(X_tr, tr["total_runs"], tr["home_win"], X_va, va["home_win"])
        # 2. Poisson
        self.pois.fit(X_tr, tr["home_runs"], tr["away_runs"])
        # 3. MLP
        self.mlp = MLPTrainer(len(FEATURE_COLS)).fit(
            X_tr, tr["total_runs"].values, tr["home_win"].values,
            X_va, va["total_runs"].values if len(va) else None)
        # 4. LSTM (solo si hay secuencias reales de momentum)
        if seqs:
            self.lstm = LSTMTrainer(seqs["train"][0].shape[-1]).fit(
                *seqs["train"], *seqs["val"])

        # --- Pesos del ensemble por desempeño en validación ---
        preds_va = self._all_preds(X_va, seqs["val"][0] if seqs else None)
        rmse_cb = np.sqrt(mean_squared_error(va["total_runs"], preds_va["cb_total"]))
        rmse_pois = np.sqrt(mean_squared_error(va["total_runs"], preds_va["pois_total"]))
        rmse_mlp = np.sqrt(mean_squared_error(va["total_runs"], preds_va["mlp_total"]))
        ll = {
            "cb": log_loss(va["home_win"], preds_va["cb_p_home"].clip(1e-6, 1-1e-6)),
            "pois": log_loss(va["home_win"], preds_va["pois_p_home"].clip(1e-6, 1-1e-6)),
            "mlp": log_loss(va["home_win"], preds_va["mlp_p_win"].clip(1e-6, 1-1e-6)),
        }
        if self.lstm is not None:
            ll["lstm"] = log_loss(va["home_win"], preds_va["lstm_p_home"].clip(1e-6, 1-1e-6))

        inv = {k: 1.0 / max(v, 1e-4) for k, v in ll.items()}
        s = sum(inv.values())
        self.ensemble_weights = {k: v / s for k, v in inv.items()}
        print(f"[pipeline] Pesos ensemble: {self.ensemble_weights}")
        print(f"[pipeline] RMSE val — CatBoost:{rmse_cb:.3f} Poisson:{rmse_pois:.3f} MLP:{rmse_mlp:.3f}")

        self.save()
        return self

    def _all_preds(self, X, seqs_va=None) -> pd.DataFrame:
        out = self.cb.predict(X).join(self.pois.predict(X))
        t, w = self.mlp.predict(X)
        out["mlp_total"], out["mlp_p_win"] = t, w
        if self.lstm is not None and seqs_va is not None:
            out["lstm_p_home"] = self.lstm.predict_proba(seqs_va)
        else:
            out["lstm_p_home"] = out["cb_p_home"]  # fallback si no hay LSTM
        return out

    # -------------------- INFERENCIA PREPARTIDO --------------------

    def predict_pregame(self, X: pd.DataFrame, seqs: np.ndarray | None,
                        consensus: pd.DataFrame) -> pd.DataFrame:
        """Genera proyección final con EDGE sobre líneas reales de consenso.

        consensus: DataFrame de consensus_line() con columnas
        ['event_id','label','median_line','median_price'].
        """
        preds = self._all_preds(X, seqs)
        w = self.ensemble_weights

        preds["proj_total"] = (w["cb"] * preds["cb_total"]
                               + w["pois"] * preds["pois_total"]
                               + w["mlp"] * preds["mlp_total"])
        p_home = sum(w[k] * preds[f"{k}_p_home" if k != "mlp" else "mlp_p_win"]
                     for k in w)
        preds["p_home_win"] = p_home

        results = []
        for _, row in preds.iterrows():
            ev = consensus[consensus["event_id"] == row["event_id"]]
            # --- Edge en Total (Over/Under) ---
            tot = ev[ev["label"].isin(["Over", "Under"])]
            if not tot.empty:
                line = float(tot["median_line"].iloc[0])
                over_price = float(tot[tot["label"] == "Over"]["median_price"].iloc[0])
                p_over_model = 1 - _poisson_cdf(line, row["proj_total"])
                p_over_impl = no_vig_prob(over_price)
                results.append({
                    "event_id": row["event_id"], "market": "total",
                    "line": line, "model_prob": p_over_model,
                    "implied_prob": p_over_impl,
                    "edge": p_over_model - p_over_impl,
                    "fair_odds": (100 * p_over_model / (1 - p_over_model)
                                  if p_over_model < 0.5 else
                                  -100 * (1 - p_over_model) / p_over_model),
                    "recommended": "Over" if row["proj_total"] > line else "Under",
                    "proj_total": row["proj_total"],
                })
            # --- Edge en Moneyline ---
            ml = ev[ev["label"].isin(["home", "away"])]
            if not ml.empty:
                home_price = float(ml[ml["label"] == "home"]["median_price"].iloc[0])
                p_impl_home = no_vig_prob(home_price)
                results.append({
                    "event_id": row["event_id"], "market": "moneyline",
                    "line": 0, "model_prob": p_home,
                    "implied_prob": p_impl_home,
                    "edge": p_home - p_impl_home,
                    "fair_odds": (100 * p_home / (1 - p_home) if p_home < 0.5
                                  else -100 * (1 - p_home) / p_home),
                    "recommended": "home" if p_home > p_impl_home else "away",
                    "proj_total": row["proj_total"],
                })
            # --- Team Totals ---
            for side in ("home", "away"):
                tt = ev[(ev["label"] == side) & (ev["market"] == "team_total")]
                if not tt.empty:
                    lam = (row["proj_total"] * p_home if side == "home"
                           else row["proj_total"] * (1 - p_home))
                    line_tt = float(tt["median_line"].iloc[0])
                    p_over = 1 - _poisson_cdf(line_tt, lam)
                    price_tt = float(tt["median_price"].iloc[0])
                    results.append({
                        "event_id": row["event_id"], "market": f"team_total_{side}",
                        "line": line_tt, "model_prob": p_over,
                        "implied_prob": no_vig_prob(price_tt),
                        "edge": p_over - no_vig_prob(price_tt),
                        "recommended": "Over" if lam > line_tt else "Under",
                        "proj_total": lam,
                    })
        return pd.DataFrame(results)

    # -------------------- PERSISTENCIA --------------------

    def save(self):
        self.cb.total_model.save_model(str(MODELS_DIR / "cb_total.cbm"))
        self.cb.ml_model.save_model(str(MODELS_DIR / "cb_ml.cbm"))
        with open(MODELS_DIR / "poisson.pkl", "wb") as f:
            pickle.dump(self.pois, f)
        torch.save(self.mlp.model.state_dict(), MODELS_DIR / "mlp.pt")
        if self.lstm:
            torch.save(self.lstm.model.state_dict(), MODELS_DIR / "lstm.pt")
        with open(MODELS_DIR / "ensemble.json", "w") as f:
            json.dump(self.ensemble_weights, f)

    @classmethod
    def load(cls) -> "PregamePipeline":
        p = cls()
        p.cb.total_model.load_model(str(MODELS_DIR / "cb_total.cbm"))
        p.cb.ml_model.load_model(str(MODELS_DIR / "cb_ml.cbm"))
        with open(MODELS_DIR / "poisson.pkl", "rb") as f:
            p.pois = pickle.load(f)
        p.mlp = MLPTrainer(len(FEATURE_COLS))
        p.mlp.model.load_state_dict(torch.load(MODELS_DIR / "mlp.pt"))
        lstm_path = MODELS_DIR / "lstm.pt"
        if lstm_path.exists():
            dummy = np.zeros((1, MODEL_CFG.lstm_seq_len, 5), dtype=np.float32)
            p.lstm = LSTMTrainer(dummy.shape[-1])
            p.lstm.model.load_state_dict(torch.load(lstm_path))
        with open(MODELS_DIR / "ensemble.json") as f:
            p.ensemble_weights = json.load(f)
        return p


def _poisson_cdf(k: float, lam: float) -> float:
    """P(X <= k) para Poisson, con continuidad en línea .5."""
    from scipy.stats import poisson
    return poisson.cdf(np.floor(k + 0.5), lam)
