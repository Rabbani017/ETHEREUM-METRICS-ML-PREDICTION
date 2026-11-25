"""
run_experiment.py
=================
Single-command end-to-end runner for the UROP project:
"Machine Learning Analysis & Prediction of Ethereum Metrics"

Team: Shaik Rabbani (AP23110010538), P. Praneeth (AP23110010508),
      P. Prabath (AP23110010565)  |  SRM University - AP

Usage:
    python run_experiment.py --network sepolia
    python run_experiment.py --network mainnet
    python run_experiment.py --network sepolia --fetch-new 300   # pull 300 fresh blocks
    python run_experiment.py --network sepolia --live            # keep running live listener

What it does (in order):
  1. Optionally fetch fresh blocks from the real chain via public RPC.
  2. Load & preprocess the raw block CSV → build ML features.
  3. Chronological 70/15/15 train/val/test split (no shuffle).
  4. Train 5 models + persistence baseline; print scores to console.
  5. Save all results, plots and Excel to outputs/.
  6. Optionally enter live-listener mode: predict next block after each new block.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
PRED_DIR    = ROOT_DIR / "ethereum_metrics_prediction"
SRC_DIR     = PRED_DIR / "src"
DATA_DIR    = PRED_DIR / "data"
MODELS_DIR  = PRED_DIR / "models"
OUTPUTS_DIR = PRED_DIR / "outputs"
FIGS_DIR    = OUTPUTS_DIR / "figures"
DB_PATH     = ROOT_DIR / "data" / "ethereum_metrics.db"

for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIGS_DIR,
          ROOT_DIR / "data"]:
    d.mkdir(parents=True, exist_ok=True)

# Add src dirs to path
for p in [str(PRED_DIR), str(SRC_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_FILE = OUTPUTS_DIR / "experiment.log"

# Tee stdout → both console and log file
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass
    def fileno(self):
        return self.streams[0].fileno()

_log_fh = open(_LOG_FILE, "a", buffering=1, encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.__stdout__,
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw block CSV (columns: number, timestamp, gas_used, gas_limit,
    base_fee_per_gas, transaction_count, size, block_time), derive:
      - base_fee_gwei, gas_utilization, tx_per_gas
      - lag features (t-1 .. t-5)
      - rolling mean & std (window 5, 10)
      - delta (first-difference)
      - target: next_gas_used (gas_used shifted by -1)
      - direction label: UP / DOWN / SAME
    Returns the feature DataFrame with NaNs dropped.
    """
    df = df.copy().sort_values("number").reset_index(drop=True)

    # Derived columns
    df["base_fee_gwei"]     = df["base_fee_per_gas"] / 1e9
    df["gas_utilization"]   = (df["gas_used"] / df["gas_limit"].replace(0, np.nan)).fillna(0)
    df["tx_per_gas"]        = (df["transaction_count"] / df["gas_used"].replace(0, np.nan)).fillna(0)

    # Block time (seconds between consecutive blocks)
    if "block_time" not in df.columns:
        df["block_time"] = df["timestamp"].diff().fillna(12).clip(0, 60)

    # Core features used in the sliding window
    core = ["gas_used", "base_fee_gwei", "transaction_count",
            "gas_utilization", "block_time"]

    # Lag features (last 5 blocks)
    for col in ["gas_used", "base_fee_gwei", "transaction_count"]:
        for lag in range(1, 6):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Rolling stats
    for col in ["gas_used", "base_fee_gwei"]:
        df[f"{col}_roll5_mean"]  = df[col].shift(1).rolling(5).mean()
        df[f"{col}_roll10_mean"] = df[col].shift(1).rolling(10).mean()
        df[f"{col}_roll5_std"]   = df[col].shift(1).rolling(5).std()

    # First-difference (momentum)
    df["gas_used_delta"]    = df["gas_used"].diff()
    df["base_fee_delta"]    = df["base_fee_gwei"].diff()

    # Target: next block's gas_used
    df["next_gas_used"]     = df["gas_used"].shift(-1)

    # Direction label for Naive Bayes
    diff = df["next_gas_used"] - df["gas_used"]
    df["next_gas_direction"] = np.where(
        diff > 0.05 * df["gas_used"], "UP",
        np.where(diff < -0.05 * df["gas_used"], "DOWN", "SAME")
    )

    df = df.dropna().reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  TRAIN / VAL / TEST SPLIT  (chronological, no shuffle)
# ═══════════════════════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame, feature_cols: list[str], target_col: str = "next_gas_used"):
    """70 / 15 / 15 chronological split."""
    n = len(df)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, y_train = X[:t1],  y[:t1]
    X_val,   y_val   = X[t1:t2], y[t1:t2]
    X_test,  y_test  = X[t2:],  y[t2:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  METRICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """R2, MAE, RMSE, MAPE, tolerance accuracy ±5% and ±10%."""
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    with np.errstate(divide="ignore", invalid="ignore"):
        mape_arr = np.where(y_true != 0,
                            np.abs((y_true - y_pred) / y_true) * 100, 0)
    mape = float(np.mean(mape_arr))

    tol5  = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9)) <= 0.05) * 100)
    tol10 = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9)) <= 0.10) * 100)

    return {
        "split":    label,
        "r2":       float(r2),
        "mae":      float(mae),
        "rmse":     float(rmse),
        "mape":     float(mape),
        "tol5_acc": tol5,
        "tol10_acc": tol10,
    }


def banner(text: str):
    """Print a prominent section banner."""
    bar = "=" * 68
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def init_db(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS blocks (
            network TEXT, number INTEGER, timestamp INTEGER,
            gas_used INTEGER, gas_limit INTEGER, base_fee_per_gas INTEGER,
            transaction_count INTEGER, size INTEGER,
            PRIMARY KEY (network, number)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            network TEXT, model TEXT,
            block_number INTEGER, predicted REAL, actual REAL,
            prediction_error REAL, created_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS model_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            network TEXT, model TEXT, split TEXT,
            r2 REAL, mae REAL, rmse REAL, mape REAL,
            tol5_acc REAL, tol10_acc REAL, created_at TEXT
        )
    """)
    con.commit()
    return con


def insert_blocks(con: sqlite3.Connection, network: str, df: pd.DataFrame):
    rows = df[["number", "timestamp", "gas_used", "gas_limit",
               "base_fee_per_gas", "transaction_count", "size"]].copy()
    rows["network"] = network
    rows.to_sql("blocks", con, if_exists="replace", index=False)
    con.commit()
    log.info("DB: inserted %d blocks for %s", len(rows), network)


def insert_scores(con: sqlite3.Connection, network: str, model: str, scores: list[dict]):
    now = datetime.datetime.utcnow().isoformat()
    for s in scores:
        con.execute("""
            INSERT INTO model_scores
              (network, model, split, r2, mae, rmse, mape, tol5_acc, tol10_acc, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (network, model, s["split"],
              s["r2"], s["mae"], s["rmse"], s["mape"],
              s["tol5_acc"], s["tol10_acc"], now))
    con.commit()


def insert_prediction(con: sqlite3.Connection, network: str, model: str,
                      block_number: int, predicted: float,
                      actual: float | None = None,
                      error: float | None = None):
    now = datetime.datetime.utcnow().isoformat()
    con.execute("""
        INSERT INTO predictions
          (network, model, block_number, predicted, actual, prediction_error, created_at)
        VALUES (?,?,?,?,?,?,?)
    """, (network, model, block_number, predicted, actual, error, now))
    con.commit()


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_all_models(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    network: str,
    con: sqlite3.Connection,
    df_feat: pd.DataFrame,
    feature_cols: list[str],
):
    """Train RF, DT, NB, DNN, Transformer and persistence baseline."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.tree import DecisionTreeRegressor

    all_scores: dict[str, dict] = {}  # model_name -> {train, val, test} metrics

    # Scale for DL models (fit ONLY on train)
    scaler = MinMaxScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_va_sc = scaler.transform(X_val)
    X_te_sc = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / f"{network}_scaler.pkl")

    # ── Persistence baseline ────────────────────────────────────────────────
    banner("BASELINE: Persistence (previous block = next block)")
    def _persist_pred(X, df_slice):
        """Return gas_used[i] as prediction for gas_used[i+1]."""
        return X[:, 0]   # feature_cols[0] is gas_used

    for split, (X_s, y_s) in [("train", (X_train, y_train)),
                                ("val",   (X_val,   y_val)),
                                ("test",  (X_test,  y_test))]:
        pred = X_s[:, 0]
        m = compute_metrics(y_s, pred, split)
        print(f"  Persistence [{split:5s}]  R²={m['r2']:+.4f}  MAE={m['mae']:,.0f}  "
              f"Tol10={m['tol10_acc']:.1f}%")
    all_scores["Persistence"] = {}

    # ── Random Forest ───────────────────────────────────────────────────────
    banner("TRAINING: Random Forest Regressor")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15,
                               min_samples_leaf=3, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_scores = []
    for split, (X_s, y_s) in [("train", (X_train, y_train)),
                                ("val",   (X_val,   y_val)),
                                ("test",  (X_test,  y_test))]:
        pred = rf.predict(X_s)
        m = compute_metrics(y_s, pred, split)
        rf_scores.append(m)
        print(f"  RF [{split:5s}]  R²={m['r2']:+.4f}  MAE={m['mae']:,.0f}  "
              f"RMSE={m['rmse']:,.0f}  Tol10={m['tol10_acc']:.1f}%")
    joblib.dump(rf, MODELS_DIR / f"{network}_rf.pkl")
    insert_scores(con, network, "RandomForest", rf_scores)
    all_scores["RandomForest"] = rf_scores

    # ── Decision Tree ───────────────────────────────────────────────────────
    banner("TRAINING: Decision Tree Regressor")
    dt = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_scores = []
    for split, (X_s, y_s) in [("train", (X_train, y_train)),
                                ("val",   (X_val,   y_val)),
                                ("test",  (X_test,  y_test))]:
        pred = dt.predict(X_s)
        m = compute_metrics(y_s, pred, split)
        dt_scores.append(m)
        print(f"  DT [{split:5s}]  R²={m['r2']:+.4f}  MAE={m['mae']:,.0f}  "
              f"RMSE={m['rmse']:,.0f}  Tol10={m['tol10_acc']:.1f}%")
    joblib.dump(dt, MODELS_DIR / f"{network}_dt.pkl")
    insert_scores(con, network, "DecisionTree", dt_scores)
    all_scores["DecisionTree"] = dt_scores

    # ── Naive Bayes (direction classification) ──────────────────────────────
    banner("TRAINING: Gaussian Naive Bayes (direction: UP/DOWN/SAME)")
    df_nb = df_feat.copy()
    n_total = len(df_nb)
    t1 = int(n_total * 0.70); t2 = int(n_total * 0.85)
    nb_target = df_nb["next_gas_direction"].values
    y_nb_train = nb_target[:t1]; y_nb_val = nb_target[t1:t2]; y_nb_test = nb_target[t2:]

    nb = GaussianNB()
    nb.fit(X_train, y_nb_train)
    for split, (X_s, y_s) in [("train", (X_train, y_nb_train)),
                                ("val",   (X_val,   y_nb_val)),
                                ("test",  (X_test,  y_nb_test))]:
        pred = nb.predict(X_s)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_s, pred) * 100
        print(f"  NB [{split:5s}]  Classification Accuracy = {acc:.2f}%")
    joblib.dump(nb, MODELS_DIR / f"{network}_nb.pkl")
    all_scores["NaiveBayes"] = {}

    # ── Deep Neural Network ─────────────────────────────────────────────────
    banner("TRAINING: Deep Neural Network (PyTorch, 256-128-64-32)")
    dnn_scores = _train_dnn(
        X_tr_sc, y_train, X_va_sc, y_val, X_te_sc, y_test,
        network, con
    )
    all_scores["DNN"] = dnn_scores

    # ── Transformer ─────────────────────────────────────────────────────────
    banner("TRAINING: Transformer Encoder (3 layers, 4 heads)")
    trans_scores = _train_transformer(
        X_tr_sc, y_train, X_va_sc, y_val, X_te_sc, y_test,
        network, con
    )
    all_scores["Transformer"] = trans_scores

    return rf, dt, nb, all_scores, scaler


def _train_dnn(X_tr, y_tr, X_va, y_va, X_te, y_te, network, con):
    """Train a 5-hidden-layer feed-forward DNN with PyTorch."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class DNN(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, 32),      nn.ReLU(),
                nn.Linear(32, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)

    model = DNN(X_tr.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0

    print("  Epoch  TrainLoss    ValLoss")
    for epoch in range(1, 101):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_tr)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t.to(device)), y_va_t.to(device)).item()

        sched.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{epoch:3d}]   {ep_loss:12.1f}   {val_loss:12.1f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 15:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / f"{network}_dnn.pt")

    scores = []
    model.eval()
    with torch.no_grad():
        for split, (X_t, y_s) in [("train", (X_tr_t, y_tr)),
                                    ("val",   (X_va_t, y_va)),
                                    ("test",  (X_te_t, y_te))]:
            pred = model(X_t.to(device)).cpu().numpy()
            m = compute_metrics(y_s, pred, split)
            scores.append(m)
            print(f"  DNN [{split:5s}]  R²={m['r2']:+.4f}  MAE={m['mae']:,.0f}  "
                  f"Tol10={m['tol10_acc']:.1f}%")

    insert_scores(con, network, "DNN", scores)
    return scores


def _train_transformer(X_tr, y_tr, X_va, y_va, X_te, y_te, network, con):
    """Train a Transformer Encoder for time-series regression."""
    import math
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class TransformerModel(nn.Module):
        def __init__(self, in_dim, d_model=64, nhead=4, nlayers=3, dropout=0.1):
            super().__init__()
            self.proj   = nn.Linear(in_dim, d_model)
            self.pos_enc = PositionalEncoding(d_model)
            enc_layer   = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                                      dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
            self.head    = nn.Linear(d_model, 1)

        def forward(self, x):
            x = x.unsqueeze(1)           # (B, 1, in_dim)
            x = self.proj(x)             # (B, 1, d_model)
            x = self.pos_enc(x)
            x = self.encoder(x)
            return self.head(x[:, -1]).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torch.utils.data import DataLoader, TensorDataset
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)

    model = TransformerModel(X_tr.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_cnt = 0

    print("  Epoch  TrainLoss    ValLoss")
    for epoch in range(1, 101):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_tr)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t.to(device)), y_va_t.to(device)).item()

        sched.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{epoch:3d}]   {ep_loss:12.1f}   {val_loss:12.1f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 15:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS_DIR / f"{network}_transformer.pt")

    scores = []
    model.eval()
    with torch.no_grad():
        for split, (X_t, y_s) in [("train", (X_tr_t, y_tr)),
                                    ("val",   (X_va_t, y_va)),
                                    ("test",  (X_te_t, y_te))]:
            pred = model(X_t.to(device)).cpu().numpy()
            m = compute_metrics(y_s, pred, split)
            scores.append(m)
            print(f"  TRF [{split:5s}]  R²={m['r2']:+.4f}  MAE={m['mae']:,.0f}  "
                  f"Tol10={m['tol10_acc']:.1f}%")

    insert_scores(con, network, "Transformer", scores)
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  SAVE RESULTS & PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(all_scores: dict, network: str, df_feat: pd.DataFrame,
                 feature_cols: list[str], rf_model, dt_model):
    """Save CSV comparison table and generate key plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Scores summary CSV ─────────────────────────────────────────────────
    rows = []
    for model_name, score_list in all_scores.items():
        if not score_list:
            continue
        for s in score_list:
            rows.append({
                "network":  network,
                "model":    model_name,
                "split":    s.get("split", ""),
                "r2":       s.get("r2", np.nan),
                "mae":      s.get("mae", np.nan),
                "rmse":     s.get("rmse", np.nan),
                "mape":     s.get("mape", np.nan),
                "tol5_acc":  s.get("tol5_acc", np.nan),
                "tol10_acc": s.get("tol10_acc", np.nan),
            })
    scores_df = pd.DataFrame(rows)
    csv_out = OUTPUTS_DIR / "metrics" / f"scores_{network}.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(csv_out, index=False)
    print(f"\n  ✅ Scores saved → {csv_out}")

    # ── Test-set bar charts ────────────────────────────────────────────────
    test_df = scores_df[scores_df["split"] == "test"].copy()
    if not test_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(test_df["model"], test_df["r2"], color="steelblue")
        ax.set_title(f"[{network.upper()}] Test R² by Model")
        ax.set_ylabel("R²")
        ax.set_xlabel("Model")
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
        plt.tight_layout()
        fig.savefig(FIGS_DIR / f"{network}_test_r2_comparison.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(test_df["model"], test_df["mae"], color="tomato")
        ax.set_title(f"[{network.upper()}] Test MAE by Model")
        ax.set_ylabel("MAE (gas units)")
        ax.set_xlabel("Model")
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
        plt.tight_layout()
        fig.savefig(FIGS_DIR / f"{network}_test_mae_comparison.png", dpi=150)
        plt.close(fig)

    # ── Actual vs Predicted (last 500 test blocks, Random Forest) ─────────
    n = len(df_feat)
    t2 = int(n * 0.85)
    X_test = df_feat[feature_cols].values[t2:]
    y_test = df_feat["next_gas_used"].values[t2:]
    last500 = min(500, len(y_test))
    y_pred  = rf_model.predict(X_test[-last500:])
    y_true  = y_test[-last500:]
    block_nums = df_feat["number"].values[t2:][-last500:]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(block_nums, y_true,  label="Actual",    color="steelblue", lw=1)
    ax.plot(block_nums, y_pred,  label="Predicted", color="tomato",    lw=1, alpha=0.8)
    ax.set_title(f"[{network.upper()}] Actual vs Predicted Gas Used — Last 500 Test Blocks (RF)")
    ax.set_xlabel("Block Number"); ax.set_ylabel("Gas Used")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGS_DIR / f"{network}_actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    print(f"  ✅ Plots saved → {FIGS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  BEST MODEL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

def print_best_model_verdict(all_scores: dict, network: str):
    banner("BEST MODEL VERDICT")
    test_rows = []
    for name, score_list in all_scores.items():
        if not score_list:
            continue
        for s in score_list:
            if s.get("split") == "test":
                test_rows.append({"model": name, **s})

    if not test_rows:
        print("  (No test scores to compare)")
        return

    df = pd.DataFrame(test_rows).sort_values("r2", ascending=False)
    print(f"\n  {'Model':<18} {'TestR²':>8} {'MAE':>14} {'Tol10%':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*14} {'-'*8}")
    for _, row in df.iterrows():
        print(f"  {row['model']:<18} {row['r2']:>+8.4f} {row['mae']:>14,.0f} "
              f"{row['tol10_acc']:>7.1f}%")

    best = df.iloc[0]
    print(f"\n  🏆 WINNER: {best['model']}")
    print(f"     Test R²={best['r2']:+.4f}  MAE={best['mae']:,.0f}  Tol10={best['tol10_acc']:.1f}%")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  LIVE LISTENER MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_live_listener(network: str, rf_model, scaler, feature_cols: list[str],
                      con: sqlite3.Connection):
    """Watch the chain for new blocks and predict the next one in real time."""
    sys.path.insert(0, str(SRC_DIR))
    from fetch_blocks import live_listener, RpcClient

    banner(f"LIVE LISTENER — {network.upper()} (Ctrl+C to stop)")

    def predict_next(window_df: pd.DataFrame) -> float:
        """Feature-engineer the latest window and predict next gas_used."""
        window_df = window_df.copy()
        # Add required columns if missing
        if "base_fee_gwei" not in window_df.columns:
            window_df["base_fee_gwei"] = window_df["base_fee_per_gas"] / 1e9
        if "gas_utilization" not in window_df.columns:
            window_df["gas_utilization"] = (
                window_df["gas_used"] / window_df["gas_limit"].replace(0, np.nan)
            ).fillna(0)
        if "block_time" not in window_df.columns:
            window_df["block_time"] = window_df["timestamp"].diff().fillna(12)
        if "transaction_count" not in window_df.columns:
            window_df["transaction_count"] = 0

        # Build feature row from the LAST block in the window
        last = window_df.iloc[-1]
        # Lag values from window history
        row = {}
        for col in feature_cols:
            if col in window_df.columns:
                # Try to construct from shift of the window
                if "_lag" in col:
                    base, lag_str = col.rsplit("_lag", 1)
                    lag = int(lag_str)
                    idx = -(lag + 1)
                    row[col] = float(window_df[base].iloc[idx]) if len(window_df) > lag else 0.0
                elif "_roll" in col or "_delta" in col or "_std" in col:
                    row[col] = float(window_df[col].iloc[-1]) if col in window_df.columns else 0.0
                else:
                    row[col] = float(last.get(col, 0))
            else:
                row[col] = 0.0

        X_live = np.array([[row[c] for c in feature_cols]])
        return float(rf_model.predict(X_live)[0])

    def db_writer(block_num, predicted, actual, error):
        insert_prediction(con, network, "RandomForest",
                          block_num, predicted, actual, error)

    live_listener(
        network=network,
        model_fn=predict_next,
        db_fn=db_writer,
        poll_interval=12,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ethereum ML Experiment Runner")
    parser.add_argument("--network", default="sepolia",
                        choices=["mainnet", "sepolia"],
                        help="Blockchain network (default: sepolia)")
    parser.add_argument("--fetch-new", type=int, default=0, metavar="N",
                        help="Fetch N fresh blocks from the real chain before training")
    parser.add_argument("--live", action="store_true",
                        help="After training, enter live block-listener mode")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training (re-use saved models)")
    args = parser.parse_args()

    network = args.network

    banner(f"ETHEREUM ML EXPERIMENT — {network.upper()}")
    print(f"  Team: Shaik Rabbani, P. Praneeth, P. Prabath | SRM University - AP")
    print(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── STEP 1: (Optionally) fetch fresh blocks ────────────────────────────
    if args.fetch_new > 0:
        banner(f"STEP 1: Fetching {args.fetch_new} fresh {network} blocks from chain")
        sys.path.insert(0, str(SRC_DIR))
        from fetch_blocks import fetch_blocks, save_blocks
        try:
            blocks = fetch_blocks(network=network, n_blocks=args.fetch_new)
            if blocks:
                csv_path = save_blocks(blocks, network)
                print(f"  ✅ Fetched {len(blocks)} blocks → {csv_path}")
            else:
                print("  ⚠️  No blocks fetched — using existing data")
        except Exception as exc:
            log.warning("Block fetch failed (%s) — using existing data", exc)
    else:
        banner("STEP 1: Loading existing block data (use --fetch-new N to refresh)")

    # ── STEP 2: Load & preprocess ──────────────────────────────────────────
    banner("STEP 2: Loading & Feature Engineering")
    csv_path = DATA_DIR / f"{network}_blocks_raw.csv"
    if not csv_path.exists():
        log.error("No data at %s — run with --fetch-new 500 first!", csv_path)
        sys.exit(1)

    raw_df = pd.read_csv(csv_path)
    print(f"  Loaded {len(raw_df):,} raw blocks  "
          f"(#{int(raw_df['number'].min())} → #{int(raw_df['number'].max())})")

    df_feat = build_features(raw_df)
    print(f"  After feature engineering: {len(df_feat):,} usable rows, "
          f"{df_feat.shape[1]} columns")

    # Feature columns: everything except meta, target, direction
    exclude = {"number", "timestamp", "miner", "size", "extra_data_len",
               "next_gas_used", "next_gas_direction",
               "gas_used", "base_fee_per_gas"}  # exclude raw targets but keep derived
    feature_cols = [c for c in df_feat.columns if c not in exclude]
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols[:6]} …")

    # ── STEP 3: Split ──────────────────────────────────────────────────────
    banner("STEP 3: Chronological 70/15/15 Split (no shuffle)")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        df_feat, feature_cols
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ── STEP 4: DB init ────────────────────────────────────────────────────
    con = init_db(DB_PATH)
    insert_blocks(con, network, raw_df)

    # ── STEP 5: Train ──────────────────────────────────────────────────────
    if not args.skip_training:
        banner("STEP 4: TRAINING STARTED")
        rf, dt, nb, all_scores, scaler = train_all_models(
            X_train, y_train, X_val, y_val, X_test, y_test,
            network, con, df_feat, feature_cols
        )
    else:
        banner("STEP 4: Skipping training — loading saved models")
        rf     = joblib.load(MODELS_DIR / f"{network}_rf.pkl")
        dt     = joblib.load(MODELS_DIR / f"{network}_dt.pkl")
        scaler = joblib.load(MODELS_DIR / f"{network}_scaler.pkl")
        nb     = joblib.load(MODELS_DIR / f"{network}_nb.pkl")
        all_scores = {}

    # ── STEP 6: Results ────────────────────────────────────────────────────
    banner("STEP 5: TESTING COMPLETE — Saving Results")
    save_results(all_scores, network, df_feat, feature_cols, rf, dt)

    # ── STEP 7: Verdict ────────────────────────────────────────────────────
    print_best_model_verdict(all_scores, network)

    # ── STEP 8: Next-block prediction (all models) ─────────────────────────
    banner("STEP 6: NEXT-BLOCK PREDICTION (from latest data)")
    last_window = df_feat.tail(50).copy()
    last_block  = int(df_feat["number"].iloc[-1])
    next_block  = last_block + 1

    # Persistence baseline
    last_gas = float(df_feat["gas_used"].iloc[-1])
    print(f"\n  Predicting gas_used for block #{next_block}:")
    print(f"  {'Model':<18} {'Prediction':>18}")
    print(f"  {'-'*18} {'-'*18}")
    print(f"  {'Persistence':<18} {last_gas:>18,.0f}")

    # RF prediction
    last_X = df_feat[feature_cols].iloc[[-1]].values
    rf_pred = float(rf.predict(last_X)[0])
    print(f"  {'RandomForest':<18} {rf_pred:>18,.0f}")
    insert_prediction(con, network, "RandomForest", next_block, rf_pred)

    dt_pred = float(dt.predict(last_X)[0])
    print(f"  {'DecisionTree':<18} {dt_pred:>18,.0f}")
    insert_prediction(con, network, "DecisionTree", next_block, dt_pred)

    # DNN prediction
    try:
        import torch, torch.nn as nn
        X_sc = scaler.transform(last_X)
        X_t  = torch.tensor(X_sc, dtype=torch.float32)

        class DNN(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(64, 32),      nn.ReLU(),
                    nn.Linear(32, 1),
                )
            def forward(self, x): return self.net(x).squeeze(-1)

        dnn = DNN(X_sc.shape[1])
        dnn_path = MODELS_DIR / f"{network}_dnn.pt"
        if dnn_path.exists():
            dnn.load_state_dict(torch.load(str(dnn_path), map_location="cpu"))
            dnn.eval()
            with torch.no_grad():
                dnn_pred = float(dnn(X_t).item())
            print(f"  {'DNN':<18} {dnn_pred:>18,.0f}")
            insert_prediction(con, network, "DNN", next_block, dnn_pred)
    except Exception as exc:
        log.warning("DNN prediction skipped: %s", exc)

    con.close()

    banner("EXPERIMENT COMPLETE")
    print(f"  Results → {OUTPUTS_DIR}")
    print(f"  Plots   → {FIGS_DIR}")
    print(f"  DB      → {DB_PATH}")
    print()

    # ── STEP 9: Live listener ──────────────────────────────────────────────
    if args.live:
        con = init_db(DB_PATH)
        run_live_listener(network, rf, scaler, feature_cols, con)


if __name__ == "__main__":
    main()
