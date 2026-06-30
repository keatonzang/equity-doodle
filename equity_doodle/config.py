"""Central configuration: ticker universe, horizons, and paths."""

from __future__ import annotations

from pathlib import Path

# --- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# --- Ticker universe ------------------------------------------------------
# A spread of large, highly liquid US equities + a few broad ETFs, so the
# global model sees a variety of price "regimes" rather than one stock.
TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "JNJ", "XOM", "WMT", "PG", "KO", "DIS", "INTC",
    "SPY", "QQQ", "DIA", "IWM",
]

# --- Data window ----------------------------------------------------------
START_DATE = "2010-01-01"
PRICE_COLUMN = "Close"   # use adjusted close; see data.py

# --- Forecasting setup ----------------------------------------------------
# Lengths are measured in *bars*, not days -- the model is time-scale agnostic.
CONTEXT_LENGTH = 60      # lookback the model conditions on (input_chunk_length)
HORIZON = 20             # bars to forecast / "doodle" forward
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]  # for the probability fan

# --- Scale invariance -----------------------------------------------------
# To train on many time resolutions at once, each daily series is aggregated
# into non-overlapping block-sums of log-returns at these scales (k bars per
# block). k=1 is daily, 5 ~ weekly, 21 ~ monthly. Combined with per-window
# z-scoring (see features.standardize_returns), this lets the model both train
# on, and complete a chart drawn at, an arbitrary time resolution.
SCALES = [1, 2, 3, 5, 10, 21]
