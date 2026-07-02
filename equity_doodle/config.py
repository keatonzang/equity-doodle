"""Central configuration: ticker universe, horizons, and paths."""

from __future__ import annotations

from pathlib import Path

# --- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# --- Ticker universe ------------------------------------------------------
# A broad spread of highly liquid US equities + sector/asset ETFs, so the global
# model sees many price "regimes" rather than one stock. A bigger, more varied
# universe gives a deep learner far more shape-examples to learn from.
TICKERS: list[str] = [
    # mega-cap / large tech + semis + software
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "ADBE", "CRM", "ORCL", "CSCO", "AMD", "QCOM", "TXN", "INTC", "NFLX", "IBM",
    "NOW", "INTU", "AMAT", "MU", "LRCX", "ADI",
    # financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP", "BLK", "SCHW", "SPGI",
    # health care
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "CVS",
    # consumer
    "WMT", "PG", "KO", "PEP", "MCD", "NKE", "HD", "COST", "DIS", "SBUX", "LOW",
    "TGT", "CL", "MDLZ", "BKNG", "TJX",
    # industrials / energy / materials
    "XOM", "CVX", "COP", "SLB", "BA", "CAT", "GE", "HON", "UNP", "UPS", "LMT",
    "RTX", "DE", "MMM", "FCX", "NEM",
    # communications
    "T", "VZ", "CMCSA", "TMUS",
    # broad / sector / asset ETFs
    "SPY", "QQQ", "DIA", "IWM", "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP",
    "GLD", "SLV", "TLT", "HYG", "EEM", "EFA",

    # --- New domains: added for tail/regime DIVERSITY (better fan coverage) ---
    # The scaling study showed extra data buys calibration, not direction, and
    # uncorrelated / new-domain assets carry the most novel tail behaviour.

    # US mid- & small-cap equities (higher idiosyncratic vol, sharper drawdowns)
    "DKS", "BURL", "DECK", "WSM", "WING", "CROX", "CIEN", "JBL", "EME", "MANH",
    "SAIA", "ANF", "SKX", "TPR", "RL", "PSTG", "CVLT", "EXP", "KBH", "TOL",
    "PII", "UFPI", "GT", "AA", "CLF", "X", "MOS", "CF", "LSCC", "AMKR",
    "COHR", "ONTO", "IJR", "MDY", "VB",
    # International (liquid ADRs + country / region ETFs)
    "TSM", "ASML", "BABA", "NVO", "TM", "SAP", "SHOP", "MELI", "SE", "NVS",
    "AZN", "BP", "SHEL", "TTE", "RIO", "BHP", "UL", "SONY", "PDD", "JD",
    "STLA", "RY", "TD", "INFY",
    "EWJ", "EWZ", "EWG", "EWU", "EWY", "EWT", "FXI", "INDA", "EWW", "VGK",
    "VWO", "ILF",
    # Crypto (24/7, fat non-Gaussian tails) + volatility products + proxies
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
    "LTC-USD", "BCH-USD", "LINK-USD", "AVAX-USD",
    "VIXY", "UVXY", "VXX", "COIN", "MSTR", "MARA",
    # Commodities (distinct vol structure)
    "USO", "UNG", "DBC", "DBA", "DBB", "CORN", "WEAT", "PPLT", "PALL", "UGA",
    "GDX", "GDXJ", "URA", "LIT", "COPX",
    # Rates / bonds (low-vol, different drawdown shape)
    "IEF", "SHY", "LQD", "AGG", "BND", "EMB", "MUB", "TIP", "JNK", "IEI",
    # FX (uncorrelated, mean-reverting)
    "UUP", "FXE", "FXY", "FXB", "FXF", "FXA", "FXC",
]

# --- Data window ----------------------------------------------------------
# Pushed back to 2005 to roughly double the history for long-lived names; newer
# tickers simply contribute from their own inception (per-ticker dropna).
START_DATE = "2005-01-01"
PRICE_COLUMN = "Close"   # use adjusted close; see data.py

# --- Forecasting setup ----------------------------------------------------
# Lengths are measured in *bars*, not days -- the model is time-scale agnostic.
CONTEXT_LENGTH = 60      # lookback the model conditions on (input_chunk_length)
HORIZON = 20             # bars to forecast / "doodle" forward
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]  # for the probability fan

# --- Model selection ------------------------------------------------------
# "tide"     -> TiDE, a Torch deep learner (encoder-decoder) with quantile
#               regression. Trains on GPU when available.
# "lightgbm" -> the original gradient-boosted baseline (CPU, fast).
# Both expose the same probabilistic interface, so the web app, forecast layer,
# and backtest are model-agnostic.
MODEL_TYPE = "nhits"

# Deep-learner (TiDE) hyper-parameters. Kept modest so it trains in minutes on a
# single GPU over the multi-scale series pool.
DL_HIDDEN_SIZE = 128
DL_NUM_ENCODER_LAYERS = 2
DL_NUM_DECODER_LAYERS = 2
DL_DECODER_OUTPUT_DIM = 16
DL_TEMPORAL_DECODER_HIDDEN = 64
DL_DROPOUT = 0.1
DL_BATCH_SIZE = 1024
DL_N_EPOCHS = 60          # generous budget; early stopping (patience 8) decides
DL_LEARNING_RATE = 1e-3
RANDOM_STATE = 42

# --- Fan calibration ------------------------------------------------------
# Quantile models are systematically overconfident here: the nominal 10-90%
# band only covers ~60% of realized moves. We widen each quantile's deviation
# from the median by this conformal-style factor (fit on a held-out slice by
# scripts/calibrate.py) so the displayed fan is honest. 1.0 = no calibration.
CALIBRATION_LAMBDA = 1.5

# --- Scale invariance -----------------------------------------------------
# To train on many time resolutions at once, each daily series is aggregated
# into non-overlapping block-sums of log-returns at these scales (k bars per
# block). k=1 is daily, 5 ~ weekly, 21 ~ monthly. Combined with per-window
# z-scoring (see features.standardize_returns), this lets the model both train
# on, and complete a chart drawn at, an arbitrary time resolution.
SCALES = [1, 2, 3, 5, 10, 21]
