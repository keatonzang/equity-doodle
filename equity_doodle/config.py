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

    # --- Big-corpus expansion (2026-07): diversity + deep history ----------
    # The scaling study says redundant windows don't help; these add what does:
    # decades-longer histories (indices, futures, FX pairs), independent
    # regimes, and hundreds more idiosyncratic equity paths. Dead/renamed
    # symbols are harmless -- all-NaN columns are dropped at download time.

    # Indices (longest histories available; many pre-2005 regimes)
    "^GSPC", "^DJI", "^IXIC", "^RUT", "^NYA", "^N225", "^FTSE", "^GDAXI",
    "^FCHI", "^HSI", "^AXJO", "^BVSP", "^MXX", "^KS11", "^TWII", "^STOXX50E",
    "^VIX", "^TNX", "^TYX", "^FVX",
    # FX pairs (far longer history than the currency ETFs above)
    "EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "NZDUSD=X", "CAD=X", "CHF=X",
    "MXN=X", "ZAR=X", "SEK=X", "NOK=X", "SGD=X", "INR=X",
    "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    # Futures front months (commodities/rates with decades of daily bars)
    "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F", "PL=F", "PA=F",
    "ZC=F", "ZS=F", "ZW=F", "KC=F", "CT=F", "SB=F", "CC=F", "LE=F", "HE=F",
    "ZN=F", "ZB=F",
    # More US large caps (long histories, sectors thin in the base list)
    "BRK-B", "ACN", "MO", "PM", "KMB", "GIS", "K", "HSY", "STZ", "EL", "YUM",
    "CMG", "DPZ", "ROST", "DG", "DLTR", "ORLY", "AZO", "EBAY", "PYPL", "ADSK",
    "CDNS", "SNPS", "ANSS", "KLAC", "MCHP", "NXPI", "ON", "SWKS", "TER",
    "ZBRA", "PANW", "FTNT", "USB", "PNC", "TFC", "FITB", "KEY", "RF", "CFG",
    "HBAN", "MTB", "ALLY", "COF", "DFS", "SYF", "AIG", "MET", "PRU", "AFL",
    "ALL", "TRV", "PGR", "CB", "HIG", "BX", "KKR", "APO", "TROW", "BEN",
    "NTRS", "STT", "BK", "CBOE", "CME", "ICE", "NDAQ", "MSCI", "MCO", "EFX",
    "FICO", "VRSK", "ISRG", "SYK", "BSX", "MDT", "EW", "ZBH", "BAX", "BDX",
    "RMD", "ALGN", "DXCM", "IDXX", "A", "WAT", "MTD", "ZTS", "VRTX", "REGN",
    "BIIB", "CI", "HUM", "CNC", "ELV", "HCA", "MCK", "CAH", "WBA", "LH",
    "DGX", "EMR", "ETN", "PH", "ROK", "ITW", "CMI", "PCAR", "WAB", "IR",
    "TT", "CARR", "OTIS", "JCI", "CSX", "NSC", "ODFL", "JBHT", "FDX", "EXPD",
    "WM", "RSG", "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "ED",
    "AWK", "ALB", "VMC", "MLM", "NUE", "STLD", "DOW", "LYB", "DD", "PPG",
    "SHW", "ECL", "APD", "LIN", "IFF", "HAL", "OXY", "DVN", "FANG", "EOG",
    "MPC", "VLO", "PSX", "KMI", "WMB", "OKE", "LNG",
    # REITs (rate-sensitive, distinct drawdown dynamics)
    "PLD", "AMT", "CCI", "EQIX", "DLR", "PSA", "O", "SPG", "VICI", "WELL",
    "AVB", "EQR", "KIM", "HST", "EXR", "IRM", "SBAC", "WY",
    # Consumer / travel / autos / media (cyclical + meme-era paths)
    "F", "GM", "HMC", "RACE", "DAL", "UAL", "AAL", "LUV", "CCL", "RCL",
    "NCLH", "MAR", "HLT", "MGM", "WYNN", "LVS", "DKNG", "EXPE", "ULTA",
    "LULU", "BBY", "M", "KSS", "VFC", "PVH", "LEVI", "ADM", "BG", "TSN",
    "HRL", "CAG", "CPB", "SJM", "MKC", "SPOT", "ROKU", "TTD", "SNAP", "PINS",
    "MTCH", "EA", "TTWO", "PARA", "WBD", "FOXA",
    # High-vol growth / EV / recent-IPO shapes
    "UBER", "ABNB", "DASH", "RBLX", "PLTR", "HOOD", "SOFI", "GME", "CVNA",
    "CHWY", "W", "PTON", "NIO", "XPEV", "LI", "RIVN", "LCID", "CRWD", "DDOG",
    "NET", "SNOW", "TEAM", "WDAY", "FSLR", "ENPH", "PLUG",
    # Precious-metal miners / royalty (gold-beta regimes)
    "RGLD", "FNV", "WPM", "AEM", "KGC", "PAAS", "AG",
    # More crypto majors (24/7 fat tails)
    "BNB-USD", "TRX-USD", "DOT-USD", "XLM-USD", "ETC-USD", "XMR-USD",
    "ATOM-USD", "FIL-USD", "NEAR-USD", "ALGO-USD", "VET-USD", "HBAR-USD",
]

# --- Data window ----------------------------------------------------------
# Pushed back to 1980: indices, futures, and FX pairs carry decades of extra
# regimes (1987, dot-com, 2008...); newer tickers simply contribute from their
# own inception (per-ticker dropna).
START_DATE = "1980-01-01"
PRICE_COLUMN = "Close"   # use adjusted close; see data.py

# --- Intraday data ---------------------------------------------------------
# Pooled into training alongside the daily panel so intraday dynamics are
# in-domain (the /live page forecasts 1-minute bars). yfinance lookback limits:
# 1h <= 730 days, 15m <= 60 days.
INTRADAY = {             # interval -> (lookback period, cache filename)
    "1h": ("730d", "prices_1h.parquet"),
    "15m": ("60d", "prices_15m.parquet"),
}

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
# The expanded corpus has ~17M windows; a full pass is ~140k steps (~1 hour),
# which would make per-epoch validation (and early stopping) glacial. Cap each
# "epoch" to a random slice so validation runs every few minutes -- with the
# 60-epoch budget the model can still see up to ~28 full passes; early stopping
# decides when to quit, exactly as before.
DL_LIMIT_TRAIN_BATCHES = 8000   # steps per epoch (~8.2M samples)
DL_LIMIT_VAL_BATCHES = 400      # val steps per epoch (~410k samples)

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
