"""Data pipeline: download and cache daily equity prices via yfinance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from . import config


def download_prices(
    tickers: list[str] | None = None,
    start: str = config.START_DATE,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download daily close prices for ``tickers`` into a wide DataFrame.

    Columns are tickers, the index is the trading date. ``auto_adjust=True``
    gives split/dividend-adjusted closes, which is what we want for modeling.
    """
    tickers = tickers or config.TICKERS
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )
    # With multiple tickers yfinance returns a column MultiIndex; grab "Close".
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.sort_index().dropna(how="all")
    return close


def cache_prices(df: pd.DataFrame, name: str = "prices.parquet") -> str:
    """Persist a price frame to the local (git-ignored) data directory."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / name
    df.to_parquet(path)
    return str(path)


def load_cached(name: str = "prices.parquet") -> pd.DataFrame:
    """Load a previously cached price frame."""
    return pd.read_parquet(config.DATA_DIR / name)
