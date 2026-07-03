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
    # Drop tickers that failed to download / have no data (all-NaN columns), so a
    # few bad symbols in a large universe don't pollute the cache.
    close = close.dropna(axis=1, how="all")
    return close


def download_intraday(
    interval: str,
    period: str,
    tickers: list[str] | None = None,
    chunk: int = 40,
) -> pd.DataFrame:
    """Download intraday close prices for the universe, chunked to stay polite
    with Yahoo. Same wide (timestamp x ticker) shape as :func:`download_prices`;
    the union index interleaves 24/7 crypto bars with exchange-hours equity
    bars, and per-ticker ``dropna`` downstream handles that fine.
    """
    import time

    tickers = tickers or config.TICKERS
    frames = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        raw = yf.download(
            batch,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if raw.empty:
            continue
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        frames.append(close)
        time.sleep(1.0)
    if not frames:
        raise ValueError(f"no intraday data returned for interval={interval}")
    out = pd.concat(frames, axis=1).sort_index()
    return out.dropna(how="all").dropna(axis=1, how="all")


def load_training_frames() -> list[pd.DataFrame]:
    """All cached price panels for training: daily + any intraday caches."""
    frames = [load_cached()]
    for _interval, (_period, name) in config.INTRADAY.items():
        if (config.DATA_DIR / name).exists():
            frames.append(load_cached(name))
    return frames


def cache_prices(df: pd.DataFrame, name: str = "prices.parquet") -> str:
    """Persist a price frame to the local (git-ignored) data directory."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / name
    df.to_parquet(path)
    return str(path)


def load_cached(name: str = "prices.parquet") -> pd.DataFrame:
    """Load a previously cached price frame."""
    return pd.read_parquet(config.DATA_DIR / name)
