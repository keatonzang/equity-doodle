"""The global forecasting model.

A single Darts ``LightGBMModel`` is trained across every ticker's log-return
series at once (a "global" model). Quantile regression gives a probabilistic
fan of completions instead of a single line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features


def _build_return_series(prices: pd.DataFrame):
    """Turn a wide price frame into a list of Darts TimeSeries of log-returns."""
    from darts import TimeSeries

    series = []
    for ticker in prices.columns:
        col = prices[ticker].dropna()
        if len(col) < config.CONTEXT_LENGTH + config.HORIZON + 5:
            continue  # not enough history to be useful
        rets = features.to_log_returns(col.to_numpy())
        idx = col.index[1:]  # returns align to the later of each pair
        ts = TimeSeries.from_times_and_values(
            pd.DatetimeIndex(idx), rets.astype("float32")
        )
        series.append(ts)
    if not series:
        raise ValueError("no ticker had enough history to build a series")
    return series


def build_model():
    """Construct the untrained global quantile model."""
    from darts.models import LightGBMModel

    return LightGBMModel(
        lags=config.CONTEXT_LENGTH,
        output_chunk_length=config.HORIZON,
        likelihood="quantile",
        quantiles=config.QUANTILES,
        verbose=-1,
    )


def train(prices: pd.DataFrame):
    """Fit the global model on all tickers' log-return series."""
    model = build_model()
    model.fit(_build_return_series(prices))
    return model


def save(model, name: str = "global_lgbm.pkl") -> str:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODEL_DIR / name
    model.save(str(path))
    return str(path)


def load(name: str = "global_lgbm.pkl"):
    from darts.models import LightGBMModel

    return LightGBMModel.load(str(config.MODEL_DIR / name))
