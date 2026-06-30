"""High-level forecast entry point: a drawn price path in, a fan of completions out.

This is the function the future web app calls: the user's doodle becomes the
model's context window, and we return a price-space forecast for each quantile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features


def complete_doodle(drawn_prices, model, horizon: int = config.HORIZON) -> dict:
    """Complete a hand-drawn chart.

    Parameters
    ----------
    drawn_prices : sequence of float
        The price path the user drew (length should be >= CONTEXT_LENGTH + 1).
    model : trained Darts model
        A quantile model from :mod:`equity_doodle.model`.
    horizon : int
        Number of future steps to "doodle".

    Returns
    -------
    dict with keys:
        ``"context"``  -> the (normalized) input price path
        ``"quantiles"`` -> {q: np.ndarray price path} for each configured quantile
    """
    from darts import TimeSeries

    prices = features.normalize_drawn_path(np.asarray(drawn_prices, dtype="float64"))
    rets = features.to_log_returns(prices)

    # Wrap returns in a TimeSeries with a synthetic daily index.
    idx = pd.date_range("2000-01-01", periods=len(rets), freq="B")
    ctx = TimeSeries.from_times_and_values(idx, rets.astype("float32"))

    pred = model.predict(n=horizon, series=ctx, num_samples=500)

    last_price = float(prices[-1])
    out = {}
    for q in config.QUANTILES:
        q_rets = pred.quantile_timeseries(q).values().flatten()
        out[q] = features.from_log_returns(q_rets, last_price)

    return {"context": prices, "quantiles": out}
