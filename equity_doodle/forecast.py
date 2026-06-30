"""High-level forecast entry point: a drawn price path in, a fan of completions out.

This is the function the future web app calls: the user's doodle becomes the
model's context window, and we return a price-space forecast for each quantile.

Everything runs in *standardized log-return* space, which is what makes the
system time-scale invariant -- see :mod:`equity_doodle.features`.
"""

from __future__ import annotations

import numpy as np

from . import config, features


def predict_returns(
    model, context_returns, horizon: int = config.HORIZON, num_samples: int = 500
) -> dict:
    """Forecast future log-returns from a raw (un-normalized) return context.

    The context is z-scored, fed to the model, and the predicted quantiles are
    mapped back to the context's own volatility scale. Shared by the web app and
    the backtest so both behave identically.

    Returns ``{quantile: np.ndarray of length ``horizon``}`` in raw return space.
    """
    from darts import TimeSeries

    z, mean, std = features.standardize_returns(context_returns)
    ctx = TimeSeries.from_values(z.astype("float32"))
    pred = model.predict(n=horizon, series=ctx, num_samples=num_samples)
    return {
        q: features.destandardize_returns(pred.quantile(q).values().flatten(), mean, std)
        for q in config.QUANTILES
    }


def complete_doodle(drawn_prices, model, horizon: int = config.HORIZON) -> dict:
    """Complete a hand-drawn chart at any time resolution.

    Parameters
    ----------
    drawn_prices : sequence of float
        The price path the user drew, with arbitrary length and price scale.
    model : trained Darts model from :mod:`equity_doodle.model`.
    horizon : int
        Number of future bars to "doodle".

    Returns
    -------
    dict with keys:
        ``"context"``   -> the (normalized) input price path
        ``"quantiles"`` -> {q: np.ndarray price path} for each configured quantile
    """
    # Resample to a fixed bar-count so point-count doesn't matter, then work in
    # returns so price scale doesn't matter.
    prices = features.resample_path(
        np.asarray(drawn_prices, dtype="float64"), config.CONTEXT_LENGTH + 1
    )
    prices = features.normalize_drawn_path(prices)
    rets = features.to_log_returns(prices)

    q_rets = predict_returns(model, rets, horizon)
    last_price = float(prices[-1])
    quantiles = {q: features.from_log_returns(r, last_price) for q, r in q_rets.items()}
    return {"context": prices, "quantiles": quantiles}
