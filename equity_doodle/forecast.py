"""High-level forecast entry point: a drawn price path in, a fan of completions out.

This is the function the future web app calls: the user's doodle becomes the
model's context window, and we return a price-space forecast for each quantile.

Everything runs in *standardized log-return* space, which is what makes the
system time-scale invariant -- see :mod:`equity_doodle.features`.
"""

from __future__ import annotations

import numpy as np

from . import config, features


def _calibrate(quantiles: dict, sample=None):
    """Widen the fan around its median by ``config.CALIBRATION_LAMBDA``.

    Quantile models here are overconfident, so we scale each quantile's (and the
    representative sample's) deviation from the median by a conformal factor fit
    offline -- see :mod:`scripts.calibrate`. ``lambda == 1.0`` is a no-op.
    """
    lam = config.CALIBRATION_LAMBDA
    if lam == 1.0:
        return quantiles if sample is None else (quantiles, sample)
    median = quantiles[0.5]
    widened = {q: median + lam * (arr - median) for q, arr in quantiles.items()}
    if sample is None:
        return widened
    return widened, median + lam * (np.asarray(sample) - median)


def predict_returns(
    model,
    context_returns,
    horizon: int = config.HORIZON,
    num_samples: int = 500,
    with_sample: bool = False,
):
    """Forecast future log-returns from a raw (un-normalized) return context.

    The context is z-scored, fed to the model, and the predicted quantiles are
    mapped back to the context's own volatility scale. Shared by the web app and
    the backtest so both behave identically.

    Returns ``{quantile: np.ndarray}`` in raw return space. If ``with_sample`` is
    set, returns ``(quantiles, sample)`` where ``sample`` is one representative
    draw from the predictive distribution -- jagged, like a real price path.
    """
    from darts import TimeSeries

    z, mean, std = features.standardize_returns(context_returns)
    ctx = TimeSeries.from_values(z.astype("float32"))
    pred = model.predict(n=horizon, series=ctx, num_samples=num_samples)
    quantiles = {
        q: features.destandardize_returns(pred.quantile(q).values().flatten(), mean, std)
        for q in config.QUANTILES
    }
    if not with_sample:
        return _calibrate(quantiles)
    # one Monte-Carlo sample path: shape (time, component, samples) -> first sample
    z_sample = pred.all_values()[:, 0, 0]
    sample = features.destandardize_returns(z_sample, mean, std)
    return _calibrate(quantiles, sample)


def predict_returns_batch(
    model,
    contexts,
    horizon: int = config.HORIZON,
    num_samples: int = 200,
):
    """Vectorized :func:`predict_returns` over many contexts in one model call.

    Each context is z-scored independently, all are forecast in a single batched
    ``model.predict`` (one Torch trainer pass instead of one per origin), then
    each prediction is mapped back to its own context's volatility. Returns a
    list of ``{quantile: np.ndarray}`` dicts aligned with ``contexts``.
    """
    from darts import TimeSeries

    stats, series = [], []
    for ctx in contexts:
        z, mean, std = features.standardize_returns(ctx)
        stats.append((mean, std))
        series.append(TimeSeries.from_values(z.astype("float32")))

    preds = model.predict(n=horizon, series=series, num_samples=num_samples)
    if not isinstance(preds, list):       # single-series guard
        preds = [preds]

    out = []
    for pred, (mean, std) in zip(preds, stats):
        quantiles = {
            q: features.destandardize_returns(
                pred.quantile(q).values().flatten(), mean, std
            )
            for q in config.QUANTILES
        }
        out.append(_calibrate(quantiles))
    return out


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

    q_rets, sample_rets = predict_returns(model, rets, horizon, with_sample=True)
    last_price = float(prices[-1])
    quantiles = {q: features.from_log_returns(r, last_price) for q, r in q_rets.items()}
    sample = features.from_log_returns(sample_rets, last_price)
    return {"context": prices, "quantiles": quantiles, "sample": sample}
