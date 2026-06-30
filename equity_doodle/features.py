"""Feature transforms: convert between price paths and log-returns.

Modeling in log-return space makes series stationary and comparable across
tickers and price scales -- which is exactly what lets an arbitrary hand-drawn
chart be fed to the same global model.
"""

from __future__ import annotations

import numpy as np


def to_log_returns(prices: np.ndarray) -> np.ndarray:
    """Convert a 1-D price path to log-returns: r_t = ln(p_t / p_{t-1}).

    Returns an array of length ``len(prices) - 1``.
    """
    prices = np.asarray(prices, dtype="float64")
    if prices.ndim != 1 or prices.size < 2:
        raise ValueError("prices must be a 1-D array with at least 2 points")
    if np.any(prices <= 0):
        raise ValueError("prices must be strictly positive for log-returns")
    return np.diff(np.log(prices))


def from_log_returns(returns: np.ndarray, last_price: float) -> np.ndarray:
    """Integrate log-returns back into a price path anchored at ``last_price``.

    The returned path has length ``len(returns)`` and excludes the anchor.
    """
    returns = np.asarray(returns, dtype="float64")
    return last_price * np.exp(np.cumsum(returns))


def standardize_returns(returns: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Z-score a return window: ``z = (r - mean) / std``.

    Returns ``(z, mean, std)`` so the transform can be inverted. This is what
    makes the model volatility/scale invariant -- the drawn window is reduced to
    a unit-variance *shape*, regardless of whether it is daily, weekly, etc.
    """
    returns = np.asarray(returns, dtype="float64")
    mean = float(returns.mean())
    std = float(returns.std())
    std = std if std > 1e-9 else 1.0   # guard a flat line
    return (returns - mean) / std, mean, std


def destandardize_returns(z: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Invert :func:`standardize_returns`: ``r = z * std + mean``."""
    return np.asarray(z, dtype="float64") * std + mean


def aggregate_returns(returns: np.ndarray, k: int, offset: int = 0) -> np.ndarray:
    """Non-overlapping block-sum of log-returns -> returns at a ``k``-bar scale.

    Summing log-returns over k bars gives the genuine log-return of the coarser
    resolution (e.g. k=5 ~ a weekly bar). Used for multi-scale training.
    """
    r = np.asarray(returns, dtype="float64")[offset:]
    n = (len(r) // k) * k
    if n == 0:
        return np.empty(0)
    return r[:n].reshape(-1, k).sum(axis=1)


def resample_path(y: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a drawn path to exactly ``n`` points.

    Gives invariance to *how many* points the user drew -- the path is reduced
    to a fixed bar-count before being fed to the model.
    """
    y = np.asarray(y, dtype="float64")
    if len(y) == n:
        return y
    xp = np.linspace(0.0, 1.0, len(y))
    x = np.linspace(0.0, 1.0, n)
    return np.interp(x, xp, y)


def normalize_drawn_path(y: np.ndarray, base: float = 100.0) -> np.ndarray:
    """Rescale a hand-drawn path so it starts at ``base`` (price scale is
    irrelevant once we move to returns, but this keeps numbers sane)."""
    y = np.asarray(y, dtype="float64")
    if y[0] == 0:
        raise ValueError("drawn path cannot start at zero")
    return y / y[0] * base
