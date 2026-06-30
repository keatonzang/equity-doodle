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


def normalize_drawn_path(y: np.ndarray, base: float = 100.0) -> np.ndarray:
    """Rescale a hand-drawn path so it starts at ``base`` (price scale is
    irrelevant once we move to returns, but this keeps numbers sane)."""
    y = np.asarray(y, dtype="float64")
    if y[0] == 0:
        raise ValueError("drawn path cannot start at zero")
    return y / y[0] * base
