"""Honest out-of-sample backtest of the global, scale-invariant model.

Trains a *fresh* multi-scale model on the first part of history and evaluates it
only on the held-out tail (no leakage). To demonstrate time-scale invariance the
same model is scored at both daily and weekly resolution. Reports directional
accuracy and MAE vs a random-walk baseline, and saves a sample fan-chart.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equity_doodle import config, data, features, forecast, model

SPLIT_FRAC = 0.8     # first 80% of history (by date) is training
STRIDE = 10          # evaluate every Nth origin in the test region


def evaluate(m, prices, split_idx, scale, scale_name):
    """Directional accuracy + MAE of cumulative H-step return, out-of-sample,
    at a given time ``scale`` (k bars aggregated per step)."""
    H, C = config.HORIZON, config.CONTEXT_LENGTH
    hits = total = 0
    model_err, naive_err = [], []
    for ticker in prices.columns:
        col = prices[ticker].dropna()
        rets = features.aggregate_returns(features.to_log_returns(col.to_numpy()), scale)
        start = int(len(rets) * SPLIT_FRAC)
        for o in range(max(start, C), len(rets) - H, STRIDE):
            ctx = rets[o - C : o]
            pred = forecast.predict_returns(m, ctx, H, num_samples=150)
            pred_cum = float(pred[0.5].sum())                 # median path
            actual_cum = float(rets[o : o + H].sum())
            hits += np.sign(pred_cum) == np.sign(actual_cum)
            total += 1
            model_err.append(abs(pred_cum - actual_cum))
            naive_err.append(abs(actual_cum))                 # RW predicts 0
    print(f"\n=== Out-of-sample @ {scale_name} (k={scale}) ===")
    print(f"  forecasts evaluated : {total}")
    print(f"  directional accuracy: {hits / total:.3f}  (random-walk ~0.50)")
    print(f"  model MAE (cum ret) : {np.mean(model_err):.4f}")
    print(f"  naive MAE (cum ret) : {np.mean(naive_err):.4f}")


def sample_chart(m, prices, ticker="AAPL", out="docs/sample_forecast.png"):
    """Reconstruct a price-space fan chart for the tail of one real ticker."""
    H, C = config.HORIZON, config.CONTEXT_LENGTH
    col = prices[ticker].dropna().to_numpy()
    rets = features.to_log_returns(col)
    ctx = rets[-(C + H) : -H]
    q_rets = forecast.predict_returns(m, ctx, H)

    anchor = float(col[-(H + 1)])
    ctx_price = col[-(C + H + 1) : -H]
    actual_price = col[-(H + 1):]

    fig, ax = plt.subplots(figsize=(10, 5))
    cx = np.arange(len(ctx_price))
    fx = np.arange(len(ctx_price) - 1, len(ctx_price) - 1 + H + 1)
    ax.plot(cx, ctx_price, color="#222", lw=2, label="drawn context")
    ax.plot(fx, np.r_[anchor, actual_price[1:]], color="#1a7f37", lw=2,
            label="actual future")
    med = features.from_log_returns(q_rets[0.5], anchor)
    lo = features.from_log_returns(q_rets[0.1], anchor)
    hi = features.from_log_returns(q_rets[0.9], anchor)
    ax.plot(fx, np.r_[anchor, med], color="#cf222e", lw=2, label="model median")
    ax.fill_between(fx, np.r_[anchor, lo], np.r_[anchor, hi], color="#cf222e",
                    alpha=0.15, label="10-90% band")
    ax.axvline(len(ctx_price) - 1, color="#999", ls="--", lw=1)
    ax.set_title(f"Equity Doodle — completion vs reality ({ticker})")
    ax.legend(loc="upper left")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nSaved sample chart -> {out}")


def main():
    prices = data.load_cached()
    split_idx = int(len(prices) * SPLIT_FRAC)
    train_prices = prices.iloc[:split_idx]
    print(f"Training multi-scale model on {prices.shape[1]} tickers x scales "
          f"{config.SCALES} (train {int(SPLIT_FRAC*100)}% by date)...")
    m = model.train(train_prices)
    evaluate(m, prices, split_idx, scale=1, scale_name="daily")
    evaluate(m, prices, split_idx, scale=5, scale_name="weekly")
    sample_chart(m, prices)


if __name__ == "__main__":
    main()
