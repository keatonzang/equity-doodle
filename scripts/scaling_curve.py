"""Data-scaling learning curve for the LightGBM baseline.

Question: is the directional ceiling (~0.50 daily / ~0.56 weekly we keep hitting)
set by *too little data* or by *market efficiency*? To find out, we hold the test
set fixed and grow the training data, then watch how the metrics move.

* test set  : last 20% (by date) of ALL tickers -- identical for every point, so
              the only thing changing is training-data volume.
* train set : first 80% of a *nested* random subset of tickers (5% -> 100%), so
              each point strictly adds data to the previous one.

If directional accuracy stays flat while data grows ~20x, the ceiling is the
market, not the model. If pinball / coverage improve, that quantifies what more
data *does* buy (calibration, not direction).

Usage:  python scripts/scaling_curve.py
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

from equity_doodle import config, data, features

SPLIT_FRAC = 0.8
STRIDE = 20            # test-origin stride (20 keeps the curve fast, still ~6k pts)
NUM_SAMPLES = 100
FRACTIONS = [0.05, 0.1, 0.25, 0.5, 1.0]
SCALES = [(1, "daily"), (5, "weekly")]


def build_lgbm():
    from darts.models import LightGBMModel

    return LightGBMModel(
        lags=config.CONTEXT_LENGTH,
        output_chunk_length=config.HORIZON,
        likelihood="quantile",
        quantiles=config.QUANTILES,
        verbose=-1,
    )


def build_series(prices, tickers, split_frac):
    """Standardized multi-scale series from the first ``split_frac`` of each
    ticker. Returns (series, n_windows) where n_windows is the true training
    sample count (a faithful 'data volume' x-axis)."""
    from darts import TimeSeries

    min_len = config.CONTEXT_LENGTH + config.HORIZON + 5
    window = config.CONTEXT_LENGTH + config.HORIZON
    series, n_windows = [], 0
    for t in tickers:
        col = prices[t].dropna().to_numpy()
        col = col[: int(len(col) * split_frac)]
        rets = features.to_log_returns(col)
        for k in config.SCALES:
            agg = features.aggregate_returns(rets, k)
            if len(agg) < min_len:
                continue
            z, _, _ = features.standardize_returns(agg)
            series.append(TimeSeries.from_values(z.astype("float32")))
            n_windows += len(agg) - window + 1
    return series, n_windows


def gather(prices, tickers, scale):
    """Fixed test contexts + realized cumulative H-step returns (test region)."""
    H, C = config.HORIZON, config.CONTEXT_LENGTH
    ctxs, acts = [], []
    for t in tickers:
        col = prices[t].dropna()
        rets = features.aggregate_returns(features.to_log_returns(col.to_numpy()), scale)
        start = int(len(rets) * SPLIT_FRAC)
        for o in range(max(start, C), len(rets) - H, STRIDE):
            ctxs.append(rets[o - C : o])
            acts.append(float(rets[o : o + H].sum()))
    return ctxs, acts


def sample_cum(model, contexts, chunk=1500):
    """Chunked probabilistic forecast -> cumulative H-step return samples per
    context. Chunking keeps memory bounded for the big LightGBM sample draws."""
    from darts import TimeSeries

    H = config.HORIZON
    out = []
    for i in range(0, len(contexts), chunk):
        batch = contexts[i : i + chunk]
        stats, ts = [], []
        for ctx in batch:
            z, m, s = features.standardize_returns(ctx)
            stats.append((m, s))
            ts.append(TimeSeries.from_values(z.astype("float32")))
        preds = model.predict(n=H, series=ts, num_samples=NUM_SAMPLES)
        if not isinstance(preds, list):
            preds = [preds]
        for pred, (m, s) in zip(preds, stats):
            raw = pred.all_values()[:, 0, :] * s + m
            out.append(raw.sum(axis=0))
    return out


def metrics(cum_samples, actuals):
    hits = cov = 0
    mae, pin = [], []
    for cs, y in zip(cum_samples, actuals):
        med = float(np.median(cs))
        hits += np.sign(med) == np.sign(y)
        mae.append(abs(med - y))
        cov += np.quantile(cs, 0.1) <= y <= np.quantile(cs, 0.9)
        pin.append(np.mean([max(q * (y - np.quantile(cs, q)), (q - 1) * (y - np.quantile(cs, q)))
                            for q in config.QUANTILES]))
    n = len(actuals)
    return {"n": n, "dir": hits / n, "mae": float(np.mean(mae)),
            "pinball": float(np.mean(pin)), "cov": cov / n}


def plot(rows, out="docs/scaling_curve.png"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, color in [("daily", "#cf222e"), ("weekly", "#1a7f37")]:
        r = [x for x in rows if x["scale"] == name]
        x = [x["train_windows"] for x in r]
        a1.plot(x, [v["dir"] for v in r], "o-", color=color, label=name)
        a2.plot(x, [v["pinball"] for v in r], "o-", color=color, label=name)
    a1.axhline(0.5, color="#999", ls="--", lw=1, label="coin flip")
    a1.set_xscale("log"); a2.set_xscale("log")
    a1.set_title("Directional accuracy vs training data"); a1.set_ylabel("dir. accuracy")
    a2.set_title("Pinball loss vs training data"); a2.set_ylabel("pinball (lower=better)")
    for a in (a1, a2):
        a.set_xlabel("training windows (log scale)"); a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); Path(out).parent.mkdir(exist_ok=True); fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")


def main():
    prices = data.load_cached()
    all_t = list(prices.columns)
    shuffled = list(np.random.default_rng(0).permutation(all_t))

    # Fixed test set (all tickers), gathered once.
    test = {name: gather(prices, all_t, s) for s, name in SCALES}

    rows = []
    for f in FRACTIONS:
        N = max(3, int(round(len(all_t) * f)))
        tks = shuffled[:N]
        series, n_win = build_series(prices, tks, SPLIT_FRAC)
        print(f"\n[frac={f}] {N} tickers, {len(series)} series, {n_win:,} train windows -> training",
              flush=True)
        m = build_lgbm()
        m.fit(series)
        for name, (ctxs, acts) in test.items():
            r = metrics(sample_cum(m, ctxs), acts)
            r.update(frac=f, tickers=N, train_series=len(series), train_windows=n_win, scale=name)
            rows.append(r)
            print(f"  [{name}] dir={r['dir']:.3f}  mae={r['mae']:.4f}  "
                  f"pinball={r['pinball']:.4f}  cov={r['cov']:.3f}  n={r['n']}", flush=True)

    print("\n" + "=" * 84)
    print(f"{'tickers':>8}{'windows':>12}{'scale':>9}{'dir_acc':>9}{'MAE':>9}{'pinball':>9}{'cov80':>8}")
    print("-" * 84)
    for r in rows:
        print(f"{r['tickers']:>8}{r['train_windows']:>12,}{r['scale']:>9}"
              f"{r['dir']:>9.3f}{r['mae']:>9.4f}{r['pinball']:>9.4f}{r['cov']:>8.3f}")
    print("=" * 84)
    plot(rows)


if __name__ == "__main__":
    main()
