"""Calibrate the predictive fan so its bands are honest.

The bake-off showed every backend is *overconfident*: the nominal 10-90% band
only covers ~56-65% of realized moves instead of 80%. This script fits a single
multiplicative widening factor ``lambda`` (conformal-style) on a held-out
calibration slice and verifies it on a separate test slice.

Calibrated quantile:  q_cal = median + lambda * (q_raw - median)

A 3-way split keeps it honest:
    train      [0,   0.70)   -> fit the model
    calibrate  [0.70, 0.85)  -> choose lambda to hit 80% coverage
    test       [0.85, 1.0)   -> report coverage / pinball before vs after

Because the model works in per-window standardized space, scaling deviations
around the median is scale-agnostic, so one lambda serves every draw resolution.

Usage:  python scripts/calibrate.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np

from equity_doodle import config, data, features
import scripts.compare_models as cmp

TRAIN_FRAC = 0.70
CAL_FRAC = 0.85          # calibrate on [TRAIN_FRAC, CAL_FRAC), test on [CAL_FRAC, 1)
STRIDE = 25              # origin stride; the expanded corpus has ~4x the tickers
LAMBDA_GRID = np.round(np.arange(1.0, 3.01, 0.05), 2)
TARGET = 0.80            # nominal coverage of the 10-90% band


def train_nhits():
    """Train the bake-off winner on the first TRAIN_FRAC of each series, over
    the same pooled panels (daily + intraday) the production model sees."""
    print(f"[calibrate] training nhits on first {int(TRAIN_FRAC*100)}% ...", flush=True)
    model, _ = cmp.build_nhits()
    frames = data.load_training_frames()
    series = cmp.build_train_series(frames, split_frac=TRAIN_FRAC)
    val = series[::8]
    train = [s for i, s in enumerate(series) if i % 8 != 0]
    model.fit(train, val_series=val)
    return model


def eval_sets():
    """(frame, scale, name) evaluation slices: daily + weekly from the daily
    panel, plus hourly if the intraday cache exists — the /live page runs on
    intraday bars, so its coverage is audited here too."""
    sets = [(data.load_cached(), 1, "daily"), (data.load_cached(), 5, "weekly")]
    _, name = config.INTRADAY.get("1h", (None, "prices_1h.parquet"))
    if (config.DATA_DIR / name).exists():
        sets.append((data.load_cached(name), 1, "hourly"))
    return sets


def gather(prices, scale, lo_frac, hi_frac):
    """Contexts + realized cumulative H-step returns for origins in a window."""
    H, C = config.HORIZON, config.CONTEXT_LENGTH
    contexts, actuals = [], []
    for ticker in prices.columns:
        col = prices[ticker].dropna()
        col = col[col > 0]  # bad prints would break log-returns
        if len(col) < 2:
            continue
        rets = features.aggregate_returns(features.to_log_returns(col.to_numpy()), scale)
        lo = max(int(len(rets) * lo_frac), C)
        hi = min(int(len(rets) * hi_frac), len(rets) - H)
        for o in range(lo, hi, STRIDE):
            contexts.append(rets[o - C : o])
            actuals.append(float(rets[o : o + H].sum()))
    return contexts, actuals


def cum_stats(model, contexts):
    """Per-context: (median, sorted cumulative samples) in raw return space."""
    cum_samples, _ = cmp._sample_paths(model, contexts, config.HORIZON)
    return [(float(np.median(cs)), cs) for cs in cum_samples]


def coverage(stats, actuals, lam):
    cov = 0
    for (m, cs), y in zip(stats, actuals):
        lo = m + lam * (np.quantile(cs, 0.1) - m)
        hi = m + lam * (np.quantile(cs, 0.9) - m)
        cov += lo <= y <= hi
    return cov / len(actuals)


def pinball(stats, actuals, lam):
    losses = []
    for (m, cs), y in zip(stats, actuals):
        per_q = []
        for q in config.QUANTILES:
            qv = m + lam * (np.quantile(cs, q) - m)
            d = y - qv
            per_q.append(max(q * d, (q - 1) * d))
        losses.append(np.mean(per_q))
    return float(np.mean(losses))


def main():
    model = train_nhits()
    sets = eval_sets()

    # --- calibration: pool all scales, pick lambda closest to TARGET coverage
    cal = {name: (cum_stats(model, c), a)
           for frame, s, name in sets
           for c, a in [gather(frame, s, TRAIN_FRAC, CAL_FRAC)]}

    def pooled_cov(lam):
        covs, ns = [], []
        for stats, acts in cal.values():
            covs.append(coverage(stats, acts, lam) * len(acts))
            ns.append(len(acts))
        return sum(covs) / sum(ns)

    best_lam = min(LAMBDA_GRID, key=lambda L: abs(pooled_cov(L) - TARGET))
    print(f"\n[calibrate] pooled coverage by lambda:")
    for L in LAMBDA_GRID[::4]:
        print(f"    lambda={L:.2f}  cov={pooled_cov(L):.3f}")
    print(f"[calibrate] chosen lambda = {best_lam:.2f}  "
          f"(pooled cal coverage {pooled_cov(best_lam):.3f})")

    # --- verify on the untouched test slice
    print("\n" + "=" * 70)
    print(f"{'scale':<8}{'cov_before':>12}{'cov_after':>12}"
          f"{'pin_before':>12}{'pin_after':>12}")
    print("-" * 70)
    for frame, s, name in sets:
        c, a = gather(frame, s, CAL_FRAC, 1.0)
        st = cum_stats(model, c)
        print(f"{name:<8}{coverage(st, a, 1.0):>12.3f}{coverage(st, a, best_lam):>12.3f}"
              f"{pinball(st, a, 1.0):>12.4f}{pinball(st, a, best_lam):>12.4f}")
    print("=" * 70)
    print(f"Target coverage = {TARGET:.2f}.  Recommended CALIBRATION_LAMBDA = {best_lam:.2f}")


if __name__ == "__main__":
    main()
