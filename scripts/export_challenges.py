"""Export a static challenge bank for the beat-the-model game (web/game.html).

Each challenge is a real historical window sampled from the cached price
panel: CONTEXT_LENGTH+1 prices the player sees, plus HORIZON outcome prices
revealed after they guess. Windows are sampled at several time scales by
taking every k-th close, and the reveal stats (annualized context vol,
outcome return, outcome max drawdown) are precomputed here so the page
stays dumb.

Usage:  python scripts/export_challenges.py [n_challenges]
Writes: web/challenges.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equity_doodle import config, data

# Bars-per-year factors per sampling scale, for annualizing context vol.
# Approximate on purpose (crypto trades 365 days; equities 252) -- the reveal
# stat is descriptive, not used by the model.
SCALES = {1: ("daily", 252), 5: ("weekly", 52), 21: ("monthly", 12)}

N_DEFAULT = 400
SEED = 20260703


def sig5(x: float) -> float:
    return float(f"{x:.5g}")


def sample_challenge(prices: pd.DataFrame, rng: np.random.Generator) -> dict | None:
    n_bars = config.CONTEXT_LENGTH + 1 + config.HORIZON
    symbol = str(rng.choice(prices.columns))
    k = int(rng.choice(list(SCALES)))
    series = prices[symbol].dropna()
    span = n_bars * k
    if len(series) < span + 1:
        return None
    start = int(rng.integers(0, len(series) - span))
    window = series.iloc[start : start + span : k]
    if len(window) < n_bars:
        return None
    window = window.iloc[:n_bars]
    vals = window.to_numpy(dtype="float64")
    if np.any(vals <= 0):
        return None
    ctx, out = vals[: config.CONTEXT_LENGTH + 1], vals[config.CONTEXT_LENGTH :]
    ctx_rets = np.diff(np.log(ctx))
    if ctx_rets.std() <= 1e-9:            # dead/illiquid stretch, no shape to read
        return None

    scale_name, per_year = SCALES[k]
    running_max = np.maximum.accumulate(out)
    return {
        "symbol": symbol,
        "scale": scale_name,
        "dates": [
            str(window.index[0].date()),                       # context start
            str(window.index[config.CONTEXT_LENGTH].date()),   # last visible bar
            str(window.index[-1].date()),                      # outcome end
        ],
        "prices": [sig5(v) for v in vals],
        "stats": {
            "ctx_vol_ann_pct": round(float(ctx_rets.std() * np.sqrt(per_year)) * 100, 1),
            "out_ret_pct": round(float(out[-1] / out[0] - 1) * 100, 2),
            "out_maxdd_pct": round(float((out / running_max - 1).min()) * 100, 2),
        },
    }


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    prices = data.load_cached()
    rng = np.random.default_rng(SEED)
    challenges, tries = [], 0
    while len(challenges) < n and tries < n * 50:
        tries += 1
        c = sample_challenge(prices, rng)
        if c is not None:
            challenges.append(c)

    out_path = Path(__file__).resolve().parent.parent / "web" / "challenges.json"
    payload = {
        "context_length": config.CONTEXT_LENGTH,
        "horizon": config.HORIZON,
        "challenges": challenges,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    kb = out_path.stat().st_size / 1024
    by_scale = pd.Series([c["scale"] for c in challenges]).value_counts().to_dict()
    print(f"Wrote {len(challenges)} challenges ({kb:.0f} KB) -> {out_path}")
    print(f"By scale: {by_scale}")


if __name__ == "__main__":
    main()
