# Equity Doodle ✏️📈

**Draw the start of a chart — let the AI doodle the rest.**

Equity Doodle is a case-study project that frames stock-chart "completion" as a
time-series forecasting problem: the segment you draw is the model's **input
context window**, and the continuation it sketches is the **forecast horizon**.

A single *global* model is trained across many liquid equities (daily data), so
it learns the general "grammar" of price paths rather than memorizing one ticker.

> ⚠️ **This is an educational experiment, not a trading system.** See the
> [Disclaimer](#-disclaimer) below. It is not expected to be profitable, and
> that is fine — the goal is to study the modeling and product idea end-to-end.

---

## How it works

```
        you draw this                    the model completes this
   ┌───────────────────────┐        ┌─────────────────────────────┐
   price path (context)            quantile "fan" of continuations
   └───────────────────────┘        └─────────────────────────────┘
            │                                      ▲
            ▼                                      │
   convert to log-returns ──► global ML model ──► future returns ──► rebuild price
```

1. **Data** — daily OHLCV for a basket of liquid equities via `yfinance`.
2. **Features** — prices are converted to **log-returns** (stationary and
   comparable across tickers), so an arbitrary hand-drawn chart at any price
   scale can be fed in directly.
3. **Model** — a **global gradient-boosted model** (Darts `LightGBMModel`) with
   lagged features and **quantile regression**, so we get a probabilistic "fan"
   of possible completions rather than a single line.
4. **Reconstruct** — predicted returns are integrated back into a price path,
   starting from the last point you drew.

## Roadmap

- [x] Project scaffold, license, disclaimer
- [ ] Data pipeline (`equity_doodle/data.py`) — download + cache daily OHLCV
- [ ] Feature transforms (`equity_doodle/features.py`) — price ⇄ log-returns
- [ ] Global quantile model (`equity_doodle/model.py`) — train + backtest
- [ ] Forecast API (`equity_doodle/forecast.py`) — drawn path → completion
- [ ] Web app — a canvas where you draw a chart and the AI finishes it

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage (work in progress)

```bash
# 1. Download daily data for the default ticker universe
python scripts/download_data.py

# 2. Train the global model
python scripts/train.py
```

## ⚖️ Disclaimer

Equity Doodle is provided for **educational and research purposes only**. It is
**not financial, investment, trading, or any other kind of advice**. Nothing it
produces is a recommendation to buy or sell any security. Forecasts are the
output of a statistical model fit on historical data and are very likely to be
**wrong**; past patterns do not predict future prices. **Do not make financial
decisions based on this software.** You use it entirely at your own risk. See
[`LICENSE`](LICENSE) — the software is provided **as is, without warranty of any
kind, and the author accepts no liability** for any damages or losses.

## License

Licensed under the **PolyForm Noncommercial License 1.0.0** — free for
noncommercial use only. See [`LICENSE`](LICENSE). For any commercial use, please
contact the author.

Required Notice: Copyright © 2026 Keaton Zang
