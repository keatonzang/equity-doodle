/* Serverless proxy for live chart data (the ONLY code here that talks to the
 * outside world — the drawing pages stay fully local).
 *
 * Default provider is Yahoo Finance's public chart API, which needs no key.
 * To swap providers, set CHART_PROVIDER=finnhub and FINNHUB_API_KEY in the
 * Vercel env vars — this file is the one-file change.
 *
 * GET /api/chart?symbol=AAPL&interval=1m
 * -> { symbol, name, currency, exchange, interval, bars: [{ t, o, h, l, c }] }
 *    t = unix seconds, o/h/l/c = OHLC (o/h/l may be null on sparse bars).
 *    Bars with a null close (halts, gaps) are dropped.
 */
"use strict";

const SYMBOL_RE = /^[A-Za-z0-9^.=\-]{1,15}$/;

// interval -> how much history to pull (enough for 60 bars of context + slack)
const YAHOO_RANGE = { "1m": "1d", "5m": "5d", "15m": "5d", "1d": "6mo" };

async function fromYahoo(symbol, interval) {
  const url =
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}` +
    `?interval=${interval}&range=${YAHOO_RANGE[interval]}&includePrePost=false`;
  const r = await fetch(url, {
    headers: { "User-Agent": "Mozilla/5.0 (compatible; equity-doodle)" },
  });
  if (!r.ok) throw new Error(`quote service replied ${r.status}`);
  const body = await r.json();
  const res = body && body.chart && body.chart.result && body.chart.result[0];
  if (!res) {
    const why = body && body.chart && body.chart.error && body.chart.error.description;
    throw new Error(why || "no data for that symbol");
  }
  const ts = res.timestamp || [];
  const q = (res.indicators && res.indicators.quote && res.indicators.quote[0]) || {};
  const bars = [];
  for (let i = 0; i < ts.length; i++) {
    if (q.close && q.close[i] != null) {
      bars.push({
        t: ts[i],
        o: q.open ? q.open[i] : null,
        h: q.high ? q.high[i] : null,
        l: q.low ? q.low[i] : null,
        c: q.close[i],
      });
    }
  }
  const meta = res.meta || {};
  return {
    symbol: meta.symbol || symbol.toUpperCase(),
    name: meta.longName || meta.shortName || null,
    currency: meta.currency || null,
    exchange: meta.exchangeName || null,
    interval,
    bars,
  };
}

const FINNHUB_RES = { "1m": "1", "5m": "5", "15m": "15", "1d": "D" };
const FINNHUB_SPAN = { "1m": 86400, "5m": 5 * 86400, "15m": 5 * 86400, "1d": 180 * 86400 };

async function fromFinnhub(symbol, interval) {
  const key = process.env.FINNHUB_API_KEY;
  if (!key) throw new Error("FINNHUB_API_KEY is not configured");
  const to = Math.floor(Date.now() / 1000);
  const from = to - FINNHUB_SPAN[interval];
  const url =
    `https://finnhub.io/api/v1/stock/candle?symbol=${encodeURIComponent(symbol.toUpperCase())}` +
    `&resolution=${FINNHUB_RES[interval]}&from=${from}&to=${to}&token=${key}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`quote service replied ${r.status}`);
  const body = await r.json();
  if (body.s !== "ok") throw new Error("no data for that symbol");
  const bars = [];
  for (let i = 0; i < body.t.length; i++) {
    if (body.c[i] != null) {
      bars.push({ t: body.t[i], o: body.o[i], h: body.h[i], l: body.l[i], c: body.c[i] });
    }
  }
  return {
    symbol: symbol.toUpperCase(),
    name: null,
    currency: null,
    exchange: null,
    interval,
    bars,
  };
}

module.exports = async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  const symbol = (req.query.symbol || "").trim();
  const interval = req.query.interval || "1m";
  if (!SYMBOL_RE.test(symbol)) {
    return res.status(400).json({ error: "bad or missing symbol" });
  }
  if (!(interval in YAHOO_RANGE)) {
    return res.status(400).json({
      error: `interval must be one of ${Object.keys(YAHOO_RANGE).join(", ")}`,
    });
  }
  const provider = process.env.CHART_PROVIDER === "finnhub" ? fromFinnhub : fromYahoo;
  try {
    const data = await provider(symbol, interval);
    // Brief CDN cache so a crowd of pollers becomes ~1 upstream request per window.
    res.setHeader("Cache-Control", "s-maxage=25, stale-while-revalidate=30");
    return res.status(200).json(data);
  } catch (err) {
    return res.status(502).json({ error: String((err && err.message) || err) });
  }
};
