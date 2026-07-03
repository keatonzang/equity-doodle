/* In-browser port of the Python forecast pipeline (equity_doodle/forecast.py).
 *
 * The trained N-HiTS network ships as model.onnx and runs locally via
 * onnxruntime-web — nothing you draw ever leaves the page. The pipeline
 * mirrors the Python reference exactly:
 *
 *   drawn pixels -> resample to CONTEXT+1 bars -> normalize -> log-returns
 *   -> z-score -> network -> de-standardize -> calibrate fan -> integrate
 *   back to prices anchored at the last drawn point.
 *
 * The network emits exact quantiles for OUTPUT_CHUNK steps. Longer horizons
 * use a Monte-Carlo autoregressive rollout (sample a path per simulation,
 * feed it back), matching darts' probabilistic predict().
 */
"use strict";

const MC_SAMPLES = 200;      // rollout simulations for horizon > OUTPUT_CHUNK
const STD_FLOOR = 1e-9;      // flat-line guard, as in features.standardize_returns
const FLAT_VOL_FALLBACK = 0.01;  // assumed vol for a flat doodle (features.FLAT_VOL_FALLBACK)

let META = null;             // model_meta.json content
let _session = null;

async function loadModel() {
  if (_session) return _session;
  const metaRes = await fetch("model_meta.json");
  META = await metaRes.json();
  _session = await ort.InferenceSession.create("model.onnx", {
    executionProviders: ["wasm"],
  });
  return _session;
}

/* ------------------------- feature transforms --------------------------- */

// features.resample_path: linear resample of a drawn path to exactly n points.
function resamplePath(y, n) {
  if (y.length === n) return y.slice();
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * (y.length - 1);
    const j = Math.min(Math.floor(t), y.length - 2);
    const f = t - j;
    out[i] = y[j] * (1 - f) + y[j + 1] * f;
  }
  return out;
}

// features.normalize_drawn_path: rescale so the path starts at 100.
function normalizePath(y, base = 100.0) {
  if (y[0] === 0) throw new Error("drawn path cannot start at zero");
  return y.map(v => (v / y[0]) * base);
}

// features.to_log_returns
function toLogReturns(prices) {
  if (prices.some(p => p <= 0)) {
    throw new Error("prices must be strictly positive");
  }
  const out = new Array(prices.length - 1);
  for (let i = 1; i < prices.length; i++) out[i - 1] = Math.log(prices[i] / prices[i - 1]);
  return out;
}

// features.standardize_returns
function standardize(returns) {
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const varSum = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length;
  let std = Math.sqrt(varSum);
  // A perfectly flat doodle has no volatility scale of its own; falling back to
  // std=1.0 would rescale predictions to 100%-per-bar volatility and blow up
  // the fan. Assume a typical daily-equity vol instead, so the model's samples
  // stay lively but sane (and continuous with an almost-flat hand drawing).
  if (std <= STD_FLOOR) std = FLAT_VOL_FALLBACK;
  return { z: returns.map(r => (r - mean) / std), mean, std };
}

/* --------------------------- network wrapper ---------------------------- */

// Run the net on a batch of z-scored contexts (each CONTEXT_LENGTH long).
// Returns [batch][step][quantile] exact quantile values in z-space.
async function runNet(session, contexts) {
  const C = META.context_length, Q = META.quantiles.length;
  const B = contexts.length;
  const data = new Float32Array(B * C);
  for (let b = 0; b < B; b++) data.set(contexts[b], b * C);
  const out = await session.run({ x: new ort.Tensor("float32", data, [B, C, 1]) });
  const flat = out.quantiles.data;           // (B, OUTPUT_CHUNK, 1, Q)
  const H = META.output_chunk_length;
  const res = new Array(B);
  for (let b = 0; b < B; b++) {
    const steps = new Array(H);
    for (let h = 0; h < H; h++) {
      const off = (b * H + h) * Q;
      steps[h] = Array.from(flat.slice(off, off + Q));
    }
    res[b] = steps;
  }
  return res;
}

/* ------------------------- probabilistic helpers ------------------------ */

// Inverse-CDF sample from a step's quantile fan: piecewise-linear between the
// quantile levels, clamped to the outermost quantiles (as darts does).
function sampleFromQuantiles(levels, values, u) {
  if (u <= levels[0]) return values[0];
  if (u >= levels[levels.length - 1]) return values[values.length - 1];
  let i = 0;
  while (u > levels[i + 1]) i++;
  const f = (u - levels[i]) / (levels[i + 1] - levels[i]);
  return values[i] * (1 - f) + values[i + 1] * f;
}

// Empirical quantile (linear interpolation, matches np.quantile default).
function empiricalQuantile(sorted, q) {
  const t = q * (sorted.length - 1);
  const j = Math.min(Math.floor(t), sorted.length - 2);
  const f = t - j;
  return sorted[j] * (1 - f) + sorted[j + 1] * f;
}

/* ----------------------------- forecasting ------------------------------ */

// Forecast `horizon` steps in z-space. Returns:
//   { fan: [step][quantile], drawSample: () => [step] }
// Steps within the first chunk use the network's exact quantiles; steps
// beyond it use empirical quantiles over MC_SAMPLES autoregressive rollouts.
// drawSample is cheap and can be called repeatedly ("roll another future"):
// short horizons draw fresh inverse-CDF samples, long horizons hand out a
// random one of the already-simulated MC paths.
async function forecastZ(session, z, horizon) {
  const OC = META.output_chunk_length, L = META.quantiles;
  const first = (await runNet(session, [z]))[0];

  if (horizon <= OC) {
    const fan = first.slice(0, horizon);
    const drawSample = () => fan.map(vals => sampleFromQuantiles(L, vals, Math.random()));
    return { fan, drawSample };
  }

  // Monte-Carlo rollout: each simulation keeps its own context and path.
  const C = META.context_length;
  const contexts = [], paths = [];
  for (let s = 0; s < MC_SAMPLES; s++) {
    contexts.push(z.slice());
    paths.push([]);
  }
  let stepQ = contexts.map(() => first);   // chunk quantiles per simulation
  let done = 0;
  while (done < horizon) {
    const take = Math.min(OC, horizon - done);
    for (let s = 0; s < MC_SAMPLES; s++) {
      for (let h = 0; h < take; h++) {
        paths[s].push(sampleFromQuantiles(L, stepQ[s][h], Math.random()));
      }
      const ctx = contexts[s].concat(paths[s].slice(-take));
      contexts[s] = ctx.slice(ctx.length - C);
    }
    done += take;
    if (done < horizon) stepQ = await runNet(session, contexts);
  }

  const fan = new Array(horizon);
  for (let h = 0; h < horizon; h++) {
    if (h < OC) {
      fan[h] = first[h];                   // exact within the first chunk
    } else {
      const vals = paths.map(p => p[h]).sort((a, b) => a - b);
      fan[h] = L.map(q => empiricalQuantile(vals, q));
    }
  }
  const drawSample = () => paths[Math.floor(Math.random() * paths.length)];
  return { fan, drawSample };
}

/* ------------------------------ public API ------------------------------ */

// State of the most recent forecast, kept so the UI can cheaply draw fresh
// sample paths ("roll another future") without re-running the network.
let _lastForecast = null;

// Turn one z-space sample draw into an anchor-prefixed price path, widening
// its deviation from the median by the calibration factor (as _calibrate does).
function _buildSamplePath(state) {
  const { drawSample, medRets, mean, std, anchor, lam, horizon } = state;
  const zs = drawSample();
  const path = [anchor];
  let cum = 0;
  for (let h = 0; h < horizon; h++) {
    const r = zs[h] * std + mean;
    cum += medRets[h] + lam * (r - medRets[h]);
    path.push(anchor * Math.exp(cum));
  }
  return path;
}

// Draw a fresh sample path from the most recent forecast (null if none yet).
function resampleDoodle() {
  return _lastForecast ? _buildSamplePath(_lastForecast) : null;
}

// Mirrors the old POST /api/forecast response shape, so the canvas code is
// unchanged: quantile keys "0.05".."0.95", anchor-prefixed paths.
async function completeDoodle(drawnPrices, horizon) {
  if (drawnPrices.length < 2) throw new Error("draw a longer line");
  const session = await loadModel();
  const lam = META.calibration_lambda;
  const L = META.quantiles;
  const medIdx = L.indexOf(0.5);

  const prices = normalizePath(resamplePath(drawnPrices, META.context_length + 1));
  const { z, mean, std } = standardize(toLogReturns(prices));
  const { fan, drawSample } = await forecastZ(session, z, horizon);

  // z -> returns, then widen the fan around the median by the conformal
  // calibration factor, per step (forecast._calibrate).
  const anchor = prices[prices.length - 1];
  const medRets = fan.map(vals => vals[medIdx] * std + mean);
  const qPaths = {};
  for (const q of L) qPaths[q.toFixed(2)] = [anchor];
  let cum = {};
  for (const q of L) cum[q.toFixed(2)] = 0;

  for (let h = 0; h < horizon; h++) {
    const rets = fan[h].map(v => v * std + mean);
    const med = medRets[h];
    L.forEach((q, i) => {
      const key = q.toFixed(2);
      cum[key] += med + lam * (rets[i] - med);
      qPaths[key].push(anchor * Math.exp(cum[key]));
    });
  }

  _lastForecast = { drawSample, medRets, mean, std, anchor, lam, horizon };

  return {
    context: { x: prices.map((_, i) => i), y: prices },
    forecast_x: Array.from({ length: horizon + 1 }, (_, i) => prices.length - 1 + i),
    quantiles: qPaths,
    sample: _buildSamplePath(_lastForecast),
  };
}
