"""Architecture / capacity bake-off for the global doodle-completion model.

Trains several candidate backends on the SAME honest split (first 80% of each
series by date) and scores them out-of-sample on the held-out tail with metrics
that actually matter for a *probabilistic* shape-completer:

* directional accuracy  -- did the median get the sign of the H-step move right?
* MAE (cum. return)     -- magnitude error of the median vs a random walk
* pinball loss          -- proper score for the whole quantile fan (lower better)
* 10-90% coverage       -- is the 80% band calibrated? (target ~0.80)

All candidates are univariate (they only see the drawn shape), so nothing here
relies on covariates the web app could not supply at draw time.

Usage:  python scripts/compare_models.py [candidate ...]
        (default: all candidates)
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np

from equity_doodle import config, data, features

SPLIT_FRAC = 0.8
STRIDE = 10
NUM_SAMPLES = 300
EVAL_QUANTILES = config.QUANTILES


# --------------------------------------------------------------------------- #
# Candidate model builders (untrained). All quantile-probabilistic, univariate. #
# --------------------------------------------------------------------------- #
def _trainer_kwargs(early_stop: bool):
    import torch
    from pytorch_lightning.callbacks import EarlyStopping

    kw = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    if early_stop:
        kw["callbacks"] = [
            EarlyStopping(monitor="val_loss", patience=6, min_delta=1e-4, mode="min")
        ]
    return kw


def build_lightgbm():
    from darts.models import LightGBMModel

    return LightGBMModel(
        lags=config.CONTEXT_LENGTH,
        output_chunk_length=config.HORIZON,
        likelihood="quantile",
        quantiles=config.QUANTILES,
        verbose=-1,
    ), False  # (model, needs_val)


def build_tide():
    from darts.models import TiDEModel
    from darts.utils.likelihood_models.torch import QuantileRegression

    return TiDEModel(
        input_chunk_length=config.CONTEXT_LENGTH,
        output_chunk_length=config.HORIZON,
        hidden_size=config.DL_HIDDEN_SIZE,
        num_encoder_layers=config.DL_NUM_ENCODER_LAYERS,
        num_decoder_layers=config.DL_NUM_DECODER_LAYERS,
        decoder_output_dim=config.DL_DECODER_OUTPUT_DIM,
        temporal_decoder_hidden=config.DL_TEMPORAL_DECODER_HIDDEN,
        dropout=config.DL_DROPOUT,
        use_layer_norm=True,
        use_static_covariates=False,
        batch_size=config.DL_BATCH_SIZE,
        n_epochs=config.DL_N_EPOCHS,
        likelihood=QuantileRegression(quantiles=config.QUANTILES),
        optimizer_kwargs={"lr": config.DL_LEARNING_RATE},
        random_state=config.RANDOM_STATE,
        pl_trainer_kwargs=_trainer_kwargs(early_stop=False),
    ), False


def build_tide_big():
    """More capacity + early stopping: train long, let val_loss decide."""
    from darts.models import TiDEModel
    from darts.utils.likelihood_models.torch import QuantileRegression

    return TiDEModel(
        input_chunk_length=config.CONTEXT_LENGTH,
        output_chunk_length=config.HORIZON,
        hidden_size=256,
        num_encoder_layers=3,
        num_decoder_layers=3,
        decoder_output_dim=32,
        temporal_decoder_hidden=128,
        dropout=0.2,
        use_layer_norm=True,
        use_static_covariates=False,
        batch_size=config.DL_BATCH_SIZE,
        n_epochs=80,
        likelihood=QuantileRegression(quantiles=config.QUANTILES),
        optimizer_kwargs={"lr": config.DL_LEARNING_RATE},
        random_state=config.RANDOM_STATE,
        pl_trainer_kwargs=_trainer_kwargs(early_stop=True),
    ), True


def build_nhits():
    """N-HiTS: multi-rate hierarchical interpolation, strong on pure shape."""
    from darts.models import NHiTSModel
    from darts.utils.likelihood_models.torch import QuantileRegression

    return NHiTSModel(
        input_chunk_length=config.CONTEXT_LENGTH,
        output_chunk_length=config.HORIZON,
        num_stacks=3,
        num_blocks=1,
        num_layers=2,
        layer_widths=256,
        dropout=0.1,
        batch_size=config.DL_BATCH_SIZE,
        n_epochs=80,
        likelihood=QuantileRegression(quantiles=config.QUANTILES),
        optimizer_kwargs={"lr": config.DL_LEARNING_RATE},
        random_state=config.RANDOM_STATE,
        pl_trainer_kwargs=_trainer_kwargs(early_stop=True),
    ), True


CANDIDATES = {
    "lightgbm": build_lightgbm,
    "tide": build_tide,
    "tide_big": build_tide_big,
    "nhits": build_nhits,
}


# --------------------------------------------------------------------------- #
# Training / evaluation                                                        #
# --------------------------------------------------------------------------- #
def build_train_series(prices, split_frac=SPLIT_FRAC):
    """Standardized, multi-scale return series from the first ``split_frac`` of
    each ticker's history (the honest training region)."""
    from darts import TimeSeries

    min_len = config.CONTEXT_LENGTH + config.HORIZON + 5
    series = []
    for ticker in prices.columns:
        col = prices[ticker].dropna().to_numpy()
        col = col[: int(len(col) * split_frac)]
        rets = features.to_log_returns(col)
        for k in config.SCALES:
            agg = features.aggregate_returns(rets, k)
            if len(agg) < min_len:
                continue
            z, _, _ = features.standardize_returns(agg)
            series.append(TimeSeries.from_values(z.astype("float32")))
    return series


def fit_candidate(name):
    print(f"\n[{name}] building + training...", flush=True)
    model, needs_val = CANDIDATES[name]()
    prices = data.load_cached()
    series = build_train_series(prices)
    if needs_val:
        # Hold out every 8th series as a validation signal for early stopping.
        val = series[::8]
        train = [s for i, s in enumerate(series) if i % 8 != 0]
        model.fit(train, val_series=val)
    else:
        model.fit(series)
    return model


def _sample_paths(model, contexts, horizon):
    """Batched probabilistic forecast -> (cum_samples, per_q_cum) per context.

    Returns, for every context: the array of sampled cumulative H-step returns
    (raw return space) and the empirical quantile of that cumulative at each
    EVAL_QUANTILES level.
    """
    from darts import TimeSeries

    stats, ts = [], []
    for ctx in contexts:
        z, mean, std = features.standardize_returns(ctx)
        stats.append((mean, std))
        ts.append(TimeSeries.from_values(z.astype("float32")))

    preds = model.predict(n=horizon, series=ts, num_samples=NUM_SAMPLES)
    if not isinstance(preds, list):
        preds = [preds]

    cum_samples, per_q = [], []
    for pred, (mean, std) in zip(preds, stats):
        z = pred.all_values()[:, 0, :]               # (H, num_samples)
        raw = z * std + mean                         # de-standardize each sample
        cs = raw.sum(axis=0)                         # (num_samples,) cum return
        cum_samples.append(cs)
        per_q.append({q: float(np.quantile(cs, q)) for q in EVAL_QUANTILES})
    return cum_samples, per_q


def _pinball(actual, pred_q, q):
    d = actual - pred_q
    return max(q * d, (q - 1) * d)


def evaluate(model, prices, scale, scale_name):
    H, C = config.HORIZON, config.CONTEXT_LENGTH
    contexts, actuals = [], []
    for ticker in prices.columns:
        col = prices[ticker].dropna()
        rets = features.aggregate_returns(features.to_log_returns(col.to_numpy()), scale)
        start = int(len(rets) * SPLIT_FRAC)
        for o in range(max(start, C), len(rets) - H, STRIDE):
            contexts.append(rets[o - C : o])
            actuals.append(float(rets[o : o + H].sum()))

    cum_samples, per_q = _sample_paths(model, contexts, H)

    hits = covered = 0
    mae, pinball = [], []
    for cs, pq, actual in zip(cum_samples, per_q, actuals):
        median = float(np.median(cs))
        hits += np.sign(median) == np.sign(actual)
        mae.append(abs(median - actual))
        covered += pq[0.1] <= actual <= pq[0.9]
        pinball.append(np.mean([_pinball(actual, pq[q], q) for q in EVAL_QUANTILES]))
    n = len(actuals)
    return {
        "scale": scale_name,
        "n": n,
        "dir_acc": hits / n,
        "mae": float(np.mean(mae)),
        "pinball": float(np.mean(pinball)),
        "coverage": covered / n,
    }


def main():
    names = sys.argv[1:] or list(CANDIDATES)
    prices = data.load_cached()
    rows = []
    for name in names:
        model = fit_candidate(name)
        for scale, sname in [(1, "daily"), (5, "weekly")]:
            r = evaluate(model, prices, scale, sname)
            r["model"] = name
            rows.append(r)
            print(f"[{name} @ {sname}] dir={r['dir_acc']:.3f}  mae={r['mae']:.4f}  "
                  f"pinball={r['pinball']:.4f}  cov(10-90)={r['coverage']:.3f}  n={r['n']}",
                  flush=True)

    # Final table
    print("\n" + "=" * 78)
    print(f"{'model':<10}{'scale':<8}{'dir_acc':>9}{'MAE':>9}{'pinball':>9}{'cov80':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['model']:<10}{r['scale']:<8}{r['dir_acc']:>9.3f}{r['mae']:>9.4f}"
              f"{r['pinball']:>9.4f}{r['coverage']:>8.3f}")
    print("=" * 78)
    print("Random-walk reference: dir_acc=0.500, MAE=naive (predict 0), "
          "well-calibrated cov80~0.80")


if __name__ == "__main__":
    main()
