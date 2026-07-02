"""The global forecasting model.

A single global Darts model is trained across every ticker's log-return series
at once. Two backends are available (see ``config.MODEL_TYPE``):

* ``"tide"``     -- TiDE, a Torch deep learner (encoder-decoder) with quantile
  regression. Trains on GPU when available.
* ``"lightgbm"`` -- the original gradient-boosted baseline.

Both use quantile regression, giving a probabilistic *fan* of completions
instead of a single line, behind one common ``predict`` interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features


def _build_return_series(prices: pd.DataFrame):
    """Turn a wide price frame into a list of standardized, multi-scale
    log-return TimeSeries (a plain integer index; the calendar carries no
    signal we rely on).

    For every ticker we emit one series per scale in ``config.SCALES``, each
    z-scored to unit variance. Pooling these teaches one global model the shape
    of price paths across time resolutions.
    """
    from darts import TimeSeries

    min_len = config.CONTEXT_LENGTH + config.HORIZON + 5
    series = []
    for ticker in prices.columns:
        col = prices[ticker].dropna()
        rets = features.to_log_returns(col.to_numpy())
        for k in config.SCALES:
            agg = features.aggregate_returns(rets, k)
            if len(agg) < min_len:
                continue
            z, _, _ = features.standardize_returns(agg)
            series.append(TimeSeries.from_values(z.astype("float32")))
    if not series:
        raise ValueError("no ticker had enough history to build a series")
    return series


def _default_name(model_type: str | None = None) -> str:
    """Filename for a given backend, e.g. ``global_tide.pkl``."""
    return f"global_{model_type or config.MODEL_TYPE}.pkl"


def _build_lightgbm():
    from darts.models import LightGBMModel

    return LightGBMModel(
        lags=config.CONTEXT_LENGTH,
        output_chunk_length=config.HORIZON,
        likelihood="quantile",
        quantiles=config.QUANTILES,
        verbose=-1,
    )


DEEP_BACKENDS = ("tide", "nhits")


def _pl_kwargs():
    """PyTorch-Lightning trainer kwargs: GPU when present, plus early stopping on
    validation loss so a generous epoch budget doesn't overfit."""
    import torch
    from pytorch_lightning.callbacks import EarlyStopping

    return {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "enable_progress_bar": True,
        "enable_model_summary": False,
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=8, min_delta=1e-4, mode="min")
        ],
    }


def _build_tide():
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
        pl_trainer_kwargs=_pl_kwargs(),
    )


def _build_nhits():
    """N-HiTS: multi-rate hierarchical net -- the bake-off winner on the proper
    (pinball) and calibration scores."""
    from darts.models import NHiTSModel
    from darts.utils.likelihood_models.torch import QuantileRegression

    return NHiTSModel(
        input_chunk_length=config.CONTEXT_LENGTH,
        output_chunk_length=config.HORIZON,
        num_stacks=3,
        num_blocks=1,
        num_layers=2,
        layer_widths=256,
        dropout=config.DL_DROPOUT,
        batch_size=config.DL_BATCH_SIZE,
        n_epochs=config.DL_N_EPOCHS,
        likelihood=QuantileRegression(quantiles=config.QUANTILES),
        optimizer_kwargs={"lr": config.DL_LEARNING_RATE},
        random_state=config.RANDOM_STATE,
        pl_trainer_kwargs=_pl_kwargs(),
    )


def build_model(model_type: str | None = None):
    """Construct the untrained global quantile model for ``config.MODEL_TYPE``."""
    model_type = model_type or config.MODEL_TYPE
    if model_type == "lightgbm":
        return _build_lightgbm()
    if model_type == "tide":
        return _build_tide()
    if model_type == "nhits":
        return _build_nhits()
    raise ValueError(f"unknown MODEL_TYPE: {model_type!r}")


def train(prices: pd.DataFrame, model_type: str | None = None):
    """Fit the global model on all tickers' log-return series.

    Deep backends hold out every 10th series as a validation set so the
    early-stopping callback (see :func:`_pl_kwargs`) can pick the best epoch.
    """
    model_type = model_type or config.MODEL_TYPE
    model = build_model(model_type)
    series = _build_return_series(prices)
    if model_type in DEEP_BACKENDS:
        val = series[::10]
        tr = [s for i, s in enumerate(series) if i % 10 != 0]
        model.fit(tr, val_series=val)
    else:
        model.fit(series)
    return model


def save(model, name: str | None = None) -> str:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODEL_DIR / (name or _default_name())
    # Torch models additionally write a sibling ``<path>.ckpt``; Darts handles
    # both files transparently on save/load.
    model.save(str(path))
    return str(path)


def load(name: str | None = None, model_type: str | None = None):
    """Load a saved global model, dispatching on the backend type."""
    model_type = model_type or config.MODEL_TYPE
    path = str(config.MODEL_DIR / (name or _default_name(model_type)))
    if model_type == "lightgbm":
        from darts.models import LightGBMModel

        return LightGBMModel.load(path)
    if model_type == "tide":
        from darts.models import TiDEModel

        return TiDEModel.load(path)
    if model_type == "nhits":
        from darts.models import NHiTSModel

        return NHiTSModel.load(path)
    raise ValueError(f"unknown MODEL_TYPE: {model_type!r}")
