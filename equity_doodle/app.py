"""FastAPI app: draw a chart in the browser, the model doodles the rest.

Run with:  uvicorn equity_doodle.app:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config, forecast, model

app = FastAPI(title="Equity Doodle", version="0.1.0")
WEB_DIR = Path(__file__).resolve().parent / "web"

_model = None


def get_model():
    """Lazy-load the trained model once per process."""
    global _model
    if _model is None:
        _model = model.load()
    return _model


class DrawRequest(BaseModel):
    prices: list[float]
    horizon: int | None = None


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/forecast")
def api_forecast(req: DrawRequest):
    if len(req.prices) < 2:
        raise HTTPException(400, "draw a longer line (need at least 2 points)")
    horizon = req.horizon or config.HORIZON
    try:
        out = forecast.complete_doodle(req.prices, get_model(), horizon=horizon)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    ctx = out["context"]
    n = len(ctx)
    anchor = float(ctx[-1])
    # Forecast paths are prefixed with the anchor so lines join the context.
    quantiles = {
        f"{q:.2f}": [anchor] + [float(v) for v in arr]
        for q, arr in out["quantiles"].items()
    }
    return {
        "context": {"x": list(range(n)), "y": [float(v) for v in ctx]},
        "forecast_x": list(range(n - 1, n - 1 + horizon + 1)),
        "quantiles": quantiles,
    }
