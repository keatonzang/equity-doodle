"""Export the trained global model to ONNX for in-browser inference.

The web app runs the network directly in the browser via onnxruntime-web, so
the whole product ships as a static site. This script:

1. loads the trained deep model (``config.MODEL_TYPE``),
2. exports the bare network to ``web/model.onnx`` with a dynamic batch axis
   (the browser batches Monte-Carlo rollouts for horizons > OUTPUT_CHUNK),
3. writes ``web/model_meta.json`` — everything the JS pipeline needs to
   replicate :mod:`equity_doodle.forecast` (context length, horizon chunk,
   quantile levels, calibration lambda),
4. checks numerical parity between the ONNX session and the torch network.

Usage:  python scripts/export_onnx.py
"""

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import torch

from equity_doodle import config, model

WEB_DIR = ROOT / "web"
ONNX_PATH = WEB_DIR / "model.onnx"
META_PATH = WEB_DIR / "model_meta.json"

PARITY_TOLERANCE = 1e-4   # max abs diff allowed between torch and ONNX outputs


def export():
    m = model.load()
    net = m.model.eval()

    WEB_DIR.mkdir(exist_ok=True)
    dummy = torch.zeros(1, config.CONTEXT_LENGTH, 1)
    torch.onnx.export(
        net,
        ((dummy, None, None),),
        str(ONNX_PATH),
        input_names=["x"],
        output_names=["quantiles"],
        dynamic_axes={"x": {0: "batch"}, "quantiles": {0: "batch"}},
        opset_version=17,
    )

    META_PATH.write_text(json.dumps({
        "model_type": config.MODEL_TYPE,
        "context_length": config.CONTEXT_LENGTH,
        "output_chunk_length": config.HORIZON,
        "quantiles": config.QUANTILES,
        "calibration_lambda": config.CALIBRATION_LAMBDA,
    }, indent=2) + "\n")
    return net


def check_parity(net):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_PATH))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, config.CONTEXT_LENGTH, 1)).astype("float32")

    with torch.no_grad():
        want = net((torch.from_numpy(x), None, None)).numpy()
    got = sess.run(None, {"x": x})[0]

    if want.shape != got.shape:
        raise SystemExit(f"shape mismatch: torch {want.shape} vs onnx {got.shape}")
    diff = float(np.abs(want - got).max())
    print(f"parity: torch vs ONNX max abs diff = {diff:.2e} "
          f"(batch=8, shape={got.shape})")
    if diff > PARITY_TOLERANCE:
        raise SystemExit(f"parity check FAILED (> {PARITY_TOLERANCE})")
    print(f"OK -> {ONNX_PATH.relative_to(ROOT)} "
          f"({ONNX_PATH.stat().st_size / 1e6:.2f} MB), "
          f"{META_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    net = export()
    check_parity(net)
