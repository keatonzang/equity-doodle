"""Train the global quantile model on all cached price panels (daily + intraday)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equity_doodle import data, model


def main() -> None:
    frames = data.load_training_frames()
    for f in frames:
        print(f"panel: {f.shape[1]} tickers x {f.shape[0]} rows")
    m = model.train(frames)
    path = model.save(m)
    print(f"Saved trained model -> {path}")


if __name__ == "__main__":
    main()
