"""Download and cache prices for the configured ticker universe.

Daily bars for the full history, plus the intraday panels configured in
``config.INTRADAY`` (pooled into training so intraday dynamics are in-domain).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equity_doodle import config, data


def main() -> None:
    prices = data.download_prices()
    path = data.cache_prices(prices)
    print(f"daily: {prices.shape[1]} tickers x {prices.shape[0]} rows -> {path}")

    for interval, (period, name) in config.INTRADAY.items():
        try:
            intraday = data.download_intraday(interval, period)
        except ValueError as err:
            print(f"{interval}: SKIPPED ({err})")
            continue
        path = data.cache_prices(intraday, name)
        print(f"{interval}: {intraday.shape[1]} tickers x {intraday.shape[0]} rows -> {path}")


if __name__ == "__main__":
    main()
