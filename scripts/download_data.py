"""Download and cache daily prices for the configured ticker universe."""

from equity_doodle import data


def main() -> None:
    prices = data.download_prices()
    path = data.cache_prices(prices)
    print(f"Saved {prices.shape[1]} tickers x {prices.shape[0]} rows -> {path}")


if __name__ == "__main__":
    main()
