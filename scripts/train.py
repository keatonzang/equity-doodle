"""Train the global quantile model on cached prices."""

from equity_doodle import data, model


def main() -> None:
    prices = data.load_cached()
    print(f"Training on {prices.shape[1]} tickers, {prices.shape[0]} rows...")
    m = model.train(prices)
    path = model.save(m)
    print(f"Saved trained model -> {path}")


if __name__ == "__main__":
    main()
