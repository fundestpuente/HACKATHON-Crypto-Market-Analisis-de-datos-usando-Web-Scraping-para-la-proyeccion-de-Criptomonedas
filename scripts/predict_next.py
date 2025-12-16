from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from app.config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict next-step return and price for a symbol.")
    parser.add_argument("--symbol", type=str, default="BTC", help="Symbol to forecast (upper-case).")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.parquet_path,
        help="Path to Parquet dataset (default: data/market_snapshots.parquet).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/export/models/BTC_model.keras"),
        help="Path to trained Keras model.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("data/export/models/BTC_meta.json"),
        help="Path to metadata JSON produced by train_model.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/export/models/latest_prediction.json"),
        help="Optional path to write the prediction payload for the dashboard overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()

    meta = json.loads(args.meta_path.read_text())
    if meta["symbol"] != symbol:
        raise SystemExit(f"Metadata symbol {meta['symbol']} does not match requested {symbol}")

    window = int(meta["window"])
    horizon = int(meta["horizon"])
    mu = float(meta["mu"])
    sigma = float(meta["sigma"])

    df = pd.read_parquet(args.data_path)
    df["as_of"] = pd.to_datetime(df["as_of"])
    df = df.sort_values("as_of")
    df_symbol = df[df["symbol"].str.upper() == symbol].copy()
    if len(df_symbol) < window + 1:
        raise SystemExit(f"Not enough data for {symbol}; need at least {window+1} rows.")

    df_symbol["log_return"] = np.log(df_symbol["price"].astype(float)).diff()
    latest_returns = df_symbol["log_return"].dropna().to_numpy()[-window:]
    last_price = float(df_symbol["price"].iloc[-1])

    X = latest_returns.reshape(1, window, 1)
    X = (X - mu) / sigma

    model = tf.keras.models.load_model(args.model_path)
    pred_returns = model.predict(X, verbose=0)[0]  # shape (horizon,)

    forecast = []
    running_price = last_price
    for r in pred_returns:
        running_price = running_price * float(np.exp(r))
        forecast.append(running_price)

    print(f"Symbol: {symbol}")
    print(f"Last price: {last_price:.6f}")
    for idx, (ret_pred, price_pred) in enumerate(zip(pred_returns, forecast), start=1):
        print(f"T+{idx}: return_pred={ret_pred:.6f}, price_pred={price_pred:.6f}")

    payload = {
        "symbol": symbol,
        "last_price": last_price,
        "predicted_returns": [float(r) for r in pred_returns],
        "predicted_prices": [float(p) for p in forecast],
        "horizon": horizon,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote prediction payload to {args.output_json}")


if __name__ == "__main__":
    main()
