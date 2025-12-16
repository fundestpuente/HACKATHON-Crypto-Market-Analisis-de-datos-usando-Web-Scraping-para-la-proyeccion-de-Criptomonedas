from __future__ import annotations

"""
Batch trainer/predictor for multiple symbols (e.g., BTC and DOGE) to speed up demo workflows.

- Loads Parquet data, optionally filters by source.
- Trains a Conv1D + stacked LSTM model per symbol with small window defaults.
- Saves models/metadata and writes latest prediction JSON per symbol for the dashboard overlay.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from app.config import settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and predict for a list of symbols (e.g., BTC,DOGE).")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC,DOGE",
        help="Comma-separated list of symbols to train (upper-case).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.parquet_path,
        help="Path to Parquet dataset.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="CoinPaprika",
        help="Optional source filter to reduce cross-source noise.",
    )
    parser.add_argument("--window", type=int, default=12, help="Lookback window length (small default for sparse data).")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/export/models"),
        help="Directory to save models, metadata, and predictions.",
    )
    return parser.parse_args(argv)


def make_windows(series: np.ndarray, window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for idx in range(len(series) - window - horizon + 1):
        xs.append(series[idx : idx + window])
        ys.append(series[idx + window : idx + window + horizon])
    return np.array(xs), np.array(ys)


def build_model(window: int, horizon: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, 1)),
            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=3,
                padding="causal",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            ),
            tf.keras.layers.SpatialDropout1D(0.1),
            tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
            tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
            tf.keras.layers.Dense(horizon),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def sanitize_returns(df_sym: pd.DataFrame) -> np.ndarray:
    """Prepare log-returns: sort, de-dup, outlier trim, light smoothing."""
    df_sym = (
        df_sym.sort_values("as_of")
        .drop_duplicates(subset="as_of", keep="last")
        .assign(price=lambda x: pd.to_numeric(x["price"], errors="coerce"))
        .dropna(subset=["price"])
    )
    df_sym["log_return"] = np.log(df_sym["price"].astype(float)).diff()
    df_sym = df_sym.dropna(subset=["log_return"])

    med = df_sym["log_return"].median()
    mad = (df_sym["log_return"] - med).abs().median() + 1e-8
    z = 0.6745 * (df_sym["log_return"] - med) / mad
    df_sym = df_sym[np.abs(z) <= 6]

    df_sym["log_return"] = df_sym["log_return"].rolling(window=3, min_periods=1, center=True).median()
    df_sym = df_sym.dropna(subset=["log_return"])
    return df_sym["log_return"].to_numpy()


def train_single(
    df: pd.DataFrame,
    symbol: str,
    window: int,
    horizon: int,
    epochs: int,
    batch_size: int,
) -> tuple[tf.keras.Model, dict, float, float, float]:
    df_sym = df[df["symbol"].str.upper() == symbol].copy()
    df_sym = df_sym.sort_values("as_of")
    if df_sym.empty:
        raise ValueError(f"No data for symbol {symbol}.")

    returns = sanitize_returns(df_sym)

    min_needed = window + horizon + 1
    if len(returns) < min_needed:
        suggested = max(4, len(returns) - horizon - 1)
        if suggested <= 0:
            raise ValueError(
                f"Not enough data to build any window for {symbol}. Have {len(returns)} points; need at least {horizon + 2}."
            )
        print(f"[{symbol}] Data is sparse. Reducing WINDOW from {window} to {suggested}.")
        window = suggested

    X, y = make_windows(returns, window, horizon)
    X = np.expand_dims(X, axis=-1)
    n = len(X)
    if n == 0:
        raise ValueError(f"Still no windows for {symbol} after adjusting window size.")

    test_n = int(n * 0.15)
    val_n = int(n * 0.15)
    train_n = n - val_n - test_n
    if train_n < 1:
        train_n = 1
        val_n = max(0, n - train_n - test_n)
        test_n = max(0, n - train_n - val_n)

    X_train, y_train = X[:train_n], y[:train_n]
    X_val, y_val = X[train_n : train_n + val_n], y[train_n : train_n + val_n]
    X_test, y_test = X[train_n + val_n :], y[train_n + val_n :]

    mu = X_train.mean()
    sigma = X_train.std() + 1e-8
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma
    X_test = (X_test - mu) / sigma

    model = build_model(window, horizon)
    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, min_delta=1e-5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5, verbose=1),
    ]

    val_data = (X_val, y_val) if len(X_val) > 0 else None
    history = model.fit(
        X_train,
        y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    test_metrics = None
    if len(X_test) > 0:
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        test_metrics = {"loss": float(test_loss), "mae": float(test_mae)}
        last_truth = y_test.squeeze()[-1]
    else:
        last_truth = None

    meta = {
        "symbol": symbol,
        "window": window,
        "horizon": horizon,
        "mu": float(mu),
        "sigma": float(sigma),
        "test": test_metrics,
        "history": history.history,
    }
    return model, meta, mu, sigma, df_sym["price"].iloc[-1]


def predict_next(
    model: tf.keras.Model,
    df_sym: pd.DataFrame,
    window: int,
    mu: float,
    sigma: float,
    horizon: int,
    step_minutes: int = 5,
):
    predicted_at = pd.Timestamp.utcnow()
    df_sym = (
        df_sym.sort_values("as_of")
        .assign(price=lambda x: pd.to_numeric(x["price"], errors="coerce"))
        .dropna(subset=["price"])
    )
    latest_returns = sanitize_returns(df_sym)
    if len(latest_returns) < window:
        if len(latest_returns) == 0:
            raise ValueError(f"No valid returns to predict for {df_sym['symbol'].iloc[0]}")
        pad = np.repeat(latest_returns[-1], window - len(latest_returns))
        latest_returns = np.concatenate([latest_returns, pad])
    else:
        latest_returns = latest_returns[-window:]
    last_price = float(df_sym["price"].iloc[-1])
    X = latest_returns.reshape(1, window, 1)
    X = (X - mu) / sigma
    pred_returns = model.predict(X, verbose=0)[0]
    forecast_prices = []
    running = last_price
    for r in pred_returns:
        running *= float(np.exp(r))
        forecast_prices.append(running)
    last_ts = df_sym["as_of"].max()
    forecast_times = [last_ts + pd.Timedelta(minutes=step_minutes * (i + 1)) for i in range(len(pred_returns))]
    return pred_returns, forecast_prices, last_price, forecast_times, predicted_at


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols and settings.symbol_allowlist:
        symbols = [s.upper() for s in settings.symbol_allowlist]
    settings.ensure_paths()

    df = pd.read_parquet(args.data_path)
    df["as_of"] = pd.to_datetime(df["as_of"])
    df = df.sort_values("as_of")
    if settings.symbol_allowlist:
        df = df[df["symbol"].str.upper().isin({s.upper() for s in settings.symbol_allowlist})]
    if args.source:
        df = df[df["source"] == args.source]

    args.model_dir.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        try:
            model, meta, mu, sigma, last_price = train_single(
                df=df,
                symbol=sym,
                window=args.window,
                horizon=args.horizon,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        except ValueError as exc:
            print(f"[{sym}] Skipping: {exc}")
            continue

        # Save model/meta
        model_path = args.model_dir / f"{sym}_model.keras"
        meta_path = args.model_dir / f"{sym}_meta.json"
        model.save(model_path)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[{sym}] Saved model to {model_path}, meta to {meta_path}")

        # Prediction
        pred_returns, pred_prices, last_price, forecast_times, predicted_at = predict_next(
            model=model,
            df_sym=df[df["symbol"].str.upper() == sym],
            window=meta["window"],
            mu=mu,
            sigma=sigma,
            horizon=meta["horizon"],
        )
        payload = {
            "symbol": sym,
            "last_price": last_price,
            "predicted_returns": [float(r) for r in pred_returns],
            "predicted_prices": [float(p) for p in pred_prices],
            "horizon": meta["horizon"],
            "forecast_times": [t.isoformat() for t in forecast_times],
            "predicted_at": predicted_at.isoformat(),
        }
        pred_path = args.model_dir / f"latest_prediction_{sym}.json"
        pred_path.write_text(json.dumps(payload, indent=2))
        print(f"[{sym}] Wrote prediction to {pred_path}")

        # Append to history CSV for visualization of past predictions vs. realized prices
        history_path = args.model_dir / f"prediction_history_{sym}.csv"
        rows = []
        for step, (ft, price, ret) in enumerate(zip(forecast_times, pred_prices, pred_returns), start=1):
            rows.append(
                {
                    "predicted_at": predicted_at.isoformat(),
                    "forecast_time": ft.isoformat(),
                    "step": step,
                    "predicted_price": price,
                    "predicted_return": ret,
                }
            )
        hist_df = pd.DataFrame(rows)
        if history_path.exists():
            existing = pd.read_csv(history_path)
            hist_df = pd.concat([existing, hist_df], ignore_index=True)
        hist_df.to_csv(history_path, index=False)

        # Also write a generic latest_prediction.json for backward compatibility (first symbol only)
        if sym == symbols[0]:
            generic_path = args.model_dir / "latest_prediction.json"
            generic_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
