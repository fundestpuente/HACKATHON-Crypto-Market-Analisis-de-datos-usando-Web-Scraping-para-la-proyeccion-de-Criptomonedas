from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from app.config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a per-symbol LSTM forecaster on market data.")
    parser.add_argument("--symbol", type=str, default="BTC", help="Symbol to train on (upper-case).")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.parquet_path,
        help="Path to Parquet dataset (default: data/market_snapshots.parquet).",
    )
    parser.add_argument("--window", type=int, default=72, help="Lookback window length.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon (steps ahead).")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation fraction.")
    parser.add_argument("--test-split", type=float, default=0.15, help="Test fraction.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/export/models"),
        help="Directory to save the trained model and metadata.",
    )
    return parser.parse_args()


def sanitize_returns(df_sym: pd.DataFrame) -> np.ndarray:
    """Prepare log-returns: sort, drop duplicates, remove outliers, light smoothing."""
    df_sym = (
        df_sym.sort_values("as_of")
        .drop_duplicates(subset="as_of", keep="last")
        .assign(price=lambda x: pd.to_numeric(x["price"], errors="coerce"))
        .dropna(subset=["price"])
    )
    df_sym["log_return"] = np.log(df_sym["price"].astype(float)).diff()
    df_sym = df_sym.dropna(subset=["log_return"])

    # Robust outlier removal via MAD to avoid spurious spikes
    med = df_sym["log_return"].median()
    mad = (df_sym["log_return"] - med).abs().median() + 1e-8
    z = 0.6745 * (df_sym["log_return"] - med) / mad
    df_sym = df_sym[np.abs(z) <= 6]

    # Mild smoothing to reduce single-tick noise
    df_sym["log_return"] = df_sym["log_return"].rolling(window=3, min_periods=1, center=True).median()
    df_sym = df_sym.dropna(subset=["log_return"])
    return df_sym["log_return"].to_numpy()


def make_windows(series: np.ndarray, window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for idx in range(len(series) - window - horizon + 1):
        xs.append(series[idx : idx + window])
        ys.append(series[idx + window : idx + window + horizon])
    return np.array(xs), np.array(ys)


def build_model(window: int, horizon: int) -> tf.keras.Model:
    """Convolutional + stacked LSTM head for short-horizon forecasting."""
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


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()
    settings.ensure_paths()

    df = pd.read_parquet(args.data_path)
    df["as_of"] = pd.to_datetime(df["as_of"])
    df = df.sort_values("as_of")

    # Apply allowlist if configured
    if settings.symbol_allowlist and symbol not in settings.symbol_allowlist:
        raise SystemExit(f"Symbol {symbol} is not in SYMBOL_ALLOWLIST. Current allowlist: {settings.symbol_allowlist}")

    df_symbol = df[df["symbol"].str.upper() == symbol].copy()
    if df_symbol.empty or len(df_symbol) < (args.window + args.horizon + 10):
        raise SystemExit(f"Not enough data for {symbol}. Need at least window+horizon+10 rows.")

    returns = sanitize_returns(df_symbol)
    if len(returns) < (args.window + args.horizon + 5):
        raise SystemExit(f"Not enough cleaned data for {symbol} after sanitization. Have {len(returns)} points.")

    X, y = make_windows(returns, args.window, args.horizon)
    X = np.expand_dims(X, axis=-1)

    n = len(X)
    test_n = int(n * args.test_split)
    val_n = int(n * args.val_split)
    train_n = n - val_n - test_n

    X_train, y_train = X[:train_n], y[:train_n]
    X_val, y_val = X[train_n : train_n + val_n], y[train_n : train_n + val_n]
    X_test, y_test = X[train_n + val_n :], y[train_n + val_n :]

    # Standardize using training statistics
    mu = X_train.mean()
    sigma = X_train.std() + 1e-8
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma
    X_test = (X_test - mu) / sigma

    model = build_model(args.window, args.horizon)
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            min_delta=1e-5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    val_data = (X_val, y_val) if len(X_val) > 0 else None
    history = model.fit(
        X_train,
        y_train,
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    test_metrics = None
    if len(X_test) > 0:
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        test_metrics = {"loss": float(test_loss), "mae": float(test_mae)}

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / f"{symbol}_model.keras"
    meta_path = args.model_dir / f"{symbol}_meta.json"

    model.save(model_path)

    meta = {
        "symbol": symbol,
        "window": args.window,
        "horizon": args.horizon,
        "mu": float(mu),
        "sigma": float(sigma),
        "test": test_metrics,
        "history": history.history,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {meta_path}")
    if test_metrics:
        print(f"Test MAE: {test_metrics['mae']:.6f}")


if __name__ == "__main__":
    main()
