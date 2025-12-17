from __future__ import annotations

"""
Batch trainer/predictor for multiple symbols (e.g., BTC and DOGE) to speed up demo workflows.

- Loads Parquet data, optionally filters by source.
- Trains a Conv1D + stacked LSTM model per symbol with small window defaults.
- Saves models/metadata and writes latest prediction JSON per symbol for the dashboard overlay.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from app.config import settings

DEFAULT_SENTIMENT_PATH = settings.data_dir / "sentiment_coindesk.csv"


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_float(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default

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
        "--sentiment-path",
        type=Path,
        default=DEFAULT_SENTIMENT_PATH,
        help="Path to sentiment CSV (defaults to DATA_DIR/sentiment_coindesk.csv).",
    )
    parser.add_argument("--sentiment-limit", type=int, default=30, help="Max headlines to analyze when refreshing.")
    parser.add_argument(
        "--refresh-sentiment",
        action="store_true",
        help="Refresh sentiment CSV before training (optional).",
    )
    parser.add_argument(
        "--step-minutes",
        type=int,
        default=_env_int("PREDICT_STEP_MINUTES"),
        help="Override forecast step size in minutes (default: infer or env).",
    )
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=_env_float("PREDICT_BLEND_ALPHA", 0.35),
        help="Blend weight for model vs baseline returns (0-1). Lower = closer to baseline.",
    )
    parser.add_argument(
        "--return-clip-mult",
        type=float,
        default=_env_float("PREDICT_RETURN_CLIP_MULT", 3.0),
        help="Clip multiplier over recent return volatility.",
    )
    parser.add_argument(
        "--return-clip-min",
        type=float,
        default=_env_float("PREDICT_MIN_RETURN_CLIP", 0.001),
        help="Minimum absolute clip for predicted returns.",
    )
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
            tf.keras.layers.Input(shape=(window, 2)),
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

    df_sym["log_return"] = (
        df_sym["log_return"]
        .rolling(window=3, min_periods=1, center=True)
        .median()
    )

    if "daily_sentiment" not in df_sym.columns:
        df_sym["daily_sentiment"] = 0.0
    else:
        df_sym["daily_sentiment"] = pd.to_numeric(df_sym["daily_sentiment"], errors="coerce").fillna(0.0)

    df_sym = df_sym.dropna(subset=["log_return", "daily_sentiment"])

    return df_sym[["log_return", "daily_sentiment"]].to_numpy()


def load_daily_sentiment(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "daily_sentiment"])

    df = pd.read_csv(path)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["date", "daily_sentiment"])

    score_col = None
    for candidate in ("sentiment_score", "sentiment", "polarity"):
        if candidate in df.columns:
            score_col = candidate
            break
    if score_col is None:
        return pd.DataFrame(columns=["date", "daily_sentiment"])

    df["date"] = pd.to_datetime(df["date"]).dt.date

    daily = (
        df.groupby("date")[score_col]
        .mean()
        .reset_index()
        .rename(columns={score_col: "daily_sentiment"})
    )
    return daily


def infer_step_minutes(df_sym: pd.DataFrame, fallback: int = 5) -> int:
    if "as_of" not in df_sym.columns:
        return fallback
    ts = pd.to_datetime(df_sym["as_of"]).sort_values()
    if len(ts) < 2:
        return fallback
    diffs = ts.diff().dropna().dt.total_seconds() / 60.0
    if diffs.empty:
        return fallback
    median = float(diffs.median())
    if not np.isfinite(median) or median <= 0:
        return fallback
    return max(1, int(round(median)))


def calibrate_pred_returns(
    pred_returns: np.ndarray,
    recent_returns: np.ndarray,
    blend_alpha: float,
    clip_mult: float,
    clip_min: float,
) -> np.ndarray:
    if recent_returns.size == 0:
        return pred_returns

    recent_mean = float(np.mean(recent_returns))
    recent_vol = float(np.std(recent_returns)) + 1e-8

    blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    pred_returns = blend_alpha * pred_returns + (1.0 - blend_alpha) * recent_mean

    max_abs = max(float(clip_min), float(clip_mult) * recent_vol)
    pred_returns = np.clip(pred_returns, -max_abs, max_abs)
    return pred_returns


def train_single(
    df: pd.DataFrame,
    symbol: str,
    window: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    sentiment_path: Path,
) -> tuple[tf.keras.Model, dict, np.ndarray, np.ndarray, float]:
    df_sym = df[df["symbol"].str.upper() == symbol].copy()
    df_sym = df_sym.sort_values("as_of")
    # Merge daily sentiment
    sentiment_df = load_daily_sentiment(sentiment_path)

    df_sym["date"] = pd.to_datetime(df_sym["as_of"]).dt.date
    df_sym = df_sym.merge(sentiment_df, on="date", how="left")

    # Neutral sentiment if no news that day
    if "daily_sentiment" in df_sym.columns:
        df_sym["daily_sentiment"] = pd.to_numeric(df_sym["daily_sentiment"], errors="coerce").fillna(0.0)
    else:
        df_sym["daily_sentiment"] = 0.0

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

    mu = X_train.mean(axis=(0, 1))
    sigma = X_train.std(axis=(0, 1)) + 1e-8

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
    meta = {
        "symbol": symbol,
        "window": window,
        "horizon": horizon,
        "mu": [float(x) for x in np.atleast_1d(mu)],
        "sigma": [float(x) for x in np.atleast_1d(sigma)],
        "test": test_metrics,
        "history": history.history,
    }
    return model, meta, mu, sigma, df_sym["price"].iloc[-1]


def predict_next(
    model: tf.keras.Model,
    df_sym: pd.DataFrame,
    window: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    horizon: int,
    sentiment_path: Path,
    step_minutes: int | None = None,
    fallback_step_minutes: int = 5,
    blend_alpha: float = 0.35,
    return_clip_mult: float = 3.0,
    return_clip_min: float = 0.001,
):
    predicted_at = pd.Timestamp.utcnow()

    # Merge sentiment
    sentiment_df = load_daily_sentiment(sentiment_path)
    df_sym = (
        df_sym.sort_values("as_of")
        .assign(
            price=lambda x: pd.to_numeric(x["price"], errors="coerce"),
            date=lambda x: pd.to_datetime(x["as_of"]).dt.date,
        )
        .merge(sentiment_df, on="date", how="left")
        .dropna(subset=["price"])
    )

    # Neutral sentiment if no news
    if "daily_sentiment" in df_sym.columns:
        df_sym["daily_sentiment"] = pd.to_numeric(df_sym["daily_sentiment"], errors="coerce").fillna(0.0)
    else:
        df_sym["daily_sentiment"] = 0.0

    # Now returns + sentiment
    latest_features = sanitize_returns(df_sym)

    if len(latest_features) < window:
        if len(latest_features) == 0:
            raise ValueError(f"No valid data to predict for {df_sym['symbol'].iloc[0]}")
        pad = np.repeat(
            latest_features[-1].reshape(1, 2),
            window - len(latest_features),
            axis=0,
        )
        latest_features = np.vstack([latest_features, pad])
    else:
        latest_features = latest_features[-window:]

    last_price = float(df_sym["price"].iloc[-1])
    if step_minutes is None:
        step_minutes = infer_step_minutes(df_sym, fallback=fallback_step_minutes)

    # (1, window, 2)
    X = latest_features.reshape(1, window, 2)
    X = (X - mu) / sigma

    pred_returns = model.predict(X, verbose=0)[0]
    recent_returns = latest_features[:, 0].astype(float)
    pred_returns = calibrate_pred_returns(
        pred_returns=pred_returns,
        recent_returns=recent_returns,
        blend_alpha=blend_alpha,
        clip_mult=return_clip_mult,
        clip_min=return_clip_min,
    )

    forecast_prices = []
    running = last_price
    for r in pred_returns:
        running *= float(np.exp(r))
        forecast_prices.append(running)

    last_ts = df_sym["as_of"].max()
    forecast_times = [
        last_ts + pd.Timedelta(minutes=step_minutes * (i + 1))
        for i in range(len(pred_returns))
    ]

    return pred_returns, forecast_prices, last_price, forecast_times, predicted_at


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols and settings.symbol_allowlist:
        symbols = [s.upper() for s in settings.symbol_allowlist]
    settings.ensure_paths()

    if args.refresh_sentiment:
        try:
            from app.sentiment.run_sentiment_pipeline import run_pipeline

            run_pipeline(limit=args.sentiment_limit, output_path=args.sentiment_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[sentiment] Refresh failed, continuing without update: {exc}")

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
                sentiment_path=args.sentiment_path,
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
            sentiment_path=args.sentiment_path,
            step_minutes=args.step_minutes,
            blend_alpha=args.blend_alpha,
            return_clip_mult=args.return_clip_mult,
            return_clip_min=args.return_clip_min,
        )
        payload = {
            "symbol": sym,
            "source": args.source,
            "last_price": last_price,
            "predicted_returns": [float(r) for r in pred_returns],
            "predicted_prices": [float(p) for p in pred_prices],
            "horizon": meta["horizon"],
            "forecast_times": [t.isoformat() for t in forecast_times],
            "predicted_at": predicted_at.isoformat(),
            "calibration": {
                "blend_alpha": float(args.blend_alpha),
                "return_clip_mult": float(args.return_clip_mult),
                "return_clip_min": float(args.return_clip_min),
            },
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
