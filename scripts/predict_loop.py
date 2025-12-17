from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

from scripts import train_and_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Looping trainer/predictor for periodic model refresh.")
    parser.add_argument("--interval", type=int, default=600, help="Seconds between training runs (default: 600s).")
    parser.add_argument("--symbols", type=str, default="BTC,DOGE", help="Comma-separated symbols.")
    parser.add_argument("--window", type=int, default=12, help="Window size.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon.")
    parser.add_argument("--source", type=str, default="CoinPaprika", help="Source filter.")
    parser.add_argument("--data-path", type=str, default="data/market_snapshots.parquet", help="Parquet path.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--model-dir", type=str, default="data/export/models", help="Directory to save models.")
    env_step = os.getenv("PREDICT_STEP_MINUTES")
    default_step = int(env_step) if env_step else None
    parser.add_argument("--step-minutes", type=int, default=default_step, help="Override forecast step size in minutes.")
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=float(os.getenv("PREDICT_BLEND_ALPHA", "0.35")),
        help="Blend weight for model vs baseline returns (0-1).",
    )
    parser.add_argument(
        "--return-clip-mult",
        type=float,
        default=float(os.getenv("PREDICT_RETURN_CLIP_MULT", "3.0")),
        help="Clip multiplier over recent return volatility.",
    )
    parser.add_argument(
        "--return-clip-min",
        type=float,
        default=float(os.getenv("PREDICT_MIN_RETURN_CLIP", "0.001")),
        help="Minimum absolute clip for predicted returns.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logger.info("Starting predict loop: symbols=%s, interval=%ss", args.symbols, args.interval)
    while True:
        start = time.time()
        try:
            argv = [
                f"--symbols={args.symbols}",
                f"--data-path={args.data_path}",
                f"--source={args.source}",
                f"--window={args.window}",
                f"--horizon={args.horizon}",
                f"--epochs={args.epochs}",
                f"--batch-size={args.batch_size}",
                f"--model-dir={args.model_dir}",
                f"--blend-alpha={args.blend_alpha}",
                f"--return-clip-mult={args.return_clip_mult}",
                f"--return-clip-min={args.return_clip_min}",
            ]
            if args.step_minutes is not None:
                argv.append(f"--step-minutes={args.step_minutes}")
            train_and_predict.main(argv)
        except Exception as exc:
            logger.exception("Predict loop error: %s", exc)

        elapsed = time.time() - start
        sleep_for = max(0, args.interval - elapsed)
        await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    asyncio.run(main())
