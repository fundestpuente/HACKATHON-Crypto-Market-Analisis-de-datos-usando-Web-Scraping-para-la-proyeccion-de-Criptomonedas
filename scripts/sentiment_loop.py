from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from app.config import settings
from app.sentiment.run_sentiment_pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Looping sentiment pipeline for periodic refresh.")
    parser.add_argument("--interval", type=int, default=900, help="Seconds between sentiment runs.")
    parser.add_argument("--limit", type=int, default=30, help="Max number of headlines to analyze.")
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.data_dir / "sentiment_coindesk.csv",
        help="Destination CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting sentiment loop: interval=%ss, limit=%s", args.interval, args.limit)
    while True:
        start = time.time()
        try:
            df = run_pipeline(limit=args.limit, output_path=args.output)
            logger.info("Sentiment refresh: %s rows -> %s", len(df), args.output)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Sentiment loop error: %s", exc)

        elapsed = time.time() - start
        sleep_for = max(0, args.interval - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
