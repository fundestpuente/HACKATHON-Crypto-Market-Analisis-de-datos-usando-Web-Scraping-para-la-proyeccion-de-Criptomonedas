from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.config import settings
from app.sentiment.run_sentiment_pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sentiment pipeline once and write CSV output.")
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
    try:
        df = run_pipeline(limit=args.limit, output_path=args.output)
        logger.info("Sentiment saved: %s rows -> %s", len(df), args.output)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sentiment pipeline failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
