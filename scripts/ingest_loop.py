from __future__ import annotations

import argparse
import asyncio
import logging
import time

from app.ingestion.pipeline import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous ingestion loop to populate time-series history.")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Interval in seconds between runs (default: 300s = 5 minutes).",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    pipeline = build_pipeline()
    logger.info("Starting ingestion loop with interval %s seconds", args.interval)

    while True:
        start = time.time()
        summary = await pipeline.run_once()
        logger.info("Ingestion run completed: %s", summary.model_dump())

        elapsed = time.time() - start
        sleep_for = max(0, args.interval - elapsed)
        await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    asyncio.run(main())
