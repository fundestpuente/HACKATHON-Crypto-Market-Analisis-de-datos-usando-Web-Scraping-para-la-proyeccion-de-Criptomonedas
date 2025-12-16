from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.config import settings
from app.storage import ParquetMarketStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the market dataset to CSV for ML consumption.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/export/market_dataset.csv"),
        help="Destination CSV path (directories will be created).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = ParquetMarketStore(settings.parquet_path)
    rows = store.export_csv(args.output)
    logger.info("Export complete: %s rows -> %s", rows, args.output)


if __name__ == "__main__":
    main()
