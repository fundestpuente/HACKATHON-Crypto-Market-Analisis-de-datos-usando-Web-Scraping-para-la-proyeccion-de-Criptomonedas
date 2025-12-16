from __future__ import annotations

import asyncio
import json
import logging

from app.ingestion.pipeline import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    pipeline = build_pipeline()
    summary = await pipeline.run_once()
    logger.info("Ingestion summary: %s", summary.model_dump())
    print(json.dumps(summary.model_dump(), indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
