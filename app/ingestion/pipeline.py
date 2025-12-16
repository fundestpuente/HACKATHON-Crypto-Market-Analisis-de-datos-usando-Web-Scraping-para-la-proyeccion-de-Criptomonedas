from __future__ import annotations

import asyncio
import logging
from typing import Iterable, List, Tuple

import httpx

from app.config import settings
from app.ingestion.sources import MarketDataSource, default_sources
from app.models import IngestionSummary, MarketRecord
from app.storage import MarketDataStore, ParquetMarketStore

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Orchestrates concurrent pulls from multiple market data sources."""

    def __init__(self, store: MarketDataStore, sources: Iterable[MarketDataSource], request_timeout_seconds: float) -> None:
        self.store = store
        self.sources = list(sources)
        self.request_timeout_seconds = request_timeout_seconds

    async def _fetch_single(self, client: httpx.AsyncClient, source: MarketDataSource) -> Tuple[str, List[MarketRecord], str | None]:
        try:
            records = await source.fetch(client)
            return (source.name, records, None)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error fetching from %s", source.name)
            return (source.name, [], str(exc))

    async def run_once(self) -> IngestionSummary:
        """Fetch from all sources, persist, and return a summary."""
        timeout = httpx.Timeout(self.request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            results = await asyncio.gather(
                *(self._fetch_single(client, src) for src in self.sources), return_exceptions=False
            )

        combined_records: List[MarketRecord] = []
        errors: List[str] = []
        for source_name, records, error in results:
            combined_records.extend(records)
            if error:
                errors.append(f"{source_name}: {error}")

        if settings.symbol_allowlist:
            combined_records = [r for r in combined_records if r.symbol.upper() in settings.symbol_allowlist]

        written = self.store.append(combined_records)
        return IngestionSummary(
            total_records=len(combined_records),
            written_records=written,
            sources=[src.name for src in self.sources],
            errors=errors,
        )


def build_pipeline() -> DataIngestionPipeline:
    """Create a pipeline with default settings and store."""
    store = ParquetMarketStore(settings.parquet_path)
    sources = default_sources(settings.top_n_assets)
    return DataIngestionPipeline(store=store, sources=sources, request_timeout_seconds=settings.request_timeout_seconds)
