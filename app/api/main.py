from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.ingestion.pipeline import build_pipeline
from app.models import IngestionSummary, MarketRecord
from app.storage import ParquetMarketStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crypto Market AI API",
    version="0.1.0",
    description="Ingestion + analytics service for crypto market data.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = ParquetMarketStore(settings.parquet_path)
pipeline = build_pipeline()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/sources", response_model=List[str])
async def list_sources() -> List[str]:
    return [src.name for src in pipeline.sources]


@app.post("/ingest/run", response_model=IngestionSummary)
async def trigger_ingestion(background: bool = Query(False, description="Run ingestion as a background task")):
    """Trigger the ingestion pipeline. Use `background=true` to return immediately."""
    if background:
        asyncio.create_task(pipeline.run_once())
        return IngestionSummary(total_records=0, written_records=0, sources=[src.name for src in pipeline.sources])

    return await pipeline.run_once()


@app.get("/markets", response_model=List[MarketRecord])
async def list_markets(
    source: Optional[str] = Query(None, description="Filter by data source"),
    symbol: Optional[str] = Query(None, description="Filter by ticker symbol (upper-case)"),
    limit: int = Query(200, ge=1, le=1000, description="Maximum number of records to return"),
) -> List[MarketRecord]:
    filters_sources = [source] if source else None
    filters_symbols = [symbol.upper()] if symbol else None
    records = store.load(sources=filters_sources, symbols=filters_symbols, limit=limit)
    if settings.symbol_allowlist:
        allow = {s.upper() for s in settings.symbol_allowlist}
        records = [r for r in records if r.symbol.upper() in allow]
    return records
