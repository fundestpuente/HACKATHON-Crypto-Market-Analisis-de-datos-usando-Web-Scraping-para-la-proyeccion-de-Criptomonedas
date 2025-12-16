from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class MarketRecord(BaseModel):
    """Normalized market datapoint across sources."""

    symbol: str = Field(..., description="Ticker symbol (upper-case).")
    name: str = Field(..., description="Asset name.")
    price: float = Field(..., description="Last traded price in USD.")
    change_24h: Optional[float] = Field(
        None, description="Percentage change over the last 24h (can be null if source is missing)."
    )
    volume_24h: Optional[float] = Field(None, description="24h trading volume in USD.")
    market_cap: Optional[float] = Field(None, description="Market capitalization in USD.")
    source: str = Field(..., description="Source identifier.")
    as_of: datetime = Field(..., description="Timestamp of the snapshot in UTC.")


class IngestionSummary(BaseModel):
    """Outcome of a pipeline run."""

    total_records: int
    written_records: int
    sources: List[str]
    errors: List[str] = Field(default_factory=list)
