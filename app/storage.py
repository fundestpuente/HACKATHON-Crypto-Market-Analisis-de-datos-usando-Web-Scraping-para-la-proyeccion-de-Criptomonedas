from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Sequence

import pandas as pd

from app.models import MarketRecord

logger = logging.getLogger(__name__)


class MarketDataStore(Protocol):
    """Storage contract for market records."""

    def append(self, records: Sequence[MarketRecord]) -> int: ...

    def load(
        self,
        sources: Optional[Iterable[str]] = None,
        symbols: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
    ) -> List[MarketRecord]: ...

    def export_csv(self, destination: Path) -> int: ...


class ParquetMarketStore:
    """Parquet-backed store for quick local iteration."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _safe_read(self) -> pd.DataFrame:
        """Read parquet defensively; if corrupted/unreadable, return empty df and log."""
        try:
            df = pd.read_parquet(self.path)
            df["as_of"] = pd.to_datetime(df["as_of"], utc=True)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Parquet file unreadable at %s, resetting: %s", self.path, exc)
            return pd.DataFrame()

    def append(self, records: Sequence[MarketRecord]) -> int:
        if not records:
            return 0

        df_new = pd.DataFrame([r.model_dump() for r in records])
        df_new["as_of"] = pd.to_datetime(df_new["as_of"], utc=True)

        df_existing = self._safe_read() if self.path.exists() else pd.DataFrame()
        df = pd.concat([df_existing, df_new], ignore_index=True)

        df.drop_duplicates(subset=["source", "symbol", "as_of"], keep="last", inplace=True)
        df.sort_values(by=["as_of", "market_cap"], ascending=[True, False], inplace=True)
        df.to_parquet(self.path, index=False)

        logger.info("Persisted %s new records to %s", len(df_new), self.path)
        return len(df_new)

    def load(
        self,
        sources: Optional[Iterable[str]] = None,
        symbols: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
    ) -> List[MarketRecord]:
        if not self.path.exists():
            return []

        df = self._safe_read()
        if df.empty:
            return []

        if sources:
            df = df[df["source"].isin(list(sources))]
        if symbols:
            df = df[df["symbol"].isin(list(symbols))]

        df.sort_values(by=["as_of", "market_cap"], ascending=[False, False], inplace=True)
        if limit:
            df = df.head(limit)

        records: List[MarketRecord] = [MarketRecord(**row) for row in df.to_dict(orient="records")]
        return records

    def export_csv(self, destination: Path) -> int:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            logger.warning("No parquet file found at %s. Nothing to export.", self.path)
            return 0

        df = self._safe_read()
        if df.empty:
            logger.warning("Parquet file unreadable at %s. Nothing to export.", self.path)
            return 0

        df.to_csv(destination, index=False)
        logger.info("Exported %s rows to %s", len(df), destination)
        return len(df)
