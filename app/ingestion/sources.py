from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable, List, Protocol

import httpx

from app.models import MarketRecord

logger = logging.getLogger(__name__)


class MarketDataSource(Protocol):
    name: str

    async def fetch(self, client: httpx.AsyncClient) -> List[MarketRecord]: ...


class CoinGeckoSource:
    """Public CoinGecko markets endpoint (no API key)."""

    name = "CoinGecko"

    def __init__(self, top_n: int = 50, vs_currency: str = "usd") -> None:
        self.top_n = top_n
        self.vs_currency = vs_currency

    async def fetch(self, client: httpx.AsyncClient) -> List[MarketRecord]:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": self.vs_currency,
            "order": "market_cap_desc",
            "per_page": self.top_n,
            "page": 1,
            "price_change_percentage": "24h",
        }
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()

        as_of = datetime.now(timezone.utc)
        records: List[MarketRecord] = []
        for entry in payload:
            records.append(
                MarketRecord(
                    symbol=str(entry.get("symbol", "")).upper(),
                    name=entry.get("name", ""),
                    price=float(entry.get("current_price") or 0),
                    change_24h=entry.get("price_change_percentage_24h"),
                    volume_24h=entry.get("total_volume"),
                    market_cap=entry.get("market_cap"),
                    source=self.name,
                    as_of=as_of,
                )
            )
        logger.info("Fetched %s records from %s", len(records), self.name)
        return records


class CoinPaprikaSource:
    """Public CoinPaprika tickers endpoint (no API key)."""

    name = "CoinPaprika"

    def __init__(self, top_n: int = 50) -> None:
        self.top_n = top_n

    async def fetch(self, client: httpx.AsyncClient) -> List[MarketRecord]:
        url = "https://api.coinpaprika.com/v1/tickers"
        resp = await client.get(url)
        resp.raise_for_status()
        payload = resp.json()

        as_of = datetime.now(timezone.utc)
        sorted_payload = sorted(payload, key=lambda row: row.get("rank") or 9999)
        sliced = sorted_payload[: self.top_n]

        records: List[MarketRecord] = []
        for entry in sliced:
            quotes = entry.get("quotes", {}).get("USD", {})
            records.append(
                MarketRecord(
                    symbol=str(entry.get("symbol", "")).upper(),
                    name=entry.get("name", ""),
                    price=float(quotes.get("price") or 0),
                    change_24h=quotes.get("percent_change_24h"),
                    volume_24h=quotes.get("volume_24h"),
                    market_cap=quotes.get("market_cap"),
                    source=self.name,
                    as_of=as_of,
                )
            )

        logger.info("Fetched %s records from %s", len(records), self.name)
        return records


def default_sources(top_n: int) -> List[MarketDataSource]:
    """Factory for the default source list."""
    return [
        CoinGeckoSource(top_n=top_n),
        CoinPaprikaSource(top_n=top_n),
    ]
