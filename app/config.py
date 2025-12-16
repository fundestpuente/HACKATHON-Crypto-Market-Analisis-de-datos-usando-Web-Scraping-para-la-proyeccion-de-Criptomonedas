from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    """Central configuration driven by environment variables."""

    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
    top_n_assets: int = int(os.getenv("TOP_N_ASSETS", "25"))

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    parquet_filename: str = os.getenv("PARQUET_FILENAME", "market_snapshots.parquet")

    dashboard_api_base_url: str | None = os.getenv("API_BASE_URL")
    symbol_allowlist_raw: str | None = os.getenv("SYMBOL_ALLOWLIST")

    def ensure_paths(self) -> None:
        """Create expected directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "export").mkdir(parents=True, exist_ok=True)

    @property
    def parquet_path(self) -> Path:
        return self.data_dir / self.parquet_filename

    @property
    def resolved_api_base_url(self) -> str:
        if self.dashboard_api_base_url:
            return self.dashboard_api_base_url.rstrip("/")
        return f"http://{self.api_host}:{self.api_port}"

    @property
    def symbol_allowlist(self) -> set[str]:
        if not self.symbol_allowlist_raw:
            return set()
        return {token.strip().upper() for token in self.symbol_allowlist_raw.split(",") if token.strip()}


# Singleton-style settings import
settings: Final[Settings] = Settings()
settings.ensure_paths()
