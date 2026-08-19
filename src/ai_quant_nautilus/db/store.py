"""Database models for ai-quant-nautilus."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StrategyStore:
    """Persistent strategy storage via JSON file."""

    def __init__(self, db_path: str = "data/strategies.json"):
        self.path = db_path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        from pathlib import Path
        p = Path(self.path)
        if p.exists():
            try:
                raw = json.loads(p.read_text())
                self._data = {s["id"]: s for s in raw.get("strategies", [])}
            except Exception as e:
                logger.warning(f"Failed to load strategies: {e}")
                self._data = {}

    def _save(self) -> None:
        from pathlib import Path
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"strategies": list(self._data.values())}, indent=2), encoding="utf-8")

    def save(self, rec: dict) -> None:
        self._data[rec["id"]] = rec
        self._save()

    def get(self, strategy_id: str) -> Optional[dict]:
        return self._data.get(strategy_id)

    def list_all(self) -> list[dict]:
        return list(self._data.values())

    def delete(self, strategy_id: str) -> bool:
        if strategy_id in self._data:
            del self._data[strategy_id]
            self._save()
            return True
        return False


class BacktestStore:
    """Persistent backtest result storage."""

    def __init__(self, db_path: str = "data/backtests.json"):
        self.path = db_path
        self._records: list[dict] = []
        self._load()

    def _load(self) -> None:
        from pathlib import Path
        p = Path(self.path)
        if p.exists():
            try:
                raw = json.loads(p.read_text())
                self._records = raw.get("backtests", [])
            except Exception as e:
                logger.warning(f"Failed to load backtests: {e}")
                self._records = []

    def _save(self) -> None:
        from pathlib import Path
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._records, indent=2), encoding="utf-8")

    def add(self, result: dict) -> str:
        record = {"id": f"bt_{datetime.utcnow().isoformat()}", **result}
        self._records.append(record)
        self._save()
        return record["id"]

    def list_all(self, limit: int = 50) -> list[dict]:
        return self._records[:limit]

    def get(self, bt_id: str) -> Optional[dict]:
        for r in self._records:
            if r["id"] == bt_id:
                return r
        return None
