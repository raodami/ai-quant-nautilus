"""
Paper trading / dry-run simulation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PaperResult:
    """Paper trading result."""
    ok: bool = False
    strategy_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    error: str = ""


class DryRunner:
    """Simulate paper trading without real exchange connection."""

    def __init__(self):
        self._running: dict[str, bool] = {}

    def start_paper(self, strategy_id: str) -> bool:
        self._running[strategy_id] = True
        return True

    def stop_paper(self, strategy_id: str) -> Optional[PaperResult]:
        if strategy_id not in self._running:
            return None
        self._running.pop(strategy_id)

        return PaperResult(
            ok=True,
            strategy_id=strategy_id,
            start_time=time.time(),
            end_time=time.time(),
            total_return=0.0,  # mock
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            total_trades=0,
        )
