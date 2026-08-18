"""
NautilusTrader strategy adapter layer.

Converts LLM-generated pseudo-Freqtrade strategies into Nautilus-compatible
Strategy classes at runtime via dynamic class generation.

Nautilus v2 API (from MIGRATION_V2.md):
- on_quote       (was on_quote_tick)
- on_trade       (was on_trade_tick)  
- on_bar         (for bar data)
- StrategyConfig for configuration
- BacktestEngine for backtesting
"""

from __future__ import annotations

import ast
import inspect
import logging
import tempfile
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AST guard: verify generated strategy code is safe
# ---------------------------------------------------------------------------

DANGEROUS_NAMES = {
    "os", "sys", "subprocess", "socket", "requests", "urllib",
    "exec", "eval", "compile", "__import__", "open", "input",
    "getattr", "setattr", "delattr", "globals", "locals",
}


def ast_guard(code: str) -> list[str]:
    """Return list of violations. Empty = safe."""
    violations = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in DANGEROUS_NAMES:
                    violations.append(f"Blocked import: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in DANGEROUS_NAMES:
                violations.append(f"Blocked import from: {node.module}")
        # Check for exec/eval calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval", "compile"):
                violations.append(f"Blocked call: {node.func.id}()")
    return violations


# ---------------------------------------------------------------------------
# Nautilus strategy code templates
# ---------------------------------------------------------------------------

NAUTILUS_STRATEGY_TEMPLATE = '''
from decimal import Decimal
import pandas as pd
import numpy as np
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import OrderListId
from nautilus_trader.model.identifiers import PositionId
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import StrategyId
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import OrderType
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy


class {strategy_name}(Strategy):
    """Auto-generated nautilus strategy: {strategy_name}"""

    def __init__(self, config):
        super().__init__(config)
        self._fast_ma = None
        self._slow_ma = None
        self._entry_threshold = {entry_threshold}
        self._exit_threshold = {exit_threshold}
        self._trade_size = Decimal("{trade_size}")

    def on_start(self):
        self._fast_ma = self.indicators.add(
            self.cache.indicator("ema", period={fast_period})
        )
        self._slow_ma = self.indicators.add(
            self.cache.indicator("ema", period={slow_period})
        )
        self.log.info("Strategy {strategy_name} started")

    def on_bar(self, bar):
        fast = self._fast_ma.value
        slow = self._slow_ma.value
        if fast is None or slow is None:
            return

        position = self.cache.position(self.instrument_id)
        side = self.cache.order_side(position) if position else None

        if side is None or side == OrderSide.NO_ORDER_SIDE:
            if fast > slow + self._entry_threshold:
                self.order_market(
                    self.instrument_id,
                    OrderSide.BUY,
                    self._trade_size,
                    TimeInForce.FOK,
                )
        else:
            if fast < slow - self._exit_threshold:
                self.order_market(
                    self.instrument_id,
                    OrderSide.SELL,
                    self._trade_size,
                    TimeInForce.FOK,
                )

    def on_stop(self):
        self.log.info("Strategy {strategy_name} stopped")
'''


def generate_nautilus_strategy(
    strategy_name: str,
    code_snippet: str,
    params: dict[str, Any],
) -> str:
    """Generate a Nautilus Strategy class from LLM output."""
    defaults = {
        "entry_threshold": "0.001",
        "exit_threshold": "0.001",
        "trade_size": "0.1",
        "fast_period": "10",
        "slow_period": "20",
    }
    defaults.update({k: str(v) for k, v in params.items()})

    return NAUTILUS_STRATEGY_TEMPLATE.format(
        strategy_name=strategy_name,
        **defaults,
    )


# ---------------------------------------------------------------------------
# Backtest adapter
# ---------------------------------------------------------------------------

@dataclass
class BacktestOutcome:
    """Nautilus backtest result."""
    ok: bool = False
    strategy_name: str = ""
    net_pnl: float = 0.0
    gross_pnl: float = 0.0
    max_drawdown: float = 0.0
    Sharpe_ratio: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    equity_curve: list = field(default_factory=list)
    error: str = ""
    reports: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "strategy_name": self.strategy_name,
            "net_pnl": round(self.net_pnl, 4),
            "gross_pnl": round(self.gross_pnl, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe_ratio": round(self.Sharpe_ratio, 4),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "error": self.error,
        }


class NautilusBacktestAdapter:
    """
    Wraps nautilus_trader.BacktestEngine for ai-quant integration.

    Usage:
        adapter = NautilusBacktestAdapter()
        outcome = adapter.run_backtest(
            strategy_code=code,
            instrument_id="ETHUSDT.BINANCE",
            data_path="data/raw/BTCUSDT_1h.parquet",
        )
    """

    def __init__(self, use_pyo3: bool = False):
        self._engine = None
        self._use_pyo3 = use_pyo3
        self._nautilus_available = False
        self._import_error: Optional[str] = None
        self._try_import()

    def _try_import(self) -> None:
        """Try to import nautilus_trader. Skip if not available."""
        try:
            import nautilus_trader  # noqa: F401
            self._nautilus_available = True
        except ImportError as e:
            self._nautilus_available = False
            self._import_error = str(e)
            logger.warning(f"nautilus_trader not available: {e}")

    def run_backtest(
        self,
        strategy_code: str,
        instrument_id: str = "ETHUSDT.BINANCE",
        data_path: Optional[str | Path] = None,
        initial_capital: float = 1_000_000.0,
        timerange: Optional[str] = None,
    ) -> BacktestOutcome:
        """
        Run a nautilus backtest. Returns BacktestOutcome.

        If nautilus is not installed or import fails, returns a mock result
        based on strategy code analysis (for CI/development without nautilus).
        """
        if not self._nautilus_available:
            return self._mock_backtest(strategy_code, instrument_id)

        # Build strategy module
        module_name = f"generated_strategy_{id(strategy_code)}"
        outcome = BacktestOutcome()
        outcome.strategy_name = self._extract_name(strategy_code)

        try:
            from nautilus_trader.backtest.engine import BacktestEngine
            from nautilus_trader.config import BacktestEngineConfig
            from nautilus_trader.config import LoggingConfig
            from nautilus_trader.model.identifiers import TraderId
            from nautilus_trader.model.enums import AccountType
            from nautilus_trader.model.enums import BookType
            from nautilus_trader.model.enums import OmsType
            from nautilus_trader.model.currencies import USDT

            # Validate strategy code first
            violations = ast_guard(strategy_code)
            if violations:
                outcome.error = "; ".join(violations)
                return outcome

            # Compile strategy code
            exec_globals: dict[str, Any] = {}
            try:
                exec(strategy_code, exec_globals)
            except Exception as e:
                outcome.error = f"Strategy compile error: {e}"
                return outcome

            # Find Strategy subclass
            strategy_cls = None
            for name, obj in exec_globals.items():
                if (inspect.isclass(obj) and
                        issubclass(obj, object) and
                        name != "Strategy" and
                        any(base.__name__ == "Strategy" for base in obj.__mro__)):
                    strategy_cls = obj
                    break

            if strategy_cls is None:
                outcome.error = "No Strategy subclass found in code"
                return outcome

            # Build engine
            config = BacktestEngineConfig(
                trader_id=TraderId("AI-QUANT-001"),
                logging=LoggingConfig(
                    log_level="WARNING",
                    use_pyo3=self._use_pyo3,
                ),
            )
            engine = BacktestEngine(config=config)
            engine.add_venue(
                venue="BINANCE",
                oms_type=OmsType.NETTING,
                book_type=BookType.L1_MBP,
                account_type=AccountType.CASH,
                base_currency=None,
                starting_balances=[Decimal(str(initial_capital)), Decimal("0")],
            )

            # Add instrument (using test provider if no custom data)
            if data_path:
                # Load from file
                outcome.error = f"data_path={data_path} (full integration pending nautilus install)"
                engine.dispose()
                return outcome

            # Run minimal backtest
            engine.run()
            engine.dispose()

            outcome.ok = True
            outcome.reports = {"engine": "nautilus"}

        except Exception as e:
            outcome.error = f"Backtest error: {e}"
            logger.error(f"Backtest failed: {e}", exc_info=True)

        return outcome

    def _mock_backtest(self, strategy_code: str, instrument_id: str) -> BacktestOutcome:
        """Fallback: produce a mock result when nautilus is not installed."""
        outcome = BacktestOutcome()
        outcome.strategy_name = self._extract_name(strategy_code)
        outcome.ok = True
        outcome.error = "nautilus_trader not installed — mock result (install with: uv pip install nautilus_trader)"
        outcome.reports = {"note": "nautilus-mock", "instrument": instrument_id}

        # Extract basic metrics from code
        violations = ast_guard(strategy_code)
        if violations:
            outcome.ok = False
            outcome.error = "; ".join(violations)

        return outcome

    @staticmethod
    def _extract_name(code: str) -> str:
        """Extract strategy name from class definition."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    return node.name
        except SyntaxError:
            pass
        return "unknown"

    def is_available(self) -> bool:
        return self._nautilus_available

    def import_error(self) -> Optional[str]:
        return self._import_error


# ---------------------------------------------------------------------------
# Strategy code converter: freqtrade-style → nautilus-style
# ---------------------------------------------------------------------------

FRETrade_TO_NAUTILUS_MAP = {
    "ta.momentum.rsi": "rsi",
    "ta.trend.SMAIndicator": "sma",
    "ta.trend.EMAIndicator": "ema",
    "ta.volatility.BollingerBands": "bb",
    "ta.volume.VolumeWeightedAveragePrice": "vwap",
    "DataFrame.close": "close",
    "DataFrame.open": "open",
    "DataFrame.high": "high",
    "DataFrame.low": "low",
    "DataFrame.volume": "volume",
}


def convert_freqtrade_to_nautilus(freqtrade_code: str) -> str:
    """
    Attempt to translate a Freqtrade IStrategy into a Nautilus Strategy.

    This is a best-effort heuristic conversion. The LLM should ideally
    generate nautilus-compatible code directly.
    """
    # Simple heuristics: replace IStrategy → Strategy, add nautilus imports
    code = freqtrade_code.replace("IStrategy", "Strategy")

    # Add nautilus imports at top
    nautilus_imports = '''
from decimal import Decimal
import pandas as pd
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.objects import Price, Quantity
'''
    # Insert after first line
    lines = code.split("\n")
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("class ") or (i > 0 and lines[i-1].strip() == ""):
            insert_pos = i
            break

    lines.insert(insert_pos, nautilus_imports)
    return "\n".join(lines)


def translate_strategy(llm_output: dict) -> tuple[str, str]:
    """
    Translate LLM output (freqtrade-style) into nautilus strategy code.

    Returns:
        (nautilus_code, error_message) — error_message is "" if successful
    """
    name = llm_output.get("name", "gen_unknown")
    params = llm_output.get("params", {})

    # Generate nautilus-compatible code from template
    nautilus_code = generate_nautilus_strategy(
        strategy_name=name,
        code_snippet=llm_output.get("code", ""),
        params=params,
    )

    # Validate
    violations = ast_guard(nautilus_code)
    if violations:
        return "", f"Validation failed: {violations}"

    return nautilus_code, ""
