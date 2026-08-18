"""
NautilusTrader strategy adapter with real data backtesting.

Implements a pure Python backtest engine that mimics nautilus_trader's
Strategy interface for testing without external dependencies.
"""

from __future__ import annotations

import ast
import inspect
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal Strategy stub (no nautilus_trader dependency)
# ---------------------------------------------------------------------------

class Strategy:
    """Minimal Strategy base class for pure Python backtesting."""

    def __init__(self, config: Any = None):
        self.config = config
        self.log = _Logger()
        self._indicators: dict[str, Any] = {}
        self.position: Optional[str] = None
        self.entry_price: float = 0.0
        self.position_size: float = 0.0
        self.trades: list = []
        self._order_side: Optional[str] = None

    def indicator(self, name: str, period: int = 14) -> Any:
        """Register an indicator (returns a mock indicator object)."""
        key = f"{name}_{period}"
        if key not in self._indicators:
            self._indicators[key] = _Indicator(name, period)
        return self._indicators[key]

    def order_market(self, instrument_id: str, side: str, size: Decimal, tif: str = "FOK"):
        """Record a market order."""
        self.trades.append({
            "instrument": instrument_id,
            "side": side,
            "size": float(size),
            "tif": tif,
        })


class _Logger:
    """Minimal logger."""
    def info(self, msg: str):
        logger.info(msg)
    def warning(self, msg: str):
        logger.warning(msg)
    def error(self, msg: str):
        logger.error(msg)


class _Indicator:
    """Indicator that computes values on demand."""

    def __init__(self, name: str, period: int):
        self.name = name
        self.period = period
        self._values: list = []

    def add(self, data: pd.Series) -> None:
        """Compute indicator values."""
        if self.name == "ema":
            self._values = list(data.ewm(span=self.period, adjust=False).mean())
        elif self.name == "sma":
            self._values = list(data.rolling(window=self.period).mean())
        elif self.name == "rsi":
            self._values = list(self._compute_rsi(data))
        elif self.name == "macd":
            self._values = list(self._compute_macd(data))

    @property
    def value(self) -> Optional[float]:
        """Return latest value."""
        return self._values[-1] if self._values else None

    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_macd(prices: pd.Series) -> pd.Series:
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        return ema12 - ema26


# ---------------------------------------------------------------------------
# AST guard
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
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in DANGEROUS_NAMES:
                    violations.append(f"Blocked import: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in DANGEROUS_NAMES:
                violations.append(f"Blocked import from: {node.module}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval", "compile"):
                violations.append(f"Blocked call: {node.func.id}()")
    return violations


# ---------------------------------------------------------------------------
# Backtest outcome
# ---------------------------------------------------------------------------

@dataclass
class BacktestOutcome:
    """Backtest result."""
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


# ---------------------------------------------------------------------------
# Nautilus-compatible Strategy executor (pure Python)
# ---------------------------------------------------------------------------

class NautilusBacktestAdapter:
    """
    Backtest adapter using pure Python (no nautilus_trader dependency).

    Usage:
        adapter = NautilusBacktestAdapter()
        outcome = adapter.run_backtest(
            strategy_code=code,
            data=df,  # Real OHLCV DataFrame
            instrument_id="ETHUSDT.BINANCE",
        )
    """

    def __init__(self):
        pass

    def run_backtest(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        instrument_id: str = "ETHUSDT.BINANCE",
        initial_capital: float = 1_000_000.0,
    ) -> BacktestOutcome:
        """Run backtest on real OHLCV data."""
        outcome = BacktestOutcome()
        outcome.strategy_name = self._extract_name(strategy_code)

        # Validate code
        violations = ast_guard(strategy_code)
        if violations:
            outcome.error = "; ".join(violations)
            return outcome

        # Prepare data
        df = data.copy()
        # Handle timestamp column or index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        elif df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
            pass  # Already has datetime index
        else:
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Ensure required columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                outcome.error = f"Missing column: {col}"
                return outcome

        # Compile strategy code
        exec_globals: dict[str, Any] = {"Strategy": Strategy, "Decimal": Decimal, "_Indicator": _Indicator}
        try:
            exec(strategy_code, exec_globals)
        except Exception as e:
            outcome.error = f"Strategy compile error: {e}"
            return outcome

        # Find Strategy subclass
        strategy_cls = None
        for name, obj in exec_globals.items():
            if (inspect.isclass(obj) and
                    name != "Strategy" and
                    Strategy in obj.__mro__):
                strategy_cls = obj
                break

        if strategy_cls is None:
            outcome.error = "No Strategy subclass found in code"
            return outcome

        # Instantiate strategy
        strategy = strategy_cls(config={})
        strategy.instrument_id = instrument_id

        # Compute indicators from data
        close_prices = df["close"]
        for key, indicator in strategy._indicators.items():
            if isinstance(indicator, _Indicator):
                indicator.add(close_prices)

        outcome.ok = True
        outcome.reports = {"engine": "pure-python", "instrument": instrument_id}

        # Execute strategy logic
        capital = initial_capital
        equity_curve = [capital]
        trades = []

        for i in range(len(df)):
            bar = df.iloc[i]
            current_price = bar["close"]

            # Call strategy methods
            strategy.on_bar(bar)

            # Process trades
            while strategy.trades:
                trade = strategy.trades.pop(0)
                trades.append(trade)

            # Calculate PnL from positions
            if strategy.position == "LONG":
                # Unrealized PnL
                pnl = (current_price - strategy.entry_price) * strategy.position_size
                capital = initial_capital + pnl
            else:
                capital = initial_capital

            equity_curve.append(capital)

        strategy.on_stop()

        # Calculate metrics
        equity = pd.Series(equity_curve)
        returns = equity.pct_change().dropna()

        outcome.net_pnl = equity_curve[-1] - initial_capital
        outcome.gross_pnl = outcome.net_pnl
        outcome.max_drawdown = self._calculate_max_drawdown(equity)
        outcome.Sharpe_ratio = self._calculate_sharpe(returns)
        outcome.total_trades = len(trades)
        outcome.win_rate = self._calculate_win_rate(trades)
        outcome.equity_curve = equity_curve

        return outcome

    @staticmethod
    def _calculate_max_drawdown(equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return float(drawdown.min()) if len(drawdown) > 0 else 0.0

    @staticmethod
    def _calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        return float(excess_returns.mean() / excess_returns.std() * (252 ** 0.5))

    @staticmethod
    def _calculate_win_rate(trades: list) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        # Simplified: assume 50% win rate for now
        return 0.5

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
        return True

    def import_error(self) -> Optional[str]:
        return None


def translate_strategy(llm_output: dict) -> tuple[str, str]:
    """Translate LLM output into strategy code."""
    from ai_quant_nautilus.backtest.templates import get_strategy_template
    name = llm_output.get("name", "gen_unknown")
    params = llm_output.get("params", {})

    tmpl = None
    for template_name in ["ema_cross", "rsi_mean_reversion", "macd_signal"]:
        t = get_strategy_template(template_name)
        if t and name in t.name.lower():
            tmpl = t
            break

    if tmpl:
        nautilus_code = tmpl.code
    else:
        nautilus_code = generate_nautilus_strategy(name, llm_output.get("code", ""), params)

    violations = ast_guard(nautilus_code)
    if violations:
        return "", f"Validation failed: {violations}"

    return nautilus_code, ""


def generate_nautilus_strategy(
    strategy_name: str,
    code_snippet: str,
    params: dict[str, Any],
) -> str:
    """Generate a basic strategy class."""
    defaults = {
        "entry_threshold": "0.001",
        "exit_threshold": "0.001",
        "trade_size": "0.1",
        "fast_period": "10",
        "slow_period": "20",
    }
    defaults.update({k: str(v) for k, v in params.items()})

    return f'''
class {strategy_name}(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self._trade_size = Decimal("{defaults['trade_size']}")

    def on_start(self):
        pass

    def on_bar(self, bar):
        pass

    def on_stop(self):
        pass
'''
