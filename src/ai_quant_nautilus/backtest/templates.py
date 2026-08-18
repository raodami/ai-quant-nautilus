"""
Strategy templates for NautilusTrader integration.

Each template returns a complete Strategy subclass that can be
executed by the Nautilus backtest engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy Templates
# ---------------------------------------------------------------------------

STRATEGY_TEMPLATES = {
    "ema_cross": "EMA Cross Trend Following",
    "rsi_mean_reversion": "RSI Mean Reversion",
    "macd_signal": "MACD Signal Crossover",
    "bollinger_breakout": "Bollinger Band Breakout",
    "golden_cross": "Golden Cross (50/200 MA)",
    "super_trend": "SuperTrend Breakout",
}


@dataclass
class StrategyTemplate:
    """Strategy template metadata."""
    name: str
    description: str
    params: dict[str, Any]
    code: str


def get_strategy_template(template_name: str) -> Optional[StrategyTemplate]:
    """Get a strategy template by name."""
    template_fn = STRATEGY_FACTORIES.get(template_name)
    if template_fn is None:
        return None
    return template_fn()


# ---------------------------------------------------------------------------
# Template Factories
# ---------------------------------------------------------------------------

STRATEGY_FACTORIES: dict[str, Any] = {}


def register_template(name: str):
    """Decorator to register a strategy template factory."""
    def decorator(fn):
        STRATEGY_FACTORIES[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Base template code
# ---------------------------------------------------------------------------

BASE_CLASS_TEMPLATE = '''
from decimal import Decimal
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity


class {name}(Strategy):
    """{description}"""

    def __init__(self, config):
        super().__init__(config)
        self._trade_size = Decimal("{trade_size}")
        self._initialized = False

    def on_start(self):
        self._initialize()
        self.log.info("{name} started on {instrument}")

    def on_bar(self, bar):
        self._on_bar(bar)

    def on_stop(self):
        self.log.info("{name} stopped")

    def _initialize(self):
        """Override in subclasses."""
        self._initialized = True

    def _on_bar(self, bar):
        """Override in subclasses."""
        pass
'''


def _make_template(
    name: str,
    description: str,
    params: dict[str, Any],
) -> StrategyTemplate:
    full_code = f'''
from decimal import Decimal
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity


class {name}(Strategy):
    """{description}"""

    def __init__(self, config):
        super().__init__(config)
        self._fast_period = {params.get("fast_period", 10)}
        self._slow_period = {params.get("slow_period", 20)}
        self._entry_threshold = {params.get("entry_threshold", "0.001")}
        self._exit_threshold = {params.get("exit_threshold", "0.001")}
        self._trade_size = Decimal("{params.get("trade_size", "0.1")}")
        self._fast_ma = None
        self._slow_ma = None
        self._initialized = False

    def on_start(self):
        self._fast_ma = self.cache.indicator("ema", period=self._fast_period)
        self._slow_ma = self.cache.indicator("ema", period=self._slow_period)
        self._initialized = True
        self.log.info("{name} initialized with fast={{_fast_period}}, slow={{_slow_period}}")

    def on_bar(self, bar):
        if not self._initialized:
            return

        fast = self._fast_ma.value
        slow = self._slow_ma.value
        if fast is None or slow is None:
            return

        position = self.cache.position(self.instrument_id)
        side = self.cache.order_side(position) if position else None

        if side is None or side == OrderSide.NO_ORDER_SIDE:
            # Long entry
            if fast > slow + Decimal(str(self._entry_threshold)):
                self.order_market(
                    self.instrument_id,
                    OrderSide.BUY,
                    self._trade_size,
                    TimeInForce.FOK,
                )
        else:
            # Long exit
            if fast < slow - Decimal(str(self._exit_threshold)):
                self.order_market(
                    self.instrument_id,
                    OrderSide.SELL,
                    self._trade_size,
                    TimeInForce.FOK,
                )

    def on_stop(self):
        self.log.info("{name} stopped")
'''
    return StrategyTemplate(
        name=name,
        description=description,
        params=params,
        code=full_code,
    )


@register_template("ema_cross")
def _ema_cross_template() -> StrategyTemplate:
    return _make_template(
        name="EMACrossStrategy",
        description="EMA Cross Trend Following: buy when fast MA crosses above slow MA, sell on cross below",
        params={
            "fast_period": 10,
            "slow_period": 20,
            "entry_threshold": 0.001,
            "exit_threshold": 0.001,
            "trade_size": "0.1",
        },
    )


@register_template("rsi_mean_reversion")
def _rsi_template() -> StrategyTemplate:
    return _make_template(
        name="RSIMeanReversionStrategy",
        description="RSI Mean Reversion: buy when RSI < oversold, sell when RSI > overbought",
        params={
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "trade_size": "0.1",
        },
    )


@register_template("macd_signal")
def _macd_template() -> StrategyTemplate:
    return _make_template(
        name="MACDSignalStrategy",
        description="MACD Signal Crossover: buy on bullish crossover, sell on bearish crossover",
        params={
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "trade_size": "0.1",
        },
    )


@register_template("bollinger_breakout")
def _bollinger_template() -> StrategyTemplate:
    return _make_template(
        name="BollingerBreakoutStrategy",
        description="Bollinger Band Breakout: buy when price breaks upper band, sell on lower band",
        params={
            "bb_period": 20,
            "bb_std": 2.0,
            "trade_size": "0.1",
        },
    )


@register_template("golden_cross")
def _golden_cross_template() -> StrategyTemplate:
    return _make_template(
        name="GoldenCrossStrategy",
        description="Golden Cross: buy when 50 MA crosses above 200 MA, sell on death cross",
        params={
            "fast_period": 50,
            "slow_period": 200,
            "entry_threshold": 0.0005,
            "exit_threshold": 0.0005,
            "trade_size": "0.1",
        },
    )


@register_template("super_trend")
def _super_trend_template() -> StrategyTemplate:
    return _make_template(
        name="SuperTrendStrategy",
        description="SuperTrend Breakout: trend-following using ATR-based channels",
        params={
            "atr_period": 10,
            "multiplier": 3.0,
            "trade_size": "0.1",
        },
    )


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

__all__ = [
    "STRATEGY_TEMPLATES",
    "StrategyTemplate",
    "get_strategy_template",
]
