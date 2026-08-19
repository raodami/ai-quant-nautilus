"""Additional strategy templates for ai-quant-nautilus."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Additional Strategy Templates
# ---------------------------------------------------------------------------

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


STRATEGY_FACTORIES: dict[str, Any] = {}


def register_template(name: str):
    """Decorator to register a strategy template factory."""
    def decorator(fn):
        STRATEGY_FACTORIES[name] = fn
        return fn
    return decorator


def _make_template(
    name: str,
    description: str,
    params: dict[str, Any],
) -> StrategyTemplate:
    full_code = f'''
class {name}(Strategy):
    """{description}"""

    def __init__(self, config):
        super().__init__(config)
        self._trade_size = Decimal("{params.get('trade_size', '0.1')}")
        self._initialized = False

    def on_start(self):
        self._initialized = True
        self.log.info("{name} initialized")

    def on_bar(self, bar):
        if not self._initialized:
            return

        # TODO: implement trading logic
        pass

    def on_stop(self):
        self.log.info("{name} stopped")
'''
    return StrategyTemplate(
        name=name,
        description=description,
        params=params,
        code=full_code,
    )


@register_template("volume_breakout")
def _volume_breakout_template() -> StrategyTemplate:
    return _make_template(
        name="VolumeBreakoutStrategy",
        description="Volume Breakout: buy when volume spikes above average, sell on reversal",
        params={
            "volume_period": 20,
            "volume_threshold": 1.5,
            "trade_size": "0.1",
        },
    )


@register_template("mean_reversion")
def _mean_reversion_template() -> StrategyTemplate:
    return _make_template(
        name="MeanReversionStrategy",
        description="Mean Reversion: buy when price drops below moving average, sell when it rises above",
        params={
            "ma_period": 50,
            "entry_threshold": 0.02,
            "exit_threshold": 0.01,
            "trade_size": "0.1",
        },
    )


@register_template("momentum")
def _momentum_template() -> StrategyTemplate:
    return _make_template(
        name="MomentumStrategy",
        description="Momentum: buy when recent return is positive and accelerating, sell on reversal",
        params={
            "lookback": 10,
            "acceleration": 0.005,
            "trade_size": "0.1",
        },
    )


@register_template("grid_trading")
def _grid_trading_template() -> StrategyTemplate:
    return _make_template(
        name="GridTradingStrategy",
        description="Grid Trading: place buy/sell orders at fixed intervals around reference price",
        params={
            "grid_levels": 5,
            "grid_spacing_pct": 0.01,
            "trade_size": "0.05",
        },
    )


@register_template("_pairs_trading")
def _pairs_trading_template() -> str:
    return _make_template(
        name="PairsTradingStrategy",
        description="Pairs Trading: trade spread between two correlated assets",
        params={
            "lookback": 20,
            "entry_z": 2.0,
            "exit_z": 0.5,
            "trade_size": "0.1",
        },
    )


@register_template("volatility_breakout")
def _volatility_breakout_template() -> StrategyTemplate:
    return _make_template(
        name="VolatilityBreakoutStrategy",
        description="Volatility Breakout: enter on break of Bollinger Bands with high volatility",
        params={
            "bb_period": 20,
            "bb_std": 2.0,
            "atr_period": 14,
            "trade_size": "0.1",
        },
    )


@register_template("scalping")
def _scalping_template() -> StrategyTemplate:
    return _make_template(
        name="ScalpingStrategy",
        description="Scalping: quick trades capturing small price movements",
        params={
            "target_pnl": 0.001,
            "stop_pnl": -0.0005,
            "trade_size": "0.05",
        },
    )


STRATEGY_TEMPLATES = {
    "ema_cross": "EMA Cross Trend Following",
    "rsi_mean_reversion": "RSI Mean Reversion",
    "macd_signal": "MACD Signal Crossover",
    "bollinger_breakout": "Bollinger Band Breakout",
    "golden_cross": "Golden Cross (50/200 MA)",
    "super_trend": "SuperTrend Breakout",
    "volume_breakout": "Volume Breakout",
    "mean_reversion": "Mean Reversion",
    "momentum": "Momentum",
    "grid_trading": "Grid Trading",
    "pairs_trading": "Pairs Trading",
    "volatility_breakout": "Volatility Breakout",
    "scalping": "Scalping",
}


__all__ = [
    "STRATEGY_TEMPLATES",
    "StrategyTemplate",
    "get_strategy_template",
    "register_template",
]
