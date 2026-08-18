"""
Risk allocation: risk parity capital distribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyAllocation:
    """Single strategy capital allocation."""
    strategy_id: str
    allocated_capital: float
    weight: float  #占总资金权重
    volatility: float  # 策略波动率


def risk_parity_weights(volatilities: list[float]) -> list[float]:
    """
    Risk parity allocation: inverse variance weighting.

    weights ∝ 1 / volatility²
    """
    if not volatilities or all(v <= 0 for v in volatilities):
        n = len(volatilities) or 1
        return [1.0 / n] * n

    inv_var = [1.0 / (v ** 2) if v > 0 else 0 for v in volatilities]
    total = sum(inv_var)

    if total == 0:
        n = len(volatilities)
        return [1.0 / n] * n

    weights = [iv / total for iv in inv_var]
    w_sum = sum(weights)
    return [w / w_sum for w in weights] if w_sum > 0 else [1.0 / len(weights)] * len(weights)


def allocate_capital(
    total_capital: float,
    strategy_volatilities: dict[str, float],
) -> dict[str, float]:
    """
    Allocate capital across strategies using risk parity.

    Args:
        total_capital: Total capital to allocate
        strategy_volatilities: {strategy_id: annualized_volatility}

    Returns:
        {strategy_id: allocated_capital}
    """
    if not strategy_volatilities:
        return {}

    vols = list(strategy_volatilities.values())
    ids = list(strategy_volatilities.keys())
    weights = risk_parity_weights(vols)

    return {
        sid: total_capital * w
        for sid, w in zip(ids, weights)
    }
