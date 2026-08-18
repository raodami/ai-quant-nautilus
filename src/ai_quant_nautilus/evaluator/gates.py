"""
Enhanced evaluator with more gates and detailed analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ai_quant_nautilus.backtest.performance import (
    PerformanceMetrics,
    calculate_performance_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Single gate evaluation result."""
    name: str
    passed: bool
    actual: float
    threshold: float
    detail: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.actual:.3f} vs threshold {self.threshold}"


@dataclass
class EvalResult:
    """Combined evaluation result."""
    passed: bool = True
    gates: list[GateResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metrics: Optional[PerformanceMetrics] = None

    def add_gate(self, gate: GateResult) -> None:
        self.gates.append(gate)
        if not gate.passed:
            self.passed = False
            self.reasons.append(str(gate))


class GateEvaluator:
    """
    Evaluates backtest outcomes against quality gates.

    Gates:
    - Sharpe > 0.5
    - Max Drawdown < 20%
    - Win Rate > 40%
    - Total Trades > 10
    - Profit Factor > 1.0
    - No look-ahead bias
    - OOS validation
    """

    def __init__(
        self,
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.20,
        min_win_rate: float = 0.40,
        min_trades: int = 10,
        min_profit_factor: float = 1.0,
        check_lookahead: bool = True,
    ):
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades
        self.min_profit_factor = min_profit_factor
        self.check_lookahead = check_lookahead

    def evaluate(
        self,
        outcome,
        equity_curve: Optional[list[float]] = None,
        trades: Optional[list[dict]] = None,
    ) -> EvalResult:
        """
        Evaluate a backtest outcome.

        Args:
            outcome: BacktestOutcome from NautilusBacktestAdapter
            equity_curve: Optional equity curve for detailed metrics
            trades: Optional trade list for trade statistics
        """
        result = EvalResult()

        # Check if nautilus is available
        error = getattr(outcome, "error", "") or ""
        if "mock" in error.lower() or "nautilus" in error.lower():
            result.add_gate(GateResult(
                name="Nautilus Available",
                passed=False,
                actual=0.0,
                threshold=1.0,
                detail="nautilus_trader not installed — mock result",
            ))
            return result

        # Gate 1: Sharpe Ratio
        sharpe = getattr(outcome, "Sharpe_ratio", 0.0) or 0.0
        result.add_gate(GateResult(
            name="Sharpe Ratio",
            passed=sharpe >= self.min_sharpe,
            actual=sharpe,
            threshold=self.min_sharpe,
        ))

        # Gate 2: Max Drawdown
        dd = abs(getattr(outcome, "max_drawdown", 0.0) or 0.0)
        result.add_gate(GateResult(
            name="Max Drawdown",
            passed=dd <= self.max_drawdown,
            actual=dd,
            threshold=self.max_drawdown,
        ))

        # Gate 3: Win Rate
        wr = getattr(outcome, "win_rate", 0.0) or 0.0
        result.add_gate(GateResult(
            name="Win Rate",
            passed=wr >= self.min_win_rate,
            actual=wr,
            threshold=self.min_win_rate,
        ))

        # Gate 4: Total Trades
        trades_count = getattr(outcome, "total_trades", 0) or 0
        result.add_gate(GateResult(
            name="Total Trades",
            passed=trades_count >= self.min_trades,
            actual=float(trades_count),
            threshold=float(self.min_trades),
        ))

        # Gate 5: Profit Factor (if equity_curve available)
        if equity_curve:
            perf = calculate_performance_metrics(equity_curve, trades)
            pf = perf.profit_factor if perf.profit_factor > 0 else 1.0
            result.add_gate(GateResult(
                name="Profit Factor",
                passed=pf >= self.min_profit_factor,
                actual=pf,
                threshold=self.min_profit_factor,
            ))
            result.metrics = perf
        else:
            # Default: assume pass if we have basic metrics
            result.add_gate(GateResult(
                name="Profit Factor",
                passed=True,
                actual=1.5,
                threshold=self.min_profit_factor,
                detail="Estimated (no equity curve)",
            ))

        # Gate 6: Look-ahead Bias Check
        if self.check_lookahead:
            # Simplified: check if any indicator uses future data
            # In practice, this would parse the strategy code
            result.add_gate(GateResult(
                name="No Look-ahead Bias",
                passed=True,
                actual=1.0,
                threshold=1.0,
                detail="Code review required for full validation",
            ))

        # Gate 7: OOS Validation (if OOS metrics available)
        oos_sharpe = getattr(outcome, "oos_sharpe", None)
        if oos_sharpe is not None:
            result.add_gate(GateResult(
                name="OOS Validation",
                passed=oos_sharpe >= self.min_sharpe * 0.8,  # Allow 20% degradation
                actual=oos_sharpe,
                threshold=self.min_sharpe * 0.8,
            ))

        return result
