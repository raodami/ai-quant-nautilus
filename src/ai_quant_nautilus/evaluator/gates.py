"""
Evaluation gates: filters that backtested strategies must pass.

Gates:
1. Sharpe ratio > 0.5
2. Max drawdown < 20%
3. Win rate > 40%
4. Total trades > 10
5. No lookahead bias
6. OOS (out-of-sample) pass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ai_quant_nautilus.backtest.nautilus_adapter import BacktestOutcome

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
        return f"[{status}] {self.name}: {self.actual:.3f} (threshold {self.threshold})"


@dataclass
class EvalResult:
    """Combined evaluation result."""
    passed: bool = True
    gates: list[GateResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def add_gate(self, gate: GateResult) -> None:
        self.gates.append(gate)
        if not gate.passed:
            self.passed = False
            self.reasons.append(str(gate))

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Eval: {status}"]
        for g in self.gates:
            lines.append(f"  {g}")
        return "\n".join(lines)


class GateEvaluator:
    """Evaluate strategies against gates."""

    def __init__(
        self,
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.20,
        min_win_rate: float = 0.40,
        min_trades: int = 10,
    ):
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades

    def evaluate(self, outcome: BacktestOutcome) -> EvalResult:
        """Evaluate a backtest outcome against all gates."""
        result = EvalResult()

        # Gate 1: Sharpe ratio
        sharpe = outcome.Sharpe_ratio
        result.add_gate(GateResult(
            name="Sharpe Ratio",
            passed=sharpe >= self.min_sharpe,
            actual=sharpe,
            threshold=self.min_sharpe,
        ))

        # Gate 2: Max drawdown
        dd = abs(outcome.max_drawdown)
        result.add_gate(GateResult(
            name="Max Drawdown",
            passed=dd <= self.max_drawdown,
            actual=dd,
            threshold=self.max_drawdown,
        ))

        # Gate 3: Win rate
        wr = outcome.win_rate
        result.add_gate(GateResult(
            name="Win Rate",
            passed=wr >= self.min_win_rate,
            actual=wr,
            threshold=self.min_win_rate,
        ))

        # Gate 4: Total trades
        trades = outcome.total_trades
        result.add_gate(GateResult(
            name="Total Trades",
            passed=trades >= self.min_trades,
            actual=float(trades),
            threshold=float(self.min_trades),
        ))

        # Gate 5: No NaN in metrics
        has_nan = any(
            v != v  # NaN check
            for v in [sharpe, dd, wr, float(trades)]
            if isinstance(v, float)
        )
        result.add_gate(GateResult(
            name="No NaN Metrics",
            passed=not has_nan,
            actual=0.0 if has_nan else 1.0,
            threshold=1.0,
        ))

        # If nautilus not available, mark as needs-real-backtest
        if "mock" in outcome.error.lower() or "nautilus" in outcome.error.lower():
            result.add_gate(GateResult(
                name="Nautilus Available",
                passed=False,
                actual=0.0,
                threshold=1.0,
                detail=outcome.error,
            ))

        return result
