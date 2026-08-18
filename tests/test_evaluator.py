import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
from ai_quant_nautilus.evaluator.gates import GateEvaluator, GateResult, EvalResult
from ai_quant_nautilus.backtest.nautilus_adapter import BacktestOutcome


class TestGateEvaluator:
    def test_evaluator_init(self):
        evaluator = GateEvaluator()
        assert evaluator.min_sharpe == 0.5
        assert evaluator.max_drawdown == 0.20

    def test_passing_backtest(self):
        evaluator = GateEvaluator()
        outcome = BacktestOutcome(
            ok=True,
            strategy_name="test",
            Sharpe_ratio=1.5,
            max_drawdown=-0.10,
            win_rate=0.55,
            total_trades=50,
        )
        result = evaluator.evaluate(outcome)
        # All gates should pass with good metrics
        assert all(g.passed for g in result.gates if g.name != "Nautilus Available")

    def test_mock_result_fails_nautilus_gate(self):
        evaluator = GateEvaluator()
        outcome = BacktestOutcome(
            ok=True,
            strategy_name="test",
            error="nautilus_trader not installed — mock result",
        )
        result = evaluator.evaluate(outcome)
        assert not result.passed
        assert any("Nautilus Available" in str(g) for g in result.gates)

    def test_gate_results_format(self):
        gate = GateResult(name="Sharpe", passed=True, actual=1.5, threshold=0.5)
        assert "PASS" in str(gate)
        assert "1.500" in str(gate)
        assert "0.5" in str(gate)
