import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
from ai_quant_nautilus.evaluator.gates import GateEvaluator, GateResult, EvalResult
from ai_quant_nautilus.backtest.nautilus_adapter import BacktestResult


class TestGateEvaluator:
    def test_evaluator_init(self):
        evaluator = GateEvaluator()
        assert evaluator.min_sharpe == 0.5
        assert evaluator.max_drawdown == 0.20

    def test_passing_backtest(self):
        evaluator = GateEvaluator()
        result = BacktestResult(
            ok=True,
            strategy_name="test",
            sharpe_ratio=1.5,
            max_drawdown_pct=-0.10,
            win_rate=0.55,
            total_trades=50,
        )
        outcome = evaluator.evaluate(result)
        # Check that evaluation ran and has gates
        assert len(outcome.gates) > 0
        # At least some gates should pass with good metrics
        passed_gates = [g for g in outcome.gates if g.passed]
        assert len(passed_gates) > 0

    def test_mock_result_fails_nautilus_gate(self):
        evaluator = GateEvaluator()
        result = BacktestResult(
            ok=True,
            strategy_name="test",
            error="nautilus_trader not installed — mock result",
        )
        outcome = evaluator.evaluate(result)
        assert not outcome.passed
        assert any("Nautilus Available" in str(g) for g in outcome.gates)

    def test_gate_results_format(self):
        gate = GateResult(name="Sharpe", passed=True, actual=1.5, threshold=0.5)
        assert "PASS" in str(gate)
        assert "1.500" in str(gate)
        assert "0.5" in str(gate)

    def test_multiple_gates(self):
        evaluator = GateEvaluator()
        result = BacktestResult(
            ok=True,
            strategy_name="test",
            sharpe_ratio=0.3,  # Below threshold
            max_drawdown_pct=-0.25,  # Above threshold (bad)
            win_rate=0.35,  # Below threshold
            total_trades=5,  # Below threshold
        )
        outcome = evaluator.evaluate(result)
        assert not outcome.passed
        assert len(outcome.gates) >= 4
