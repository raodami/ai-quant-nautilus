import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
from ai_quant_nautilus.backtest.nautilus_adapter import ast_guard, generate_nautilus_strategy, NautilusBacktestAdapter, BacktestOutcome


class TestAstGuard:
    def test_safe_code(self):
        code = "x = 1 + 2"
        assert ast_guard(code) == []

    def test_blocked_import_os(self):
        code = "import os"
        violations = ast_guard(code)
        assert any("os" in v for v in violations)

    def test_blocked_import_subprocess(self):
        code = "import subprocess"
        violations = ast_guard(code)
        assert any("subprocess" in v for v in violations)

    def test_blocked_exec_call(self):
        code = "exec('hello')"
        violations = ast_guard(code)
        assert any("exec" in v for v in violations)

    def test_syntax_error(self):
        code = "def broken("
        violations = ast_guard(code)
        assert len(violations) > 0
        assert "SyntaxError" in violations[0]

    def test_safe_nautilus_code(self):
        code = '''
from decimal import Decimal
from nautilus_trader.trading.strategy import Strategy
class MyStrat(Strategy):
    def __init__(self, config):
        super().__init__(config)
    def on_start(self):
        pass
    def on_bar(self, bar):
        pass
'''
        assert ast_guard(code) == []


class TestGenerateNautilusStrategy:
    def test_basic_generation(self):
        code = generate_nautilus_strategy(
            strategy_name="TestStrategy",
            code_snippet="pass",
            params={"fast_period": 10, "slow_period": 20},
        )
        assert "class TestStrategy" in code
        assert "Strategy" in code
        assert "on_start" in code
        assert "on_bar" in code

    def test_params_inserted(self):
        code = generate_nautilus_strategy(
            strategy_name="MyStrat",
            code_snippet="",
            params={"entry_threshold": 0.005},
        )
        assert "0.005" in code


class TestNautilusBacktestAdapter:
    def test_mock_backtest(self):
        adapter = NautilusBacktestAdapter()
        code = '''
from nautilus_trader.trading.strategy import Strategy
class TestStrat(Strategy):
    def on_start(self):
        pass
    def on_bar(self, bar):
        pass
'''
        outcome = adapter.run_backtest(strategy_code=code)
        assert outcome.ok
        assert outcome.strategy_name == "TestStrat"

    def test_mock_returns_error_when_unavailable(self):
        adapter = NautilusBacktestAdapter()
        code = "import os"
        outcome = adapter.run_backtest(strategy_code=code)
        assert not outcome.ok
        assert "Blocked import" in outcome.error


class TestEvalResult:
    def test_all_pass(self):
        from ai_quant_nautilus.evaluator.gates import GateResult, EvalResult
        result = EvalResult()
        result.add_gate(GateResult(name="Sharpe", passed=True, actual=1.5, threshold=0.5))
        assert result.passed

    def test_one_fail(self):
        from ai_quant_nautilus.evaluator.gates import GateResult, EvalResult
        result = EvalResult()
        result.add_gate(GateResult(name="Sharpe", passed=False, actual=0.3, threshold=0.5))
        assert not result.passed
        assert len(result.reasons) == 1
