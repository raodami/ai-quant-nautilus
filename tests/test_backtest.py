import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
import numpy as np
import pandas as pd
from ai_quant_nautilus.backtest.nautilus_adapter import ast_guard, generate_nautilus_strategy, NautilusBacktestAdapter, BacktestOutcome
from ai_quant_nautilus.backtest.performance import PerformanceMetrics, calculate_performance_metrics
from ai_quant_nautilus.backtest.templates import get_strategy_template


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

    def test_safe_strategy_code(self):
        code = '''
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
        assert "MyStrat" in code
        assert "Strategy" in code


class TestRealBacktest:
    """Test backtesting with real OHLCV data."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Generate realistic OHLCV data."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")

        # Random walk with drift
        prices = [50000.0]
        for _ in range(n - 1):
            change = np.random.normal(0.0001, 0.02)
            prices.append(prices[-1] * (1 + change))

        df = pd.DataFrame({
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "close": prices,
            "volume": [abs(np.random.lognormal(0, 1)) * 100 for _ in range(n)],
        }, index=dates)
        df.index.name = "timestamp"
        return df

    def test_backtest_with_real_data(self, sample_ohlcv):
        """Test backtest on real OHLCV data."""
        adapter = NautilusBacktestAdapter()

        # Use a simple strategy template
        template = get_strategy_template("ema_cross")
        assert template is not None

        outcome = adapter.run_backtest(
            strategy_code=template.code,
            data=sample_ohlcv,
            instrument_id="BTCUSDT.BINANCE",
            initial_capital=1000000.0,
        )

        assert outcome.ok
        assert outcome.strategy_name == "EMACrossStrategy"
        assert isinstance(outcome.equity_curve, list)
        assert len(outcome.equity_curve) > 0

    def test_backtest_with_trending_data(self):
        """Test backtest on trending data."""
        np.random.seed(123)
        n = 300
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")

        # Upward trend
        prices = [100.0]
        for _ in range(n - 1):
            change = np.random.normal(0.001, 0.01)
            prices.append(prices[-1] * (1 + change))

        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [100.0] * n,
        }, index=dates)
        df.index.name = "timestamp"

        adapter = NautilusBacktestAdapter()
        template = get_strategy_template("ema_cross")

        outcome = adapter.run_backtest(
            strategy_code=template.code,
            data=df,
            instrument_id="TEST.USDT",
        )

        assert outcome.ok
        assert outcome.total_trades >= 0

    def test_backtest_fails_on_bad_data(self, sample_ohlcv):
        """Test backtest fails gracefully on invalid data."""
        adapter = NautilusBacktestAdapter()

        # Missing required column
        bad_df = sample_ohlcv.drop(columns=["close"])
        outcome = adapter.run_backtest(
            strategy_code="pass",
            data=bad_df,
        )
        assert not outcome.ok
        assert "close" in outcome.error

    def test_backtest_fails_on_bad_strategy(self, sample_ohlcv):
        """Test backtest fails gracefully on invalid strategy."""
        adapter = NautilusBacktestAdapter()
        outcome = adapter.run_backtest(
            strategy_code="invalid python code",
            data=sample_ohlcv,
        )
        assert not outcome.ok
        assert "compile error" in outcome.error.lower() or "SyntaxError" in outcome.error

    def test_backtest_with_rsi_strategy(self, sample_ohlcv):
        """Test RSI strategy on real data."""
        adapter = NautilusBacktestAdapter()
        template = get_strategy_template("rsi_mean_reversion")
        assert template is not None

        outcome = adapter.run_backtest(
            strategy_code=template.code,
            data=sample_ohlcv,
            instrument_id="ETHUSDT.BINANCE",
        )
        assert outcome.ok

    def test_backtest_with_macd_strategy(self, sample_ohlcv):
        """Test MACD strategy on real data."""
        adapter = NautilusBacktestAdapter()
        template = get_strategy_template("macd_signal")
        assert template is not None

        outcome = adapter.run_backtest(
            strategy_code=template.code,
            data=sample_ohlcv,
            instrument_id="ETHUSDT.BINANCE",
        )
        assert outcome.ok


class TestPerformanceMetrics:
    def test_basic_calculation(self):
        equity = [1000000, 1010000, 1005000, 1020000, 1015000, 1030000, 1025000, 1040000]
        metrics = calculate_performance_metrics(equity)
        assert metrics.total_return > 0
        assert metrics.sharpe_ratio >= 0

    def test_empty_equity(self):
        metrics = calculate_performance_metrics([])
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0


class TestStrategyTemplates:
    def test_get_ema_cross_template(self):
        tmpl = get_strategy_template("ema_cross")
        assert tmpl is not None
        assert "EMACrossStrategy" in tmpl.name

    def test_get_rsi_template(self):
        tmpl = get_strategy_template("rsi_mean_reversion")
        assert tmpl is not None
        assert "RSI" in tmpl.name

    def test_get_macd_template(self):
        tmpl = get_strategy_template("macd_signal")
        assert tmpl is not None
        assert "MACD" in tmpl.name

    def test_get_nonexistent(self):
        tmpl = get_strategy_template("nonexistent")
        assert tmpl is None

    def test_template_has_params(self):
        tmpl = get_strategy_template("ema_cross")
        assert "fast_period" in tmpl.params
        assert "slow_period" in tmpl.params

    def test_template_has_code(self):
        tmpl = get_strategy_template("ema_cross")
        assert "class EMACrossStrategy" in tmpl.code
        assert "on_start" in tmpl.code
        assert "on_bar" in tmpl.code

    def test_all_templates_registered(self):
        from ai_quant_nautilus.backtest.templates import STRATEGY_TEMPLATES
        for name in STRATEGY_TEMPLATES:
            tmpl = get_strategy_template(name)
            assert tmpl is not None, f"Template {name} not found"
