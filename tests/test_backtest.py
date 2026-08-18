import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
import numpy as np
import pandas as pd
from ai_quant_nautilus.backtest.nautilus_adapter import (
    BacktestEngine, BacktestConfig, ast_guard, generate_nautilus_strategy, 
    Strategy, OrderSide, PositionDirection, translate_strategy
)
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


class TestBacktestEngine:
    """Test production-grade backtest engine with real OHLCV data."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Generate realistic OHLCV data."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")

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

    def test_engine_init(self):
        engine = BacktestEngine()
        assert engine is not None

    def test_engine_with_custom_config(self):
        config = BacktestConfig(
            initial_capital=500000.0,
            commission_rate=0.0015,
            slippage_pct=0.001,
        )
        engine = BacktestEngine(config=config)
        assert engine.config.initial_capital == 500000.0
        assert engine.config.commission_rate == 0.0015

    def test_backtest_with_ema_cross(self, sample_ohlcv):
        """Test EMA Cross strategy on real data."""
        engine = BacktestEngine()
        template = get_strategy_template("ema_cross")
        assert template is not None

        result = engine.run(
            strategy_code=template.code,
            data=sample_ohlcv,
            instrument_id="BTCUSDT.BINANCE",
            initial_capital=1000000.0,
        )

        assert result.ok
        assert result.strategy_name == "EMACrossStrategy"
        assert len(result.equity_curve) > 0
        assert result.total_trades >= 0
        assert result.initial_capital == 1000000.0

    def test_backtest_with_rsi_strategy(self, sample_ohlcv):
        """Test RSI strategy on real data."""
        engine = BacktestEngine()
        template = get_strategy_template("rsi_mean_reversion")
        assert template is not None

        result = engine.run(
            strategy_code=template.code,
            data=sample_ohlcv,
            instrument_id="ETHUSDT.BINANCE",
        )

        assert result.ok
        assert result.strategy_name == "RSIMeanReversionStrategy"

    def test_backtest_with_macd_strategy(self, sample_ohlcv):
        """Test MACD strategy on real data."""
        engine = BacktestEngine()
        template = get_strategy_template("macd_signal")
        assert template is not None

        result = engine.run(
            strategy_code=template.code,
            data=sample_ohlcv,
            instrument_id="ETHUSDT.BINANCE",
        )

        assert result.ok

    def test_backtest_fails_on_bad_data(self, sample_ohlcv):
        """Test engine fails gracefully on invalid data."""
        engine = BacktestEngine()
        bad_df = sample_ohlcv.drop(columns=["close"])
        result = engine.run(strategy_code="pass", data=bad_df)
        assert not result.ok
        assert "close" in result.error

    def test_backtest_fails_on_bad_strategy(self, sample_ohlcv):
        """Test engine fails gracefully on invalid strategy."""
        engine = BacktestEngine()
        result = engine.run(strategy_code="invalid python code", data=sample_ohlcv)
        assert not result.ok
        assert "compile error" in result.error.lower() or "SyntaxError" in result.error

    def test_backtest_with_custom_capital(self, sample_ohlcv):
        """Test backtest with custom initial capital."""
        engine = BacktestEngine()
        template = get_strategy_template("ema_cross")
        
        result = engine.run(
            strategy_code=template.code,
            data=sample_ohlcv,
            initial_capital=250000.0,
        )
        
        assert result.ok
        assert result.initial_capital == 250000.0

    def test_backtest_returns_comprehensive_metrics(self, sample_ohlcv):
        """Test that results include all expected metrics."""
        engine = BacktestEngine()
        template = get_strategy_template("ema_cross")

        result = engine.run(
            strategy_code=template.code,
            data=sample_ohlcv,
        )

        assert result.ok
        # Capital metrics
        assert result.initial_capital > 0
        assert result.final_capital > 0
        assert isinstance(result.net_pnl, float)
        assert isinstance(result.gross_pnl, (float, int))
        
        # Risk metrics
        assert result.max_drawdown <= 0  # Should be negative or zero
        assert result.sharpe_ratio >= 0
        assert result.volatility >= 0
        
        # Trade metrics
        assert result.total_trades >= 0
        assert 0 <= result.win_rate <= 1
        
        # Equity curve
        assert len(result.equity_curve) > 0
        assert len(result.drawdown_curve) > 0

    def test_to_dict_format(self, sample_ohlcv):
        """Test result serialization."""
        engine = BacktestEngine()
        template = get_strategy_template("ema_cross")
        
        result = engine.run(
            strategy_code=template.code,
            data=sample_ohlcv,
        )
        
        d = result.to_dict()
        assert "ok" in d
        assert "strategy_name" in d
        assert "net_pnl" in d
        assert "sharpe_ratio" in d
        assert "total_trades" in d
        assert "win_rate" in d


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
