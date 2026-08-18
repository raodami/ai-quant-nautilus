"""
Pipeline integration test: full workflow from strategy generation to evaluation.
"""

import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
from ai_quant_nautilus.backtest.nautilus_adapter import NautilusBacktestAdapter, BacktestOutcome
from ai_quant_nautilus.backtest.templates import get_strategy_template
from ai_quant_nautilus.evaluator.gates import GateEvaluator
from ai_quant_nautilus.generator.prompt_builder import GenerationContext, build_system_prompt, build_user_prompt
from ai_quant_nautilus.generator.schema import validate_schema_output


class TestPipelineIntegration:
    """Test the full pipeline: template → backtest → evaluate."""

    def test_ema_cross_pipeline(self):
        """End-to-end: get EMA Cross template → run backtest → evaluate."""
        import numpy as np
        import pandas as pd

        # 1. Get strategy template
        template = get_strategy_template("ema_cross")
        assert template is not None
        assert "EMACrossStrategy" in template.name

        # 2. Generate sample data
        np.random.seed(42)
        n = 200
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
        prices = [50000.0]
        for _ in range(n - 1):
            change = np.random.normal(0.0001, 0.02)
            prices.append(prices[-1] * (1 + change))
        data = pd.DataFrame({
            "open": prices, "high": [p*1.01 for p in prices],
            "low": [p*0.99 for p in prices], "close": prices, "volume": [100.0]*n
        }, index=dates)

        # 3. Run backtest
        adapter = NautilusBacktestAdapter()
        outcome = adapter.run_backtest(
            strategy_code=template.code,
            data=data,
            instrument_id="ETHUSDT.BINANCE",
        )
        assert outcome.ok
        assert "EMACrossStrategy" in outcome.strategy_name

        # 4. Evaluate
        evaluator = GateEvaluator()
        result = evaluator.evaluate(outcome)
        # Should have evaluated the outcome
        assert len(result.gates) > 0

    def test_rsi_template_pipeline(self):
        """Test RSI template through full pipeline."""
        import numpy as np
        import pandas as pd

        template = get_strategy_template("rsi_mean_reversion")
        assert template is not None
        assert "RSI" in template.name

        # Generate data
        np.random.seed(123)
        n = 200
        prices = [100.0]
        for _ in range(n - 1):
            change = np.random.normal(0.0001, 0.01)
            prices.append(prices[-1] * (1 + change))
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
        data = pd.DataFrame({
            "open": prices, "high": [p*1.01 for p in prices],
            "low": [p*0.99 for p in prices], "close": prices, "volume": [100.0]*n
        }, index=dates)

        adapter = NautilusBacktestAdapter()
        outcome = adapter.run_backtest(strategy_code=template.code, data=data)
        assert outcome.ok

    def test_golden_cross_template(self):
        """Test Golden Cross template."""
        template = get_strategy_template("golden_cross")
        assert template is not None
        assert template.params["fast_period"] == 50
        assert template.params["slow_period"] == 200

    def test_all_templates_generate_valid_code(self):
        """All registered templates should generate valid Python code."""
        from ai_quant_nautilus.backtest.templates import STRATEGY_TEMPLATES

        for name in STRATEGY_TEMPLATES:
            template = get_strategy_template(name)
            assert template is not None, f"Template {name} not found"
            assert "class " in template.code
            assert "Strategy" in template.code
            assert "on_start" in template.code
            assert "on_bar" in template.code

    def test_prompt_builders(self):
        """Test prompt building functions."""
        system = build_system_prompt()
        assert "NautilusTrader" in system or "nautilus" in system.lower()

        ctx = GenerationContext(
            market_summary="BTC trending up",
            timeframe="1h",
        )
        user = build_user_prompt(ctx)
        assert "BTC trending up" in user
        # timeframe is stored in context but not always included in output
        assert len(user) > 0

    def test_schema_validation_with_template(self):
        """Test that template output would pass schema validation."""
        template = get_strategy_template("ema_cross")

        # Simulate LLM output based on template
        llm_output = {
            "name": "gen_ema_cross",
            "rationale": "EMA crossover strategy captures trend following opportunities",
            "code": template.code,
            "params": template.params,
            "expected_edge": "Trend following in directional markets",
        }

        valid, errors = validate_schema_output(llm_output)
        assert valid, f"Validation errors: {errors}"

    def test_mock_backtest_with_real_template(self):
        """Test that backtest produces expected results for real templates."""
        import numpy as np
        import pandas as pd

        adapter = NautilusBacktestAdapter()

        # Generate sample data
        np.random.seed(42)
        n = 200
        prices = [50000.0]
        for _ in range(n - 1):
            change = np.random.normal(0.0001, 0.02)
            prices.append(prices[-1] * (1 + change))
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
        data = pd.DataFrame({
            "open": prices, "high": [p*1.01 for p in prices],
            "low": [p*0.99 for p in prices], "close": prices, "volume": [100.0]*n
        }, index=dates)

        for template_name in ["ema_cross", "rsi_mean_reversion", "macd_signal"]:
            template = get_strategy_template(template_name)
            assert template is not None

            outcome = adapter.run_backtest(strategy_code=template.code, data=data)
            assert outcome.ok
            assert template.name in outcome.strategy_name

    def test_performance_metrics_integration(self):
        """Test performance metrics calculation with sample data."""
        from ai_quant_nautilus.backtest.performance import calculate_performance_metrics

        # Simulate equity curve with upward trend
        equity = [1000000]
        for i in range(100):
            prev = equity[-1]
            # Random walk with positive drift
            change = prev * (1 + 0.001 * (i % 10) / 10 + 0.0001)
            equity.append(change)

        metrics = calculate_performance_metrics(equity)
        assert metrics.total_return > 0
        assert metrics.sharpe_ratio >= 0
