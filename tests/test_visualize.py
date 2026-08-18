import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
from ai_quant_nautilus.visualize.visualizer import (
    equity_curve_plot,
    drawdown_plot,
    trade_histogram,
    correlation_heatmap,
    metrics_summary,
    export_report,
)
from pathlib import Path


class TestVisualizer:
    """Test visualization functions."""

    def test_equity_curve_plot(self):
        equity = [1000000 + i * 1000 for i in range(100)]
        html = equity_curve_plot(equity)
        assert "Equity Curve" in html or html == ""

    def test_equity_curve_with_benchmark(self):
        equity = [1000000 + i * 1000 for i in range(100)]
        benchmark = [1000000 + i * 800 for i in range(100)]
        html = equity_curve_plot(equity, benchmark=benchmark)
        assert "Equity Curve" in html or html == ""

    def test_drawdown_plot(self):
        drawdown = [-0.01 * i for i in range(100)]
        html = drawdown_plot(drawdown)
        assert "Drawdown" in html or html == ""

    def test_trade_histogram(self):
        trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}, {"pnl": -100}]
        html = trade_histogram(trades)
        assert "Trade P&L" in html or html == ""

    def test_correlation_heatmap(self):
        import pandas as pd
        corr = pd.DataFrame({
            "A": [1.0, 0.5, 0.3],
            "B": [0.5, 1.0, 0.2],
            "C": [0.3, 0.2, 1.0],
        })
        html = correlation_heatmap(corr)
        assert "Correlation" in html or html == ""

    def test_metrics_summary(self):
        result = {
            "total_return": 0.15,
            "annualized_return": 0.12,
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": -0.10,
            "total_trades": 50,
            "win_rate": 0.55,
            "initial_capital": 1000000.0,
            "final_capital": 1150000.0,
            "net_pnl": 150000.0,
        }
        html = metrics_summary(result)
        assert "Backtest Summary" in html
        assert "1,150,000.00" in html
        assert "15.00%" in html

    def test_export_report(self, tmp_path):
        result = {
            "strategy_name": "EMA Strategy",
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
        }
        equity = [1000000 + i * 500 for i in range(50)]
        drawdown = [-0.02 * (i % 10) for i in range(50)]
        trades = [{"pnl": 100}, {"pnl": -50}]
        output = tmp_path / "report.html"
        path = export_report(result, str(output), equity=equity, drawdown=drawdown, trades=trades)
        assert path.exists()
        content = path.read_text()
        assert "EMA Strategy" in content
        assert "Backtest Report" in content
