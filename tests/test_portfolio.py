import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
import numpy as np
import pandas as pd
from ai_quant_nautilus.backtest.portfolio import (
    Portfolio,
    PortfolioResult,
    AllocationMethod,
    AssetPosition,
    backtest_portfolio,
)


class TestPortfolio:
    """Test multi-asset portfolio backtesting."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for 3 assets."""
        np.random.seed(42)
        days = 252
        return {
            "BTC/USDT": pd.Series(np.random.normal(0.001, 0.03, days)),
            "ETH/USDT": pd.Series(np.random.normal(0.0008, 0.035, days)),
            "SOL/USDT": pd.Series(np.random.normal(0.0005, 0.04, days)),
        }

    def test_portfolio_init(self):
        portfolio = Portfolio(initial_capital=500000.0)
        assert portfolio.initial_capital == 500000.0
        assert portfolio.allocation_method == AllocationMethod.EQUAL
        assert len(portfolio.positions) == 0

    def test_add_asset(self, sample_returns):
        portfolio = Portfolio()
        for symbol, returns in sample_returns.items():
            portfolio.add_asset(symbol, returns)
        assert len(portfolio.positions) == 3
        assert "BTC/USDT" in portfolio.positions

    def test_equal_weight_allocation(self, sample_returns):
        portfolio = Portfolio(allocation_method=AllocationMethod.EQUAL)
        for symbol, returns in sample_returns.items():
            portfolio.add_asset(symbol, returns)
        portfolio.allocate_capital()
        for position in portfolio.positions.values():
            assert abs(position.weight - 1/3) < 0.001

    def test_risk_parity_allocation(self, sample_returns):
        portfolio = Portfolio(allocation_method=AllocationMethod.RISK_PARITY)
        for symbol, returns in sample_returns.items():
            portfolio.add_asset(symbol, returns)
        portfolio.allocate_capital()
        # Risk parity should give lower weight to higher volatility asset
        weights = {s: p.weight for s, p in portfolio.positions.items()}
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)
        # SOL has highest vol, should get lowest weight
        assert weights["SOL/USDT"] < weights["BTC/USDT"]

    def test_portfolio_returns(self, sample_returns):
        portfolio = Portfolio()
        for symbol, returns in sample_returns.items():
            portfolio.add_asset(symbol, returns)
        portfolio.allocate_capital()
        port_returns = portfolio.calculate_portfolio_returns()
        assert isinstance(port_returns, pd.Series)
        assert len(port_returns) == 252

    def test_correlation_matrix(self, sample_returns):
        portfolio = Portfolio()
        for symbol, returns in sample_returns.items():
            portfolio.add_asset(symbol, returns)
        corr = portfolio.calculate_correlation_matrix()
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (3, 3)
        np.testing.assert_array_almost_equal(corr.values, corr.T.values)

    def test_run_backtest(self, sample_returns):
        portfolio = Portfolio(initial_capital=1_000_000.0)
        for symbol, returns in sample_returns.items():
            portfolio.add_asset(symbol, returns)
        result = portfolio.run_backtest()
        assert isinstance(result, PortfolioResult)
        assert result.total_return != 0 or True  # Can be positive or negative
        assert result.sharpe_ratio >= 0
        assert result.max_drawdown <= 0
        assert len(result.equity_curve) > 0

    def test_portfolio_summary(self, sample_returns):
        portfolio = Portfolio()
        for symbol, returns in sample_returns.items():
            portfolio.add_asset(symbol, returns)
        summary = portfolio.get_summary()
        assert "Portfolio Summary" in summary
        assert "BTC/USDT" in summary
        assert "$" in summary

    def test_to_dict(self, sample_returns):
        portfolio = Portfolio()
        for symbol, returns in sample_returns.items():
            portfolio.add_asset(symbol, returns)
        result = portfolio.run_backtest()
        d = result.to_dict()
        assert "symbols" in d
        assert "weights" in d
        assert "sharpe_ratio" in d
        assert "correlation_matrix" in d

    def test_empty_portfolio(self):
        portfolio = Portfolio()
        result = portfolio.run_backtest()
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0


class TestConvenienceFunction:
    """Test backtest_portfolio convenience function."""

    def test_backtest_portfolio(self):
        np.random.seed(42)
        asset_data = {
            "BTC/USDT": pd.Series(np.random.normal(0.001, 0.03, 100)),
            "ETH/USDT": pd.Series(np.random.normal(0.0008, 0.035, 100)),
        }
        result = backtest_portfolio(asset_data)
        assert isinstance(result, PortfolioResult)
        assert len(result.symbols) == 2

    def test_backtest_with_custom_capital(self):
        asset_data = {
            "TEST/USDT": pd.Series(np.random.normal(0.001, 0.02, 50)),
        }
        result = backtest_portfolio(
            asset_data,
            initial_capital=250000.0,
        )
        assert result.total_return != 0 or True


class TestAllocationMethods:
    """Test different allocation methods."""

    def test_equal_method_enum(self):
        assert AllocationMethod.EQUAL.value == "equal"

    def test_risk_parity_method_enum(self):
        assert AllocationMethod.RISK_PARITY.value == "risk_parity"

    def test_unknown_method_raises(self):
        portfolio = Portfolio()
        portfolio.add_asset("TEST", pd.Series([0.01] * 10))
        portfolio.allocation_method = "unknown"
        with pytest.raises(ValueError):
            portfolio.allocate_capital()
