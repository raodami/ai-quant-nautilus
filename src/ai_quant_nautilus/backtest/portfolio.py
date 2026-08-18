"""
Multi-asset portfolio backtesting module.

Supports:
- Concurrent backtesting of multiple trading pairs
- Capital allocation (equal weight / risk parity)
- Portfolio return and risk calculation
- Correlation matrix
- Portfolio risk metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Capital allocation method."""
    EQUAL = "equal"
    RISK_PARITY = "risk_parity"


@dataclass
class AssetPosition:
    """Single asset position in the portfolio."""
    symbol: str
    allocated_capital: float
    weight: float
    returns_series: Optional[pd.Series] = None  # daily returns
    pnl: float = 0.0


@dataclass
class PortfolioResult:
    """Backtest result for the multi-asset portfolio."""
    symbols: list[str]
    weights: dict[str, float]
    allocated_capital: dict[str, float]
    allocation_method: str
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    correlation_matrix: Optional[pd.DataFrame] = None
    cov_matrix: Optional[np.ndarray] = None
    equity_curve: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)
    individual_returns: dict[str, float] = field(default_factory=dict)
    position_data: dict[str, AssetPosition] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "symbols": self.symbols,
            "allocation_method": self.allocation_method,
            "weights": {k: round(v, 6) for k, v in self.weights.items()},
            "allocated_capital": {k: round(v, 2) for k, v in self.allocated_capital.items()},
            "total_return": round(self.total_return, 4),
            "annualized_return": round(self.annualized_return, 4),
            "volatility": round(self.volatility, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "individual_returns": {k: round(v, 4) for k, v in self.individual_returns.items()},
        }
        if self.correlation_matrix is not None:
            result["correlation_matrix"] = self.correlation_matrix.round(4).to_dict()
        return result


class Portfolio:
    """Multi-asset portfolio for concurrent backtesting."""

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        allocation_method: AllocationMethod = AllocationMethod.EQUAL,
        risk_free_rate: float = 0.0,
        trading_days_per_year: int = 365,
    ):
        self.initial_capital = initial_capital
        self.allocation_method = allocation_method
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.positions: dict[str, AssetPosition] = {}
        self._return_data: dict[str, pd.Series] = {}

    def add_asset(
        self,
        symbol: str,
        returns: pd.Series,
        allocated_capital: Optional[float] = None,
        weight: Optional[float] = None,
    ) -> None:
        """
        Add an asset to the portfolio.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            returns: Series of daily returns for the asset
            allocated_capital: Explicit capital to allocate (optional if weight given)
            weight: Explicit weight (0-1) (optional if allocated_capital given)
        """
        if symbol in self.positions:
            logger.warning(f"Asset {symbol} already exists, replacing.")

        position = AssetPosition(
            symbol=symbol,
            allocated_capital=allocated_capital or 0.0,
            weight=weight or 0.0,
            returns_series=returns.copy(),
        )
        self.positions[symbol] = position
        self._return_data[symbol] = returns

    def allocate_capital(self) -> None:
        """Allocate capital across assets based on allocation method."""
        if not self.positions:
            return

        symbols = list(self.positions.keys())
        n = len(symbols)

        if self.allocation_method == AllocationMethod.EQUAL:
            weights = [1.0 / n] * n
        elif self.allocation_method == AllocationMethod.RISK_PARITY:
            weights = self._risk_parity_weights()
        else:
            raise ValueError(f"Unknown allocation method: {self.allocation_method}")

        for i, symbol in enumerate(symbols):
            weight = weights[i]
            position = self.positions[symbol]
            position.weight = weight
            position.allocated_capital = self.initial_capital * weight

    def _risk_parity_weights(self) -> list[float]:
        """Calculate risk parity weights (inverse variance)."""
        vols = []
        for symbol in self.positions:
            rets = self._return_data[symbol].dropna()
            vol = rets.std() * np.sqrt(self.trading_days_per_year) if len(rets) > 1 else 0.0
            vols.append(vol)

        if not vols or all(v <= 0 for v in vols):
            n = len(vols) or 1
            return [1.0 / n] * n

        inv_var = [1.0 / (v ** 2) if v > 0 else 0 for v in vols]
        total = sum(inv_var)

        if total == 0:
            n = len(vols)
            return [1.0 / n] * n

        weights = [iv / total for iv in inv_var]
        w_sum = sum(weights)
        return [w / w_sum for w in weights] if w_sum > 0 else [1.0 / n] * n

    def calculate_portfolio_returns(self) -> pd.Series:
        """
        Calculate weighted portfolio returns.

        Returns:
            Series of portfolio daily returns
        """
        if not self.positions:
            return pd.Series(dtype=float)

        # Get aligned returns across all assets
        returns_df = pd.DataFrame({
            symbol: self._return_data[symbol]
            for symbol in self.positions
        })

        # Forward fill and drop NaN
        returns_df = returns_df.ffill().dropna()

        if returns_df.empty:
            return pd.Series(dtype=float)

        # Weighted returns
        weights = np.array([
            self.positions[symbol].weight
            for symbol in returns_df.columns
        ])

        portfolio_returns = (returns_df * weights).sum(axis=1)
        return portfolio_returns

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix between assets."""
        if not self._return_data:
            return pd.DataFrame()

        returns_df = pd.DataFrame(self._return_data)
        returns_df = returns_df.ffill().dropna()

        if returns_df.empty:
            return pd.DataFrame()

        corr = returns_df.corr()
        return corr

    def calculate_covariance_matrix(self) -> np.ndarray:
        """Calculate covariance matrix between assets."""
        if not self._return_data:
            return np.array([])

        returns_df = pd.DataFrame(self._return_data)
        returns_df = returns_df.ffill().dropna()

        if returns_df.empty:
            return np.array([])

        return returns_df.cov().values

    def run_backtest(self) -> PortfolioResult:
        """
        Run the full portfolio backtest.

        Returns:
            PortfolioResult with all metrics
        """
        self.allocate_capital()

        # Calculate portfolio returns
        port_returns = self.calculate_portfolio_returns()

        if port_returns.empty:
            return PortfolioResult(
                symbols=list(self.positions.keys()),
                weights={s: p.weight for s, p in self.positions.items()},
                allocated_capital={s: p.allocated_capital for s, p in self.positions.items()},
                allocation_method=self.allocation_method.value,
            )

        # Equity curve
        equity_curve = [self.initial_capital]
        for ret in port_returns:
            equity_curve.append(equity_curve[-1] * (1 + ret))

        equity = np.array(equity_curve)

        # Individual asset returns
        individual_returns = {}
        for symbol, position in self.positions.items():
            if position.returns_series is not None:
                rets = position.returns_series.dropna()
                if len(rets) > 0:
                    individual_returns[symbol] = float((1 + rets).prod() - 1)

        # Portfolio metrics
        total_return = float((equity[-1] / equity[0]) - 1)
        periods = len(equity) - 1

        # Annualized return
        if equity[0] > 0 and periods > 0:
            annualized_return = float((equity[-1] / equity[0]) ** (self.trading_days_per_year / periods) - 1)
        else:
            annualized_return = 0.0

        # Volatility
        daily_returns = port_returns.values
        if len(daily_returns) > 1:
            volatility = float(np.std(daily_returns) * np.sqrt(self.trading_days_per_year))
        else:
            volatility = 0.0

        # Sharpe ratio
        if volatility > 0:
            excess_returns = daily_returns - (self.risk_free_rate / self.trading_days_per_year)
            sharpe = float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days_per_year))
        else:
            sharpe = 0.0

        # Sortino ratio
        if len(daily_returns) > 1:
            downside = daily_returns[daily_returns < 0]
            if len(downside) > 0:
                downside_dev = np.std(downside) * np.sqrt(self.trading_days_per_year)
                if downside_dev > 0:
                    excess_mean = np.mean(daily_returns) - (self.risk_free_rate / self.trading_days_per_year)
                    sortino = float(excess_mean / downside_dev * np.sqrt(self.trading_days_per_year))
                else:
                    sortino = 0.0
            else:
                sortino = 0.0
        else:
            sortino = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Calmar ratio
        calmar = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0

        # Correlation matrix
        corr_matrix = self.calculate_correlation_matrix()

        # Covariance matrix
        cov_matrix = self.calculate_covariance_matrix()

        result = PortfolioResult(
            symbols=list(self.positions.keys()),
            weights={s: p.weight for s, p in self.positions.items()},
            allocated_capital={s: p.allocated_capital for s, p in self.positions.items()},
            allocation_method=self.allocation_method.value,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            correlation_matrix=corr_matrix,
            cov_matrix=cov_matrix,
            equity_curve=equity_curve,
            daily_returns=daily_returns.tolist(),
            individual_returns=individual_returns,
            position_data=self.positions,
        )

        return result

    def get_summary(self) -> str:
        """Get a human-readable summary of the portfolio."""
        if not self.positions:
            return "No assets in portfolio."

        lines = [
            f"Portfolio Summary",
            f"=" * 50,
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"Allocation Method: {self.allocation_method.value}",
            f"Number of Assets: {len(self.positions)}",
            "",
        ]

        for symbol, position in self.positions.items():
            lines.append(f"  {symbol}:")
            lines.append(f"    Weight: {position.weight:.2%}")
            lines.append(f"    Capital: ${position.allocated_capital:,.2f}")
            if position.returns_series is not None:
                rets = position.returns_series.dropna()
                if len(rets) > 0:
                    lines.append(f"    Avg Daily Return: {rets.mean():.6f}")
                    lines.append(f"    Volatility: {rets.std() * np.sqrt(self.trading_days_per_year):.2%}")
            lines.append("")

        return "\n".join(lines)


def backtest_portfolio(
    asset_data: dict[str, pd.Series],
    initial_capital: float = 1_000_000.0,
    allocation_method: AllocationMethod = AllocationMethod.EQUAL,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 365,
) -> PortfolioResult:
    """
    Convenience function to backtest a portfolio.

    Args:
        asset_data: Dict of {symbol: returns_series}
        initial_capital: Starting capital
        allocation_method: Equal weight or risk parity
        risk_free_rate: Annual risk-free rate
        trading_days_per_year: Days per year for annualization

    Returns:
        PortfolioResult with all metrics
    """
    portfolio = Portfolio(
        initial_capital=initial_capital,
        allocation_method=allocation_method,
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year,
    )

    for symbol, returns in asset_data.items():
        portfolio.add_asset(symbol, returns)

    return portfolio.run_backtest()


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    days = 252

    # Generate sample returns for 3 assets
    asset_data = {
        "BTC/USDT": pd.Series(np.random.normal(0.001, 0.03, days)),
        "ETH/USDT": pd.Series(np.random.normal(0.0008, 0.035, days)),
        "SOL/USDT": pd.Series(np.random.normal(0.0005, 0.04, days)),
    }

    # Test equal weight
    print("=" * 60)
    print("Equal Weight Portfolio")
    print("=" * 60)
    result_equal = backtest_portfolio(
        asset_data,
        allocation_method=AllocationMethod.EQUAL,
    )
    summary = result_equal.to_dict()
    for key, value in summary.items():
        if key != "correlation_matrix":
            print(f"{key}: {value}")
    print(f"\nCorrelation Matrix:\n{result_equal.correlation_matrix.round(3)}")

    # Test risk parity
    print("\n" + "=" * 60)
    print("Risk Parity Portfolio")
    print("=" * 60)
    result_risk_parity = backtest_portfolio(
        asset_data,
        allocation_method=AllocationMethod.RISK_PARITY,
    )
    summary = result_risk_parity.to_dict()
    for key, value in summary.items():
        if key != "correlation_matrix":
            print(f"{key}: {value}")
    print(f"\nCorrelation Matrix:\n{result_risk_parity.correlation_matrix.round(3)}")
