"""
Performance analyzer for backtest results.

Calculates comprehensive metrics including:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Trade statistics
- Win/loss distribution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive backtest performance metrics."""
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0

    # Trades
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0

    # Risk-adjusted
    excess_return: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_return": round(self.total_return, 4),
            "annualized_return": round(self.annualized_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "volatility": round(self.volatility, 4),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "largest_win": round(self.largest_win, 4),
            "largest_loss": round(self.largest_loss, 4),
        }


def calculate_performance_metrics(
    equity_curve: list[float],
    trades: Optional[list[dict]] = None,
    risk_free_rate: float = 0.0,
    benchmark_returns: Optional[list[float]] = None,
    trading_days_per_year: int = 365,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity_curve: List of portfolio values over time
        trades: List of trade dictionaries with 'pnl' key
        risk_free_rate: Annual risk-free rate (default 0%)
        benchmark_returns: Benchmark returns for beta/alpha calculation
        trading_days_per_year: For annualization
    """
    metrics = PerformanceMetrics()

    if not equity_curve:
        return metrics

    equity = np.array(equity_curve, dtype=float)
    initial_value = equity[0]
    final_value = equity[-1]

    # Total return
    metrics.total_return = (final_value - initial_value) / initial_value

    # Period returns (log returns for Sharpe)
    if len(equity) > 1:
        period_returns = np.diff(equity) / equity[:-1]
    else:
        period_returns = np.array([0.0])

    # Annualized return
    periods = len(equity)
    if initial_value > 0 and periods > 1:
        metrics.annualized_return = (
            (final_value / initial_value) ** (trading_days_per_year / periods) - 1
        ) if periods > 0 else 0.0

    # Volatility
    if len(period_returns) > 1:
        metrics.volatility = np.std(period_returns) * np.sqrt(trading_days_per_year)

    # Sharpe ratio
    if metrics.volatility > 0:
        excess_returns = period_returns - (risk_free_rate / trading_days_per_year)
        metrics.sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(trading_days_per_year)

    # Sortino ratio (downside deviation)
    if len(period_returns) > 1:
        downside_returns = period_returns[period_returns < 0]
        if len(downside_returns) > 0:
            downside_dev = np.std(downside_returns) * np.sqrt(trading_days_per_year)
            if downside_dev > 0:
                excess_mean = np.mean(period_returns) - (risk_free_rate / trading_days_per_year)
                metrics.sortino_ratio = excess_mean / downside_dev * np.sqrt(trading_days_per_year)

    # Maximum drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    metrics.max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Calmar ratio
    if abs(metrics.max_drawdown) > 0:
        metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)

    # Trade statistics
    if trades:
        pnls = [t.get("pnl", 0.0) for t in trades]
        metrics.total_trades = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        if pnls:
            metrics.win_rate = len(wins) / len(pnls)
            metrics.avg_win = np.mean(wins) if wins else 0.0
            metrics.avg_loss = np.mean(losses) if losses else 0.0
            metrics.largest_win = max(wins) if wins else 0.0
            metrics.largest_loss = min(losses) if losses else 0.0

            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            if gross_loss > 0:
                metrics.profit_factor = gross_profit / gross_loss

    # Beta and Alpha (if benchmark available)
    if benchmark_returns and len(benchmark_returns) == len(period_returns):
        benchmark_arr = np.array(benchmark_returns)
        if np.std(benchmark_arr) > 0 and np.cov(period_returns, benchmark_arr)[0, 1] > 0:
            metrics.beta = np.cov(period_returns, benchmark_arr)[0, 1] / np.var(benchmark_arr)
            metrics.alpha = (
                metrics.total_return / periods * trading_days_per_year
                - risk_free_rate
                - metrics.beta * (np.mean(benchmark_returns) * trading_days_per_year - risk_free_rate)
            )

    return metrics


def evaluate_strategy_performance(
    equity_curve: list[float],
    trades: Optional[list[dict]] = None,
    initial_capital: float = 1_000_000.0,
) -> dict:
    """
    Evaluate strategy and return gate-pass/fail results.
    """
    metrics = calculate_performance_metrics(equity_curve, trades)
    result = metrics.to_dict()

    # Add custom gates
    gates = {
        "sharpe_pass": metrics.sharpe_ratio >= 0.5,
        "max_dd_pass": abs(metrics.max_drawdown) <= 0.20,
        "win_rate_pass": metrics.win_rate >= 0.40,
        "trades_pass": metrics.total_trades >= 10,
        "profit_factor_pass": metrics.profit_factor >= 1.0,
        "calmar_pass": metrics.calmar_ratio >= 0.5,
    }

    result["gates"] = gates
    result["all_passed"] = all(gates.values())

    return result


if __name__ == "__main__":
    # Test with sample data
    equity = [1000000, 1010000, 1005000, 1020000, 1015000, 1030000, 1025000, 1040000]
    trades = [
        {"pnl": 5000, "duration": 24},
        {"pnl": -3000, "duration": 12},
        {"pnl": 8000, "duration": 36},
        {"pnl": 4000, "duration": 18},
    ]

    result = evaluate_strategy_performance(equity, trades)
    print(f"Performance: {result}")
