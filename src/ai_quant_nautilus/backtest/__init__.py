"""Backtest module."""
from ai_quant_nautilus.backtest.nautilus_adapter import (
    NautilusBacktestAdapter,
    BacktestOutcome,
    translate_strategy,
    ast_guard,
    generate_nautilus_strategy,
)
from ai_quant_nautilus.backtest.performance import (
    PerformanceMetrics,
    calculate_performance_metrics,
    evaluate_strategy_performance,
)
from ai_quant_nautilus.backtest.templates import (
    get_strategy_template,
    STRATEGY_TEMPLATES,
    StrategyTemplate,
)

__all__ = [
    "NautilusBacktestAdapter",
    "BacktestOutcome",
    "translate_strategy",
    "ast_guard",
    "generate_nautilus_strategy",
    "PerformanceMetrics",
    "calculate_performance_metrics",
    "evaluate_strategy_performance",
    "get_strategy_template",
    "STRATEGY_TEMPLATES",
    "StrategyTemplate",
]
