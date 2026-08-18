# Genetic Algorithm Optimizer

A comprehensive genetic algorithm (GA) optimizer for strategy parameter tuning in quantitative trading systems.

## Features

- **Single & Multi-Objective Optimization**: Optimize for Sharpe ratio alone or balance Sharpe vs Maximum Drawdown
- **Multiple Encoding Schemes**: Binary, real-valued, and integer parameter encoding
- **Flexible Selection Methods**: Tournament, roulette wheel, and elite selection
- **Crossover Operators**: Single-point, two-point, uniform, and BLX-alpha crossover
- **Mutation Strategies**: Flip, Gaussian, swap, and invert mutation
- **Pareto Front Extraction**: Identify non-dominated solutions for multi-objective problems
- **Early Stopping**: Convergence-based termination to save computation time
- **Strategy Templates**: Pre-configured parameters for EMA, RSI, and MACD strategies

## Quick Start

```python
from ai_quant_nautilus.optimizer import (
    GeneticAlgorithm,
    optimize_ema,
    optimize_rsi,
    optimize_macd,
    get_ema_parameters,
)

# Define fitness function (returns tuple of sharpe_ratio, max_drawdown)
def fitness_fn(params: dict) -> tuple:
    # Backtest strategy with given parameters
    result = backtest_strategy(params)
    return result.sharpe_ratio, abs(result.max_drawdown)

# Option 1: Use pre-built optimizers
result = optimize_ema(
    fitness_fn=fitness_fn,
    population_size=100,
    max_generations=50,
    seed=42,
)

# Option 2: Custom GA with full control
ga = GeneticAlgorithm(
    parameters=get_ema_parameters(),
    population_size=100,
    max_generations=50,
    selection_method="tournament",
    crossover_method="two_point",
    mutation_method="gaussian",
    seed=42,
)

result = ga.optimize(fitness_fn)

# Access results
best_params = result.best_individual.parameter_values
pareto_front = result.pareto_front
print(f"Best Sharpe: {result.best_individual.fitness[0]:.4f}")
print(f"Pareto solutions: {len(result.pareto_front)}")
```

## Strategy Parameters

### EMA Crossover
```python
params = get_ema_parameters()
# fast_period: 5-50
# slow_period: 20-200
# entry_threshold: 0.001-0.01
# exit_threshold: 0.001-0.01
```

### RSI Strategy
```python
params = get_rsi_parameters()
# rsi_period: 7-30
# oversold: 10-40
# overbought: 60-90
```

### MACD Strategy
```python
params = get_macd_parameters()
# fast_period: 8-21
# slow_period: 18-40
# signal_period: 5-15
```

### Combined Multi-Indicator
```python
params = get_combined_parameters()
# All EMA + RSI + MACD parameters combined
```

## Integration with Backtest Engine

```python
from ai_quant_nautilus.backtest import BacktestEngine, BacktestConfig
from ai_quant_nautilus.optimizer import optimize_ema

# Load historical data
import pandas as pd
df = pd.read_parquet("data/ethusdt_daily.parquet")

# Define fitness function
def evaluate_params(params):
    config = BacktestConfig(initial_capital=1_000_000)
    engine = BacktestEngine(config)
    
    # Generate strategy code with parameters
    strategy_code = f'''
class EmaStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self._fast = {params['fast_period']}
        self._slow = {params['slow_period']}
        self._entry = {params['entry_threshold']}
        
    def on_start(self):
        self.fast_ma = self.indicator("ema", period=self._fast)
        self.slow_ma = self.indicator("ema", period=self._slow)
    
    def on_bar(self, bar):
        fast = self.fast_ma.value
        slow = self.slow_ma.value
        if fast is None or slow is None:
            return
        if fast > slow + self._entry:
            self.order_market(self.instrument_id, "BUY", Decimal("0.1"))
        elif fast < slow - self._entry:
            self.order_market(self.instrument_id, "SELL", Decimal("0.1"))
'''
    
    result = engine.run(strategy_code, df)
    return result.sharpe_ratio, abs(result.max_drawdown_pct)

# Run optimization
result = optimize_ema(
    fitness_fn=evaluate_params,
    population_size=50,
    max_generations=30,
    seed=42,
)

# Get best parameters
print(f"Optimal fast_period: {result.best_individual.parameter_values['fast_period']}")
print(f"Optimal slow_period: {result.best_individual.parameter_values['slow_period']}")
```

## API Reference

### GeneticAlgorithm

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parameters` | list | - | List of Parameter definitions |
| `population_size` | int | 100 | Number of individuals in population |
| `max_generations` | int | 200 | Maximum generations to evolve |
| `crossover_rate` | float | 0.8 | Probability of crossover |
| `mutation_rate` | float | 0.1 | Probability of mutation |
| `selection_method` | SelectionMethod | TOURNAMENT | Selection algorithm |
| `crossover_method` | CrossoverMethod | TWO_POINT | Crossover algorithm |
| `mutation_method` | MutationMethod | GAUSSIAN | Mutation algorithm |
| `elite_size` | int | 2 | Number of elites to preserve |
| `tournament_size` | int | 3 | Tournament selection size |
| `early_stop_threshold` | float | 1e-6 | Convergence threshold |
| `early_stop_patience` | int | 20 | Generations before early stop |
| `seed` | int | None | Random seed for reproducibility |

### Parameter

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | - | Parameter name |
| `min_val` | float | 0.0 | Minimum value |
| `max_val` | float | 100.0 | Maximum value |
| `default` | float | 50.0 | Default/initial value |
| `encoding` | Encoding | REAL | Encoding scheme |
| `bits` | int | 8 | Bits for binary encoding |

### OptimizationResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `best_individual` | Individual | Best solution found |
| `pareto_front` | list | Pareto-optimal solutions |
| `generations_run` | int | Number of generations executed |
| `execution_time` | float | Total optimization time (seconds) |
| `convergence_history` | list | Per-generation metrics |
| `parameter_names` | list | Parameter names in order |

## Running Tests

```bash
uv run python tests/test_optimizer.py
```
