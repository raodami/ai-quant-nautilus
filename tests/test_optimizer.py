"""
Test script for Genetic Algorithm Optimizer.
"""

import sys
import os
import logging

# Add source to path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), 'src')))

from ai_quant_nautilus.optimizer.optimizer import (
    GeneticAlgorithm,
    OptimizationResult,
    Parameter,
    Encoding,
    SelectionMethod,
    CrossoverMethod,
    MutationMethod,
    optimize_ema,
    optimize_rsi,
    optimize_macd,
    get_ema_parameters,
    get_rsi_parameters,
    get_macd_parameters,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mock_fitness_fn(params: dict) -> tuple:
    """Mock fitness function for testing."""
    # Simulate Sharpe and MaxDD based on parameters
    base_sharpe = 1.5
    base_dd = -0.15
    
    # Adjust based on parameter values
    if 'fast_period' in params:
        base_sharpe += (params['fast_period'] / 50.0) * 0.5
        base_dd -= (params['fast_period'] / 50.0) * 0.05
    
    if 'slow_period' in params:
        base_sharpe += (params['slow_period'] / 200.0) * 0.3
        base_dd -= (params['slow_period'] / 200.0) * 0.03
    
    if 'rsi_period' in params:
        base_sharpe += (params['rsi_period'] / 30.0) * 0.4
        base_dd -= (params['rsi_period'] / 30.0) * 0.04
    
    return (base_sharpe, abs(base_dd))


def test_ga_basic():
    """Test basic GA functionality."""
    logger.info("=== Testing Basic GA ===")
    
    parameters = [
        Parameter(name="param1", min_val=0, max_val=100, default=50, encoding=Encoding.REAL),
        Parameter(name="param2", min_val=1, max_val=20, default=10, encoding=Encoding.INTEGER),
    ]
    
    ga = GeneticAlgorithm(
        parameters=parameters,
        population_size=50,
        max_generations=30,
        crossover_rate=0.8,
        mutation_rate=0.1,
        seed=42,
    )
    
    result = ga.optimize(mock_fitness_fn, verbose=True)
    
    assert result is not None
    assert result.best_individual is not None
    assert len(result.pareto_front) > 0
    assert result.generations_run > 0
    
    logger.info(f"Best params: {result.best_individual.parameter_values}")
    logger.info(f"Best fitness: Sharpe={result.best_individual.fitness[0]:.4f}, DD={result.best_individual.fitness[1]:.4f}")
    logger.info(f"Pareto front size: {len(result.pareto_front)}")
    logger.info(f"Generations: {result.generations_run}")
    
    return result


def test_ema_optimization():
    """Test EMA strategy optimization."""
    logger.info("\n=== Testing EMA Optimization ===")
    
    result = optimize_ema(
        fitness_fn=mock_fitness_fn,
        population_size=30,
        max_generations=20,
        seed=123,
    )
    
    logger.info(f"Optimized EMA params: {result.best_individual.parameter_values}")
    logger.info(f"Best Sharpe: {result.best_individual.fitness[0]:.4f}")
    
    return result


def test_multi_objective():
    """Test multi-objective optimization with Pareto front."""
    logger.info("\n=== Testing Multi-Objective (Pareto) ===")
    
    parameters = [
        Parameter(name="alpha", min_val=0.01, max_val=0.5, default=0.1, encoding=Encoding.REAL),
        Parameter(name="beta", min_val=0.5, max_val=5.0, default=1.0, encoding=Encoding.REAL),
    ]
    
    def multi_objective_fn(params: dict) -> tuple:
        """Multi-objective: maximize sharpe, minimize dd."""
        sharpe = params.get('alpha', 0.1) * 10 + params.get('beta', 1.0) * 2
        dd = params.get('alpha', 0.1) * 5 + params.get('beta', 1.0) * 1
        return (sharpe, dd)
    
    ga = GeneticAlgorithm(
        parameters=parameters,
        population_size=40,
        max_generations=25,
        seed=456,
    )
    
    result = ga.optimize(multi_objective_fn, verbose=True)
    
    logger.info(f"Pareto front solutions: {len(result.pareto_front)}")
    for i, ind in enumerate(result.pareto_front[:5]):
        logger.info(f"  Solution {i+1}: {ind.parameter_values} -> Sharpe={ind.fitness[0]:.2f}, DD={ind.fitness[1]:.2f}")
    
    return result


def test_convergence():
    """Test convergence tracking."""
    logger.info("\n=== Testing Convergence Tracking ===")
    
    parameters = [
        Parameter(name="x", min_val=0, max_val=10, default=5, encoding=Encoding.REAL),
    ]
    
    def simple_fn(params: dict) -> tuple:
        x = params.get('x', 5)
        return (10 - abs(x - 7), 0.1)  # Optimum at x=7
    
    ga = GeneticAlgorithm(
        parameters=parameters,
        population_size=20,
        max_generations=50,
        seed=789,
    )
    
    result = ga.optimize(simple_fn, verbose=False)
    
    assert len(result.convergence_history) > 0
    logger.info(f"Convergence history entries: {len(result.convergence_history)}")
    logger.info(f"First gen sharpe: {result.convergence_history[0]['best_sharpe']:.4f}")
    logger.info(f"Last gen sharpe: {result.convergence_history[-1]['best_sharpe']:.4f}")
    
    return result


def test_serialization():
    """Test result serialization."""
    logger.info("\n=== Testing Serialization ===")
    
    result = test_ga_basic()
    serialized = result.to_dict()
    
    assert 'best_params' in serialized
    assert 'best_fitness' in serialized
    assert 'pareto_front_size' in serialized
    assert 'generations' in serialized
    assert 'execution_time_sec' in serialized
    
    logger.info(f"Serialized result keys: {list(serialized.keys())}")
    logger.info(f"Best params: {serialized['best_params']}")
    
    return serialized


if __name__ == "__main__":
    print("=" * 60)
    print("Genetic Algorithm Optimizer - Test Suite")
    print("=" * 60)
    
    try:
        test_ga_basic()
        test_ema_optimization()
        test_multi_objective()
        test_convergence()
        test_serialization()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
