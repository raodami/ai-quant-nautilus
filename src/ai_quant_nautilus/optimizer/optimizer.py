"""
Genetic Algorithm Optimizer for Strategy Parameters.

Supports single-objective (Sharpe) and multi-objective (Sharpe + MaxDD)
optimization for EMA, RSI, MACD, and other technical indicator strategies.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------

class Encoding(str, Enum):
    """Parameter encoding scheme."""
    BINARY = "binary"
    REAL = "real"
    INTEGER = "integer"


class SelectionMethod(str, Enum):
    """Selection algorithm."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    ELITE = "elite"


class CrossoverMethod(str, Enum):
    """Crossover algorithm."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    BLX_ALPHA = "blx_alpha"


class MutationMethod(str, Enum):
    """Mutation algorithm."""
    FLIP = "flip"
    GAUSSIAN = "gaussian"
    SWAP = "swap"
    INVERT = "invert"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Parameter:
    """Single optimization parameter definition."""
    name: str
    value_type: str = "continuous"  # "continuous", "integer", "categorical"
    min_val: float = 0.0
    max_val: float = 100.0
    default: float = 50.0
    encoding: Encoding = Encoding.REAL
    bits: int = 8  # for binary encoding

    def encode(self, value: float) -> np.ndarray:
        """Encode parameter value to chromosome segment."""
        if self.encoding == Encoding.BINARY:
            normalized = (value - self.min_val) / (self.max_val - self.min_val)
            normalized = max(0.0, min(1.0, normalized))
            binary = format(int(normalized * (2**self.bits - 1)), f'0{self.bits}b')
            return np.array([int(b) for b in binary])
        elif self.encoding == Encoding.INTEGER:
            int_val = int(round(value))
            int_val = max(self.min_val, min(self.max_val, int_val))
            return np.array([int_val])
        else:  # REAL
            return np.array([value])

    def decode(self, segment: np.ndarray) -> float:
        """Decode chromosome segment to parameter value."""
        if self.encoding == Encoding.BINARY:
            binary_str = ''.join(str(int(b)) for b in segment[:self.bits])
            normalized = int(binary_str, 2) / (2**self.bits - 1)
            return self.min_val + normalized * (self.max_val - self.min_val)
        elif self.encoding == Encoding.INTEGER:
            return float(segment[0])
        else:  # REAL
            return float(segment[0])


@dataclass
class Individual:
    """Chromosome representing a parameter set."""
    genes: np.ndarray = field(default_factory=np.ndarray)
    fitness: Tuple[float, float] = (0.0, 0.0)  # (sharpe, max_dd_neg)
    rank: int = 0  # Pareto rank
    dominance_count: int = 0  # How many individuals dominate this one
    crowd_distance: float = 0.0  # Crowding distance for diversity
    parameter_values: dict = field(default_factory=dict)
    _parameters: list = field(default_factory=list, repr=False)

    def set_parameters(self, parameters: list) -> None:
        """Set the parameter definitions for decoding."""
        self._parameters = parameters

    def evaluate(self, fitness_fn: Callable) -> None:
        """Evaluate fitness using provided function."""
        params = self.decode_parameters()
        self.fitness = fitness_fn(params)
        if not self.parameter_values:
            self.parameter_values = params

    def decode_parameters(self) -> dict:
        """Decode chromosome to named parameters."""
        if self.parameter_values:
            return self.parameter_values
        if not self._parameters:
            return {}
        params = {}
        idx = 0
        for param in self._parameters:
            segment_len = param.bits if param.encoding == Encoding.BINARY else 1
            segment = self.genes[idx:idx + segment_len]
            params[param.name] = param.decode(segment)
            idx += segment_len
        return params

    def crossover(self, other: 'Individual', method: CrossoverMethod = CrossoverMethod.SINGLE_POINT) -> Tuple['Individual', 'Individual']:
        """Crossover with another individual."""
        child1_genes, child2_genes = _crossover(self.genes, other.genes, method)
        child1 = Individual(genes=child1_genes)
        child2 = Individual(genes=child2_genes)
        return child1, child2

    def mutate(self, method: MutationMethod = MutationMethod.GAUSSIAN, **kwargs) -> None:
        """Mutate genes in place."""
        self.genes = _mutation(self.genes, method, **kwargs)

    def clone(self) -> 'Individual':
        """Create deep copy."""
        return Individual(
            genes=self.genes.copy(),
            fitness=self.fitness,
            rank=self.rank,
            dominance_count=self.dominance_count,
            crowd_distance=self.crowd_distance,
            parameter_values=self.parameter_values.copy(),
        )


@dataclass
class OptimizationResult:
    """Results from genetic algorithm optimization."""
    best_individual: Optional[Individual] = None
    pareto_front: list = field(default_factory=list)
    generations_run: int = 0
    execution_time: float = 0.0
    convergence_history: list = field(default_factory=list)
    parameter_names: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize results."""
        result = {
            "best_params": self.best_individual.parameter_values if self.best_individual else {},
            "best_fitness": self.best_individual.fitness if self.best_individual else (0, 0),
            "pareto_front_size": len(self.pareto_front),
            "generations": self.generations_run,
            "execution_time_sec": round(self.execution_time, 2),
        }
        if self.pareto_front:
            result["pareto_solutions"] = [
                {"params": ind.parameter_values, "fitness": ind.fitness}
                for ind in self.pareto_front[:20]  # Top 20
            ]
        return result


# ---------------------------------------------------------------------------
# Genetic Algorithm Engine
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """
    Generic GA engine for parameter optimization.
    
    Supports:
    - Single and multi-objective optimization
    - Tournament, roulette wheel, and elite selection
    - Single-point, two-point, uniform, and BLX-alpha crossover
    - Flip, Gaussian, swap, and invert mutation
    - Niching and crowding for diversity preservation
    - Early stopping based on convergence criteria
    """

    def __init__(
        self,
        parameters: list,
        population_size: int = 100,
        max_generations: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
        crossover_method: CrossoverMethod = CrossoverMethod.TWO_POINT,
        mutation_method: MutationMethod = MutationMethod.GAUSSIAN,
        elite_size: int = 2,
        tournament_size: int = 3,
        niching_radius: Optional[float] = None,
        early_stop_threshold: float = 1e-6,
        early_stop_patience: int = 20,
        seed: Optional[int] = None,
    ):
        self.parameters = [Parameter(**p) if isinstance(p, dict) else p for p in parameters]
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.niching_radius = niching_radius
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.gene_length = sum(p.bits if p.encoding == Encoding.BINARY else 1 for p in self.parameters)

        # Tracking
        self.convergence_history = []
        self.generation = 0
        self.start_time = 0.0

    def optimize(
        self,
        fitness_fn: Callable[[dict], Tuple[float, float]],
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Run genetic algorithm optimization.
        
        Args:
            fitness_fn: Function mapping parameter dict to (sharpe_ratio, max_drawdown).
            verbose: Print progress information.
            
        Returns:
            OptimizationResult with best solution and Pareto front.
        """
        self.start_time = time.time()
        self.generation = 0
        self.convergence_history = []

        # Initialize population
        population = self._initialize_population()

        # Evaluate initial population
        for ind in population:
            ind.evaluate(fitness_fn)

        best = self._get_best(population)
        pareto = self._get_pareto_front(population)

        if verbose:
            logger.info(f"GA started: pop={self.population_size}, gen={self.max_generations}")
            logger.info(f"Initial best Sharpe: {best.fitness[0]:.4f}, MaxDD: {best.fitness[1]:.4f}")

        # Main evolution loop
        for gen in range(self.max_generations):
            self.generation = gen + 1

            # Selection
            parents = self._select_parents(population)

            # Crossover
            offspring = self._crossover(parents)

            # Mutation
            offspring = self._mutate(offspring)

            # Evaluation
            for ind in offspring:
                ind.evaluate(fitness_fn)

            # Combine and select next generation
            combined = population + offspring
            population = self._selection(combined)

            # Track progress
            gen_best = self._get_best(population)
            gen_pareto = self._get_pareto_front(population)
            self.convergence_history.append({
                "generation": gen + 1,
                "best_sharpe": gen_best.fitness[0],
                "best_maxdd": gen_best.fitness[1],
                "pareto_size": len(gen_pareto),
                "avg_sharpe": np.mean([i.fitness[0] for i in population]),
            })

            # Log progress
            if verbose and (gen % 10 == 0 or gen == self.max_generations - 1):
                logger.info(
                    f"Gen {gen + 1}/{self.max_generations}: "
                    f"Sharpe={gen_best.fitness[0]:.4f}, DD={gen_best.fitness[1]:.4f}, "
                    f"Pareto={len(gen_pareto)}"
                )

            # Early stopping check
            if self._check_early_stopping():
                if verbose:
                    logger.info(f"Early stopping at generation {gen + 1}")
                break

        # Final results
        final_best = self._get_best(population)
        final_pareto = self._get_pareto_front(population)
        exec_time = time.time() - self.start_time

        result = OptimizationResult(
            best_individual=final_best,
            pareto_front=final_pareto,
            generations_run=self.generation,
            execution_time=exec_time,
            convergence_history=self.convergence_history,
            parameter_names=[p.name for p in self.parameters],
        )

        if verbose:
            logger.info(f"Optimization complete in {exec_time:.2f}s")
            logger.info(f"Best Sharpe: {final_best.fitness[0]:.4f}, MaxDD: {final_best.fitness[1]:.4f}")

        return result

    def _initialize_population(self) -> list:
        """Create random initial population."""
        population = []
        for _ in range(self.population_size):
            genes = []
            param_values = {}
            for param in self.parameters:
                # Random value within bounds
                if param.encoding == Encoding.BINARY:
                    value = param.min_val + random.random() * (param.max_val - param.min_val)
                elif param.encoding == Encoding.INTEGER:
                    value = random.randint(int(param.min_val), int(param.max_val))
                else:
                    value = param.min_val + random.random() * (param.max_val - param.min_val)

                param_values[param.name] = value
                genes.extend(param.encode(value))

            individual = Individual(
                genes=np.array(genes, dtype=float),
                parameter_values=param_values,
                _parameters=self.parameters,
            )
            population.append(individual)
        return population

    def _select_parents(self, population: list) -> list:
        """Select parents for reproduction."""
        parents = []
        for _ in range(len(population)):
            if self.selection_method == SelectionMethod.TOURNAMENT:
                parents.append(self._tournament_select(population))
            elif self.selection_method == SelectionMethod.ROULETTE:
                parents.append(self._roulette_select(population))
            else:  # ELITE
                parents.append(self._get_best(population))
        return parents

    def _tournament_select(self, population: list) -> Individual:
        """Tournament selection."""
        contenders = random.sample(population, min(self.tournament_size, len(population)))
        return max(contenders, key=lambda x: x.fitness[0])

    def _roulette_select(self, population: list) -> Individual:
        """Roulette wheel selection (fitness proportional)."""
        fitnesses = np.array([ind.fitness[0] for ind in population])
        # Ensure non-negative for roulette
        fitnesses = fitnesses - np.min(fitnesses) + 1e-10
        probabilities = fitnesses / np.sum(fitnesses)
        return random.choices(population, weights=probabilities, k=1)[0]

    def _crossover(self, parents: list) -> list:
        """Perform crossover to generate offspring."""
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            if random.random() < self.crossover_rate:
                child1_genes, child2_genes = _crossover(parent1.genes, parent2.genes, self.crossover_method)
                child1 = Individual(
                    genes=child1_genes,
                    parameter_values=self._merge_parameters(parent1.parameter_values, parent2.parameter_values),
                    _parameters=self.parameters,
                )
                child2 = Individual(
                    genes=child2_genes,
                    parameter_values=self._merge_parameters(parent2.parameter_values, parent1.parameter_values),
                    _parameters=self.parameters,
                )
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.clone(), parent2.clone()])
        return offspring

    def _mutate(self, offspring: list) -> list:
        """Perform mutation on offspring."""
        for ind in offspring:
            if random.random() < self.mutation_rate:
                ind.mutate(self.mutation_method, 
                          gene_length=len(ind.genes),
                          param_ranges=self._get_param_ranges())
        return offspring

    def _selection(self, population: list) -> list:
        """Select next generation with elitism and niching."""
        # Sort by fitness (higher is better for sharpe)
        sorted_pop = sorted(population, key=lambda x: x.fitness[0], reverse=True)
        
        # Keep elites
        elites = sorted_pop[:self.elite_size]
        
        # Fill rest from non-elites
        next_gen = elites + sorted_pop[self.elite_size:self.population_size - len(elites)]
        
        return next_gen[:self.population_size]

    def _get_best(self, population: list) -> Individual:
        """Get best individual by Sharpe ratio."""
        return max(population, key=lambda x: x.fitness[0])

    def _get_pareto_front(self, population: list) -> list:
        """Extract Pareto-optimal front."""
        pareto = []
        for ind in population:
            is_dominated = False
            for other in population:
                if other is ind:
                    continue
                # Check if 'other' dominates 'ind'
                if (other.fitness[0] >= ind.fitness[0] and 
                    other.fitness[1] >= ind.fitness[1] and
                    (other.fitness[0] > ind.fitness[0] or other.fitness[1] > ind.fitness[1])):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto.append(ind)
        return pareto

    def _check_early_stopping(self) -> bool:
        """Check if optimization should stop early."""
        if len(self.convergence_history) < self.early_stop_patience:
            return False
        
        recent = self.convergence_history[-self.early_stop_patience:]
        sharpes = [h["best_sharpe"] for h in recent]
        improvement = max(sharpes) - min(sharpes)
        
        return improvement < self.early_stop_threshold

    def _merge_parameters(self, params1: dict, params2: dict) -> dict:
        """Merge parameter values from two parents."""
        merged = {}
        for key in params1:
            if random.random() < 0.5:
                merged[key] = params1.get(key, params2.get(key))
            else:
                merged[key] = params2.get(key, params1.get(key))
        return merged

    def _get_param_ranges(self) -> list:
        """Get parameter ranges for mutation."""
        return [(p.min_val, p.max_val) for p in self.parameters]


# ---------------------------------------------------------------------------
# Crossover and Mutation Functions
# ---------------------------------------------------------------------------

def _crossover(
    genes1: np.ndarray, 
    genes2: np.ndarray, 
    method: CrossoverMethod
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two gene vectors."""
    length = min(len(genes1), len(genes2))
    genes1, genes2 = genes1[:length], genes2[:length]
    
    if method == CrossoverMethod.SINGLE_POINT:
        point = random.randint(1, length - 1)
        child1 = np.concatenate([genes1[:point], genes2[point:]])
        child2 = np.concatenate([genes2[:point], genes1[point:]])
    elif method == CrossoverMethod.TWO_POINT:
        if length >= 3:
            points = sorted(random.sample(range(1, length), 2))
            child1 = np.concatenate([genes1[:points[0]], genes2[points[0]:points[1]], genes1[points[1]:]])
            child2 = np.concatenate([genes2[:points[0]], genes1[points[0]:points[1]], genes2[points[1]:]])
        else:
            # Fallback to single-point for small genes
            point = random.randint(1, max(1, length - 1))
            child1 = np.concatenate([genes1[:point], genes2[point:]])
            child2 = np.concatenate([genes2[:point], genes1[point:]])
    elif method == CrossoverMethod.UNIFORM:
        mask = np.random.random(length) < 0.5
        child1 = np.where(mask, genes1, genes2)
        child2 = np.where(mask, genes2, genes1)
    elif method == CrossoverMethod.BLX_ALPHA:
        alpha = 0.5
        min_vals = np.minimum(genes1, genes2)
        max_vals = np.maximum(genes1, genes2)
        range_vals = max_vals - min_vals
        child1 = min_vals + np.random.random(length) * (1 + alpha) * range_vals
        child2 = min_vals + np.random.random(length) * (1 + alpha) * range_vals
        # Clip to valid range
        child1 = np.clip(child1, min_vals, max_vals)
        child2 = np.clip(child2, min_vals, max_vals)
    else:
        child1, child2 = genes1.copy(), genes2.copy()
    
    return child1, child2


def _mutation(
    genes: np.ndarray, 
    method: MutationMethod,
    gene_length: int = 0,
    param_ranges: Optional[list] = None,
    **kwargs
) -> np.ndarray:
    """Perform mutation on gene vector."""
    genes = genes.copy()
    length = len(genes)
    
    if method == MutationMethod.FLIP:
        for i in range(length):
            if random.random() < kwargs.get('flip_rate', 0.01):
                genes[i] = 1 - genes[i]
    elif method == MutationMethod.GAUSSIAN:
        sigma = kwargs.get('sigma', 0.1)
        genes += np.random.normal(0, sigma, length)
        # Clip if ranges provided
        if param_ranges and len(param_ranges) == len(genes):
            for i, (min_val, max_val) in enumerate(param_ranges):
                genes[i] = max(min_val, min(max_val, genes[i]))
    elif method == MutationMethod.SWAP:
        if length >= 2 and random.random() < 0.1:
            i, j = random.sample(range(length), 2)
            genes[i], genes[j] = genes[j], genes[i]
    elif method == MutationMethod.INVERT:
        if length >= 2 and random.random() < 0.1:
            i, j = sorted(random.sample(range(length), 2))
            genes[i:j+1] = genes[i:j+1][::-1]
    
    return genes


# ---------------------------------------------------------------------------
# Strategy-Specific Parameter Definitions
# ---------------------------------------------------------------------------

def get_ema_parameters() -> list:
    """Get parameter definitions for EMA crossover strategy."""
    return [
        Parameter(
            name="fast_period",
            min_val=5,
            max_val=50,
            default=10,
            encoding=Encoding.INTEGER,
        ),
        Parameter(
            name="slow_period",
            min_val=20,
            max_val=200,
            default=20,
            encoding=Encoding.INTEGER,
        ),
        Parameter(
            name="entry_threshold",
            min_val=0.001,
            max_val=0.01,
            default=0.001,
            encoding=Encoding.REAL,
        ),
        Parameter(
            name="exit_threshold",
            min_val=0.001,
            max_val=0.01,
            default=0.001,
            encoding=Encoding.REAL,
        ),
    ]


def get_rsi_parameters() -> list:
    """Get parameter definitions for RSI strategy."""
    return [
        Parameter(
            name="rsi_period",
            min_val=7,
            max_val=30,
            default=14,
            encoding=Encoding.INTEGER,
        ),
        Parameter(
            name="oversold",
            min_val=10,
            max_val=40,
            default=30,
            encoding=Encoding.INTEGER,
        ),
        Parameter(
            name="overbought",
            min_val=60,
            max_val=90,
            default=70,
            encoding=Encoding.INTEGER,
        ),
    ]


def get_macd_parameters() -> list:
    """Get parameter definitions for MACD strategy."""
    return [
        Parameter(
            name="fast_period",
            min_val=8,
            max_val=21,
            default=12,
            encoding=Encoding.INTEGER,
        ),
        Parameter(
            name="slow_period",
            min_val=18,
            max_val=40,
            default=26,
            encoding=Encoding.INTEGER,
        ),
        Parameter(
            name="signal_period",
            min_val=5,
            max_val=15,
            default=9,
            encoding=Encoding.INTEGER,
        ),
    ]


def get_combined_parameters() -> list:
    """Get combined parameter definitions for multi-indicator strategy."""
    return get_ema_parameters() + get_rsi_parameters() + get_macd_parameters()


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def optimize_ema(
    fitness_fn: Callable,
    population_size: int = 100,
    max_generations: int = 100,
    seed: Optional[int] = None,
) -> OptimizationResult:
    """Optimize EMA strategy parameters."""
    ga = GeneticAlgorithm(
        parameters=get_ema_parameters(),
        population_size=population_size,
        max_generations=max_generations,
        seed=seed,
    )
    return ga.optimize(fitness_fn)


def optimize_rsi(
    fitness_fn: Callable,
    population_size: int = 100,
    max_generations: int = 100,
    seed: Optional[int] = None,
) -> OptimizationResult:
    """Optimize RSI strategy parameters."""
    ga = GeneticAlgorithm(
        parameters=get_rsi_parameters(),
        population_size=population_size,
        max_generations=max_generations,
        seed=seed,
    )
    return ga.optimize(fitness_fn)


def optimize_macd(
    fitness_fn: Callable,
    population_size: int = 100,
    max_generations: int = 100,
    seed: Optional[int] = None,
) -> OptimizationResult:
    """Optimize MACD strategy parameters."""
    ga = GeneticAlgorithm(
        parameters=get_macd_parameters(),
        population_size=population_size,
        max_generations=max_generations,
        seed=seed,
    )
    return ga.optimize(fitness_fn)


__all__ = [
    "GeneticAlgorithm",
    "OptimizationResult",
    "Individual",
    "Parameter",
    "Encoding",
    "SelectionMethod",
    "CrossoverMethod",
    "MutationMethod",
    "optimize_ema",
    "optimize_rsi",
    "optimize_macd",
    "get_ema_parameters",
    "get_rsi_parameters",
    "get_macd_parameters",
    "get_combined_parameters",
]
