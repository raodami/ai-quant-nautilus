"""
Configuration management for ai-quant-nautilus.

Loads YAML config files and provides sensible defaults.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.001
    slippage_pct: float = 0.0005
    max_position_size_pct: float = 0.1
    fill_model: str = "market"


@dataclass
class DataConfig:
    """Data source configuration."""
    default_exchange: str = "binance"
    cache_dir: str = "data/cache"
    limit: int = 1000


@dataclass
class OptimizerConfig:
    """Genetic algorithm configuration."""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5


@dataclass
class EvaluatorConfig:
    """Evaluation gate configuration."""
    min_sharpe: float = 0.5
    max_drawdown: float = 0.20
    min_win_rate: float = 0.40
    min_trades: int = 10


@dataclass
class AppSettings:
    """Application-wide settings."""
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    log_level: str = "INFO"
    output_dir: str = "output"


class Config:
    """Configuration loader and manager."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.settings = AppSettings()
        self._config_path = config_path
        
        if config_path and config_path.exists():
            self.load(config_path)
    
    def load(self, path: Path) -> None:
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed, using defaults")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        
        self._apply(data)
        logger.info(f"Loaded config from {path}")
    
    def _apply(self, data: dict) -> None:
        """Apply config values to settings."""
        if 'backtest' in data:
            for k, v in data['backtest'].items():
                if hasattr(self.settings.backtest, k):
                    setattr(self.settings.backtest, k, v)
        
        if 'data' in data:
            for k, v in data['data'].items():
                if hasattr(self.settings.data, k):
                    setattr(self.settings.data, k, v)
        
        if 'optimizer' in data:
            for k, v in data['optimizer'].items():
                if hasattr(self.settings.optimizer, k):
                    setattr(self.settings.optimizer, k, v)
        
        if 'evaluator' in data:
            for k, v in data['evaluator'].items():
                if hasattr(self.settings.evaluator, k):
                    setattr(self.settings.evaluator, k, v)
        
        if 'log_level' in data:
            self.settings.log_level = data['log_level']
        
        if 'output_dir' in data:
            self.settings.output_dir = data['output_dir']
    
    def to_dict(self) -> dict:
        """Serialize config to dict."""
        return {
            'backtest': {
                'initial_capital': self.settings.backtest.initial_capital,
                'commission_rate': self.settings.backtest.commission_rate,
                'slippage_pct': self.settings.backtest.slippage_pct,
            },
            'data': {
                'default_exchange': self.settings.data.default_exchange,
                'cache_dir': self.settings.data.cache_dir,
            },
            'optimizer': {
                'population_size': self.settings.optimizer.population_size,
                'generations': self.settings.optimizer.generations,
            },
            'evaluator': {
                'min_sharpe': self.settings.evaluator.min_sharpe,
                'max_drawdown': self.settings.evaluator.max_drawdown,
            },
        }
    
    def save(self, path: Path) -> None:
        """Save current config to YAML file."""
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not installed, cannot save config")
            return
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        
        logger.info(f"Config saved to {path}")


def get_default_config() -> AppSettings:
    """Return default application settings."""
    return AppSettings()


def load_config(path: Optional[str] = None) -> Config:
    """Load config from path or return defaults."""
    config_path = Path(path) if path else None
    return Config(config_path)
