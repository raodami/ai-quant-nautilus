"""
Configuration manager: loads YAML config files with defaults and overrides.

Supports nested config sections for backtest, generator, evaluator, risk, and data.
Config precedence (highest wins): CLI args > env vars > config file > built-in defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Built-in defaults applied when no config file is loaded.
_DEFAULTS: dict[str, Any] = {
    "backtest": {
        "instrument": "ETHUSDT.BINANCE",
        "initial_capital": 1_000_000.0,
        "commission_bps": 10,
        "slippage_bps": 5,
    },
    "generator": {
        "strategy_limit": 3,
        "max_iterations": 10,
        "model": "gpt-4o",
        "temperature": 0.7,
    },
    "evaluator": {
        "sharpe_min": 0.5,
        "max_drawdown_max": 0.25,
        "win_rate_min": 0.45,
        "min_trades": 20,
    },
    "risk": {
        "max_position_pct": 0.1,
        "max_total_exposure": 0.5,
        "stop_loss_pct": 0.02,
    },
    "data": {
        "raw_dir": "data/raw",
        "cache_dir": "data/cache",
        "exchange": "binance",
        "symbols": ["ETHUSDT"],
        "interval": "1h",
    },
    "logging": {
        "level": "INFO",
        "format": "json",
        "log_dir": "logs",
    },
    "experiment": {
        "registry_path": "data/registry.json",
        "results_dir": "data/experiments",
        "cache_enabled": True,
        "cache_ttl_hours": 24,
    },
}


@dataclass
class Config:
    """Flat config namespace backed by a nested YAML dict."""

    # Backtest
    instrument: str = "ETHUSDT.BINANCE"
    initial_capital: float = 1_000_000.0
    commission_bps: int = 10
    slippage_bps: int = 5

    # Generator
    strategy_limit: int = 3
    max_iterations: int = 10
    model: str = "gpt-4o"
    temperature: float = 0.7

    # Evaluator gates
    sharpe_min: float = 0.5
    max_drawdown_max: float = 0.25
    win_rate_min: float = 0.45
    min_trades: int = 20

    # Risk
    max_position_pct: float = 0.1
    max_total_exposure: float = 0.5
    stop_loss_pct: float = 0.02

    # Data
    raw_dir: str = "data/raw"
    cache_dir: str = "data/cache"
    exchange: str = "binance"
    symbols: list[str] = field(default_factory=lambda: ["ETHUSDT"])
    interval: str = "1h"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_dir: str = "logs"

    # Experiment tracking
    registry_path: str = "data/registry.json"
    results_dir: str = "data/experiments"
    cache_enabled: bool = True
    cache_ttl_hours: int = 24

    # Raw source (for introspection / serialization)
    _source_path: Optional[Path] = field(default=None, repr=False)
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build a Config from a flat or nested dict (merged with defaults first)."""
        merged = _merge_defaults(_DEFAULTS, data)
        return cls(**_extract_flat(merged))

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load config from a YAML file, merging on top of built-in defaults."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", path)
        cfg = cls.from_dict(data)
        cfg._source_path = path.resolve()
        cfg._raw = data
        return cfg

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from environment variables prefixed with AQN_."""
        env_data: dict[str, Any] = {}
        prefix = "AQN_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            inner = key[len(prefix):].lower()
            _set_nested(env_data, inner, _parse_env_value(value))
        return cls.from_dict(env_data)

    @classmethod
    def load(
        cls,
        config_path: Optional[Path] = None,
        env_prefix: str = "AQN_",
    ) -> "Config":
        """
        Load config with precedence: explicit path > env vars > defaults.

        If config_path is None, only env vars (AQN_ prefix) and defaults are used.
        """
        cfg = cls()  # built-in defaults
        if config_path is not None:
            try:
                cfg = cls.from_yaml(config_path)
            except FileNotFoundError:
                logger.warning("Config file not found, using defaults + env: %s", config_path)
        # Env vars override file config
        env_cfg = cls.from_env()
        if env_cfg._raw:
            cfg = cls.from_dict(_merge_flat(cfg.to_dict(), env_cfg.to_dict()))
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict (excluding internal fields)."""
        import dataclasses
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if not f.name.startswith("_")
        }

    def save_yaml(self, path: Path) -> None:
        """Dump current config to a YAML file (for reproducibility)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info("Saved config to %s", path)

    def section(self, key: str) -> dict[str, Any]:
        """Return a nested dict slice for a given top-level section."""
        flat = self.to_dict()
        result: dict[str, Any] = {}
        for k, v in flat.items():
            if k == key and isinstance(v, dict):
                result = v
            elif k.startswith(f"{key}."):
                result[k[len(key) + 1:]] = v
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merge_defaults(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override into base; override wins."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge_defaults(merged[k], v)
        else:
            merged[k] = v
    return merged


def _merge_flat(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Flat dict merge; b wins on conflict."""
    merged = dict(a)
    merged.update(b)
    return merged


def _extract_flat(nested: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested config dict to top-level keys the dataclass expects."""
    flat: dict[str, Any] = {}
    for section, values in nested.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat[f"{section}_{k}" if section not in ("logging",) else k] = v
        else:
            flat[section] = values
    # Special mapping for common keys
    renaming = {
        "backtest_instrument": "instrument",
        "backtest_initial_capital": "initial_capital",
        "backtest_commission_bps": "commission_bps",
        "backtest_slippage_bps": "slippage_bps",
        "generator_strategy_limit": "strategy_limit",
        "generator_max_iterations": "max_iterations",
        "generator_model": "model",
        "generator_temperature": "temperature",
        "evaluator_sharpe_min": "sharpe_min",
        "evaluator_max_drawdown_max": "max_drawdown_max",
        "evaluator_win_rate_min": "win_rate_min",
        "evaluator_min_trades": "min_trades",
        "risk_max_position_pct": "max_position_pct",
        "risk_max_total_exposure": "max_total_exposure",
        "risk_stop_loss_pct": "stop_loss_pct",
        "data_raw_dir": "raw_dir",
        "data_cache_dir": "cache_dir",
        "data_exchange": "exchange",
        "data_symbols": "symbols",
        "data_interval": "interval",
        "logging_level": "log_level",
        "logging_format": "log_format",
        "logging_log_dir": "log_dir",
        "experiment_registry_path": "registry_path",
        "experiment_results_dir": "results_dir",
        "experiment_cache_enabled": "cache_enabled",
        "experiment_cache_ttl_hours": "cache_ttl_hours",
    }
    out: dict[str, Any] = {}
    for k, v in flat.items():
        out[renaming.get(k, k)] = v
    return out


def _set_nested(d: dict[str, Any], key: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated keys."""
    parts = key.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _parse_env_value(value: str) -> Any:
    """Parse an environment variable string into an appropriate Python type."""
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
