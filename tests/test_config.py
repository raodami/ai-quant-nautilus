import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
from pathlib import Path
from ai_quant_nautilus.config.config import Config, get_default_config, load_config


class TestConfig:
    """Test configuration management."""

    def test_default_config(self):
        cfg = Config()
        # Default nested structure
        assert cfg.settings.backtest.initial_capital == 1_000_000.0
        assert cfg.settings.optimizer.population_size == 50
        assert cfg.settings.evaluator.min_sharpe == 0.5

    def test_to_dict(self):
        cfg = Config()
        d = cfg.to_dict()
        assert "backtest" in d
        assert "optimizer" in d
        assert "evaluator" in d

    def test_from_yaml(self, tmp_path):
        yaml_content = """
backtest:
  initial_capital: 500000.0
  commission_rate: 0.0015
evaluator:
  min_sharpe: 1.0
  max_drawdown: 0.15
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)
        cfg = Config(config_file)
        assert cfg.settings.backtest.initial_capital == 500000.0
        assert cfg.settings.evaluator.min_sharpe == 1.0
        assert cfg.settings.evaluator.max_drawdown == 0.15

    def test_from_yaml_not_found(self, tmp_path):
        # Should use defaults when file not found
        cfg = Config(tmp_path / "nonexistent.yaml")
        assert cfg is not None

    def test_save_yaml(self, tmp_path):
        cfg = Config()
        cfg.settings.backtest.initial_capital = 250000.0
        out_path = tmp_path / "output" / "config.yaml"
        cfg.save(out_path)
        assert out_path.exists()

    def test_section(self):
        cfg = Config()
        bt = cfg.to_dict().get("backtest", {})
        assert "initial_capital" in bt

    def test_load_config(self, tmp_path):
        yaml_content = """
backtest:
  initial_capital: 750000.0
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)
        cfg = load_config(str(config_file))
        assert cfg.settings.backtest.initial_capital == 750000.0


class TestLoadConfig:
    """Test config loading convenience functions."""

    def test_get_default_config(self):
        settings = get_default_config()
        assert settings is not None
        assert settings.backtest.initial_capital == 1_000_000.0

    def test_load_config_no_path(self):
        cfg = load_config()
        assert cfg is not None
        assert cfg.settings.backtest.initial_capital == 1_000_000.0
