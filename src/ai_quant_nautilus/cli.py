"""CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="aqn",
        description="AI-Quant-Nautilus: AI-driven crypto quant closed-loop",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (default: look for config.yaml in project root)",
    )

    sub = parser.add_subparsers(dest="command")

    # run: full simulation
    p_run = sub.add_parser("run", help="Run one full iteration (generate → backtest → evaluate)")
    p_run.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    p_run.add_argument("--strategy-limit", type=int, default=3, help="Max strategies per iteration")
    p_run.add_argument("--data-dir", type=Path, default=None, help="OHLCV data dir (overrides config)")
    p_run.add_argument("--registry", type=Path, default=None, help="Strategy registry path (overrides config)")

    # backtest: run nautilus backtest on a strategy file
    p_bt = sub.add_parser("backtest", help="Run nautilus backtest on a strategy")
    p_bt.add_argument("--strategy", type=Path, required=True, help="Strategy Python file")
    p_bt.add_argument("--instrument", type=str, default=None, help="Instrument ID (overrides config)")
    p_bt.add_argument("--data-path", type=Path, required=True, help="CSV/parquet data file")
    p_bt.add_argument("--initial-capital", type=float, default=None, help="Starting USDT (overrides config)")

    # status: show current registry
    sub.add_parser("status", help="Show strategy registry status")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    # Load config: explicit path > auto-discover > defaults
    config_path = args.config
    if config_path is None:
        auto_path = PROJECT_ROOT / "config.yaml"
        if auto_path.exists():
            config_path = auto_path

    from ai_quant_nautilus.config import Config
    cfg = Config.load(config_path) if config_path else Config()

    # Apply CLI overrides
    if args.data_dir is not None:
        cfg.raw_dir = str(args.data_dir)
    if args.registry is not None:
        cfg.registry_path = str(args.registry)
    if args.instrument is not None:
        cfg.instrument = args.instrument
    if args.initial_capital is not None:
        cfg.initial_capital = args.initial_capital
    if hasattr(args, "strategy_limit") and args.strategy_limit != 3:
        cfg.strategy_limit = args.strategy_limit
    if hasattr(args, "iterations") and args.iterations != 1:
        cfg.max_iterations = args.iterations

    # Setup structured logging
    from ai_quant_nautilus.logger import setup_logging
    setup_logging(
        level=cfg.log_level,
        log_format=cfg.log_format,
        log_dir=cfg.log_dir,
    )

    from ai_quant_nautilus.cli_commands import dispatch
    # Attach config to args so commands can access it
    args.config = cfg
    return dispatch(args)


if __name__ == "__main__":
    sys.exit(main())
