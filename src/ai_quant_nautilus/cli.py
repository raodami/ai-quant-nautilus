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
    sub = parser.add_subparsers(dest="command")

    # run: full simulation
    p_run = sub.add_parser("run", help="Run one full iteration (generate → backtest → evaluate)")
    p_run.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    p_run.add_argument("--strategy-limit", type=int, default=3, help="Max strategies per iteration")
    p_run.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw", help="OHLCV data dir")
    p_run.add_argument("--registry", type=Path, default=PROJECT_ROOT / "data" / "registry.json", help="Strategy registry path")

    # backtest: run nautilus backtest on a strategy file
    p_bt = sub.add_parser("backtest", help="Run nautilus backtest on a strategy")
    p_bt.add_argument("--strategy", type=Path, required=True, help="Strategy Python file")
    p_bt.add_argument("--instrument", type=str, default="ETHUSDT.BINANCE", help="Instrument ID")
    p_bt.add_argument("--data-path", type=Path, required=True, help="CSV/parquet data file")
    p_bt.add_argument("--initial-capital", type=float, default=1_000_000.0, help="Starting USDT")

    # status: show current registry
    sub.add_parser("status", help="Show strategy registry status")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    from ai_quant_nautilus.cli_commands import dispatch
    return dispatch(args)


if __name__ == "__main__":
    sys.exit(main())
