"""CLI command dispatch."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def dispatch(args) -> int:
    cfg = getattr(args, "config", None)
    if args.command == "status":
        return cmd_status(args, cfg)
    elif args.command == "run":
        return cmd_run(args, cfg)
    elif args.command == "backtest":
        return cmd_backtest(args, cfg)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


def cmd_status(args, cfg=None) -> int:
    registry_path = Path(cfg.registry_path) if cfg else Path("data/registry.json")
    if not registry_path.exists():
        print("No registry found. Run `aqn run` first.")
        return 1
    with open(registry_path) as f:
        reg = json.load(f)
    strategies = reg.get("strategies", [])
    print(f"Total strategies: {len(strategies)}")
    from collections import Counter
    counts = Counter(s.get("status", "unknown") for s in strategies)
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    return 0


def cmd_run(args, cfg=None) -> int:
    from ai_quant_nautilus.orchestrator import Orchestrator
    orch = Orchestrator(
        data_dir=Path(cfg.raw_dir) if cfg else None,
        registry_path=Path(cfg.registry_path) if cfg else None,
    )
    iterations = getattr(args, "iterations", 1)
    strategy_limit = getattr(args, "strategy_limit", cfg.strategy_limit if cfg else 3)
    for i in range(iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {i+1}/{iterations}")
        print(f"{'='*50}")
        result = orch.run_iteration(strategy_limit=strategy_limit)
        print(f"Generated: {result.get('generated', 0)}")
        print(f"Backtested: {result.get('backtested', 0)}")
        print(f"Passed eval: {result.get('passed', 0)}")
    return 0


def cmd_backtest(args, cfg=None) -> int:
    from ai_quant_nautilus.backtest.nautilus_engine import NautilusBacktestAdapter
    instrument = args.instrument or (cfg.instrument if cfg else "ETHUSDT.BINANCE")
    initial_capital = args.initial_capital or (cfg.initial_capital if cfg else 1_000_000.0)
    adapter = NautilusBacktestAdapter()
    result = adapter.run_backtest(
        strategy_path=args.strategy,
        instrument_id=instrument,
        data_path=args.data_path,
        initial_capital=initial_capital,
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0
