"""CLI command dispatch."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def dispatch(args) -> int:
    if args.command == "status":
        return cmd_status(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "backtest":
        return cmd_backtest(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


def cmd_status(args) -> int:
    registry_path = PROJECT_ROOT / "data" / "registry.json"
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


def cmd_run(args) -> int:
    from ai_quant_nautilus.orchestrator import Orchestrator
    orch = Orchestrator(
        data_dir=args.data_dir,
        registry_path=args.registry,
    )
    for i in range(args.iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {i+1}/{args.iterations}")
        print(f"{'='*50}")
        result = orch.run_iteration(strategy_limit=args.strategy_limit)
        print(f"Generated: {result.get('generated', 0)}")
        print(f"Backtested: {result.get('backtested', 0)}")
        print(f"Passed eval: {result.get('passed', 0)}")
    return 0


def cmd_backtest(args) -> int:
    from ai_quant_nautilus.backtest.nautilus_adapter import NautilusBacktestAdapter
    adapter = NautilusBacktestAdapter()
    result = adapter.run_backtest(
        strategy_path=args.strategy,
        instrument_id=args.instrument,
        data_path=args.data_path,
        initial_capital=args.initial_capital,
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0
