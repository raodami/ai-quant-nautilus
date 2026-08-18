"""
Enhanced CLI commands for ai-quant-nautilus.

Provides commands for live monitoring, data management, and experiment tracking.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_dashboard(args, cfg=None) -> int:
    """Start the monitoring dashboard."""
    from ai_quant_nautilus.monitor import MetricsTracker, MonitorServer

    tracker = MetricsTracker()
    server = MonitorServer(tracker, port=getattr(args, "port", 8080))

    print(f"Starting monitor on http://127.0.0.1:{server.port}")
    print("Press Ctrl+C to stop\n")

    try:
        server.start(background=False)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        server.stop()
    return 0


def cmd_data_quality(args, cfg=None) -> int:
    """Validate OHLCV data quality."""
    from ai_quant_nautilus.data.validator import DataValidator
    from ai_quant_nautilus.data.collector import DataCollector
    import pandas as pd

    symbol = getattr(args, "symbol", None)
    data_dir = Path(getattr(args, "data_dir", "data/raw"))

    validator = DataValidator()

    # Find data files
    patterns = ["*.parquet", "*.csv"]
    files = []
    for pattern in patterns:
        files.extend(data_dir.glob(f"**/{pattern}"))

    if not files:
        print(f"No data files found in {data_dir}")
        return 1

    print(f"Found {len(files)} data files\n")

    for data_file in files:
        try:
            if data_file.suffix == ".parquet":
                df = pd.read_parquet(data_file)
            else:
                df = pd.read_csv(data_file, parse_dates=["timestamp"])
                df = df.set_index("timestamp")

            symbol_name = data_file.stem.replace("_", "/").replace("USDT", "/USDT")
            report = validator.validate(df, symbol_name)

            status = "✅ CLEAN" if report.is_clean() else "⚠️ ISSUES"
            print(f"{status} {symbol_name}")
            print(f"  Rows: {report.total_rows}")
            print(f"  Range: {report.date_range[0][:10]} to {report.date_range[1][:10]}")
            print(f"  Missing: {sum(report.missing_values.values())}")
            print(f"  Anomalies: {report.anomalies_detected}")
            if report.recommendations:
                for rec in report.recommendations[:3]:
                    print(f"  • {rec}")
            print()

        except Exception as e:
            print(f"❌ {data_file.name}: {e}\n")

    return 0


def cmd_experiments(args, cfg=None) -> int:
    """List and manage experiments."""
    from ai_quant_nautilus.config.experiment import ExperimentTracker

    experiments_dir = Path(getattr(cfg, "results_dir", "data/experiments")) if cfg else Path("data/experiments")
    tracker = ExperimentTracker(experiments_dir)

    if hasattr(args, "action") and args.action == "list":
        experiments = tracker.list()
        if not experiments:
            print("No experiments found.")
            return 0

        print(f"{'ID':<20} {'Strategy':<25} {'Status':<10} {'Date':<20} {'Sharpe':<10}")
        print("-" * 85)
        for exp in experiments[:20]:
            print(
                f"{exp['id']:<20} "
                f"{exp.get('strategy_name', 'N/A'):<25} "
                f"{exp.get('status', 'unknown'):<10} "
                f"{datetime.fromtimestamp(exp['timestamp']).strftime('%Y-%m-%d %H:%M'):<20} "
                f"{exp.get('metrics', {}).get('sharpe', 'N/A'):<10}"
            )
        print(f"\nTotal: {len(experiments)} experiments")

    elif hasattr(args, "action") and args.action == "best":
        metric = getattr(args, "metric", "sharpe")
        best = tracker.best_by_metric(metric)
        if best:
            print(f"\nBest experiment by {metric}:")
            print(f"  ID: {best['id']}")
            print(f"  Strategy: {best.get('strategy_name')}")
            print(f"  {metric}: {best.get('metrics', {}).get(metric, 'N/A')}")
            print(f"  Date: {datetime.fromtimestamp(best['timestamp'])}")
        else:
            print("No experiments found.")

    elif hasattr(args, "action") and args.action == "export":
        exp_id = getattr(args, "experiment_id", None)
        if exp_id:
            exp = tracker.get(exp_id)
            if exp:
                output = getattr(args, "output", "experiment.json")
                with open(output, "w") as f:
                    json.dump(exp, f, indent=2, default=str)
                print(f"Exported experiment to {output}")
            else:
                print(f"Experiment {exp_id} not found")
                return 1
        else:
            print("Use --experiment-id to specify which experiment to export")
            return 1

    return 0


def cmd_live(args, cfg=None) -> int:
    """Run live trading simulation."""
    from ai_quant_nautilus.monitor import MetricsTracker, MonitorServer
    from ai_quant_nautilus.simulation.dry_runner import DryRunSimulator

    tracker = MetricsTracker()
    server = MonitorServer(tracker, port=getattr(args, "port", 8081))
    server.start(background=True)

    print(f"Live simulation starting...")
    print(f"Dashboard: http://127.0.0.1:{server.port}/dashboard")

    simulator = DryRunSimulator(tracker=tracker)

    try:
        simulator.run_forever()
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        server.stop()
    return 0
