"""
Experiment tracker for ai-quant-nautilus.

Records backtest results, optimizations, and generates summary reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRecord:
    """Single experiment result."""
    experiment_id: str
    timestamp: str
    strategy_name: str
    result_type: str  # backtest, optimization, live
    
    # Metrics
    metrics: dict[str, Any] = field(default_factory=dict)
    
    # Config used
    config: dict[str, Any] = field(default_factory=dict)
    
    # Additional data
    notes: str = ""
    artifacts: list[str] = field(default_factory=list)


class ExperimentTracker:
    """Track and manage experiments."""
    
    def __init__(self, output_dir: str = "output/experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[ExperimentRecord] = []
    
    def log_backtest(
        self,
        strategy_name: str,
        result: Any,
        config: Optional[dict] = None,
        notes: str = "",
    ) -> str:
        """Log a backtest experiment."""
        record = ExperimentRecord(
            experiment_id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            strategy_name=strategy_name,
            result_type="backtest",
            metrics=result.to_dict() if hasattr(result, 'to_dict') else vars(result),
            config=config or {},
            notes=notes,
        )
        self._records.append(record)
        self._save_record(record)
        logger.info(f"Logged backtest: {record.experiment_id}")
        return record.experiment_id
    
    def log_optimization(
        self,
        strategy_name: str,
        result: Any,
        config: Optional[dict] = None,
        notes: str = "",
    ) -> str:
        """Log an optimization experiment."""
        record = ExperimentRecord(
            experiment_id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            strategy_name=strategy_name,
            result_type="optimization",
            metrics=result.to_dict() if hasattr(result, 'to_dict') else vars(result),
            config=config or {},
            notes=notes,
        )
        self._records.append(record)
        self._save_record(record)
        return record.experiment_id
    
    def _save_record(self, record: ExperimentRecord) -> None:
        """Save experiment record to JSON file."""
        file_path = self.output_dir / f"{record.experiment_id}.json"
        data = asdict(record)
        file_path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    
    def _generate_id(self) -> str:
        """Generate unique experiment ID."""
        import uuid
        return uuid.uuid4().hex[:8]
    
    def list_experiments(self, result_type: Optional[str] = None) -> list[dict]:
        """List all experiments."""
        records = self._records
        if result_type:
            records = [r for r in records if r.result_type == result_type]
        return [asdict(r) for r in records]
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Get a specific experiment by ID."""
        for record in self._records:
            if record.experiment_id == experiment_id:
                return record
        return None
    
    def export_summary(self, output_path: str) -> Path:
        """Export summary of all experiments."""
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_experiments": len(self._records),
            "by_type": {},
            "experiments": [asdict(r) for r in self._records],
        }
        
        for record in self._records:
            rt = record.result_type
            summary["by_type"][rt] = summary["by_type"].get(rt, 0) + 1
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
        
        logger.info(f"Exported summary to {path}")
        return path


def create_tracker(output_dir: str = "output/experiments") -> ExperimentTracker:
    """Create and return an experiment tracker."""
    return ExperimentTracker(output_dir)
