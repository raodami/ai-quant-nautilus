"""
Experiment tracker: records results with caching for idempotent re-runs.

Stores experiment metadata and metrics as JSON lines, with optional
filesystem cache to skip re-computation when inputs haven't changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRecord:
    """Single experiment run record."""

    experiment_id: str
    timestamp: float
    config_snapshot: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # running | completed | failed | cached
    error: Optional[str] = None
    cache_key: Optional[str] = None
    cached_at: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        ts = data.pop("timestamp", None)
        if ts and isinstance(ts, str):
            data["timestamp"] = datetime.fromisoformat(ts).timestamp()
        return cls(**data)


class ExperimentTracker:
    """
    Tracks experiments: records results, supports cache lookup, and persists to disk.

    Usage:
        tracker = ExperimentTracker(results_dir="data/experiments")
        with tracker.run("my-exp") as rec:
            rec.metrics["sharpe"] = 1.23
            rec.status = "completed"
    """

    def __init__(
        self,
        results_dir: str = "data/experiments",
        cache_enabled: bool = True,
        cache_ttl_hours: int = 24,
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled
        self.cache_ttl_hours = cache_ttl_hours
        self._cache: dict[str, ExperimentRecord] = {}
        self._load_cache()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    class _RunContext:
        def __init__(self, tracker: "ExperimentTracker", experiment_id: str) -> None:
            self.tracker = tracker
            self.experiment_id = experiment_id
            self.record: Optional[ExperimentRecord] = None

        def __enter__(self) -> ExperimentRecord:
            cache_key = self.tracker._make_cache_key(self.experiment_id)
            if self.tracker.cache_enabled and cache_key in self.tracker._cache:
                rec = self.tracker._cache[cache_key]
                logger.info("Cache hit for experiment %s", self.experiment_id)
                self.record = rec
                rec.status = "cached"
                rec.cached_at = time.time()
                return rec

            rec = ExperimentRecord(
                experiment_id=self.experiment_id,
                timestamp=time.time(),
                config_snapshot={},
                status="running",
                cache_key=cache_key,
            )
            self.record = rec
            return rec

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            if self.record is None:
                return
            if exc_type is not None:
                self.record.status = "failed"
                self.record.error = str(exc_val)
            else:
                self.record.status = "completed"
            self.tracker._save(self.record)
            self.tracker._write_event_log(self.record)
            logger.info(
                "Experiment %s done: status=%s",
                self.experiment_id,
                self.record.status,
            )

    def run(self, experiment_id: str) -> _RunContext:
        """Context manager for a single experiment run."""
        return self._RunContext(self, experiment_id)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Retrieve a record by ID."""
        path = self.results_dir / f"{experiment_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return ExperimentRecord.from_dict(json.load(f))

    def list_experiments(self, limit: int = 20) -> list[ExperimentRecord]:
        """List recent experiments, newest first."""
        records = []
        for path in sorted(self.results_dir.glob("*.json"), reverse=True):
            if path.name == "events.log":
                continue
            try:
                with open(path) as f:
                    records.append(ExperimentRecord.from_dict(json.load(f)))
            except (json.JSONDecodeError, KeyError):
                continue
            if len(records) >= limit:
                break
        return records

    def best_by_metric(self, metric: str, limit: int = 5) -> list[ExperimentRecord]:
        """Return experiments sorted by a numeric metric (descending)."""
        records = self.list_experiments(limit=100)
        valid = [
            r for r in records
            if r.status == "completed" and metric in r.metrics
        ]
        valid.sort(key=lambda r: r.metrics[metric], reverse=True)
        return valid[:limit]

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _make_cache_key(self, experiment_id: str) -> str:
        """Deterministic cache key from experiment ID."""
        return hashlib.sha256(experiment_id.encode()).hexdigest()[:16]

    def _load_cache(self) -> None:
        """Load cache index from disk if available."""
        if not self.cache_enabled:
            return
        cache_file = self.results_dir / ".cache_index.json"
        if not cache_file.exists():
            return
        try:
            with open(cache_file) as f:
                data = json.load(f)
            now = time.time()
            ttl_seconds = self.cache_ttl_hours * 3600
            self._cache = {}
            for entry in data:
                if now - entry.get("cached_at", 0) > ttl_seconds:
                    continue
                self._cache[entry["cache_key"]] = ExperimentRecord.from_dict(entry)
            logger.debug("Loaded %d cache entries", len(self._cache))
        except Exception as e:
            logger.warning("Failed to load cache index: %s", e)

    def _save_cache_index(self) -> None:
        """Persist cache index to disk."""
        if not self.cache_enabled:
            return
        cache_file = self.results_dir / ".cache_index.json"
        entries = [r.to_dict() for r in self._cache.values()]
        with open(cache_file, "w") as f:
            json.dump(entries, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, record: ExperimentRecord) -> None:
        """Save a single experiment record to JSON."""
        path = self.results_dir / f"{record.experiment_id}.json"
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2, default=str)
        # Update cache
        if self.cache_enabled and record.cache_key:
            self._cache[record.cache_key] = record
            self._save_cache_index()

    def _write_event_log(self, record: ExperimentRecord) -> None:
        """Append a JSON line to the experiment event log."""
        log_path = self.results_dir / "events.log"
        with open(log_path, "a") as f:
            f.write(json.dumps(record.to_dict(), default=str) + "\n")

    def export_summary(self, output_path: Optional[Path] = None) -> dict[str, Any]:
        """Generate a summary report of all experiments."""
        records = self.list_experiments(limit=100)
        total = len(records)
        completed = sum(1 for r in records if r.status == "completed")
        failed = sum(1 for r in records if r.status == "failed")
        cached = sum(1 for r in records if r.status == "cached")

        # Aggregate common metrics
        metric_sums: dict[str, list[float]] = {}
        for r in records:
            if r.status != "completed":
                continue
            for k, v in r.metrics.items():
                if isinstance(v, (int, float)):
                    metric_sums.setdefault(k, []).append(v)

        aggregates = {
            k: {
                "mean": sum(v) / len(v),
                "min": min(v),
                "max": max(v),
                "count": len(v),
            }
            for k, v in metric_sums.items()
        }

        summary = {
            "total_experiments": total,
            "completed": completed,
            "failed": failed,
            "cached": cached,
            "aggregates": aggregates,
            "top_strategies": self.best_by_metric("sharpe_ratio", limit=5),
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info("Summary written to %s", output_path)

        return summary
