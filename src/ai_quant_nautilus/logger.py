"""
Structured logger: JSON-formatted logs with context enrichment.

Provides:
  - get_logger(name) → Logger with consistent JSON formatting
  - setup_logging(config) → Configure root logger from Config object
  - log_experiment(event) → Write a structured experiment event to log file
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Log formatter that emits one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Attach exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exc"] = self.formatException(record.exc_info)
        # Attach any extra fields that were passed via extra=
        for key in ("experiment_id", "strategy_id", "iteration", "metric", "value"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class PrettyFormatter(logging.Formatter):
    """Human-readable multi-line formatter for console output."""

    _LEVEL_COLORS = {
        "DEBUG": "\033[36m",   # cyan
        "INFO": "\033[32m",    # green
        "WARNING": "\033[33m", # yellow
        "ERROR": "\033[31m",   # red
        "CRITICAL": "\033[35m",# magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        color = self._LEVEL_COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname}{self._RESET}"
        msg = record.getMessage()
        # Include experiment/strategy context if present
        ctx_parts = []
        for attr in ("experiment_id", "strategy_id", "iteration"):
            if hasattr(record, attr):
                ctx_parts.append(f"{attr}={getattr(record, attr)}")
        ctx = f" [{', '.join(ctx_parts)}]" if ctx_parts else ""
        exc = ""
        if record.exc_info and record.exc_info[0] is not None:
            exc = "\n" + self.formatException(record.exc_info)
        return f"{ts} {level} {record.name}{ctx}\n  {msg}{exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with the project's standard handlers."""
    logger = logging.getLogger(name)
    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(PrettyFormatter())
    logger.addHandler(console)

    # File handler (rotating) — added only if setup_logging was called
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler("logs/app.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_dir: str = "logs",
    enable_file: bool = True,
) -> None:
    """
    Configure the root logger based on a Config object or individual params.

    Creates the log directory if it doesn't exist.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Determine formatter
    fmt = JSONFormatter() if log_format == "json" else PrettyFormatter()

    # Root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove old handlers to avoid duplicates on re-init
    root.handlers.clear()

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(PrettyFormatter())
    root.addHandler(console)

    # File
    if enable_file:
        log_file = Path(log_dir) / "app.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

        # Separate experiment log
        exp_file = Path(log_dir) / "experiments.log"
        eh = logging.FileHandler(exp_file, encoding="utf-8")
        eh.setLevel(logging.DEBUG)
        eh.setFormatter(JSONFormatter())
        eh.addFilter(lambda r: hasattr(r, "experiment_id"))
        root.addHandler(eh)

    logging.info("Logging initialized: level=%s format=%s", level, log_format)


def log_experiment(
    experiment_id: str,
    event: str,
    **kwargs: Any,
) -> None:
    """
    Write a structured experiment event to the experiment log.

    Usage:
        log_experiment("exp-001", "backtest_done", sharpe=1.23, trades=47)
    """
    logger = logging.getLogger("ai_quant_nautilus.experiment")
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "",
        0,
        f"event={event}",
        (),
        None,
    )
    record.experiment_id = experiment_id
    for k, v in kwargs.items():
        setattr(record, k, v)
    logger.handle(record)
    logger.info(
        "experiment_id=%s event=%s %s",
        experiment_id,
        event,
        json.dumps(kwargs, default=str),
        extra={"experiment_id": experiment_id},
    )
