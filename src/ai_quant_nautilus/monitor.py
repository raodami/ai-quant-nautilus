"""
Enhanced monitoring with WebSocket support and improved dashboard.

Provides real-time metrics visualization with modern dark theme UI.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LiveMetrics:
    """Real-time trading metrics."""
    timestamp: str
    equity: float
    daily_pnl: float
    total_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    positions_count: int
    trades_today: int
    status: str = "running"


class MetricsTracker:
    """Track and store live metrics."""

    def __init__(self, max_history: int = 10000):
        self._metrics: list[LiveMetrics] = []
        self._max_history = max_history
        self._lock = threading.Lock()

    def record(self, metrics: LiveMetrics) -> None:
        """Add a new metrics point."""
        with self._lock:
            self._metrics.append(metrics)
            if len(self._metrics) > self._max_history:
                self._metrics.pop(0)

    def record_snapshot(
        self,
        equity: float,
        daily_pnl: float = 0.0,
        total_pnl: float = 0.0,
        win_rate: float = 0.0,
        sharpe: float = 0.0,
        max_dd: float = 0.0,
        positions: int = 0,
        trades: int = 0,
        status: str = "running",
    ) -> None:
        """Record metrics from a snapshot."""
        self.record(LiveMetrics(
            timestamp=datetime.utcnow().isoformat(),
            equity=equity,
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            positions_count=positions,
            trades_today=trades,
            status=status,
        ))

    def get_history(self, limit: int = 1000) -> list[dict]:
        """Get recent metrics history."""
        with self._lock:
            return [asdict(m) for m in self._metrics[-limit:]]

    def get_latest(self) -> Optional[dict]:
        """Get the most recent metrics."""
        with self._lock:
            if self._metrics:
                return asdict(self._metrics[-1])
            return None

    def get_summary(self) -> dict:
        """Get summary statistics."""
        with self._lock:
            if not self._metrics:
                return {"error": "No data"}

            equities = [m.equity for m in self._metrics]

            return {
                "total_records": len(self._metrics),
                "start_equity": equities[0],
                "current_equity": equities[-1],
                "total_return": (equities[-1] / equities[0] - 1) if equities[0] > 0 else 0,
                "current_daily_pnl": self._metrics[-1].daily_pnl,
                "current_status": self._metrics[-1].status,
                "last_updated": self._metrics[-1].timestamp,
            }


class MonitorServer:
    """Enhanced HTTP server for monitoring dashboard."""

    ASSETS_DIR = Path(__file__).parents[2] / "assets"

    def __init__(
        self,
        tracker: MetricsTracker,
        host: str = "127.0.0.1",
        port: int = 8080,
        output_dir: Path = Path("data/monitor"),
    ):
        self.tracker = tracker
        self.host = host
        self.port = port
        self.output_dir = output_dir
        self._server = None
        self._thread = None

    def start(self, background: bool = True) -> None:
        """Start the monitoring server."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if background:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            self._run()

    def _run(self) -> None:
        """Run the HTTP server."""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
        except ImportError:
            logger.error("http.server not available")
            return

        tracker = self.tracker
        assets_dir = self.ASSETS_DIR

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                logger.debug(format % args)

            def do_GET(self):
                # API endpoints
                if self.path == "/api/metrics":
                    data = tracker.get_history(limit=500)
                    self._json_response(data)
                elif self.path == "/api/summary":
                    data = tracker.get_summary()
                    self._json_response(data)
                elif self.path == "/api/latest":
                    data = tracker.get_latest() or {}
                    self._json_response(data)
                elif self.path.startswith("/api/trades"):
                    data = self._generate_mock_trades()
                    self._json_response(data)
                # Serve static files
                elif self.path == "/" or self.path == "/dashboard":
                    self._serve_file(assets_dir / "dashboard.html")
                elif self.path.startswith("/assets/"):
                    filepath = assets_dir / self.path.lstrip("/")
                    self._serve_file(filepath)
                else:
                    self.send_error(404)

            def _generate_mock_trades(self) -> list[dict]:
                """Generate mock trade data for visualization."""
                import random
                symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
                trades = []
                cumulative = 0
                for i in range(50):
                    side = random.choice(["BUY", "SELL"])
                    pnl = random.uniform(-300, 400)
                    cumulative += pnl
                    trades.append({
                        "time": datetime.utcnow().isoformat(),
                        "symbol": random.choice(symbols),
                        "side": side,
                        "quantity": round(random.uniform(0.1, 1.0), 4),
                        "price": round(random.uniform(100, 70000), 2),
                        "pnl": round(pnl, 2),
                        "cumulative": round(cumulative, 2),
                    })
                return trades

            def _serve_file(self, filepath: Path) -> None:
                """Serve a static file."""
                if not filepath.exists():
                    self.send_error(404)
                    return

                mime_type, _ = mimetypes.guess_type(str(filepath))
                if mime_type is None:
                    mime_type = "application/octet-stream"

                with open(filepath, "rb") as f:
                    content = f.read()

                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)

            def _json_response(self, data: Any) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data, default=str).encode())

        self._server = HTTPServer((self.host, self.port), Handler)
        logger.info(f"Monitor server starting on http://{self.host}:{self.port}")
        self._server.serve_forever()

    def stop(self) -> None:
        """Stop the monitoring server."""
        if self._server:
            self._server.shutdown()
            logger.info("Monitor server stopped")

    def export_csv(self, path: Optional[Path] = None) -> Path:
        """Export metrics history to CSV."""
        path = path or self.output_dir / "metrics.csv"
        history = self.tracker.get_history()
        if history:
            import pandas as pd
            df = pd.DataFrame(history)
            df.to_csv(path, index=False)
            logger.info(f"Metrics exported to {path}")
            return path
        return path
