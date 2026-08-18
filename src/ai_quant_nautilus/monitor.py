"""
Live monitoring dashboard for backtest and trading results.

Provides real-time metrics visualization via web interface.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

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
        self._start_time = time.time()

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
            pnls = [m.total_pnl for m in self._metrics]

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
    """Simple HTTP server for monitoring dashboard."""

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

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/metrics":
                    data = tracker.get_history(limit=500)
                    self._json_response(data)
                elif self.path == "/api/summary":
                    data = tracker.get_summary()
                    self._json_response(data)
                elif self.path == "/api/latest":
                    data = tracker.get_latest() or {}
                    self._json_response(data)
                elif self.path == "/dashboard":
                    self._serve_dashboard()
                else:
                    self.send_error(404)

            def _json_response(self, data: Any) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data, default=str).encode())

            def _serve_dashboard(self) -> None:
                html = self._generate_html()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())

            def _generate_html(self) -> str:
                return """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>AI Quant Monitor</title>
                    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
                    <style>
                        body { font-family: -apple-system, sans-serif; margin: 0; padding: 20px; background: #0f172a; color: #e2e8f0; }
                        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
                        .card { background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
                        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
                        .metric { background: #334155; padding: 15px; border-radius: 8px; }
                        .metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; }
                        .metric-value { font-size: 24px; font-weight: bold; color: #6366f1; }
                        .metric-value.positive { color: #22c55e; }
                        .metric-value.negative { color: #ef4444; }
                        #chart { width: 100%; height: 400px; }
                        .status { padding: 8px 16px; border-radius: 20px; font-size: 14px; }
                        .status.running { background: #22c55e33; color: #22c55e; }
                        .status.stopped { background: #ef444433; color: #ef4444; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>AI Quant Monitor</h1>
                        <span id="status" class="status running">Running</span>
                    </div>
                    <div class="card">
                        <div class="metrics">
                            <div class="metric"><div class="metric-label">Equity</div><div class="metric-value" id="equity">-</div></div>
                            <div class="metric"><div class="metric-label">Daily PnL</div><div class="metric-value" id="pnl">-</div></div>
                            <div class="metric"><div class="metric-label">Win Rate</div><div class="metric-value" id="winrate">-</div></div>
                            <div class="metric"><div class="metric-label">Sharpe</div><div class="metric-value" id="sharpe">-</div></div>
                            <div class="metric"><div class="metric-label">Max DD</div><div class="metric-value negative" id="dd">-</div></div>
                        </div>
                    </div>
                    <div class="card">
                        <div id="chart"></div>
                    </div>
                    <script>
                        let data = [];
                        async function fetchMetrics() {
                            const resp = await fetch('/api/metrics');
                            data = await resp.json();
                            updateDisplay();
                        }
                        function updateDisplay() {
                            if (!data.length) return;
                            const latest = data[data.length - 1];
                            document.getElementById('equity').textContent = '$' + latest.equity.toFixed(2);
                            const pnlEl = document.getElementById('pnl');
                            pnlEl.textContent = (latest.daily_pnl >= 0 ? '+' : '') + '$' + latest.daily_pnl.toFixed(2);
                            pnlEl.className = 'metric-value ' + (latest.daily_pnl >= 0 ? 'positive' : 'negative');
                            document.getElementById('winrate').textContent = (latest.win_rate * 100).toFixed(1) + '%';
                            document.getElementById('sharpe').textContent = latest.sharpe_ratio.toFixed(2);
                            document.getElementById('dd').textContent = (latest.max_drawdown * 100).toFixed(2) + '%';
                            const equities = data.map(d => d.equity);
                            const timestamps = data.map(d => d.timestamp.slice(11, 19));
                            Plotly.newPlot('chart', [{x: timestamps, y: equities, type: 'scatter', mode: 'lines'}], {paper_bgcolor: '#1e293b', plot_bgcolor: '#1e293b', font: {color: '#e2e8f0'}});
                        }
                        fetchMetrics();
                        setInterval(fetchMetrics, 5000);
                    </script>
                </body>
                </html>
                """

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
            df = pd.DataFrame(history)
            df.to_csv(path, index=False)
            logger.info(f"Metrics exported to {path}")
            return path
        return path
