import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
from unittest.mock import MagicMock, patch
from ai_quant_nautilus.monitor import MetricsTracker, LiveMetrics, MonitorServer
from datetime import datetime


class TestMetricsTracker:
    """Test metrics tracking."""

    def test_record_and_get(self):
        """Test recording and retrieving metrics."""
        tracker = MetricsTracker()
        metrics = LiveMetrics(
            timestamp=datetime.utcnow().isoformat(),
            equity=1000000.0,
            daily_pnl=5000.0,
            total_pnl=10000.0,
            win_rate=0.55,
            sharpe_ratio=1.2,
            max_drawdown=-0.05,
            positions_count=2,
            trades_today=10,
        )
        tracker.record(metrics)

        history = tracker.get_history()
        assert len(history) == 1
        assert history[0]["equity"] == 1000000.0

    def test_get_latest(self):
        """Test getting latest metrics."""
        tracker = MetricsTracker()
        tracker.record(LiveMetrics(
            timestamp="2024-01-01T00:00:00",
            equity=1000000.0,
            daily_pnl=0.0,
            total_pnl=0.0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            positions_count=0,
            trades_today=0,
        ))
        tracker.record(LiveMetrics(
            timestamp="2024-01-01T01:00:00",
            equity=1050000.0,
            daily_pnl=5000.0,
            total_pnl=50000.0,
            win_rate=0.55,
            sharpe_ratio=1.2,
            max_drawdown=-0.05,
            positions_count=2,
            trades_today=10,
        ))

        latest = tracker.get_latest()
        assert latest is not None
        assert latest["equity"] == 1050000.0

    def test_get_empty(self):
        """Test getting latest from empty tracker."""
        tracker = MetricsTracker()
        assert tracker.get_latest() is None

    def test_summary(self):
        """Test summary statistics."""
        tracker = MetricsTracker()
        tracker.record(LiveMetrics(
            timestamp="2024-01-01T00:00:00",
            equity=1000000.0,
            daily_pnl=0.0,
            total_pnl=0.0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            positions_count=0,
            trades_today=0,
        ))
        tracker.record(LiveMetrics(
            timestamp="2024-01-01T01:00:00",
            equity=1100000.0,
            daily_pnl=10000.0,
            total_pnl=100000.0,
            win_rate=0.6,
            sharpe_ratio=1.5,
            max_drawdown=-0.08,
            positions_count=3,
            trades_today=20,
        ))

        summary = tracker.get_summary()
        assert summary["start_equity"] == 1000000.0
        assert summary["current_equity"] == 1100000.0
        assert summary["total_return"] == pytest.approx(0.1, abs=0.001)

    def test_max_history_limit(self):
        """Test history length limit."""
        tracker = MetricsTracker(max_history=5)
        for i in range(10):
            tracker.record(LiveMetrics(
                timestamp="2024-01-01",
                equity=float(i),
                daily_pnl=0.0,
                total_pnl=0.0,
                win_rate=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                positions_count=0,
                trades_today=0,
            ))

        history = tracker.get_history()
        assert len(history) <= 5

    def test_record_snapshot(self):
        """Test convenience snapshot recording."""
        tracker = MetricsTracker()
        tracker.record_snapshot(
            equity=1000000.0,
            daily_pnl=5000.0,
            total_pnl=10000.0,
            win_rate=0.5,
            sharpe=1.5,
            max_dd=-0.1,
            positions=3,
            trades=15,
        )

        latest = tracker.get_latest()
        assert latest is not None
        assert latest["equity"] == 1000000.0
        assert latest["positions_count"] == 3

    def test_multiple_records(self):
        """Test recording multiple metrics."""
        tracker = MetricsTracker()
        for i in range(100):
            tracker.record_snapshot(
                equity=1000000 + i * 1000,
            )

        history = tracker.get_history()
        assert len(history) == 100
        assert history[-1]["equity"] == 1099000.0


class TestMonitorServer:
    """Test monitor server."""

    @patch("ai_quant_nautilus.monitor.HTTPServer")
    def test_server_creation(self, mock_server_class):
        """Test server can be created."""
        tracker = MetricsTracker()
        server = MonitorServer(tracker, port=9999)

        assert server.port == 9999
        assert server.host == "127.0.0.1"

    def test_assets_directory(self):
        """Test assets directory exists."""
        assets_dir = MonitorServer.ASSETS_DIR
        dashboard = assets_dir / "dashboard.html"
        # If assets dir doesn't exist yet, skip this check
        if not assets_dir.exists():
            pytest.skip("Assets directory not found")
        assert dashboard.exists()

    def test_export_csv_no_data(self, tmp_path):
        """Test CSV export with no data."""
        tracker = MetricsTracker()
        server = MonitorServer(tracker, output_dir=tmp_path)
        result = server.export_csv()
        assert result.suffix == ".csv"

    def test_export_csv_with_data(self, tmp_path):
        """Test CSV export with data."""
        tracker = MetricsTracker()
        tracker.record_snapshot(equity=1000000.0)

        server = MonitorServer(tracker, output_dir=tmp_path)
        result = server.export_csv(tmp_path / "test.csv")

        assert result.exists()
        content = result.read_text()
        assert "equity" in content
        assert "1000000" in content
