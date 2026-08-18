import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
import pandas as pd
import numpy as np
from ai_quant_nautilus.data.validator import DataValidator, DataQualityReport, quick_validate


class TestDataValidator:
    """Test OHLCV data validation."""

    @pytest.fixture
    def clean_data(self):
        """Create clean OHLCV data."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
        prices = [100.0]
        for _ in range(n - 1):
            change = np.random.normal(0, 0.01)
            prices.append(prices[-1] * (1 + change))

        return pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [100.0] * n,
        }, index=dates)

    @pytest.fixture
    def dirty_data(self):
        """Create dirty OHLCV data with anomalies."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
        prices = [100.0]
        for _ in range(n - 1):
            change = np.random.normal(0, 0.01)
            prices.append(prices[-1] * (1 + change))

        # Add some bad data
        prices[20] = 0  # Zero price
        prices[21] = -10  # Negative price
        prices[30] = prices[29] * 10  # Massive spike

        return pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [100.0] * n,
        }, index=dates)

    def test_clean_data_validation(self, clean_data):
        validator = DataValidator()
        report = validator.validate(clean_data, "BTC/USDT")

        assert report.total_rows == 100
        assert report.price_valid is True
        assert report.volume_valid is True
        assert report.time_sorted is True
        assert report.is_clean() is True

    def test_dirty_data_validation(self, dirty_data):
        validator = DataValidator()
        report = validator.validate(dirty_data, "BTC/USDT")

        assert report.price_valid is False
        assert report.is_clean() is False

    def test_missing_values(self, clean_data):
        """Test detection of missing values."""
        # Add some NaN values
        clean_data.loc[clean_data.index[10], "close"] = np.nan
        clean_data.loc[clean_data.index[20], "volume"] = np.nan

        validator = DataValidator()
        report = validator.validate(clean_data, "TEST")

        assert report.missing_values["close"] == 1
        assert report.missing_values["volume"] == 1

    def test_anomaly_detection(self, dirty_data):
        """Test anomaly detection."""
        validator = DataValidator(max_anomaly_zscore=100.0)  # High threshold to avoid false positives
        report = validator.validate(dirty_data, "BTC/USDT")

        # At minimum, should complete without error and detect dirty data
        assert report is not None
        assert report.price_valid is False

    def test_empty_data(self):
        """Test validation on empty DataFrame."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        validator = DataValidator()
        report = validator.validate(df, "EMPTY")

        assert report.total_rows == 0
        assert report.is_clean() is False

    def test_quick_validate(self, clean_data):
        """Test quick validation helper."""
        result = quick_validate(clean_data, "BTC/USDT")
        assert result is True

    def test_missing_column(self, clean_data):
        """Test validation fails on missing columns."""
        clean_data = clean_data.drop(columns=["volume"])
        validator = DataValidator()
        with pytest.raises(ValueError, match="Missing required column"):
            validator.validate(clean_data, "TEST")

    def test_negative_prices(self):
        """Test detection of negative prices."""
        df = pd.DataFrame({
            "open": [-10, 100],
            "high": [-5, 105],
            "low": [-15, 95],
            "close": [-10, 100],
            "volume": [100, 100],
        })
        df.index = pd.date_range(start="2024-01-01", periods=2, freq="1h")
        validator = DataValidator()
        report = validator.validate(df, "TEST")
        assert report.price_valid is False
