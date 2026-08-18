import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
from ai_quant_nautilus.risk.allocator import risk_parity_weights, allocate_capital


class TestRiskParity:
    def test_equal_vols(self):
        vols = [0.2, 0.2, 0.2]
        weights = risk_parity_weights(vols)
        assert abs(sum(weights) - 1.0) < 0.001
        assert all(abs(w - 1/3) < 0.001 for w in weights)

    def test_different_vols(self):
        vols = [0.1, 0.2, 0.4]
        weights = risk_parity_weights(vols)
        # Lower volatility should get higher weight
        assert weights[0] > weights[1] > weights[2]
        assert abs(sum(weights) - 1.0) < 0.001

    def test_empty_list(self):
        assert risk_parity_weights([]) == [1.0]

    def test_zero_volatility(self):
        vols = [0, 0.2, 0.3]
        weights = risk_parity_weights(vols)
        # Zero vol gets zero weight
        assert weights[0] == 0.0
        assert abs(sum(weights) - 1.0) < 0.001

    def test_allocate_capital(self):
        result = allocate_capital(1000000.0, {"s1": 0.2, "s2": 0.4})
        assert abs(sum(result.values()) - 1000000.0) < 0.01
        assert result["s1"] > result["s2"]  # lower vol gets more
