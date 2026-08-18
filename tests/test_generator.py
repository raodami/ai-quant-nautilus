import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
from ai_quant_nautilus.generator.schema import validate_schema_output, STRATEGY_SCHEMA


class TestSchemaValidation:
    def test_valid_output(self):
        result = {
            "name": "gen_test_strategy",
            "rationale": "This is a valid strategy rationale with enough text",
            "code": "class TestStrat:\n    def method(self):\n        return 1 + 1\n",
            "params": {"rsi_window": 14},
            "expected_edge": "Momentum edge in trending markets",
        }
        valid, errors = validate_schema_output(result)
        assert valid, f"Errors: {errors}"
        assert len(errors) == 0

    def test_missing_fields(self):
        result = {"name": "gen_test"}
        valid, errors = validate_schema_output(result)
        assert not valid
        assert len(errors) >= 3

    def test_invalid_name_pattern(self):
        result = {
            "name": "invalid-name",
            "rationale": "x" * 25,
            "code": "class X:\n    def method(self):\n        return 1\n",
            "params": {},
            "expected_edge": "edge",
        }
        valid, errors = validate_schema_output(result)
        assert not valid
        assert any("格式错误" in e for e in errors) or any("format" in e.lower() for e in errors)

    def test_short_code(self):
        result = {
            "name": "gen_test",
            "rationale": "x" * 25,
            "code": "x=1",
            "params": {},
            "expected_edge": "edge",
        }
        valid, errors = validate_schema_output(result)
        assert not valid
        assert any("过短" in e for e in errors)
