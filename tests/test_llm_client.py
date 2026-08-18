import sys
sys.path.insert(0, "D:/ai-quant-nautilus/src")

import pytest
from ai_quant_nautilus.generator.llm_client import (
    LLMResponse,
    BaseLLMClient,
    DeepSeekClient,
    OpenAICompatClient,
    MockLLMClient,
    create_llm_client,
)
import json


class TestLLMClient:
    """Test LLM client interfaces."""

    def test_mock_client_basic(self):
        """Test mock LLM client returns valid strategies."""
        client = MockLLMClient()
        response = client.generate("system prompt", "user prompt")

        assert isinstance(response, LLMResponse)
        assert response.model == "mock-llm"
        assert response.content
        assert response.usage is not None

    def test_mock_client_json(self):
        """Test mock client returns valid JSON."""
        client = MockLLMClient()
        result = client.generate_json("system", "user")

        assert isinstance(result, dict)
        assert "name" in result
        assert "code" in result
        assert "params" in result
        assert "rationale" in result
        assert "expected_edge" in result

    def test_mock_client_sequential(self):
        """Test mock client cycles through strategies."""
        client = MockLLMClient()

        result1 = client.generate_json("system", "user")
        result2 = client.generate_json("system", "user")

        # Should get different strategies
        assert result1["name"] != result2["name"]

    def test_mock_client_reset(self):
        """Test mock client reset."""
        client = MockLLMClient()
        client.generate_json("system", "user")
        client.reset()

        result = client.generate_json("system", "user")
        assert result["name"] == client.MOCK_STRATEGIES[0]["name"]

    def test_create_client_mock(self):
        """Test factory function with mock provider."""
        client = create_llm_client(provider="mock")
        assert isinstance(client, MockLLMClient)

    def test_create_client_deepseek(self):
        """Test factory function with deepseek provider."""
        with pytest.raises(ValueError, match="DeepSeek API key"):
            create_llm_client(provider="deepseek")

    def test_create_client_invalid(self):
        """Test factory function with invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client(provider="invalid")

    def test_response_to_dict(self):
        """Test LLMResponse serialization."""
        response = LLMResponse(
            content="test content",
            model="test-model",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
        d = response.to_dict()
        assert d["content"] == "test content"
        assert d["model"] == "test-model"
        assert d["usage"]["prompt_tokens"] == 10
        assert d["usage"]["completion_tokens"] == 20

    def test_mock_strategy_format(self):
        """Test that mock strategies have correct format."""
        client = MockLLMClient()
        result = client.generate_json("system", "user")

        # Verify all required fields
        assert isinstance(result["name"], str)
        assert result["name"].startswith("gen_")
        assert len(result["rationale"]) >= 30
        assert "class" in result["code"]
        assert "Strategy" in result["code"]
        assert isinstance(result["params"], dict)
        assert len(result["expected_edge"]) > 0


class TestMockStrategies:
    """Test the mock strategy templates."""

    def test_has_three_strategies(self):
        """Test that we have exactly 3 mock strategies."""
        assert len(MockLLMClient.MOCK_STRATEGIES) == 3

    def test_ema_strategy(self):
        """Test EMA strategy template."""
        strategies = MockLLMClient.MOCK_STRATEGIES
        ema = next(s for s in strategies if "ema" in s["name"].lower())
        assert "EMA" in ema["code"] or "ema" in ema["code"]
        assert "fast_period" in ema["params"]
        assert "slow_period" in ema["params"]

    def test_rsi_strategy(self):
        """Test RSI strategy template."""
        strategies = MockLLMClient.MOCK_STRATEGIES
        rsi = next(s for s in strategies if "rsi" in s["name"].lower())
        assert "RSI" in rsi["code"] or "rsi" in rsi["code"]
        assert "rsi_period" in rsi["params"]
        assert "oversold" in rsi["params"]
        assert "overbought" in rsi["params"]

    def test_macd_strategy(self):
        """Test MACD strategy template."""
        strategies = MockLLMClient.MOCK_STRATEGIES
        macd = next(s for s in strategies if "macd" in s["name"].lower())
        assert "MACD" in macd["code"] or "macd" in macd["code"]
        assert "fast" in macd["params"]
        assert "slow" in macd["params"]
        assert "signal" in macd["params"]
