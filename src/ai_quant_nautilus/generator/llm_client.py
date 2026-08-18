"""
LLM client interfaces for strategy generation.

Supports: DeepSeek, OpenAI, Anthropic, and local models via OpenAI-compatible API.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    model: str
    finish_reason: str = "stop"
    usage: dict[str, int] | None = None

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
        }


class BaseLLMClient(ABC):
    """Base class for LLM clients."""

    def __init__(self, model: str = "deepseek-chat", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Generate and parse JSON response."""
        response = self.generate(system_prompt, user_prompt)
        content = response.content.strip()

        # Extract JSON from markdown code blocks if present
        if "```" in content:
            lines = content.split("\n")
            in_code = False
            json_lines = []
            for line in lines:
                if line.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    json_lines.append(line)
            content = "\n".join(json_lines)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {content[:500]}")
            raise


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
    ):
        super().__init__(model=model, temperature=temperature)
        self.api_key = api_key or self._get_api_key()
        self.base_url = base_url

    @staticmethod
    def _get_api_key() -> str:
        """Get API key from environment or config file."""
        import os
        key = os.environ.get("DEEPSEEK_API_KEY")
        if key:
            return key

        # Check config file
        config_path = Path.home() / ".ai_quant" / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                key = config.get("deepseek_api_key")
                if key:
                    return key

        raise ValueError(
            "DeepSeek API key not found. Set DEEPSEEK_API_KEY env var or add to config."
        )

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate response using DeepSeek API."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for DeepSeek client. Install with: pip install httpx")

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"} if "JSON" in system_prompt else None,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=self.model,
            finish_reason=data["choices"][0]["finish_reason"],
            usage=usage,
        )


class OpenAICompatClient(BaseLLMClient):
    """OpenAI-compatible API client (works with any OpenAI-compatible server)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ):
        super().__init__(model=model, temperature=temperature)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate response using OpenAI-compatible API."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required. Install with: pip install httpx")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }

        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=self.model,
            finish_reason=data["choices"][0]["finish_reason"],
            usage=usage,
        )


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without real API calls."""

    MOCK_STRATEGIES = [
        {
            "name": "gen_ema_trend",
            "rationale": "EMA crossover strategy captures medium-term trends. Fast EMA(20) crosses above Slow EMA(50) for longs, below for shorts.",
            "code": """from decimal import Decimal
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import OrderSide, TimeInForce

class EMATrendStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self._fast_period = config.get("fast_period", 20)
        self._slow_period = config.get("slow_period", 50)
        self._trade_size = Decimal("0.1")
        self._ema_fast = None
        self._ema_slow = None

    def on_start(self):
        self._ema_fast = self.indicators.ema(
            self.symbol, self._fast_period
        )
        self._ema_slow = self.indicators.ema(
            self.symbol, self._slow_period
        )

    def on_bar(self, bar):
        fast_val = self._ema_fast.value[-1]
        slow_val = self._ema_slow.value[-1]

        if fast_val > slow_val and self.portfolio.is_flat(self.symbol):
            self.order_market(
                self.symbol, OrderSide.BUY,
                self._trade_size, TimeInForce.FOK
            )
        elif fast_val < slow_val and self.portfolio.is_flat(self.symbol):
            self.order_market(
                self.symbol, OrderSide.SELL,
                self._trade_size, TimeInForce.FOK
            )

    def on_stop(self):
        self.log.info("EMA Trend Strategy stopped")
""",
            "params": {"fast_period": 20, "slow_period": 50},
            "expected_edge": "Trend following in directional markets",
        },
        {
            "name": "gen_rsi_reversion",
            "rationale": "RSI mean reversion strategy. Buys when RSI < 30 (oversold), sells when RSI > 70 (overbought).",
            "code": """from decimal import Decimal
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import OrderSide, TimeInForce

class RSIMeanReversion(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self._rsi_period = config.get("rsi_period", 14)
        self._oversold = config.get("oversold", 30)
        self._overbought = config.get("overbought", 70)
        self._trade_size = Decimal("0.1")
        self._rsi = None

    def on_start(self):
        self._rsi = self.indicators.rsi(
            self.symbol, self._rsi_period
        )

    def on_bar(self, bar):
        rsi_val = self._rsi.value[-1]

        if rsi_val < self._oversold and self.portfolio.is_flat(self.symbol):
            self.order_market(
                self.symbol, OrderSide.BUY,
                self._trade_size, TimeInForce.FOK
            )
        elif rsi_val > self._overbought and self.portfolio.is_flat(self.symbol):
            self.order_market(
                self.symbol, OrderSide.SELL,
                self._trade_size, TimeInForce.FOK
            )

    def on_stop(self):
        self.log.info("RSI Mean Reversion stopped")
""",
            "params": {"rsi_period": 14, "oversold": 30, "overbought": 70},
            "expected_edge": "Mean reversion in ranging markets",
        },
        {
            "name": "gen_macd_signal",
            "rationale": "MACD histogram momentum strategy. Entry on histogram crossing zero from negative to positive.",
            "code": """from decimal import Decimal
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import OrderSide, TimeInForce

class MACDSignalStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self._fast = config.get("fast", 12)
        self._slow = config.get("slow", 26)
        self._signal = config.get("signal", 9)
        self._trade_size = Decimal("0.1")
        self._macd = None

    def on_start(self):
        self._macd = self.indicators.macd(
            self.symbol, self._fast, self._slow, self._signal
        )

    def on_bar(self, bar):
        hist = self._macd.histogram.value[-1]
        prev_hist = self._macd.histogram.value[-2] if len(self._macd.histogram.value) > 1 else 0

        if hist > 0 and prev_hist <= 0 and self.portfolio.is_flat(self.symbol):
            self.order_market(
                self.symbol, OrderSide.BUY,
                self._trade_size, TimeInForce.FOK
            )
        elif hist < 0 and prev_hist >= 0 and self.portfolio.is_flat(self.symbol):
            self.order_market(
                self.symbol, OrderSide.SELL,
                self._trade_size, TimeInForce.FOK
            )

    def on_stop(self):
        self.log.info("MACD Signal Strategy stopped")
""",
            "params": {"fast": 12, "slow": 26, "signal": 9},
            "expected_edge": "Momentum capture on trend changes",
        },
    ]

    _index = 0

    def reset(self) -> None:
        """Reset the mock strategy index."""
        self._index = 0

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a mock strategy."""
        strategy = self.MOCK_STRATEGIES[self._index % len(self.MOCK_STRATEGIES)]
        self._index += 1

        content = json.dumps(strategy, indent=2, ensure_ascii=False)
        return LLMResponse(
            content=content,
            model="mock-llm",
            usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        )


def create_llm_client(provider: str = "deepseek", **kwargs) -> BaseLLMClient:
    """Factory function to create LLM client."""
    if provider == "deepseek":
        return DeepSeekClient(**kwargs)
    elif provider == "openai_compat":
        if "base_url" not in kwargs:
            raise ValueError("base_url is required for openai_compat")
        return OpenAICompatClient(**kwargs)
    elif provider == "mock":
        return MockLLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
