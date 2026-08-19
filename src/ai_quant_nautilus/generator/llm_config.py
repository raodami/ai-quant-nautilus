"""LLM integration configuration for ai-quant-nautilus."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "deepseek"  # deepseek, openai, openai_compat
    model: str = "deepseek-chat"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load config from environment variables."""
        import os
        return cls(
            provider=os.environ.get("LLM_PROVIDER", "deepseek"),
            model=os.environ.get("LLM_MODEL", "deepseek-chat"),
            api_key=os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=os.environ.get("LLM_BASE_URL"),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
        )

    @classmethod
    def from_file(cls, path: str = "~/.ai_quant/config.json") -> "LLMConfig":
        """Load config from JSON file."""
        p = Path(path).expanduser()
        if not p.exists():
            return cls.from_env()
        with open(p) as f:
            data = json.load(f)
        return cls(
            provider=data.get("provider", "deepseek"),
            model=data.get("model", "deepseek-chat"),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            temperature=data.get("temperature", 0.7),
        )

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key[:8] + "..." if self.api_key else None,
            "base_url": self.base_url,
            "temperature": self.temperature,
        }


def get_llm_client(provider: str = None, **kwargs):
    """Factory function to create LLM client."""
    from ai_quant_nautilus.generator.llm_client import create_llm_client
    
    if provider is None:
        config = LLMConfig.from_env()
        provider = config.provider
    
    return create_llm_client(provider=provider, **kwargs)


def generate_strategy(market_summary: str = "", max_iterations: int = 3):
    """Generate a trading strategy using LLM.
    
    Args:
        market_summary: Optional market context
        max_iterations: Maximum retry attempts
    
    Returns:
        dict with strategy details or None if failed
    """
    try:
        from ai_quant_nautilus.generator.prompt_builder import (
            build_system_prompt,
            build_user_prompt,
            GenerationContext,
        )
        from ai_quant_nautilus.generator.llm_client import create_llm_client
        
        config = LLMConfig.from_env()
        client = create_llm_client(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
        )
        
        ctx = GenerationContext(market_summary=market_summary)
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(ctx)
        
        for attempt in range(max_iterations):
            try:
                response = client.generate(system_prompt, user_prompt)
                strategy = client.generate_json(system_prompt, user_prompt)
                logger.info(f"Strategy generated: {strategy.get('name', 'unknown')}")
                return strategy
            except Exception as e:
                logger.warning(f"Generation attempt {attempt+1} failed: {e}")
                if attempt < max_iterations - 1:
                    continue
                raise
        
        return None
    except ImportError as e:
        logger.error(f"Failed to import LLM modules: {e}")
        return None
    except Exception as e:
        logger.error(f"Strategy generation failed: {e}")
        return None


def test_llm_connection() -> bool:
    """Test if LLM API is configured and reachable."""
    try:
        config = LLMConfig.from_env()
        if not config.api_key:
            logger.warning("No API key configured")
            return False
        
        client = create_llm_client(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
        )
        
        # Simple test request
        response = client.generate("You are a helpful assistant.", "Say hello")
        return response and len(response.content) > 0
    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return False
