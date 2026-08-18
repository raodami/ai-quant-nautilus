"""
LLM prompt builders for strategy generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Nautilus-specific system prompt
NAUTILUS_SYSTEM_PROMPT = """你是一个加密货币量化策略工程师。你的任务是生成可以被 NautilusTrader 回测引擎加载的 Python 策略代码。

## NautilusStrategy 模板要求

```python
from decimal import Decimal
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity

class MyStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self._trade_size = Decimal("0.1")

    def on_start(self):
        # 初始化指标
        pass

    def on_bar(self, bar):
        # 交易逻辑
        # 买入: self.order_market(self.instrument_id, OrderSide.BUY, self._trade_size, TimeInForce.FOK)
        # 卖出: self.order_market(self.instrument_id, OrderSide.SELL, self._trade_size, TimeInForce.FOK)
        pass

    def on_stop(self):
        pass
```

## 输出格式
你必须只返回一个合法的 JSON 对象:
- "name": 策略名 (gen_ 前缀)
- "rationale": 策略逻辑说明 (至少30字)
- "code": 完整 Python 类代码
- "params": 可调超参数字典
- "expected_edge": 预期优势与市场状态

## 代码约束
1. 必须继承 Strategy
2. 必须实现 on_start, on_bar, on_stop
3. 只能用 pandas, numpy, decimal
4. 严禁: os, sys, subprocess, socket, requests, eval, exec, open
5. 严禁前视偏差
6. 信号明确: order_market 调用

## 可用数据
- bar.open, bar.high, bar.low, bar.close, bar.volume
- self.cache.price(instrument_id, Aggregation.ASK, ...) 获取价格
"""


def build_system_prompt() -> str:
    return NAUTILUS_SYSTEM_PROMPT


@dataclass
class GenerationContext:
    """Strategy generation context."""
    market_summary: str = ""
    top_strategies: list[dict] = field(default_factory=list)
    failure_examples: list[dict] = field(default_factory=list)
    existing_hashes: set[str] = field(default_factory=set)
    timeframe: str = "1h"
    can_short: bool = False


def build_user_prompt(ctx: GenerationContext) -> str:
    parts = []

    # Market context
    parts.append(f"# 市场环境\n{ctx.market_summary or '暂无市场摘要，请自行设计通用策略。'}\n")

    # Available data
    parts.append("""# 可用数据列
- bar.open, bar.high, bar.low, bar.close, bar.volume
- 可通过 self.cache.indicator() 获取技术指标
""")

    # Top strategies
    if ctx.top_strategies:
        parts.append("# 表现较好的策略 (可借鉴逻辑, 不要抄袭代码)")
        for s in ctx.top_strategies[:3]:
            parts.append(f"- {s['name']}: {s.get('rationale', 'N/A')}")
            parts.append(f"  Sharpe={s.get('sharpe', 'N/A')}, WinRate={s.get('win_rate', 'N/A')}%")
        parts.append("")

    # Existing hashes
    if ctx.existing_hashes:
        parts.append(f"# 已存在的策略名 (避免重复): {', '.join(ctx.existing_hashes)}")
        parts.append("")

    return "\n".join(parts)
