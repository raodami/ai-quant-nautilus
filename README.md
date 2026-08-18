# AI-Quant-Nautilus

AI-driven crypto quant closed-loop system fusing:
- **ai-quant**: LLM strategy generation + AST guard + state machine orchestrator
- **nautilus_trader**: Production-grade Rust-native trading engine (v2)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator (状态机)                         │
│   读注册表 → 选策略 → LLM生成 → Nautilus回测 → 评估 → 模拟      │
└───┬───────────┬───────────────┬───────────┬───────────┬────────┘
    ▼           ▼               ▼           ▼           ▼
┌────────┐  ┌──────────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ 生成层 │─▶│ Nautilus回测 │─▶│ 评估层 │─▶│ 模拟层 │─▶│ 风控层 │
│ LLM+  │  │ 引擎(替代     │  │ 门槛+  │  │ dry   │  │ 闸门   │
│ AST   │  │  Freqtrade)  │  │ 反过拟合│  │ run   │  │ 人工   │
└────────┘  └──────────────┘  └────────┘  └────────┘  └────────┘
```

## Quick Start

```bash
# Install
cd D:/ai-quant-nautilus
uv sync

# Run without nautilus (mock mode)
uv run python -m ai_quant_nautilus.cli run --iterations 3

# With nautilus (requires Rust toolchain)
uv pip install nautilus_trader
uv run python -m ai_quant_nautilus.cli run --iterations 3
```

## CLI Commands

```bash
aqn run --iterations 5          # Run 5 iterations
aqn status                      # Show registry
aqn backtest --strategy main.py --data-path data.csv
```

## Directory Structure

```
ai-quant-nautilus/
├── src/ai_quant_nautilus/
│   ├── orchestrator.py       # Main state machine
│   ├── backtest/
│   │   └── nautilus_adapter.py  # Nautilus integration layer
│   ├── evaluator/
│   │   └── gates.py          # Strategy evaluation gates
│   ├── generator/
│   │   ├── prompt_builder.py # LLM prompts (nautilus-aware)
│   │   └── schema.py         # Output validation
│   ├── risk/
│   │   └── allocator.py      # Risk parity allocation
│   ├── sandbox/
│   │   └── ast_guard.py      # AST safety check
│   └── data/
│       └── collector.py      # OHLCV data fetcher
├── tests/
├── data/raw/                 # Cached OHLCV data
└── pyproject.toml
```

## NautilusTrader Integration

The key difference from ai-quant:
- **Backtest engine**: NautilusTrader (Rust-native, event-driven) instead of Freqtrade
- **Strategy format**: Nautilus `Strategy` class instead of Freqtrade `IStrategy`
- **Live trading**: Nautilus live execution pipeline (future)

See `src/ai_quant_nautilus/backtest/nautilus_adapter.py` for the adapter layer.

## Testing

```bash
uv run pytest tests/ -v
```

## License

MIT
