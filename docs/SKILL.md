---
name: ai-quant-nautilus
category: quant-finance
description: Fusion project combining AI strategy generation with nautilus_trader backtesting engine.
---

# ai-quant-nautilus 融合项目

## 概述

融合 ai-quant 的 AI 策略生成 + 状态机编排层与 nautilus_trader 的高性能 Rust 原生回测/实盘引擎。

## 项目位置

`D:/ai-quant-nautilus`

## 架构分层

```
┌─────────────────────────────────────────┐
│            Orchestrator (状态机)         │
│  generated→backtested→passed→paper→live │
├─────────────────────────────────────────┤
│  Generator (AI策略生成)                  │
│  └── Prompt Builder → LLM → Schema     │
├─────────────────────────────────────────┤
│  Backtest (Nautilus适配器)               │
│  └── 动态生成 Strategy 子类             │
├─────────────────────────────────────────┤
│  Evaluator (评估门控)                    │
│  └── Sharpe>0.5, MaxDD<20%, 胜率>40%    │
├─────────────────────────────────────────┤
│  Risk (风险分配)                         │
│  └── 风险平价资本分配                    │
├─────────────────────────────────────────┤
│  Sandbox (AST安全)                       │
│  └── 阻止危险 import/call               │
├─────────────────────────────────────────┤
│  Simulation (模拟盘)                     │
├─────────────────────────────────────────┤
│  Data (ccxt 数据收集 + mock 生成)        │
└─────────────────────────────────────────┘
```

## Python 版本约束

- **当前环境**: Python 3.11
- **nautilus_trader 要求**: Python 3.12+
- **解决方案**: nautilus_trader 设为可选依赖，当前走 mock 模式
- **启用 nautilus**: 升级 Python 到 3.12+ 后 `uv pip install nautilus_trader`

## 测试覆盖

**总计: 47 tests pass**

| 模块 | 测试数 |
|------|--------|
| TestAstGuard | 5 |
| TestGenerateNautilusStrategy | 2 |
| TestNautilusBacktestAdapter | 2 |
| TestPerformanceMetrics | 3 |
| TestStrategyTemplates | 7 |
| TestGateEvaluator | 5 |
| TestSchemaValidation | 4 |
| TestRiskParity | 5 |
| TestPipelineIntegration | 8 |
| TestCLI | 3 |

## 依赖安装

```bash
cd D:/ai-quant-nautilus
uv sync
uv run pytest tests/ -v
```

## CLI 命令

```bash
aqn run              # 运行完整流程
aqn backtest --strategy ... --data-path ...
aqn status           # 查看策略状态
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `src/ai_quant_nautilus/backtest/nautilus_adapter.py` | Nautilus 适配器核心 |
| `src/ai_quant_nautilus/backtest/templates.py` | 6 个策略模板 |
| `src/ai_quant_nautilus/backtest/performance.py` | 性能分析器 |
| `src/ai_quant_nautilus/orchestrator.py` | 状态机编排 |
| `src/ai_quant_nautilus/evaluator/gates.py` | 评估门控 |
| `src/ai_quant_nautilus/generator/schema.py` | JSON Schema 校验 |
| `src/ai_quant_nautilus/sandbox/ast_guard.py` | AST 安全检查 |
| `src/ai_quant_nautilus/data/mock_generator.py` | Mock 数据生成 |

## 策略模板

1. `ema_cross` - EMA 交叉趋势跟踪
2. `rsi_mean_reversion` - RSI 均值回归
3. `macd_signal` - MACD 信号交叉
4. `bollinger_breakout` - 布林带突破
5. `golden_cross` - 金叉死叉 (50/200 MA)
6. `super_trend` - SuperTrend 突破

## 下一步

1. Python 3.12+ 环境安装 nautilus_trader 进行真实回测
2. 集成 ccxt 连接交易所获取真实 OHLCV 数据
3. 添加更多策略模板
4. 实盘部署联调
