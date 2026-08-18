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
│  Data (ccxt 数据收集)                    │
└─────────────────────────────────────────┘
```

## Python 版本约束

- **当前环境**: Python 3.11
- **nautilus_trader 要求**: Python 3.12+
- **解决方案**: nautilus_trader 设为可选依赖，当前走 mock 模式
- **启用 nautilus**: 升级 Python 到 3.12+ 后 `uv pip install nautilus_trader`

## 依赖安装

```bash
cd D:/ai-quant-nautilus
uv sync
uv run pytest tests/ -v
```

## CLI 命令

```bash
ai-quant-nautilus init        # 初始化配置
ai-quant-nautilus generate    # AI 策略生成
ai-quant-nautilus backtest    # Nautilus 回测
ai-quant-nautilus evaluate    # 评估门控
ai-quant-nautilus paper       # 模拟盘运行
ai-quant-nautilus live        # 实盘部署
```

## 测试覆盖

- **TestAstGuard**: 5 tests — AST 安全检查
- **TestGenerateNautilusStrategy**: 2 tests — 策略生成模板
- **TestNautilusBacktestAdapter**: 2 tests — 回测适配器（mock 模式）
- **TestGateEvaluator**: 4 tests — 评估门控
- **TestSchemaValidation**: 4 tests — Schema 校验
- **TestRiskParity**: 5 tests — 风险分配

**总计**: 25 tests pass

## 关键文件

| 文件 | 说明 |
|------|------|
| `src/ai_quant_nautilus/backtest/nautilus_adapter.py` | Nautilus 适配器核心（策略转换） |
| `src/ai_quant_nautilus/orchestrator.py` | 状态机编排器 |
| `src/ai_quant_nautilus/evaluator/gates.py` | 评估门控逻辑 |
| `src/ai_quant_nautilus/generator/schema.py` | JSON Schema 定义与校验 |
| `src/ai_quant_nautilus/sandbox/ast_guard.py` | AST 安全检查 |

## 下一步

1. Python 3.12+ 环境上安装 nautilus_trader 进行真实回测
2. 集成 ccxt 连接交易所获取真实 OHLCV 数据
3. 添加 EMA Cross 等更多策略模板
4. 实盘部署联调
