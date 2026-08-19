"""
FastAPI backend for ai-quant-nautilus.
Serves strategies, backtests, metrics, and WebSocket events.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai_quant_nautilus.db.store import StrategyStore, BacktestStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class StrategyCreate(BaseModel):
    name: str
    code: str = ""
    status: str = "draft"
    performance: float = 0.0
    sharpe: float = 0.0
    maxDrawdown: float = 0.0
    winRate: float = 0.0

class StrategyUpdate(BaseModel):
    id: str
    status: Optional[str] = None
    performance: Optional[float] = None
    sharpe: Optional[float] = None
    maxDrawdown: Optional[float] = None
    winRate: Optional[float] = None

class BacktestCreate(BaseModel):
    strategy_name: str
    instrument: str = "BTCUSDT.BINANCE"
    timeframe: str = "1h"
    initial_capital: float = 1000000
    final_capital: float = 0
    total_return: float = 0
    sharpe_ratio: float = 0
    max_drawdown: float = 0
    win_rate: float = 0
    total_trades: int = 0
    trades: list = []
    equity_curve: list = []

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Quant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stores
strategy_store = StrategyStore()
backtest_store = BacktestStore()

# WebSocket clients
active_connections: list[WebSocket] = []


@app.get("/api/status")
async def get_status():
    strategies = strategy_store.list_all()
    backtests = backtest_store.list_all(limit=10)
    
    total_return = sum(b.get("total_return", 0) for b in backtests) / max(len(backtests), 1)
    avg_sharpe = sum(b.get("sharpe_ratio", 0) for b in backtests) / max(len(backtests), 1)
    
    return {
        "status": "running",
        "version": "1.0.0",
        "uptime": int(time.time()),
        "strategies": len(strategies),
        "backtests": len(backtests),
        "summary": {
            "totalReturn": round(total_return, 2),
            "avgSharpe": round(avg_sharpe, 2),
            "activeStrategies": len([s for s in strategies if s.get("status") == "running"]),
        }
    }

@app.get("/api/strategies")
async def list_strategies():
    strategies = strategy_store.list_all()
    return {"strategies": strategies}

@app.post("/api/strategies")
async def create_strategy(data: StrategyCreate):
    strategy = {
        "id": f"strat_{int(time.time() * 1000)}",
        "name": data.name,
        "code": data.code,
        "status": data.status,
        "performance": data.performance,
        "sharpe": data.sharpe,
        "maxDrawdown": data.maxDrawdown,
        "winRate": data.winRate,
        "createdAt": datetime.utcnow().isoformat(),
        "updatedAt": datetime.utcnow().isoformat(),
    }
    strategy_store.save(strategy)
    await broadcast_event({"type": "strategy_created", "data": strategy})
    return {"strategy": strategy}

@app.put("/api/strategies")
async def update_strategy(data: StrategyUpdate):
    strategy = strategy_store.get(data.id)
    if not strategy:
        return {"error": "Not found"}, 404
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    strategy.update(updates)
    strategy["updatedAt"] = datetime.utcnow().isoformat()
    strategy_store.save(strategy)
    return {"strategy": strategy}

@app.delete("/api/strategies")
async def delete_strategy(strategy_id: str):
    if strategy_store.delete(strategy_id):
        return {"success": True}
    return {"error": "Not found"}, 404

@app.get("/api/backtests")
async def list_backtests(limit: int = 20):
    backtests = backtest_store.list_all(limit=limit)
    return {"backtests": backtests}

@app.post("/api/backtests")
async def create_backtest(data: BacktestCreate):
    record = {
        "id": f"bt_{int(time.time() * 1000)}",
        "strategyName": data.strategy_name,
        "instrument": data.instrument,
        "timeframe": data.timeframe,
        "initialCapital": data.initial_capital,
        "finalCapital": data.final_capital,
        "totalReturn": data.total_return,
        "sharpeRatio": data.sharpe_ratio,
        "maxDrawdown": data.max_drawdown,
        "winRate": data.win_rate,
        "totalTrades": data.total_trades,
        "trades": data.trades,
        "equityCurve": data.equity_curve,
        "timestamp": datetime.utcnow().isoformat(),
    }
    bt_id = backtest_store.add(record)
    await broadcast_event({"type": "backtest_completed", "data": record})
    return {"backtest": record, "id": bt_id}

@app.get("/api/metrics")
async def get_metrics():
    backtests = backtest_store.list_all(limit=50)
    strategies = strategy_store.list_all()
    
    if not backtests:
        return {
            "metrics": [],
            "summary": {
                "equity": 1000000,
                "totalReturn": "0.00",
                "sharpeRatio": "0.00",
                "maxDrawdown": "0.00",
                "winRate": "0.00",
                "trades": 0,
            },
            "activeStrategies": 0,
        }
    
    latest = backtests[0]
    equity = latest.get("finalCapital", 1000000)
    
    equity_curve = [
        {"timestamp": bt["timestamp"], "equity": bt.get("finalCapital", equity)}
        for bt in backtests[:100]
    ][::-1]
    
    return {
        "metrics": equity_curve,
        "summary": {
            "equity": equity,
            "totalReturn": str(latest.get("total_return", 0)),
            "sharpeRatio": str(latest.get("sharpe_ratio", 0)),
            "maxDrawdown": str(latest.get("max_drawdown", 0)),
            "winRate": str(latest.get("win_rate", 0)),
            "trades": latest.get("total_trades", 0),
        },
        "activeStrategies": len([s for s in strategies if s.get("status") == "running"]),
    }

@app.get("/api/trades")
async def get_trades(limit: int = 20):
    backtests = backtest_store.list_all(limit=10)
    trades = []
    for bt in backtests:
        for trade in bt.get("trades", []):
            trades.append({
                "id": trade.get("id", f"trade_{len(trades)}"),
                "symbol": trade.get("symbol", bt.get("instrument", "BTCUSDT")),
                "side": trade.get("side", "buy"),
                "quantity": trade.get("quantity", "0"),
                "price": trade.get("price", "0"),
                "pnl": trade.get("pnl", 0),
                "timestamp": trade.get("timestamp", bt.get("timestamp", "")),
            })
    return {"trades": trades[:limit]}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast_event(event: dict):
    """Send event to all connected WebSocket clients."""
    if not active_connections:
        return
    disconnected = []
    for conn in active_connections:
        try:
            await conn.send_json(event)
        except Exception:
            disconnected.append(conn)
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)

# ---------------------------------------------------------------------------
# Run with: uvicorn backend:app --reload --port 8000
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
