# AI Quant Nautilus - Deployment Guide

## Quick Start (Local)

### 1. Start Backend
```bash
cd D:/ai-quant-nautilus
PYTHONPATH=D:/ai-quant-nautilus/src uvicorn ai_quant_nautilus.backend:app --host 0.0.0.0 --port 8000
```

### 2. Start Frontend (another terminal)
```bash
cd D:/ai-quant-nautilus/frontend
npm run dev -- --port 3012
```

### 3. Access
- Frontend: http://localhost:3012
- Backend API: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws

---

## Test Accounts
| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | admin |
| trader | trader123 | user |

---

## Docker Deployment

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Services:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000

---

## GitHub Push (requires token)

```bash
git remote set-url origin https://<TOKEN>@github.com/raodami/ai-quant-nautilus.git
git push origin main
```

---

## Strategy Templates

Available strategies:
1. ema_cross - EMA Cross Trend Following
2. rsi_mean_reversion - RSI Mean Reversion
3. macd_signal - MACD Signal Crossover
4. bollinger_breakout - Bollinger Band Breakout
5. golden_cross - Golden Cross (50/200 MA)
6. super_trend - SuperTrend Breakout

---

## LLM Configuration

Set environment variables:
```bash
export DEEPSEEK_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
```

Or edit config.yaml:
```yaml
generator:
  model: "deepseek-chat"
  provider: "deepseek"
```

---

## Files
- `src/ai_quant_nautilus/backend.py` - FastAPI backend
- `src/ai_quant_nautilus/db/store.py` - Database storage
- `frontend/src/app/api/` - API routes
- `frontend/src/hooks/useWebSocket.ts` - WebSocket hook
- `frontend/src/components/LiveStatus.tsx` - Live events component
