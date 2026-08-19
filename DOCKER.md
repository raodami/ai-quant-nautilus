# AI Quant Nautilus Docker Images

## Multi-stage Build

```bash
# Build and run
docker-compose up -d

# Build only
docker-compose build

# Rebuild
docker-compose up -d --build
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| frontend | 3000 | Next.js app + Python backend |

## Data Volume

```bash
# Data is persisted in ./data/
./data/backtests.json
./data/strategies.json
./data/market_data/
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| NODE_ENV | production | Node environment |
| NEXT_PUBLIC_API_URL | http://localhost:3000 | API URL |
