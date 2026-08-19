.PHONY: install dev test docker-clean

PYTHON = /d/hermes/hermes-agent/venv/Scripts/python.exe
UV = uv

install:
	$(UV) pip install -e ".[dev]"
	cd frontend && npm ci

dev-backend:
	$(PYTHON) -m ai_quant_nautilus.backend

dev-frontend:
	cd frontend && npm run dev

test:
	$(PYTHON) -m pytest tests/ -q
	cd frontend && node test-suite.js

test-py:
	$(PYTHON) -m pytest tests/ -q

test-fe:
	cd frontend && node test-suite.js

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	rm -rf data/*.json
	rm -rf frontend/.next
	rm -rf __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
