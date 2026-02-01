.PHONY: run test lint format cov compose-up compose-down compose-logs compose-rebuild train-dry-run

run:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

cov:
	uv run pytest --cov=services --cov=app --cov-report=term-missing

compose-up:
	docker compose up --build

compose-down:
	docker compose down -v

compose-logs:
	docker compose logs -f --tail=200

compose-rebuild:
	docker compose build --no-cache

train-dry-run:
	uv run python -m scripts.train --start-date $(START_DATE) --end-date $(END_DATE) --dry-run

