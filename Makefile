.PHONY: run test lint format cov compose-up compose-down compose-logs compose-rebuild

run:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest

lint:
	uv run flake8 .
	uv run isort . --check-only --filter-files

format:
	uv run isort . --filter-files

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
