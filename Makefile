.PHONY: check dashboard

check:
	uv lock --check
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pytest --cov=eventpilot --cov-report=term-missing
	uv build

dashboard:
	uv run eventpilot dashboard
