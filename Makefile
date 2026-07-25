.PHONY: check demo run

check:
	uv lock --check
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pytest --cov=eventpilot --cov-report=term-missing
	uv build

demo:
	uv run eventpilot demo

run:
	uv run eventpilot run
