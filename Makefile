.PHONY: setup test coverage smoke fit freeze evaluate report demo benchmark

# Installs the locked dependencies, including QuantLib for the cross-check.
setup:
	uv sync --all-extras --frozen

# Correctness suite, including the statistical checks.
test:
	uv run pytest

coverage:
	uv run pytest --cov=asianrisk --cov-report=term-missing

# Runs the whole pipeline at small scale. It proves the pipeline works, not any result.
smoke:
	uv run python -m asianrisk smoke

# Labels development data and fits every surface. Skips anything already saved.
fit:
	uv run python -m asianrisk fit
	uv run python -m asianrisk tune

# Pins the experiment config. Sealed labels refuse to exist without this file.
freeze:
	uv run python -m asianrisk freeze

# Labels the sealed sets and measures every route. Close other work first.
evaluate:
	uv run python -m asianrisk evaluate

# Rebuilds the report tables, plots and verdict from raw JSON. Runs no pricer.
report:
	uv run python -m asianrisk report

# Rebuilds site/index.html from the recorded run.
demo:
	uv run python -m asianrisk demo

benchmark: fit freeze evaluate report demo
