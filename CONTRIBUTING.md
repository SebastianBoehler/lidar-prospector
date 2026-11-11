# Contributing

Small, measured changes are easiest to review.

## Setup

```bash
uv sync --group dev
uv run pre-commit install
```

## Before opening a PR

```bash
uv run ruff check . --fix && uv run ruff format .
uv run pytest -W error
```

CI runs the same on Python 3.11 and 3.13.

## Guidelines

- **Measure detection changes.** If a change affects what is found, include before/after numbers from
  `python -m lidarprospector.crossval` (or a new benchmark region) in the PR. Held-out tiles only.
- **No mock data in benchmarks.** Unit tests may use small synthetic terrain; benchmarks use real, openly
  licensed data with the licence noted in the README.
- **Fail loudly.** Surface network and data errors; do not silently fall back.
- **Keep files short** (roughly under 300 lines) and modules single-purpose.
- **New data sources** must be openly licensed; add the source and licence to the README table.
- **Sensitive locations.** Do not commit coordinates of previously unrecorded sites.
