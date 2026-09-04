# TTCL Experiments

Research code for the bias-accumulation and flow-routing experiments on CIFAR-100
and ImageNet-R.

## Setup

Install [`uv`](https://docs.astral.sh/uv/), then create the locked environment:

```bash
uv sync --frozen --group dev
```

All commands are run from the repository root through the unified CLI:

```bash
uv run python -m src.cli --help
```

## Main commands

```bash
# Inspect the bias-accumulation run matrix.
uv run python -m src.cli bias list

# Generate every paper figure from the included figure data.
uv run python -m src.cli figures all

# Run the core flow-dimension experiments.
uv run python -m src.cli core dimensions

# Run the canonical-position bias experiments.
uv run python -m src.cli bias run
```

CIFAR-100 is downloaded automatically. ImageNet-R should be placed at
`data/imagenet-r/`; its deterministic split is created with:

```bash
uv run python -m src.cli prepare-imagenet-r
```

Experiment settings live in `configs/experiments.yaml`. Generated experiment runs
are written to `outputs/`. Paper figures are written to `output/`.

## Repository layout

- `src/bias_accumulation/`: canonical-task position interventions.
- `src/core/`: classifier and flow-routing experiments.
- `src/figures/`: deterministic paper-figure generation.
- `figure_data/`: compact numeric inputs needed to reproduce the included figures.
- `output/`: generated paper figures.
- `tests/`: standalone checks and optional migration-parity checks.

## Checks

```bash
uv run ruff check .
uv run pytest
```

