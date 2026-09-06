# TTCL Experiments

Research code for the bias-accumulation and flow-routing experiments on CIFAR-100
and ImageNet-R.

## Reproducibility

The experiments require Python 3.10. Install [`uv`](https://docs.astral.sh/uv/),
then create the exact locked environment:

```bash
uv sync --frozen --group dev
```

Run commands from the repository root. The default configuration is
`configs/experiments.yaml`; it fixes the datasets, model backbones, task splits,
random seeds, training settings, and output location. It expects CUDA with
bfloat16 support, so change the `runtime` section before running on CPU.

Check the available commands and planned bias-accumulation runs before launching
the experiments:

```bash
uv run python -m src.cli --help
uv run python -m src.cli bias list
```

### Data

CIFAR-100 is downloaded automatically to `data/cifar100/`. Download ImageNet-R
from its official release, extract its 200 class directories under
`data/imagenet-r/`, and create the deterministic 80/20 split:

```bash
uv run python -m src.cli prepare-imagenet-r
```

Dataset files and pretrained weights are not redistributed by this repository.
Their credits and terms are summarized in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Experiments

Run the full configured experiment matrix in this order:

```bash
# Extract and store frozen-backbone features.
uv run python -m src.cli core cache

# Compare classifier and flow routing across all configured projection dimensions.
uv run python -m src.cli core dimensions

# Run the canonical-task position experiments.
uv run python -m src.cli bias run
```

Runs are resumable by default and are written to `outputs/`. Each experiment
family records its resolved configuration and environment metadata there. Use
`--dataset`, `--backbone`, `--tasks`, and `--seed` to run a smaller subset. For
example:

```bash
uv run python -m src.cli core cache --dataset cifar100 --backbone vit_b16
uv run python -m src.cli core dimensions --dataset cifar100 --backbone vit_b16 \
  --tasks 10 --seed 1993
uv run python -m src.cli bias run --dataset cifar100 --backbone vit_b16 \
  --tasks 10 --seed 1993 --canonical-position 0
```

Use `--overwrite` only when intentionally replacing a completed run.

### Figures and checks

Plotting commands read the compact `figure_data/` bundle by default:

```bash
# Per-seed routing-error distributions.
uv run python -m src.cli figures bias-accumulation

# Heatmap of routing errors as the canonical task changes position.
uv run python -m src.cli figures canonical-position

# Accuracy curves and final method-comparison tables.
uv run python -m src.cli figures method-comparison
```

Figures are written to `output/`. To plot new runs, add
`--runs-root outputs/bias_accumulation/runs` or
`--experiment-root outputs/core`, as applicable. Run the checks with:

```bash
uv run ruff check .
uv run python -m pytest tests/test_smoke.py
```

## Repository layout

- `src/bias_accumulation/`: canonical-task position interventions.
- `src/core/`: classifier and flow-routing experiments.
- `src/figures/`: deterministic paper-figure generation.
- `figure_data/`: compact numeric inputs needed to reproduce the included figures.
- `output/`: generated paper figures.
- `tests/`: standalone correctness and smoke checks.

## License

This project is licensed under the [MIT License](LICENSE). Third-party datasets,
model weights, and software remain subject to the terms in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
