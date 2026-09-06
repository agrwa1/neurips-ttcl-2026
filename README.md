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

The full configured experiment matrix is:

```bash
uv run python -m src.cli core cache
uv run python -m src.cli core dimensions
uv run python -m src.cli bias run
```

Runs are resumable by default and are written to `outputs/`. Each experiment
family records its resolved configuration and environment metadata there. Use
the CLI's filters for smaller runs and `--overwrite` only when replacing a
completed run intentionally.

### Figures and checks

If the compact `figure_data/` artifact bundle is present, reproduce all paper
figures with:

```bash
uv run python -m src.cli figures all
```

To render the freshly generated experiment results while retaining the supplied
t-SNE coordinates:

```bash
uv run python -m src.cli figures all \
  --bias-runs-root outputs/bias_accumulation/runs \
  --experiment-root outputs/core \
  --tsne-root figure_data/tsne
```

Figures are written to `output/`. Run the checks with:

```bash
uv run ruff check .
uv run pytest
```

## Repository layout

- `src/bias_accumulation/`: canonical-task position interventions.
- `src/core/`: classifier and flow-routing experiments.
- `src/figures/`: deterministic paper-figure generation.
- `figure_data/`: compact numeric inputs needed to reproduce the included figures.
- `output/`: generated paper figures.
- `tests/`: standalone checks and optional migration-parity checks.

## License

This project is licensed under the [MIT License](LICENSE). Third-party datasets,
model weights, and software remain subject to the terms in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
