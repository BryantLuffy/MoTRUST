# MoTRUST

**Adaptive semantic protection for mosaic single-cell multi-omics.**

MoTRUST integrates partially observed RNA, ATAC and ADT measurements, predicts
missing molecular profiles, and generates conditional RNA samples around a
fixed point estimate. Model implementations, data preparation, integration
metrics and evaluation workflows are part of the `motrust` Python package.

## Installation

Python 3.10 or later is required. From the repository root:

```bash
python -m pip install -e ".[dev,reproduce]"
motrust --help
python examples/quickstart.py
```

The quickstart uses small synthetic arrays on CPU and needs no downloaded
data. For full molecular recovery and RNA diffusion fitting, install a
CUDA-enabled PyTorch build appropriate for your machine. Data conversion
also uses R with `jsonlite` and `Matrix`.

## Workflows

| Component | Purpose | Entry point |
|---|---|---|
| Multimodal integration | Protect shared biological structure while aligning neural or spectral representations | `motrust integrate` |
| Molecular point recovery | Fit three members and average their RNA predictions | `motrust recover` |
| Conditional RNA diffusion | Generate residual samples, preserve the point mean and evaluate the distribution | `motrust rna-diffusion` |
| Data and benchmark utilities | Build semantic/spectral representations and evaluate common embedding outputs | `motrust --help` |

The integration workflow consumes prepared representations and their task
metadata. The RNA workflow covers ATAC-to-RNA recovery on TEA,
BMMC-Multiome and Retina. See [the workflow guide](docs/workflows.md) for
input contracts, preparation commands and the recorded settings.

```bash
# Check the inputs for one integration task.
motrust integrate --task-id TEA_s1__random1 --check

# Execute after supplying its required representations and metadata.
motrust integrate --task-id TEA_s1__random1 --run

# Prepare, fit the point ensemble, and evaluate conditional RNA samples.
motrust rna-diffusion --stage prepare --run
motrust recover --run
motrust rna-diffusion --stage evaluate --run
```

Without `--run`, these three workflow commands check inputs only. Missing
inputs are listed with exit code 2. Data and trained weights are obtained
separately; installing the package does not download them.

## Data and output locations

The current directory is the default experiment directory. Select another
directory with `--workdir` or `MOTRUST_WORKDIR`:

```bash
motrust --workdir /path/to/experiment integrate --task-id TEA_s1__random1 --check
```

Inputs live in `data/`, intermediate representations in `runs/`, and results
in `results/` inside that directory. Installation files remain separate from
experimental outputs. Bundled protocols are read from package resources.

See [dataset sources](docs/datasets.md) and [input formats](docs/data-format.md).

## Repository structure

```text
src/motrust/
  models/             Multimodal encoders, decoders and fusion layers
  preprocessing/      Shared-feature mapping and semantic anchors
  integration/        Alignment and representation-fusion primitives
  recovery/           Recovery and uncertainty primitives
  data/               Task data access and output contracts
  benchmark/          Integration metrics and evaluation
  workflows/          Integration, point recovery and RNA diffusion
  resources/          Protocols, task definitions and R conversion helpers
tests/                Numerical and workflow contract tests
examples/             Runnable examples
docs/                 Data and workflow documentation
scripts/              Source validation and developer utilities
```

## Testing

```bash
python scripts/verify_source.py
python scripts/smoke_test.py
python -m pytest tests
```

The smoke test runs 30 small CPU tests without experimental data. Workflow
tests additionally cover module interfaces and numerical contracts. These
checks validate software behavior; reproducing benchmark results requires
the corresponding data, representations and training runs.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff); a final paper
citation can be added when available. MoTRUST uses the existing
[MIT license](LICENSE). External methods, dependencies and datasets retain
their own terms; see [third-party notices](docs/third-party-notices.md).

[Contributing](CONTRIBUTING.md) · [GitHub upload instructions](docs/upload-zh.md)
