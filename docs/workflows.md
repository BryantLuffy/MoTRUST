# Running MoTRUST

## Experiment directory

All paths below are relative to the selected experiment directory. Use the
current directory, set `MOTRUST_WORKDIR`, or pass `motrust --workdir PATH ...`.
Code and protocol resources are loaded from the installed package, so the
experiment directory need not be the source checkout.

## Integration

Convert the supplied processed data and prepare metadata with the bundled
conversion helpers:

```bash
motrust prepare-data --help
motrust prepare-data cache --data-dir /path/to/processed
motrust prepare-data metadata --data-dir /path/to/processed --task-id TEA_s1__random1
```

These commands use the original input directory structure specified in the
data protocol. They convert processed matrices and annotations; they do not
start from raw sequencing reads.

The integration workflow starts from a semantic anchor, a candidate
representation and task-level mixing settings. It aligns the candidate into
the semantic coordinate frame, applies neighborhood-based bounded semantic
protection, mixes the representations, and aligns modality/batch effects.

```bash
motrust integrate --task-id TEA_s1__random1 --check
motrust integrate --task-id TEA_s1__random1 --run
```

Use `--no-evaluate` when producing coordinates without evaluation labels.
`--domain cortex` selects the cortical task definitions and data paths.
Custom task definitions can be supplied through the data contract described
in [data-format.md](data-format.md).

Preparation commands expose the package's semantic-anchor construction,
multimodal encoder, spectral representation and candidate routing:

```bash
motrust prepare-semantic --help
motrust train-integration --help
motrust build-spectral --help
motrust finalize-candidate --help
motrust compose --help
motrust evaluate-integration --help
```

Supply the arguments listed by each command for the chosen input data.
Gene annotation for RNA/ATAC feature mapping is an external input. A
semantic/candidate pair alone does not reconstruct the entire original data
preparation. Keep cell order and identifiers aligned across representations.

## ATAC-to-RNA point recovery and diffusion

Prepare paired RNA/ATAC matrices and library metadata in the cache format,
then run the steps in order:

```bash
motrust rna-diffusion --stage prepare --check
motrust rna-diffusion --stage prepare --run
motrust recover --check
motrust recover --run
motrust rna-diffusion --stage evaluate --check
motrust rna-diffusion --stage evaluate --run
```

`motrust recover` is equivalent to `motrust rna-diffusion --stage points`.
The preparation step deduplicates copies of the same source library and
barcode. It constructs fixed within-library train/validation/evaluation
splits and selects features from the training observations. The point
step trains or reloads three members, averages RNA predictions, and saves
source posterior statistics. The evaluation step fits or reloads the
residual DDPM and matched conditional stochastic head, generates samples,
and writes distribution scores. It can train models; it is not a read-only
scoring command.

| Setting | Value |
|---|---|
| Data sources | TEA, BMMC-Multiome, Retina |
| Direction | ATAC to RNA |
| Split | Seed 913; 60% training, 20% validation, 20% evaluation within library |
| Features | Up to 2,000 RNA and 5,000 ATAC features selected by training prevalence |
| Point ensemble | Seeds 42, 0 and 1 |
| Point fitting | 100 encoder epochs, 100 head epochs, 50 adaptation epochs, 20 refinement epochs |
| Distribution fitting | 100 epochs, seeds 42, 0 and 1 |
| Sampling | 100 diffusion steps, 32 samples per cell |
| Sample constraint | Common bounded mean projection |

Full fitting and synchronized sampling require CUDA. Metadata export uses
Rscript and `jsonlite`; set the `RSCRIPT` environment variable if Rscript is
not on PATH. Results are written below `results/rna_diffusion/`.

The three sources use cell holdouts within libraries. This protocol does
not represent independent-donor or zero-shot external-cohort testing.
Input hashes and protocol records identify each run; keep completed runs
separate when changing source, inputs or settings.

## Python interfaces

Core APIs are available under `motrust.models`, `motrust.preprocessing`,
`motrust.integration` and `motrust.recovery`. Workflow APIs live under
`motrust.workflows`; metrics and common output validation are under
`motrust.benchmark` and `motrust.data`. The CPU example in
`examples/quickstart.py` illustrates the numerical interface with synthetic
inputs.

The additional `motrust-recovery` command exposes the separate latent
recovery interface. It has its own input/configuration contract and is not
the conditional RNA residual-diffusion workflow above.
