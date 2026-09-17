# Data and representation contracts

Paths are relative to the experiment directory selected with `--workdir`.
Data files, model weights and generated outputs are not committed to Git.

## Integration task data

```text
data/
  cache/cache_index.json
  tasks/<task_id>/metadata_training.csv
  tasks/<task_id>/metadata_evaluation.csv
runs/integration/
  semantic/<task_id>/
  candidate/<task_id>/
  composition/<task_id>/composition_manifest.json
  reference/<task_id>/
results/integration/
```

Semantic and candidate folders contain `embedding.npy` and `metadata.csv`.
The semantic folder also supplies `semantic_reliability.npy`. Embeddings
are cells by latent dimensions; metadata cell IDs must have exactly the same
row order. Training metadata must not contain cell-type evaluation labels.

Composition settings have the following structure:

```json
{
  "adaptive_mixing": {
    "semantic_intervention_fraction": 0.5,
    "effective_strength": 1.5
  },
  "effective_neural_fraction": 0.9
}
```

These values illustrate the schema, not an alternative tuning protocol.
Use the settings produced for the task. Zero-intervention tasks use the
corresponding `reference/<task_id>/embedding.npy` without applying new
semantic displacement. The reference must use float32 and include its
`metadata.csv` in the identical task cell order. Final results are written to
`results/integration/<domain>/<task_id>/`. Training metadata must include
`cell_id`, `modality` and `instance_batch`; evaluation metadata additionally
contains `cell_type`. Cell-type, fine-cluster, broad-class and source-cell
annotations are excluded from model outputs.

Bundled tasks are in the package's `resources/integration/tasks.json`.
Supply custom task definitions in `data/tasks/tasks.json` using the same
schema. Cortex tasks use `data/cortex/` and
`runs/integration/cortex/`, with cortical definitions supplied in
`data/cortex/tasks/tasks.json` or the corresponding bundled resource.
Cortex counts use sparse cells-by-features NPZ matrices in
`data/cortex/prepared/{rna,atac}_counts.npz` with matching
`{rna,atac}_feature_names.npy`. Each cortical task also requires
`data/cortex/indices/<task_id>/indices.npz` containing `task_order`,
`rna_observed`, `atac_observed`, `bridge`, `rna_query` and `atac_query` indices.

Each task record includes `task_id`, `scenario`, `replicate`, `modalities`
and `observations`. Each benchmark observation identifies `base_batch`,
`instance_batch`, `observed_modalities`, `identity_policy` and, for the R
metadata converter, `source_subdir`. Cortex observations additionally name
their `role`. Bundled benchmark designs retain the originating Palette
script identifiers and checksums for attribution and protocol traceability.

From processed per-library RDS matrices, the packaged R converters can be
called through `motrust prepare-data cache` and
`motrust prepare-data metadata --task-id TASK`. They require R with
`jsonlite` and `Matrix`; use `RSCRIPT` to select the executable. The metadata
converter expects the scenario-specific annotation columns recorded in the
task protocol, rather than inferring cell types. It produces separate
training and evaluation tables. These converters do not download raw data.

## Molecular input cache

`data/cache/cache_index.json` has a top-level `scenarios` object. Each
scenario maps library IDs to records containing `source_subdir`, `metadata`
and `modalities`. For example:

```json
{
  "scenarios": {
    "TEA_s1": {
      "B2": {
        "source_subdir": "TEA/B2",
        "metadata": "data/processed/TEA/B2/metadata.rds",
        "modalities": {
          "rna": {
            "matrix": "data/cache/TEA/B2/rna.mtx",
            "features": "data/cache/TEA/B2/rna_features.tsv",
            "barcodes": "data/cache/TEA/B2/rna_barcodes.tsv"
          },
          "atac": {
            "matrix": "data/cache/TEA/B2/atac.mtx",
            "features": "data/cache/TEA/B2/atac_features.tsv",
            "barcodes": "data/cache/TEA/B2/atac_barcodes.tsv"
          }
        }
      }
    }
  }
}
```

This is a format example, not a complete experiment. The three-source RNA
workflow requires TEA_s1, BMMC_s1 and Retina, with all intended paired
libraries. Matrices use MatrixMarket format in features-by-cells orientation.
Feature/barcode text files have one name per line; a one-column CSV is also
supported. RNA and ATAC barcode order must agree within each paired library.
Metadata must preserve the original barcodes and annotations. Cache paths
must resolve within the selected experiment directory or be valid absolute
paths on the local machine.

Relative paths use the experiment directory by default. Set the optional
top-level `"path_base": "index"` field to resolve them relative to the
directory containing `cache_index.json` instead.

The preparation workflow writes training-selected feature identities,
split indices and input hashes. Preserve these files together with the
trained model and result records.
