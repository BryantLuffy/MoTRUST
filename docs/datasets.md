# Public data sources

Experimental matrices, raw reads, fragments, checkpoints and generated samples are obtained or produced separately. Dataset accessions identify the original studies; the processed input release and saved cell/feature identities identify what was actually analyzed. Downloading an original study alone does not recreate the prepared task cache.

## Primary integration and RNA-distribution sources

| Source | Original study | Analyzed processed input |
|---|---|---|
| TEA | [GSE158013](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158013) | Four PBMC libraries with RNA/ATAC/ADT; 25,517 unique paired cells in the ATAC-to-RNA workflow. |
| BMMC | [GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122) | Integration uses CITE and Multiome libraries. The RNA-distribution workflow uses only the three Multiome libraries, totaling 17,243 unique paired cells. |
| Retina | [GSE196235](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE196235) | Eight RNA/ATAC libraries; 50,312 unique paired cells. |
| Ab-seq | [Proteogenomic reference collection](https://figshare.com/projects/Single-cell_proteogenomic_reference_maps_of_the_human_hematopoietic_system/94469) | Six RNA/ADT libraries; integration only in these workflows. |

The processed matrices, supplied feature vocabularies and annotations used for these integration tasks came from the [data release accompanying Palette](https://doi.org/10.5281/zenodo.18045027). This is an input-data provenance statement, not a requirement to run Palette to train MoTRUST. Dataset use remains subject to each provider's conditions.

The three-source RNA workflow deduplicates integration scenario copies using original library and barcode. It splits each original library into 60% training, 20% validation and 20% evaluation with seed 913, then selects 2,000 RNA and 5,000 ATAC features from training observations. The recorded evaluation counts are 5,105 TEA, 3,449 BMMC-Multiome and 10,065 Retina cells. These are cell holdouts within libraries, not held-out donors.

## Point-recovery datasets

| Display name | Public source / import route |
|---|---|
| PBMC-10x | [10x healthy-donor PBMC Multiome](https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-1-0-0), imported through `SingleCellMultiModal::scMultiome("pbmc_10x")`. The resource is described in [Eckenrode et al.](https://doi.org/10.1371/journal.pcbi.1011324). |
| Chen-2019 | [GSE126074](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126074), mouse cortical SNARE-seq. |
| Ma-2020 | [GSE140203](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203), mouse skin SHARE-seq. |
| DOGMA | [GSE166188](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166188); the point-recovery evaluation uses the measured RNA/ADT subset. |
| Donor-separated PBMC evaluation | [GSE297529](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE297529). |

PBMC-10x has the historical executable key `MAE`. The analyzed package input contains **10,032 matched profiles, 36,549 genes and 108,344 peaks** before task-specific feature selection. The original 10x release summary reports 11,909 nuclei; do not substitute that larger original matrix and assume byte-identical input. No undocumented package filtering procedure is inferred here. Ma-2020 belongs to point-recovery development and is not one of the three primary RNA-DDPM sources above.

## Cortical and paired cross-species sources

- Human cortical Multiome: [BICCN challenge data](https://ucdnjj.github.io/data/), `10XMultiome/Human`. The adopted cortical analysis uses fixed cells and fragment-reconstructed ATAC inputs; the raw release matrix alone is not the full frozen input contract.
- Human motor-cortex SNARE-seq: [NeMO motor-cortex archive](https://assets.nemoarchive.org/dat-ek5dbmu).
- Mouse motor-cortex Multiome: [GSE229169](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE229169). The paired human–mouse analysis used processed inputs from the Palette data release.

These sources document experimental provenance. Dataset downloading and prepared-input archives are managed separately. Use `motrust integrate ... --check` or `motrust rna-diffusion ... --check` to inspect the inputs required by a workflow, and retain original cell/feature identities and observation masks when preparing them.
