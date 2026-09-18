"""Tests for the no-bridge shared semantic anchor."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from motrust.preprocessing import (
    build_shared_semantic_anchor,
    nearest_tss_mapping,
    parse_peak_names,
)


def test_antibody_suffix_is_normalized() -> None:
    from motrust.preprocessing import normalize_gene_symbol

    assert normalize_gene_symbol("CD14-AB") == "CD14"


def test_peak_parsing_and_nearest_tss_mapping() -> None:
    peaks = np.asarray(["chr1-90-110", "chr1:490-510", "invalid"])
    parsed = parse_peak_names(peaks)
    assert parsed["feature_index"].tolist() == [0, 1]
    annotation = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 500],
            "end": [150, 550],
            "symbol": ["G1", "G2"],
            "strand": ["+", "+"],
            "tss": [100, 500],
        }
    )
    mapping = nearest_tss_mapping(peaks, np.asarray(["G1", "G2"]), annotation)
    assert mapping.shape == (3, 2)
    assert mapping.nnz == 2
    assert np.array_equal(mapping.nonzero()[0], np.asarray([0, 1]))


def test_anchor_uses_shared_features_without_paired_cells(tmp_path) -> None:
    annotation = tmp_path / "genes.bed"
    annotation.write_text("chr1\t100\t150\tG1\t+\nchr1\t500\t550\tG2\t+\n", encoding="utf-8")
    rna = sparse.csr_matrix(
        [[8, 1], [7, 1], [0, 0], [0, 0]], dtype=np.float32
    )
    atac = sparse.csr_matrix(
        [[0, 0], [0, 0], [9, 1], [8, 1]], dtype=np.float32
    )
    result = build_shared_semantic_anchor(
        {"rna": rna, "atac": atac},
        {
            "rna": np.asarray([1, 1, 0, 0], dtype=np.float32),
            "atac": np.asarray([0, 0, 1, 1], dtype=np.float32),
        },
        {
            "rna": np.asarray(["G1", "G2"]),
            "atac": np.asarray(["chr1-90-110", "chr1-490-510"]),
        },
        annotation_path=annotation,
        dim=2,
        n_genes=2,
        seed=7,
    )
    assert result.embedding.shape == (4, 2)
    assert np.isfinite(result.embedding).all()
    assert (result.reliability > 0).all()
    assert result.manifest["cell_correspondence_required"] is False
