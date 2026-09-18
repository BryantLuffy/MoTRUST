"""Unit tests for the unequal-row shared-gene anchor."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from motrust.preprocessing import build_unpaired_shared_gene_anchor


def test_unpaired_anchor_accepts_unequal_cell_counts() -> None:
    genes = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "chromStart": [100, 500, 100],
            "chromEnd": [200, 600, 200],
            "strand": ["+", "-", "+"],
        }
    )
    peaks = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "chromStart": [90, 490, 90, 300],
            "chromEnd": [120, 520, 120, 330],
        }
    )
    rna = sparse.csr_matrix([[2, 0, 1], [0, 3, 1], [1, 1, 0], [2, 1, 0]])
    atac = sparse.csr_matrix([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    result = build_unpaired_shared_gene_anchor(
        rna,
        atac,
        genes,
        peaks,
        n_genes=3,
        dim=2,
        seed=3,
    )
    assert result.embedding.shape == (7, 2)
    assert result.reliability.shape == (7,)
    assert np.isfinite(result.embedding).all()
    assert result.manifest["cell_correspondence_required"] is False
