"""Tests for source-only selective transfer calibration."""

from __future__ import annotations

import numpy as np

from motrust.evaluation import SourceConditionalConformal, exact_rejection_mask


def test_exact_rejection_mask_is_tie_safe() -> None:
    rejected = exact_rejection_mask(np.ones(13), 0.1)
    assert rejected.sum() == 2


def test_source_conditional_conformal_flags_far_queries() -> None:
    rng = np.random.default_rng(17)
    source = np.vstack((rng.normal(-2, 0.2, (60, 3)), rng.normal(2, 0.2, (60, 3))))
    labels = np.repeat(["a", "b"], 60)
    known = np.vstack((rng.normal(-2, 0.2, (20, 3)), rng.normal(2, 0.2, (20, 3))))
    novel = rng.normal(8, 0.2, (40, 3))
    model = SourceConditionalConformal(neighbors=10).fit(source, labels)
    _, _, known_score = model.predict(known)
    _, _, novel_score = model.predict(novel)
    assert float(np.median(novel_score)) > float(np.median(known_score))
