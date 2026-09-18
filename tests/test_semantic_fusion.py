"""Tests for continuous no-fallback semantic integration."""

from __future__ import annotations

import numpy as np

from motrust.integration import (
    balanced_neighbor_graph_embedding,
    balanced_neighbor_smoothing,
    batch_discriminant_nullspace,
    conditional_coral_align,
    conditional_location_scale_align,
    cross_group_snn_subspace,
    local_group_separability,
    prototype_transport_align,
    reliability_kernel_subspace,
    reliability_cluster_graph_align,
    reliability_mnn_align,
    reliability_reference_mnn_align,
    reliability_residual_fusion,
    reliability_soft_prototype_align,
    sparse_cell_topology_guard,
    topology_safe_residual_fusion,
    transport_complete_modality_views,
)


def test_residual_fusion_uses_reliability_continuously() -> None:
    semantic = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    neural = semantic[:, ::-1].copy()
    neural[3] += np.asarray([0.75, -0.25])
    low = reliability_residual_fusion(semantic, neural, np.zeros(4), neural_fraction=0.5)
    high = reliability_residual_fusion(semantic, neural, np.ones(4), neural_fraction=0.5)
    assert low.shape == semantic.shape
    assert np.isfinite(high).all()
    assert not np.allclose(low, high)


def test_topology_safe_fusion_rejects_damaging_candidate() -> None:
    rng = np.random.default_rng(73)
    anchor = np.vstack(
        (rng.normal(loc=-2.0, scale=0.2, size=(40, 4)), rng.normal(loc=2.0, scale=0.2, size=(40, 4)))
    ).astype(np.float32)
    candidate = rng.permutation(anchor)
    output, audit = topology_safe_residual_fusion(
        anchor,
        candidate,
        np.ones(len(anchor)),
        neighbors=10,
        maximum_fraction=1.0,
        minimum_global_recall=0.9,
    )
    assert output.shape == anchor.shape
    assert np.isfinite(output).all()
    assert audit["fused_neighbor_recall_mean"] >= 0.9
    assert audit["global_scale"] < 1.0 or audit["rejection_rate"] > 0.0


def test_topology_safe_fusion_accepts_concordant_candidate() -> None:
    rng = np.random.default_rng(79)
    anchor = rng.normal(size=(80, 5)).astype(np.float32)
    candidate = anchor + rng.normal(scale=0.01, size=anchor.shape)
    _, audit = topology_safe_residual_fusion(
        anchor, candidate, np.ones(len(anchor)), neighbors=10
    )
    assert audit["global_scale"] == 1.0
    assert np.mean(audit["effective_weight"]) > 0.0


def test_sparse_guard_changes_only_requested_density_fraction() -> None:
    rng = np.random.default_rng(83)
    anchor = rng.normal(size=(100, 5)).astype(np.float32)
    anchor[:10] *= 5.0
    candidate = anchor + rng.normal(scale=0.5, size=anchor.shape)
    _, audit = sparse_cell_topology_guard(
        anchor,
        candidate,
        np.ones(len(anchor)),
        protected_fraction=0.2,
        neighbors=10,
    )
    assert 0.19 <= audit["protected_fraction"] <= 0.21


def test_conditional_alignment_is_finite_and_shape_stable() -> None:
    rng = np.random.default_rng(3)
    embedding = rng.normal(size=(60, 4)).astype(np.float32)
    groups = np.repeat(["a", "b", "c"], 20)
    embedding[groups == "b"] += 2.0
    corrected = conditional_location_scale_align(
        embedding, groups, clusters=3, iterations=2, random_state=5
    )
    assert corrected.shape == embedding.shape
    assert np.isfinite(corrected).all()


def test_prototype_transport_is_finite_and_shape_stable() -> None:
    rng = np.random.default_rng(23)
    first = rng.normal(size=(50, 4)).astype(np.float32)
    second = first + np.asarray([3.0, -2.0, 1.0, 0.5], dtype=np.float32)
    embedding = np.vstack([first, second])
    groups = np.repeat(["a", "b"], 50)
    corrected = prototype_transport_align(
        embedding, groups, prototypes=5, random_state=4
    )
    assert corrected.shape == embedding.shape
    assert np.isfinite(corrected).all()


def test_balanced_graph_embedding_is_finite_and_shape_stable() -> None:
    rng = np.random.default_rng(31)
    first = rng.normal(size=(30, 4)).astype(np.float32)
    second = first + 1.5
    embedding = np.vstack([first, second])
    groups = np.repeat(["a", "b"], 30)
    corrected = balanced_neighbor_graph_embedding(
        embedding, groups, neighbors_per_group=3, dimensions=4
    )
    assert corrected.shape == embedding.shape
    assert np.isfinite(corrected).all()


def test_balanced_neighbor_smoothing_is_finite_and_shape_stable() -> None:
    rng = np.random.default_rng(37)
    first = rng.normal(size=(30, 4)).astype(np.float32)
    embedding = np.vstack([first, first + 2.0])
    groups = np.repeat(["a", "b"], 30)
    corrected = balanced_neighbor_smoothing(
        embedding, groups, neighbors_per_group=3, iterations=1, strength=0.5
    )
    assert corrected.shape == embedding.shape
    assert np.isfinite(corrected).all()


def test_conditional_coral_is_finite_and_shape_stable() -> None:
    rng = np.random.default_rng(13)
    embedding = rng.normal(size=(120, 4)).astype(np.float32)
    groups = np.repeat(["a", "b", "c"], 40)
    embedding[groups == "b"] = embedding[groups == "b"] @ np.diag([2.0, 0.5, 1.5, 0.7])
    corrected = conditional_coral_align(
        embedding, groups, clusters=3, iterations=1, random_state=2
    )
    assert corrected.shape == embedding.shape
    assert np.isfinite(corrected).all()


def test_reliability_kernel_subspace_is_finite_and_deterministic() -> None:
    rng = np.random.default_rng(7)
    biology = np.repeat([[-2.0, 0.0], [2.0, 0.0]], 80, axis=0)
    groups = np.tile(np.repeat(["a", "b"], 40), 2)
    offsets = np.where(groups[:, None] == "a", [0.0, -1.0], [0.0, 1.0])
    values = biology + offsets + rng.normal(scale=0.25, size=biology.shape)
    reliability = np.ones(len(values), dtype=np.float32)
    first = reliability_kernel_subspace(
        values,
        groups,
        reliability,
        clusters_per_group=2,
        dimensions=2,
        representatives_per_cluster=24,
        random_state=11,
    )
    second = reliability_kernel_subspace(
        values,
        groups,
        reliability,
        clusters_per_group=2,
        dimensions=2,
        representatives_per_cluster=24,
        random_state=11,
    )
    assert first.shape == values.shape
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_transport_completion_fills_diagonal_mosaic_without_bridge() -> None:
    rng = np.random.default_rng(19)
    semantic = rng.normal(size=(40, 4)).astype(np.float32)
    modality_embeddings = {
        "rna": semantic[:20] + 0.05,
        "atac": semantic[20:] - 0.05,
    }
    observed_rows = {"rna": np.arange(20), "atac": np.arange(20, 40)}
    modality_reliability = {
        "rna": np.r_[np.ones(20), np.zeros(20)],
        "atac": np.r_[np.zeros(20), np.ones(20)],
    }
    completed = transport_complete_modality_views(
        semantic,
        modality_embeddings,
        observed_rows,
        modality_reliability,
        neighbors=5,
        dimensions=4,
        random_state=3,
    )
    assert completed.shape == semantic.shape
    assert np.isfinite(completed).all()


def test_soft_prototype_alignment_is_finite() -> None:
    rng = np.random.default_rng(29)
    biology = np.repeat([[-2.0, 0.0], [2.0, 0.0]], 60, axis=0)
    groups = np.tile(np.repeat(["a", "b"], 30), 2)
    offsets = np.where(groups[:, None] == "a", [0.0, -1.5], [0.0, 1.5])
    values = biology + offsets + rng.normal(scale=0.3, size=biology.shape)
    output = reliability_soft_prototype_align(
        values,
        groups,
        np.ones(len(values)),
        prototypes=2,
        iterations=3,
        random_state=4,
    )
    assert output.shape == values.shape
    assert np.isfinite(output).all()


def test_reliability_mnn_alignment_is_finite() -> None:
    rng = np.random.default_rng(41)
    first = rng.normal(size=(50, 4))
    values = np.vstack((first, first + np.asarray([0.5, -0.5, 0.25, 0.0])))
    groups = np.repeat(["a", "b"], 50)
    output = reliability_mnn_align(
        values,
        groups,
        np.ones(len(values)),
        neighbors=5,
        smoothing_neighbors=10,
    )
    assert output.shape == values.shape
    assert np.isfinite(output).all()


def test_reference_mnn_uses_fixed_matching_geometry() -> None:
    rng = np.random.default_rng(61)
    reference = rng.normal(size=(50, 4))
    matching = np.vstack((reference, reference))
    shifted = np.vstack((reference, reference + np.asarray([2.0, -1.0, 0.5, 0.0])))
    groups = np.repeat(["a", "b"], 50)
    output = reliability_reference_mnn_align(
        shifted,
        matching,
        groups,
        np.ones(len(shifted)),
        neighbors=5,
        smoothing_neighbors=10,
    )
    assert output.shape == shifted.shape
    assert np.isfinite(output).all()
    before = np.linalg.norm(shifted[:50].mean(axis=0) - shifted[50:].mean(axis=0))
    after = np.linalg.norm(output[:50].mean(axis=0) - output[50:].mean(axis=0))
    assert after < before


def test_cross_group_snn_subspace_is_finite() -> None:
    rng = np.random.default_rng(43)
    values = rng.normal(size=(80, 8))
    groups = np.repeat(["a", "b"], 40)
    values[groups == "b"] += 0.5
    output = cross_group_snn_subspace(
        values,
        groups,
        np.ones(len(values)),
        graph_neighbors=10,
        cross_group_neighbors=3,
        dimensions=4,
    )
    assert output.shape == (80, 4)
    assert np.isfinite(output).all()


def test_cluster_graph_alignment_is_finite() -> None:
    rng = np.random.default_rng(47)
    first = rng.normal(size=(60, 4))
    values = np.vstack((first, first + np.asarray([1.0, -0.5, 0.0, 0.25])))
    groups = np.repeat(["a", "b"], 60)
    output = reliability_cluster_graph_align(
        values,
        groups,
        np.ones(len(values)),
        clusters_per_group=4,
        iterations=2,
    )
    assert output.shape == values.shape
    assert np.isfinite(output).all()


def test_batch_discriminant_nullspace_is_finite() -> None:
    rng = np.random.default_rng(53)
    values = rng.normal(size=(100, 6))
    groups = np.repeat(["a", "b"], 50)
    values[groups == "b", :2] += 2.0
    output = batch_discriminant_nullspace(values, groups)
    assert output.shape == values.shape
    assert np.isfinite(output).all()


def test_local_group_separability_detects_group_offsets() -> None:
    rng = np.random.default_rng(59)
    mixed = rng.normal(size=(100, 4))
    groups = np.repeat(["a", "b"], 50)
    separated = mixed.copy()
    separated[groups == "b", 0] += 8.0
    assert local_group_separability(separated, groups) > local_group_separability(mixed, groups)
