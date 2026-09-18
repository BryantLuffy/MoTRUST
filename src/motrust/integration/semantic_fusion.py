"""Continuous semantic residual fusion and prototype-conditional alignment."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.linalg import orthogonal_procrustes
from scipy.linalg import eigh as generalized_eigh
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _standardize(values: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(values, dtype=np.float32)).astype(np.float32)


def local_group_separability(
    embedding: np.ndarray,
    groups: np.ndarray,
    *,
    neighbors: int = 30,
) -> float:
    """Return chance-corrected local group purity in ``[0, 1]``."""

    values = _standardize(embedding)
    groups = np.asarray(groups, dtype=str)
    if len(groups) != len(values):
        raise ValueError("groups must contain one value per cell")
    if len(np.unique(groups)) < 2 or len(values) < 3:
        return 0.0
    k = min(max(1, int(neighbors)), len(values) - 1)
    indices = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(values).kneighbors(
        values, return_distance=False
    )[:, 1:]
    observed = float(np.mean(groups[indices] == groups[:, None]))
    _, counts = np.unique(groups, return_counts=True)
    expected = float(np.sum(np.square(counts / len(groups))))
    return float(np.clip((observed - expected) / max(1.0 - expected, 1e-8), 0.0, 1.0))


def reliability_residual_fusion(
    semantic: np.ndarray,
    neural: np.ndarray,
    reliability: np.ndarray,
    *,
    neural_fraction: float = 0.2,
) -> np.ndarray:
    """Fuse every cell through one semantic-to-neural residual path.

    The neural coordinates are orthogonally aligned to the no-bridge semantic
    anchor. Reliability continuously controls the residual contribution; no
    dataset or cell is switched to a separate fallback representation.
    """

    semantic_z = _standardize(semantic)
    neural_z = _standardize(neural)
    if semantic_z.shape != neural_z.shape:
        raise ValueError("semantic and neural embeddings must have the same shape")
    rho = np.asarray(reliability, dtype=np.float32).reshape(-1)
    if len(rho) != len(semantic_z):
        raise ValueError("reliability must contain one value per cell")
    rotation, _ = orthogonal_procrustes(neural_z, semantic_z)
    aligned_neural = neural_z @ rotation
    weight = np.clip(float(neural_fraction) * rho, 0.0, 1.0)[:, None]
    return ((1.0 - weight) * semantic_z + weight * aligned_neural).astype(np.float32)


def _neighbor_recall(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return per-row recall of a fixed reference neighbor set."""

    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate neighbor arrays must have the same shape")
    return np.asarray(
        [len(set(left).intersection(right)) / max(len(left), 1) for left, right in zip(reference, candidate)],
        dtype=np.float32,
    )


def topology_safe_residual_fusion(
    anchor: np.ndarray,
    candidate: np.ndarray,
    modality_reliability: np.ndarray,
    *,
    neighbors: int = 30,
    maximum_fraction: float = 0.5,
    minimum_local_trust: float = 0.15,
    minimum_global_recall: float = 0.9,
    distortion_temperature: float = 0.5,
    sparse_cell_protection: float = 0.5,
    maximum_backtracking_steps: int = 8,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    """Fuse a candidate view while preserving the label-free anchor topology.

    The candidate is first orthogonally aligned to the RNA (or other trusted)
    anchor.  Its cell-wise contribution is then limited by modality quality,
    anchor-neighborhood recall, local edge distortion, and anchor density.
    A global backtracking step guarantees a minimum mean neighborhood recall.
    Cells whose topology trust is unsupported receive exactly zero residual and
    therefore fall back to the anchor without consulting biological labels.
    """

    anchor_z = _standardize(anchor)
    candidate_z = _standardize(candidate)
    if anchor_z.shape != candidate_z.shape:
        raise ValueError("anchor and candidate embeddings must have the same shape")
    if len(anchor_z) < 3:
        raise ValueError("at least three cells are required for topology-safe fusion")
    reliability = np.clip(
        np.asarray(modality_reliability, dtype=np.float32).reshape(-1), 0.0, 1.0
    )
    if len(reliability) != len(anchor_z):
        raise ValueError("modality_reliability must contain one value per cell")

    rotation, _ = orthogonal_procrustes(candidate_z, anchor_z)
    aligned = (candidate_z @ rotation).astype(np.float32)
    k = min(max(1, int(neighbors)), len(anchor_z) - 1)
    anchor_model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1).fit(anchor_z)
    anchor_distances, anchor_indices = anchor_model.kneighbors(anchor_z)
    anchor_distances = anchor_distances[:, 1:]
    anchor_indices = anchor_indices[:, 1:]
    candidate_indices = NearestNeighbors(
        n_neighbors=k + 1, metric="euclidean", n_jobs=1
    ).fit(aligned).kneighbors(aligned, return_distance=False)[:, 1:]
    local_recall = _neighbor_recall(anchor_indices, candidate_indices)

    rows = np.arange(len(anchor_z))[:, None]
    candidate_edge_distance = np.linalg.norm(
        aligned[rows] - aligned[anchor_indices], axis=2
    )
    ratio = candidate_edge_distance / np.clip(anchor_distances, 1e-6, None)
    edge_distortion = np.median(np.abs(np.log(np.clip(ratio, 1e-6, 1e6))), axis=1)
    distortion_trust = np.exp(
        -edge_distortion / max(float(distortion_temperature), 1e-6)
    ).astype(np.float32)

    local_radius = anchor_distances[:, -1]
    low, high = np.quantile(local_radius, [0.1, 0.9])
    sparse_risk = np.clip((local_radius - low) / max(float(high - low), 1e-8), 0.0, 1.0)
    density_guard = 1.0 - np.clip(float(sparse_cell_protection), 0.0, 1.0) * sparse_risk
    topology_trust = np.clip(local_recall * distortion_trust * density_guard, 0.0, 1.0)
    rejected = topology_trust < float(minimum_local_trust)
    topology_trust[rejected] = 0.0
    cell_weight = (
        np.clip(float(maximum_fraction), 0.0, 1.0) * reliability * topology_trust
    ).astype(np.float32)

    global_scale = 1.0
    fused = anchor_z.copy()
    achieved_recall = 1.0
    steps = 0
    for steps in range(max(0, int(maximum_backtracking_steps)) + 1):
        effective_weight = global_scale * cell_weight
        fused = anchor_z + effective_weight[:, None] * (aligned - anchor_z)
        fused_indices = NearestNeighbors(
            n_neighbors=k + 1, metric="euclidean", n_jobs=1
        ).fit(fused).kneighbors(fused, return_distance=False)[:, 1:]
        achieved_recall = float(_neighbor_recall(anchor_indices, fused_indices).mean())
        if achieved_recall >= float(minimum_global_recall):
            break
        global_scale *= 0.5

    effective_weight = (global_scale * cell_weight).astype(np.float32)
    diagnostics: dict[str, np.ndarray | float | int] = {
        "local_neighbor_recall": local_recall,
        "edge_distortion": edge_distortion.astype(np.float32),
        "sparse_risk": sparse_risk.astype(np.float32),
        "topology_trust": topology_trust.astype(np.float32),
        "effective_weight": effective_weight,
        "rejected": rejected,
        "candidate_neighbor_recall_mean": float(local_recall.mean()),
        "fused_neighbor_recall_mean": achieved_recall,
        "global_scale": float(global_scale),
        "backtracking_steps": int(steps),
        "rejection_rate": float(rejected.mean()),
    }
    return _standardize(fused), diagnostics


def sparse_cell_topology_guard(
    anchor: np.ndarray,
    candidate: np.ndarray,
    modality_reliability: np.ndarray,
    *,
    protected_fraction: float = 0.25,
    **topology_kwargs,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    """Apply anchor protection only to label-free low-density candidate cells."""

    safe, diagnostics = topology_safe_residual_fusion(
        anchor, candidate, modality_reliability, **topology_kwargs
    )
    candidate_z = _standardize(candidate)
    rotation, _ = orthogonal_procrustes(safe, candidate_z)
    safe_aligned = safe @ rotation
    fraction = np.clip(float(protected_fraction), 0.0, 1.0)
    if fraction == 0.0:
        protected = np.zeros(len(candidate_z), dtype=bool)
    else:
        threshold = np.quantile(diagnostics["sparse_risk"], 1.0 - fraction)
        protected = np.asarray(diagnostics["sparse_risk"]) >= threshold
    guarded = candidate_z.copy()
    guarded[protected] = safe_aligned[protected]
    diagnostics = dict(diagnostics)
    diagnostics["protected"] = protected
    diagnostics["protected_fraction"] = float(protected.mean())
    return _standardize(guarded), diagnostics


def transport_complete_modality_views(
    semantic: np.ndarray,
    modality_embeddings: dict[str, np.ndarray],
    observed_rows: dict[str, np.ndarray],
    modality_reliability: dict[str, np.ndarray],
    *,
    neighbors: int = 20,
    temperature: float = 0.15,
    dimensions: int = 32,
    random_state: int = 0,
) -> np.ndarray:
    """Complete latent modality views through a shared, no-bridge semantic space.

    Each missing view is a reliability-weighted local transport barycenter of
    observed target-modality cells. The query geometry is the shared-gene
    semantic anchor, so the operation remains defined for diagonal mosaics
    without paired bridge cells. Observed views are never overwritten.
    """

    semantic_z = _standardize(semantic)
    semantic_unit = semantic_z / np.clip(
        np.linalg.norm(semantic_z, axis=1, keepdims=True), 1e-8, None
    )
    n_cells = len(semantic_z)
    completed: list[np.ndarray] = []
    for modality in sorted(modality_embeddings):
        rows = np.asarray(observed_rows[modality], dtype=np.int64)
        values = np.asarray(modality_embeddings[modality], dtype=np.float32)
        if len(rows) != len(values):
            raise ValueError(f"Observed row count differs for {modality}")
        if len(rows) < 2:
            continue
        quality = np.asarray(modality_reliability[modality], dtype=np.float32).reshape(-1)
        if len(quality) != n_cells:
            raise ValueError(f"Reliability row count differs for {modality}")
        values = _standardize(values)
        full = np.zeros((n_cells, values.shape[1]), dtype=np.float32)
        full[rows] = values
        missing = np.ones(n_cells, dtype=bool)
        missing[rows] = False
        missing_rows = np.flatnonzero(missing)
        if len(missing_rows):
            k = min(max(1, int(neighbors)), len(rows))
            distances, indices = NearestNeighbors(
                n_neighbors=k,
                metric="cosine",
                n_jobs=1,
            ).fit(semantic_unit[rows]).kneighbors(semantic_unit[missing_rows])
            target_rows = rows[indices]
            weights = np.exp(-distances / max(float(temperature), 1e-4)).astype(np.float32)
            weights *= np.clip(quality[target_rows], 0.05, 1.0)
            weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
            full[missing_rows] = np.einsum("ij,ijk->ik", weights, values[indices])
        completed.append(_standardize(full))
    if len(completed) < 2:
        raise ValueError("At least two modality views are required for completion")
    concatenated = np.concatenate(completed, axis=1)
    output_dimensions = min(int(dimensions), concatenated.shape[1], concatenated.shape[0] - 1)
    output = PCA(
        n_components=output_dimensions,
        svd_solver="randomized",
        random_state=int(random_state),
    ).fit_transform(concatenated)
    return _standardize(output)


def conditional_location_scale_align(
    embedding: np.ndarray,
    groups: np.ndarray,
    *,
    clusters: int = 40,
    iterations: int = 2,
    strength: float = 1.0,
    shrinkage: float = 20.0,
    scale_strength: float = 1.0,
    random_state: int = 0,
) -> np.ndarray:
    """Align group distributions inside label-free semantic prototypes.

    Local group moments are shrunk toward each prototype's pooled moments.
    The operation targets modality/batch offsets conditional on inferred
    biology, avoiding a global correction that would erase cell states.
    """

    corrected = _standardize(embedding)
    groups = np.asarray(groups, dtype=str)
    if len(groups) != len(corrected):
        raise ValueError("groups must contain one value per cell")
    for iteration in range(int(iterations)):
        assignments = MiniBatchKMeans(
            n_clusters=min(int(clusters), len(corrected)),
            batch_size=2048,
            n_init=5,
            random_state=int(random_state) + iteration,
        ).fit_predict(corrected)
        updated = corrected.copy()
        for cluster in np.unique(assignments):
            cluster_mask = assignments == cluster
            pooled = corrected[cluster_mask]
            if len(pooled) < 3:
                continue
            pooled_mean = pooled.mean(axis=0)
            pooled_std = pooled.std(axis=0).clip(0.1, None)
            for group in np.unique(groups[cluster_mask]):
                selected = cluster_mask & (groups == group)
                count = int(selected.sum())
                if count < 2:
                    continue
                local = corrected[selected]
                local_mean = local.mean(axis=0)
                local_std = local.std(axis=0).clip(0.1, None)
                centered = local - local_mean + pooled_mean
                scaled = (local - local_mean) * (pooled_std / local_std) + pooled_mean
                target = (1.0 - float(scale_strength)) * centered + float(scale_strength) * scaled
                reliability = count / (count + float(shrinkage))
                amount = np.clip(float(strength) * reliability, 0.0, 1.0)
                updated[selected] = (1.0 - amount) * local + amount * target
        corrected = updated
    return corrected.astype(np.float32)


def reliability_soft_prototype_align(
    embedding: np.ndarray,
    groups: np.ndarray,
    reliability: np.ndarray | None = None,
    *,
    prototypes: int = 40,
    iterations: int = 10,
    assignment_temperature: float = 0.2,
    diversity_strength: float = 1.0,
    correction_strength: float = 1.0,
    shrinkage: float = 20.0,
    random_state: int = 0,
) -> np.ndarray:
    """Remove supported group offsets inside reliability-weighted soft prototypes.

    The soft assignments are encouraged to use evidence from all groups, but a
    group correction is estimated only where its effective prototype support is
    sufficient. This targets modality mixing without requiring cell labels or
    forcing unmatched biological populations to overlap.
    """

    original = _standardize(embedding).astype(np.float64)
    corrected = original.copy()
    groups = np.asarray(groups, dtype=str)
    if len(groups) != len(original):
        raise ValueError("groups must contain one value per cell")
    unique_groups, group_index = np.unique(groups, return_inverse=True)
    if len(unique_groups) < 2:
        return original.astype(np.float32)
    if reliability is None:
        rho = np.ones(len(original), dtype=np.float64)
    else:
        rho = np.clip(np.asarray(reliability, dtype=np.float64).reshape(-1), 0.05, 1.0)
        if len(rho) != len(original):
            raise ValueError("reliability must contain one value per cell")
    k = min(max(2, int(prototypes)), len(original))
    group_one_hot = np.eye(len(unique_groups), dtype=np.float64)[group_index]
    group_frequency = group_one_hot.mean(axis=0)
    model = MiniBatchKMeans(
        n_clusters=k,
        batch_size=2048,
        n_init=10,
        random_state=int(random_state),
    ).fit(corrected)
    centers = model.cluster_centers_.astype(np.float64)
    previous = corrected.copy()
    for _ in range(int(iterations)):
        unit = corrected / np.clip(np.linalg.norm(corrected, axis=1, keepdims=True), 1e-8, None)
        center_unit = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8, None)
        distance = np.clip(2.0 - 2.0 * (unit @ center_unit.T), 0.0, None)
        logits = -distance / max(2.0 * float(assignment_temperature) ** 2, 1e-4)
        logits -= logits.max(axis=1, keepdims=True)
        responsibilities = np.exp(logits) * rho[:, None]
        observed = responsibilities.T @ group_one_hot
        expected = responsibilities.sum(axis=0)[:, None] * group_frequency[None, :]
        diversity = ((expected + 1.0) / (observed + 1.0)) ** float(diversity_strength)
        responsibilities *= diversity[:, group_index].T
        responsibilities /= np.clip(responsibilities.sum(axis=1, keepdims=True), 1e-12, None)

        prototype_mass = responsibilities.sum(axis=0)
        pooled_means = (responsibilities.T @ original) / np.clip(
            prototype_mass[:, None], 1e-8, None
        )
        correction = np.zeros_like(original)
        for group_value in range(len(unique_groups)):
            selected = group_index == group_value
            local_responsibilities = responsibilities[selected]
            local_mass = local_responsibilities.sum(axis=0)
            local_means = (local_responsibilities.T @ original[selected]) / np.clip(
                local_mass[:, None], 1e-8, None
            )
            support = local_mass / (local_mass + float(shrinkage))
            offsets = (local_means - pooled_means) * support[:, None]
            correction[selected] = local_responsibilities @ offsets
        corrected = original - float(correction_strength) * correction
        centers = (responsibilities.T @ corrected) / np.clip(
            responsibilities.sum(axis=0)[:, None], 1e-8, None
        )
        change = float(np.mean(np.square(corrected - previous)))
        previous = corrected.copy()
        if change < 1e-7:
            break
    return corrected.astype(np.float32)


def _symmetric_matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-4, None) ** power
    return (vectors * values) @ vectors.T


def conditional_coral_align(
    embedding: np.ndarray,
    groups: np.ndarray,
    *,
    clusters: int = 40,
    iterations: int = 2,
    strength: float = 1.0,
    shrinkage: float = 20.0,
    covariance_regularization: float = 0.05,
    random_state: int = 0,
) -> np.ndarray:
    """Align full group covariance inside label-free semantic prototypes."""

    corrected = _standardize(embedding).astype(np.float64)
    groups = np.asarray(groups, dtype=str)
    dimensions = corrected.shape[1]
    identity = np.eye(dimensions)
    for iteration in range(int(iterations)):
        assignments = MiniBatchKMeans(
            n_clusters=min(int(clusters), len(corrected)),
            batch_size=2048,
            n_init=5,
            random_state=int(random_state) + iteration,
        ).fit_predict(corrected)
        updated = corrected.copy()
        for cluster in np.unique(assignments):
            cluster_mask = assignments == cluster
            pooled = corrected[cluster_mask]
            if len(pooled) < 3:
                continue
            pooled_mean = pooled.mean(axis=0)
            pooled_covariance = np.cov(pooled, rowvar=False)
            pooled_covariance = (
                (1.0 - covariance_regularization) * pooled_covariance
                + covariance_regularization * identity
            )
            pooled_root = _symmetric_matrix_power(pooled_covariance, 0.5)
            for group in np.unique(groups[cluster_mask]):
                selected = cluster_mask & (groups == group)
                count = int(selected.sum())
                if count < max(3, dimensions // 2):
                    continue
                local = corrected[selected]
                local_mean = local.mean(axis=0)
                local_covariance = np.cov(local, rowvar=False)
                local_covariance = (
                    (1.0 - covariance_regularization) * local_covariance
                    + covariance_regularization * identity
                )
                target = (
                    (local - local_mean)
                    @ _symmetric_matrix_power(local_covariance, -0.5)
                    @ pooled_root
                    + pooled_mean
                )
                reliability = count / (count + float(shrinkage))
                amount = np.clip(float(strength) * reliability, 0.0, 1.0)
                updated[selected] = (1.0 - amount) * local + amount * target
        corrected = updated
    return corrected.astype(np.float32)


def prototype_transport_align(
    embedding: np.ndarray,
    groups: np.ndarray,
    *,
    prototypes: int = 40,
    strength: float = 1.0,
    residual_scale: float = 1.0,
    random_state: int = 0,
) -> np.ndarray:
    """Transport independently learned group prototypes to a common reference.

    Group-wise standardization removes global technical moments before matching.
    Hungarian prototype correspondence then supplies a discrete, label-free
    approximation to unbalanced transport; cells retain their local residuals.
    """

    source = _standardize(embedding)
    groups = np.asarray(groups, dtype=str)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return source
    reference_group = max(unique_groups, key=lambda value: int(np.sum(groups == value)))
    group_models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    k = min(int(prototypes), min(int(np.sum(groups == value)) for value in unique_groups))
    if k < 2:
        return source
    for offset, group in enumerate(unique_groups):
        rows = np.flatnonzero(groups == group)
        local = source[rows]
        local_z = StandardScaler().fit_transform(local).astype(np.float32)
        assignments = MiniBatchKMeans(
            n_clusters=k,
            batch_size=2048,
            n_init=10,
            random_state=int(random_state) + offset,
        ).fit_predict(local_z)
        standardized_centers = np.vstack(
            [local_z[assignments == cluster].mean(axis=0) for cluster in range(k)]
        )
        original_centers = np.vstack(
            [local[assignments == cluster].mean(axis=0) for cluster in range(k)]
        )
        group_models[str(group)] = (assignments, standardized_centers, original_centers)

    reference_assignments, reference_z_centers, reference_centers = group_models[str(reference_group)]
    del reference_assignments
    corrected = source.copy()
    for group in unique_groups:
        if group == reference_group:
            continue
        rows = np.flatnonzero(groups == group)
        assignments, standardized_centers, original_centers = group_models[str(group)]
        cost = np.square(
            standardized_centers[:, None, :] - reference_z_centers[None, :, :]
        ).mean(axis=2)
        source_index, target_index = linear_sum_assignment(cost)
        target_lookup = np.empty(k, dtype=np.int64)
        target_lookup[source_index] = target_index
        for cluster in range(k):
            selected = rows[assignments == cluster]
            if len(selected) == 0:
                continue
            local_center = original_centers[cluster]
            target_center = reference_centers[target_lookup[cluster]]
            residual = source[selected] - local_center
            target = target_center + float(residual_scale) * residual
            corrected[selected] = (
                (1.0 - float(strength)) * source[selected] + float(strength) * target
            )
    return corrected.astype(np.float32)


def reliability_cluster_graph_align(
    embedding: np.ndarray,
    groups: np.ndarray,
    reliability: np.ndarray | None = None,
    *,
    clusters_per_group: int = 20,
    reciprocal_neighbors: int = 2,
    minimum_similarity: float = 0.2,
    strength: float = 1.0,
    residual_scale: float = 1.0,
    iterations: int = 1,
    random_state: int = 0,
) -> np.ndarray:
    """Align many-to-many batch clusters supported by reciprocal graph links."""

    corrected = _standardize(embedding)
    groups = np.asarray(groups, dtype=str)
    unique_groups = np.unique(groups)
    if reliability is None:
        rho = np.ones(len(corrected), dtype=np.float32)
    else:
        rho = np.clip(np.asarray(reliability, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if len(groups) != len(corrected) or len(rho) != len(corrected):
        raise ValueError("groups and reliability must contain one value per cell")
    if len(unique_groups) < 2:
        return corrected

    for iteration in range(max(1, int(iterations))):
        nodes: list[dict[str, object]] = []
        group_nodes: dict[str, list[int]] = {}
        for offset, group in enumerate(unique_groups):
            rows = np.flatnonzero(groups == group)
            local = StandardScaler().fit_transform(corrected[rows])
            k = min(max(2, int(clusters_per_group)), len(rows))
            labels = MiniBatchKMeans(
                n_clusters=k,
                batch_size=2048,
                n_init=10,
                random_state=int(random_state) + iteration * len(unique_groups) + offset,
            ).fit_predict(local)
            group_nodes[str(group)] = []
            for cluster in range(k):
                selected = rows[labels == cluster]
                if len(selected) == 0:
                    continue
                weight = np.clip(rho[selected], 0.05, 1.0)
                match_center = np.average(local[labels == cluster], axis=0, weights=weight)
                match_center /= max(float(np.linalg.norm(match_center)), 1e-8)
                center = np.average(corrected[selected], axis=0, weights=weight)
                node = len(nodes)
                nodes.append(
                    {
                        "group": str(group),
                        "rows": selected,
                        "match_center": match_center,
                        "center": center,
                        "mass": float(weight.sum()),
                    }
                )
                group_nodes[str(group)].append(node)

        edges_left: list[int] = []
        edges_right: list[int] = []
        for left_offset, left_group in enumerate(unique_groups[:-1]):
            left_nodes = group_nodes[str(left_group)]
            left_centers = np.vstack([nodes[index]["match_center"] for index in left_nodes])
            for right_group in unique_groups[left_offset + 1 :]:
                right_nodes = group_nodes[str(right_group)]
                right_centers = np.vstack([nodes[index]["match_center"] for index in right_nodes])
                similarity = left_centers @ right_centers.T
                left_k = min(max(1, int(reciprocal_neighbors)), similarity.shape[1])
                right_k = min(max(1, int(reciprocal_neighbors)), similarity.shape[0])
                left_top = np.argpartition(-similarity, kth=left_k - 1, axis=1)[:, :left_k]
                right_top = np.argpartition(-similarity, kth=right_k - 1, axis=0)[:right_k, :]
                right_sets = [set(right_top[:, column].tolist()) for column in range(len(right_nodes))]
                for left_local in range(len(left_nodes)):
                    for right_local in left_top[left_local]:
                        if left_local not in right_sets[int(right_local)]:
                            continue
                        if similarity[left_local, right_local] < float(minimum_similarity):
                            continue
                        edges_left.extend((left_nodes[left_local], right_nodes[int(right_local)]))
                        edges_right.extend((right_nodes[int(right_local)], left_nodes[left_local]))
        if not edges_left:
            break
        graph = sparse.csr_matrix(
            (np.ones(len(edges_left)), (edges_left, edges_right)), shape=(len(nodes), len(nodes))
        )
        _, components = connected_components(graph, directed=False)
        updated = corrected.copy()
        for component in np.unique(components):
            members = np.flatnonzero(components == component)
            represented_groups = {nodes[index]["group"] for index in members}
            if len(represented_groups) < 2:
                continue
            centers = np.vstack([nodes[index]["center"] for index in members])
            masses = np.asarray([nodes[index]["mass"] for index in members])
            target = np.average(centers, axis=0, weights=masses)
            for index in members:
                rows = np.asarray(nodes[index]["rows"], dtype=np.int64)
                center = np.asarray(nodes[index]["center"])
                residual = corrected[rows] - center
                aligned = target + float(residual_scale) * residual
                amount = np.clip(float(strength) * rho[rows], 0.0, 1.0)[:, None]
                updated[rows] = (1.0 - amount) * corrected[rows] + amount * aligned
        corrected = updated
    return corrected.astype(np.float32)


def reliability_mnn_align(
    embedding: np.ndarray,
    groups: np.ndarray,
    reliability: np.ndarray | None = None,
    *,
    neighbors: int = 20,
    smoothing_neighbors: int = 40,
    bandwidth: float = 1.0,
    strength: float = 1.0,
) -> np.ndarray:
    """Align groups using only reliability-supported mutual nearest neighbors."""

    corrected = _standardize(embedding).astype(np.float64)
    groups = np.asarray(groups, dtype=str)
    unique_groups = sorted(np.unique(groups), key=lambda value: -int(np.sum(groups == value)))
    if len(unique_groups) < 2:
        return corrected.astype(np.float32)
    if reliability is None:
        rho = np.ones(len(corrected), dtype=np.float64)
    else:
        rho = np.clip(np.asarray(reliability, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if len(rho) != len(corrected):
            raise ValueError("reliability must contain one value per cell")
    reference_rows = np.flatnonzero(groups == unique_groups[0])
    for group in unique_groups[1:]:
        query_rows = np.flatnonzero(groups == group)
        k = min(max(1, int(neighbors)), len(reference_rows), len(query_rows))
        reference_unit = corrected[reference_rows]
        reference_unit /= np.clip(np.linalg.norm(reference_unit, axis=1, keepdims=True), 1e-8, None)
        query_unit = corrected[query_rows]
        query_unit /= np.clip(np.linalg.norm(query_unit, axis=1, keepdims=True), 1e-8, None)
        query_to_reference = NearestNeighbors(
            n_neighbors=k, metric="cosine", n_jobs=1
        ).fit(reference_unit).kneighbors(query_unit, return_distance=False)
        reference_to_query = NearestNeighbors(
            n_neighbors=k, metric="cosine", n_jobs=1
        ).fit(query_unit).kneighbors(reference_unit, return_distance=False)
        reverse_sets = [set(values.tolist()) for values in reference_to_query]
        anchor_query: list[int] = []
        anchor_correction: list[np.ndarray] = []
        anchor_weight: list[float] = []
        for query_local, candidates in enumerate(query_to_reference):
            mutual = [ref_local for ref_local in candidates if query_local in reverse_sets[ref_local]]
            if not mutual:
                continue
            ref_rows = reference_rows[np.asarray(mutual, dtype=np.int64)]
            pair_weight = np.clip(rho[ref_rows], 0.05, 1.0)
            target = np.average(corrected[ref_rows], axis=0, weights=pair_weight)
            query_row = int(query_rows[query_local])
            anchor_query.append(query_local)
            anchor_correction.append(target - corrected[query_row])
            anchor_weight.append(float(np.sqrt(rho[query_row] * np.mean(pair_weight))))
        if anchor_query:
            anchor_query_array = np.asarray(anchor_query, dtype=np.int64)
            correction_values = np.asarray(anchor_correction, dtype=np.float64)
            support_weights = np.asarray(anchor_weight, dtype=np.float64)
            smooth_k = min(max(1, int(smoothing_neighbors)), len(anchor_query_array))
            distances, indices = NearestNeighbors(
                n_neighbors=smooth_k, metric="euclidean", n_jobs=1
            ).fit(query_unit[anchor_query_array]).kneighbors(query_unit)
            local_scale = np.median(distances[:, -1])
            scale = max(float(bandwidth) * float(local_scale), 1e-4)
            weights = np.exp(-0.5 * np.square(distances / scale))
            weights *= support_weights[indices]
            weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
            correction = np.einsum("ij,ijk->ik", weights, correction_values[indices])
            corrected[query_rows] += float(strength) * correction
        reference_rows = np.concatenate((reference_rows, query_rows))
    return corrected.astype(np.float32)


def reliability_reference_mnn_align(
    embedding: np.ndarray,
    matching_embedding: np.ndarray,
    groups: np.ndarray,
    reliability: np.ndarray | None = None,
    *,
    neighbors: int = 20,
    smoothing_neighbors: int = 40,
    bandwidth: float = 1.0,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply MNN corrections supported by a fixed semantic matching space.

    Mutual neighbors are discovered in ``matching_embedding`` while their
    correction vectors are measured in ``embedding``.  This separation keeps
    modality drift from defining its own correspondences and lets a shared-gene
    or bridge-supported semantic anchor control which corrections are allowed.
    """

    corrected = _standardize(embedding).astype(np.float64)
    matching = _standardize(matching_embedding).astype(np.float64)
    groups = np.asarray(groups, dtype=str)
    if corrected.shape != matching.shape:
        raise ValueError("embedding and matching_embedding must have the same shape")
    if len(groups) != len(corrected):
        raise ValueError("groups must contain one value per cell")
    unique_groups = sorted(np.unique(groups), key=lambda value: -int(np.sum(groups == value)))
    if len(unique_groups) < 2:
        return corrected.astype(np.float32)
    if reliability is None:
        rho = np.ones(len(corrected), dtype=np.float64)
    else:
        rho = np.clip(np.asarray(reliability, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if len(rho) != len(corrected):
            raise ValueError("reliability must contain one value per cell")

    reference_rows = np.flatnonzero(groups == unique_groups[0])
    for group in unique_groups[1:]:
        query_rows = np.flatnonzero(groups == group)
        k = min(max(1, int(neighbors)), len(reference_rows), len(query_rows))
        reference_match = matching[reference_rows]
        reference_match /= np.clip(
            np.linalg.norm(reference_match, axis=1, keepdims=True), 1e-8, None
        )
        query_match = matching[query_rows]
        query_match /= np.clip(np.linalg.norm(query_match, axis=1, keepdims=True), 1e-8, None)
        query_to_reference = NearestNeighbors(
            n_neighbors=k, metric="cosine", n_jobs=1
        ).fit(reference_match).kneighbors(query_match, return_distance=False)
        reference_to_query = NearestNeighbors(
            n_neighbors=k, metric="cosine", n_jobs=1
        ).fit(query_match).kneighbors(reference_match, return_distance=False)
        reverse_sets = [set(values.tolist()) for values in reference_to_query]
        anchor_query: list[int] = []
        anchor_correction: list[np.ndarray] = []
        anchor_weight: list[float] = []
        for query_local, candidates in enumerate(query_to_reference):
            mutual = [ref_local for ref_local in candidates if query_local in reverse_sets[ref_local]]
            if not mutual:
                continue
            ref_rows = reference_rows[np.asarray(mutual, dtype=np.int64)]
            pair_weight = np.clip(rho[ref_rows], 0.05, 1.0)
            target = np.average(corrected[ref_rows], axis=0, weights=pair_weight)
            query_row = int(query_rows[query_local])
            anchor_query.append(query_local)
            anchor_correction.append(target - corrected[query_row])
            anchor_weight.append(float(np.sqrt(rho[query_row] * np.mean(pair_weight))))
        if anchor_query:
            anchor_query_array = np.asarray(anchor_query, dtype=np.int64)
            correction_values = np.asarray(anchor_correction, dtype=np.float64)
            support_weights = np.asarray(anchor_weight, dtype=np.float64)
            smooth_k = min(max(1, int(smoothing_neighbors)), len(anchor_query_array))
            distances, indices = NearestNeighbors(
                n_neighbors=smooth_k, metric="euclidean", n_jobs=1
            ).fit(query_match[anchor_query_array]).kneighbors(query_match)
            local_scale = np.median(distances[:, -1])
            scale = max(float(bandwidth) * float(local_scale), 1e-4)
            weights = np.exp(-0.5 * np.square(distances / scale))
            weights *= support_weights[indices]
            weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
            correction = np.einsum("ij,ijk->ik", weights, correction_values[indices])
            corrected[query_rows] += float(strength) * correction
        reference_rows = np.concatenate((reference_rows, query_rows))
    return corrected.astype(np.float32)


def batch_discriminant_nullspace(
    embedding: np.ndarray,
    groups: np.ndarray,
    *,
    directions: int | None = None,
    strength: float = 1.0,
    regularization: float = 0.1,
) -> np.ndarray:
    """Suppress the smallest linear subspace that discriminates observed groups."""

    source = _standardize(embedding).astype(np.float64)
    groups = np.asarray(groups, dtype=str)
    unique_groups = np.unique(groups)
    if len(groups) != len(source):
        raise ValueError("groups must contain one value per cell")
    if len(unique_groups) < 2:
        return source.astype(np.float32)
    global_mean = source.mean(axis=0)
    between = np.zeros((source.shape[1], source.shape[1]), dtype=np.float64)
    within = np.zeros_like(between)
    for group in unique_groups:
        values = source[groups == group]
        mean = values.mean(axis=0)
        delta = mean - global_mean
        between += len(values) * np.outer(delta, delta)
        centered = values - mean
        within += centered.T @ centered
    trace_scale = float(np.trace(within) / max(source.shape[1], 1))
    within += max(float(regularization) * trace_scale, 1e-6) * np.eye(source.shape[1])
    values, vectors = generalized_eigh(between, within)
    count = min(directions or (len(unique_groups) - 1), len(unique_groups) - 1, source.shape[1])
    axes = vectors[:, np.argsort(values)[::-1][:count]]
    axes, _ = np.linalg.qr(axes)
    batch_component = (source @ axes) @ axes.T
    return (source - float(strength) * batch_component).astype(np.float32)


def balanced_neighbor_graph_embedding(
    embedding: np.ndarray,
    groups: np.ndarray,
    *,
    neighbors_per_group: int = 5,
    dimensions: int | None = None,
    temperature: float = 0.25,
    random_state: int = 0,
) -> np.ndarray:
    """Embed a graph with an equal neighbor budget from every observed group."""

    del random_state  # ARPACK initialization is deterministic through v0 below.
    source = _standardize(embedding)
    groups = np.asarray(groups, dtype=str)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return source
    normalized = source.copy()
    for group in unique_groups:
        selected = groups == group
        normalized[selected] = StandardScaler().fit_transform(source[selected])

    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    all_rows = np.arange(len(source), dtype=np.int64)
    for group in unique_groups:
        target_rows = np.flatnonzero(groups == group)
        desired = min(int(neighbors_per_group), max(0, len(target_rows) - 1))
        query_k = min(desired + 1, len(target_rows))
        if desired <= 0:
            continue
        distances, local_neighbors = NearestNeighbors(
            n_neighbors=query_k,
            metric="cosine",
            n_jobs=1,
        ).fit(normalized[target_rows]).kneighbors(normalized)
        neighbors = target_rows[local_neighbors]
        keep_neighbors = np.empty((len(source), desired), dtype=np.int64)
        keep_distances = np.empty((len(source), desired), dtype=np.float32)
        for row in range(len(source)):
            keep = neighbors[row] != row
            available_neighbors = neighbors[row][keep]
            available_distances = distances[row][keep]
            if len(available_neighbors) < desired:
                available_neighbors = neighbors[row][:desired]
                available_distances = distances[row][:desired]
            keep_neighbors[row] = available_neighbors[:desired]
            keep_distances[row] = available_distances[:desired]
        neighbors, distances = keep_neighbors, keep_distances
        row_parts.append(np.repeat(all_rows, neighbors.shape[1]))
        column_parts.append(neighbors.reshape(-1))
        value_parts.append(
            np.exp(-distances.reshape(-1) / max(float(temperature), 1e-4)).astype(np.float32)
        )
    adjacency = sparse.csr_matrix(
        (np.concatenate(value_parts), (np.concatenate(row_parts), np.concatenate(column_parts))),
        shape=(len(source), len(source)),
        dtype=np.float32,
    )
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_root = np.divide(
        1.0,
        np.sqrt(degree),
        out=np.zeros_like(degree, dtype=np.float32),
        where=degree > 0,
    )
    normalized_adjacency = sparse.diags(inverse_root) @ adjacency @ sparse.diags(inverse_root)
    output_dim = min(dimensions or source.shape[1], len(source) - 2)
    generator = np.random.default_rng(0)
    values, vectors = eigsh(
        normalized_adjacency,
        k=output_dim + 1,
        which="LA",
        tol=1e-3,
        v0=generator.normal(size=len(source)),
    )
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order]
    values = values[order]
    coordinates = vectors[:, 1 : output_dim + 1] * np.sqrt(np.clip(values[1 : output_dim + 1], 0.0, None))
    if output_dim < (dimensions or source.shape[1]):
        coordinates = np.pad(
            coordinates,
            ((0, 0), (0, (dimensions or source.shape[1]) - output_dim)),
        )
    return _standardize(coordinates)


def cross_group_snn_subspace(
    embedding: np.ndarray,
    groups: np.ndarray,
    reliability: np.ndarray | None = None,
    *,
    graph_neighbors: int = 30,
    cross_group_neighbors: int = 5,
    geometry_penalty: float = 0.5,
    dimensions: int = 20,
) -> np.ndarray:
    """Project features using a reliability-weighted cross-group SNN kernel."""

    source = _standardize(embedding).astype(np.float64)
    groups = np.asarray(groups, dtype=str)
    if len(groups) != len(source):
        raise ValueError("groups must contain one value per cell")
    if reliability is None:
        rho = np.ones(len(source), dtype=np.float64)
    else:
        rho = np.clip(np.asarray(reliability, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if len(rho) != len(source):
            raise ValueError("reliability must contain one value per cell")
    if len(np.unique(groups)) < 2:
        return source.astype(np.float32)
    n_neighbors = min(max(2, int(graph_neighbors)) + 1, len(source))
    indices = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_jobs=1,
    ).fit(source).kneighbors(source, return_distance=False)
    rows = np.repeat(np.arange(len(source)), n_neighbors - 1)
    columns = np.empty((len(source), n_neighbors - 1), dtype=np.int64)
    for row in range(len(source)):
        selected = indices[row][indices[row] != row][: n_neighbors - 1]
        if len(selected) < n_neighbors - 1:
            selected = indices[row][: n_neighbors - 1]
        columns[row] = selected
    neighbor_graph = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns.reshape(-1))),
        shape=(len(source), len(source)),
    )
    intersections = (neighbor_graph @ neighbor_graph.T).tocsr()
    intersections.setdiag(0.0)
    intersections.eliminate_zeros()
    intersections.data /= np.clip(
        2.0 * (n_neighbors - 1) - intersections.data, 1.0, None
    )

    selected_rows: list[int] = []
    selected_columns: list[int] = []
    selected_values: list[float] = []
    keep_per_group = max(1, int(cross_group_neighbors))
    for row in range(len(source)):
        start, end = intersections.indptr[row], intersections.indptr[row + 1]
        candidates = intersections.indices[start:end]
        values = intersections.data[start:end]
        if len(candidates) == 0:
            continue
        for target_group in np.unique(groups[candidates]):
            if target_group == groups[row]:
                continue
            mask = groups[candidates] == target_group
            target_columns = candidates[mask]
            target_values = values[mask]
            count = min(keep_per_group, len(target_columns))
            if count == 0:
                continue
            top = np.argpartition(-target_values, kth=count - 1)[:count]
            edge_values = target_values[top] * np.sqrt(rho[row] * rho[target_columns[top]])
            positive = edge_values > 0
            selected_rows.extend([row] * int(positive.sum()))
            selected_columns.extend(target_columns[top][positive].tolist())
            selected_values.extend(edge_values[positive].tolist())
    if not selected_values:
        return source.astype(np.float32)
    kernel = sparse.csr_matrix(
        (selected_values, (selected_rows, selected_columns)),
        shape=(len(source), len(source)),
        dtype=np.float64,
    )
    row_sum = np.asarray(kernel.sum(axis=1)).ravel()
    kernel = sparse.diags(
        np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    ) @ kernel
    kernel = (kernel + kernel.T) * 0.5
    degree = np.asarray(kernel.sum(axis=1)).ravel()
    objective_kernel = kernel - float(geometry_penalty) * sparse.diags(degree)
    objective = source.T @ (objective_kernel @ source)
    objective = (objective + objective.T) * 0.5
    values, vectors = np.linalg.eigh(objective)
    output_dimensions = min(max(1, int(dimensions)), source.shape[1])
    order = np.argsort(values)[::-1][:output_dimensions]
    output = source @ vectors[:, order]
    return _standardize(output)


def balanced_neighbor_smoothing(
    embedding: np.ndarray,
    groups: np.ndarray,
    *,
    neighbors_per_group: int = 5,
    iterations: int = 1,
    strength: float = 0.5,
) -> np.ndarray:
    """Smooth cells toward an equal number of neighbors from every group."""

    corrected = _standardize(embedding)
    groups = np.asarray(groups, dtype=str)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return corrected
    for _ in range(int(iterations)):
        normalized = corrected.copy()
        for group in unique_groups:
            selected = groups == group
            normalized[selected] = StandardScaler().fit_transform(corrected[selected])
        targets = np.zeros_like(normalized)
        contributions = np.zeros(len(normalized), dtype=np.float32)
        for group in unique_groups:
            target_rows = np.flatnonzero(groups == group)
            desired = min(int(neighbors_per_group), max(0, len(target_rows) - 1))
            if desired <= 0:
                continue
            query_k = min(desired + 1, len(target_rows))
            neighbors = target_rows[
                NearestNeighbors(n_neighbors=query_k, metric="cosine", n_jobs=1)
                .fit(normalized[target_rows])
                .kneighbors(normalized, return_distance=False)
            ]
            for row in range(len(normalized)):
                selected_neighbors = neighbors[row][neighbors[row] != row][:desired]
                if len(selected_neighbors) < desired:
                    selected_neighbors = neighbors[row][:desired]
                targets[row] += normalized[selected_neighbors].mean(axis=0)
                contributions[row] += 1.0
        valid = contributions > 0
        targets[valid] /= contributions[valid, None]
        corrected[valid] = (
            (1.0 - float(strength)) * normalized[valid]
            + float(strength) * targets[valid]
        )
    return corrected.astype(np.float32)


def _median_cosine(values: np.ndarray, *, maximum: int, rng: np.random.Generator) -> float:
    """Estimate within-cluster cosine tightness without a quadratic memory spike."""

    if len(values) < 2:
        return 1.0
    if len(values) > maximum:
        values = values[rng.choice(len(values), size=maximum, replace=False)]
    similarities = values @ values.T
    upper = similarities[np.triu_indices(len(values), k=1)]
    return float(np.median(upper)) if len(upper) else 1.0


def reliability_kernel_subspace(
    embedding: np.ndarray,
    groups: np.ndarray,
    reliability: np.ndarray | None = None,
    *,
    clusters_per_group: int = 20,
    dimensions: int | None = None,
    representatives_per_cluster: int = 64,
    angle_tolerance_degrees: float = 15.0,
    maximum_angle_degrees: float = 50.0,
    geometry_penalty: float = 0.8,
    links_per_representative: int = 2,
    random_state: int = 0,
) -> np.ndarray:
    """Learn a reliability-weighted cross-group discriminative subspace.

    Cells are clustered independently inside each observed group. Cross-group
    clusters are connected only when their median cosine similarity is
    compatible with both clusters' internal angular spread. The resulting
    sparse kernel rewards supported cross-group links, while the degree term
    preserves within-group geometry. This is a label-free adaptation of the
    cluster-kernel principle used by Palette, extended with per-cell reliability.
    """

    source = _standardize(embedding).astype(np.float64)
    groups = np.asarray(groups, dtype=str)
    if len(groups) != len(source):
        raise ValueError("groups must contain one value per cell")
    if reliability is None:
        rho = np.ones(len(source), dtype=np.float64)
    else:
        rho = np.clip(np.asarray(reliability, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if len(rho) != len(source):
            raise ValueError("reliability must contain one value per cell")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return source.astype(np.float32)

    rng = np.random.default_rng(int(random_state))
    norm = np.linalg.norm(source, axis=1, keepdims=True)
    unit = source / np.clip(norm, 1e-8, None)
    representative_rows: dict[tuple[str, int], np.ndarray] = {}
    tightness: dict[tuple[str, int], float] = {}

    for offset, group in enumerate(unique_groups):
        rows = np.flatnonzero(groups == group)
        k = min(max(2, int(clusters_per_group)), len(rows))
        labels = MiniBatchKMeans(
            n_clusters=k,
            batch_size=2048,
            n_init=10,
            random_state=int(random_state) + offset,
        ).fit_predict(source[rows])
        for cluster in range(k):
            selected = rows[labels == cluster]
            if len(selected) == 0:
                continue
            key = (str(group), cluster)
            centroid = unit[selected].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
            order = np.argsort(-(unit[selected] @ centroid))
            representative_rows[key] = selected[order[: int(representatives_per_cluster)]]
            tightness[key] = _median_cosine(
                unit[selected], maximum=int(representatives_per_cluster), rng=rng
            )

    all_representatives = np.unique(np.concatenate(list(representative_rows.values())))
    local_index = {int(row): index for index, row in enumerate(all_representatives)}
    edge_rows: list[int] = []
    edge_columns: list[int] = []
    edge_values: list[float] = []
    tolerance = np.deg2rad(float(angle_tolerance_degrees))
    maximum_angle = np.deg2rad(float(maximum_angle_degrees))

    for left_offset, left_group in enumerate(unique_groups[:-1]):
        left_keys = [key for key in representative_rows if key[0] == left_group]
        for right_group in unique_groups[left_offset + 1 :]:
            right_keys = [key for key in representative_rows if key[0] == right_group]
            if not left_keys or not right_keys:
                continue
            cross = np.zeros((len(left_keys), len(right_keys)), dtype=np.float64)
            accepted = np.zeros_like(cross, dtype=bool)
            for i, left_key in enumerate(left_keys):
                left = unit[representative_rows[left_key]]
                left_cutoff = max(
                    np.cos(maximum_angle),
                    np.cos(np.arccos(np.clip(tightness[left_key], -1.0, 1.0)) + tolerance),
                )
                for j, right_key in enumerate(right_keys):
                    right = unit[representative_rows[right_key]]
                    similarity = float(np.median(left @ right.T))
                    right_cutoff = max(
                        np.cos(maximum_angle),
                        np.cos(
                            np.arccos(np.clip(tightness[right_key], -1.0, 1.0)) + tolerance
                        ),
                    )
                    cross[i, j] = similarity
                    accepted[i, j] = similarity >= max(left_cutoff, right_cutoff)

            selected_pairs: set[tuple[int, int]] = set()
            for i in range(len(left_keys)):
                candidates = np.flatnonzero(accepted[i])
                if len(candidates):
                    selected_pairs.add((i, int(candidates[np.argmax(cross[i, candidates])])))
            for j in range(len(right_keys)):
                candidates = np.flatnonzero(accepted[:, j])
                if len(candidates):
                    selected_pairs.add((int(candidates[np.argmax(cross[candidates, j])]), j))

            for i, j in selected_pairs:
                left = representative_rows[left_keys[i]]
                right = representative_rows[right_keys[j]]
                pair_similarity = unit[left] @ unit[right].T
                count = min(max(1, int(links_per_representative)), len(right))
                neighbors = np.argpartition(-pair_similarity, kth=count - 1, axis=1)[:, :count]
                for row_offset, left_row in enumerate(left):
                    for right_offset in neighbors[row_offset]:
                        right_row = int(right[right_offset])
                        value = max(float(pair_similarity[row_offset, right_offset]), 0.0)
                        value *= float(np.sqrt(rho[left_row] * rho[right_row]))
                        if value <= 0:
                            continue
                        edge_rows.extend((local_index[int(left_row)], local_index[right_row]))
                        edge_columns.extend((local_index[right_row], local_index[int(left_row)]))
                        edge_values.extend((value, value))

    if not edge_values:
        return source.astype(np.float32)
    kernel = sparse.csr_matrix(
        (edge_values, (edge_rows, edge_columns)),
        shape=(len(all_representatives), len(all_representatives)),
        dtype=np.float64,
    )
    kernel.sum_duplicates()
    degree = np.asarray(kernel.sum(axis=1)).ravel()
    objective_kernel = kernel - float(geometry_penalty) * sparse.diags(degree)
    representative_source = source[all_representatives]
    objective = representative_source.T @ (objective_kernel @ representative_source)
    objective = (objective + objective.T) * 0.5
    values, vectors = np.linalg.eigh(objective)
    output_dimensions = min(dimensions or source.shape[1], source.shape[1])
    order = np.argsort(values)[::-1][:output_dimensions]
    output = source @ vectors[:, order]
    output /= np.clip(np.linalg.norm(output, axis=1, keepdims=True), 1e-8, None)
    return output.astype(np.float32)
