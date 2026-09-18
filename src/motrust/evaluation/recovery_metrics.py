"""Common array-level metrics for missing-modality recovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


EPS = 1e-8


@dataclass(frozen=True)
class MetricValue:
    category: str
    metric: str
    value: float
    higher_is_better: bool


def _validated_arrays(prediction: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(truth, dtype=np.float64)
    if pred.ndim != 2 or ref.ndim != 2 or pred.shape != ref.shape:
        raise ValueError(f"prediction/truth must be equal 2D arrays, got {pred.shape} and {ref.shape}")
    if pred.shape[0] < 3 or pred.shape[1] < 2:
        raise ValueError("recovery metrics require at least 3 cells and 2 features")
    if not np.isfinite(pred).all() or not np.isfinite(ref).all():
        raise ValueError("prediction/truth contain non-finite values")
    return pred, ref


def normalize_for_evaluation(values: np.ndarray, modality: str, *, truth: bool = False) -> np.ndarray:
    """Apply one method-independent evaluation scale for a target modality."""

    x = np.asarray(values, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    modality = modality.lower()
    if modality == "atac":
        return (x > 0).astype(np.float64) if truth else np.clip(x, 0.0, 1.0)
    x = np.clip(x, 0.0, None)
    if modality == "rna":
        library = np.sum(x, axis=1, keepdims=True)
        return np.log1p(x * (1.0e4 / np.maximum(library, EPS)))
    if modality == "adt":
        logged = np.log1p(x)
        return logged - np.mean(logged, axis=1, keepdims=True)
    raise ValueError(f"Unsupported modality: {modality}")


def standardized_mse(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Mean feature-wise MSE divided by ground-truth sample variance."""

    pred, ref = _validated_arrays(prediction, truth)
    variance = np.var(ref, axis=0, ddof=1)
    valid = variance > EPS
    if not np.any(valid):
        return float("nan")
    feature_mse = np.mean((pred[:, valid] - ref[:, valid]) ** 2, axis=0)
    return float(np.mean(feature_mse / variance[valid]))


def cosine_rows(prediction: np.ndarray, truth: np.ndarray) -> float:
    pred, ref = _validated_arrays(prediction, truth)
    numerator = np.sum(pred * ref, axis=1)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(ref, axis=1)
    valid = denominator > EPS
    return float(np.mean(numerator[valid] / denominator[valid])) if np.any(valid) else float("nan")


def top_variable_features(truth: np.ndarray, n_features: int) -> np.ndarray:
    variance = np.var(np.asarray(truth, dtype=np.float64), axis=0, ddof=1)
    n = min(max(2, int(n_features)), len(variance))
    return np.argsort(variance, kind="stable")[-n:]


def top_differential_features(
    truth: np.ndarray, labels: np.ndarray, per_class: int
) -> np.ndarray:
    """Select one-vs-rest features using a ground-truth standardized mean effect."""

    ref = np.asarray(truth, dtype=np.float64)
    y = np.asarray(labels)
    if len(y) != len(ref):
        raise ValueError("labels do not match recovery rows")
    selected: set[int] = set()
    for label in np.unique(y):
        inside = ref[y == label]
        outside = ref[y != label]
        if len(inside) < 2 or len(outside) < 2:
            continue
        delta = np.abs(np.mean(inside, axis=0) - np.mean(outside, axis=0))
        pooled = np.sqrt(np.var(inside, axis=0, ddof=1) + np.var(outside, axis=0, ddof=1) + EPS)
        score = delta / pooled
        n = min(max(1, int(per_class)), ref.shape[1])
        selected.update(np.argsort(score, kind="stable")[-n:].tolist())
    return np.asarray(sorted(selected), dtype=int)


def correlation_structure_preservation(
    prediction: np.ndarray, truth: np.ndarray, feature_indices: np.ndarray
) -> float:
    """Mean row-wise agreement between predicted and true feature correlations."""

    pred, ref = _validated_arrays(prediction, truth)
    idx = np.unique(np.asarray(feature_indices, dtype=int))
    idx = idx[(idx >= 0) & (idx < pred.shape[1])]
    if len(idx) < 2:
        return float("nan")
    pred = pred[:, idx]
    ref = ref[:, idx]
    variable = (np.var(pred, axis=0) > EPS) & (np.var(ref, axis=0) > EPS)
    if np.sum(variable) < 2:
        # A completed but constant prediction preserves no measurable
        # feature-correlation structure; it is not a missing/unsupported run.
        return 0.0
    corr_pred = np.corrcoef(pred[:, variable], rowvar=False)
    corr_ref = np.corrcoef(ref[:, variable], rowvar=False)
    row_scores: list[float] = []
    for i in range(len(corr_pred)):
        a = corr_pred[i]
        b = corr_ref[i]
        if np.std(a) > EPS and np.std(b) > EPS:
            row_scores.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(row_scores)) if row_scores else 0.0


def atac_ranking_metrics(
    prediction: np.ndarray, truth: np.ndarray, chunk_size: int = 256
) -> tuple[float, float]:
    pred, ref = _validated_arrays(prediction, truth)
    binary = ref > 0
    ap_chunks: list[np.ndarray] = []
    recall_chunks: list[np.ndarray] = []
    ranks = np.arange(1, pred.shape[1] + 1, dtype=np.float64)[None, :]
    for start in range(0, len(pred), chunk_size):
        stop = min(start + chunk_size, len(pred))
        positives = np.asarray(binary[start:stop], dtype=np.int8)
        n_positive = positives.sum(axis=1, dtype=np.int64)
        valid = (n_positive > 0) & (n_positive < positives.shape[1])
        if not valid.any():
            continue
        positives = positives[valid]
        n_positive = n_positive[valid]
        order = np.argsort(pred[start:stop][valid], axis=1)[:, ::-1]
        ranked = np.take_along_axis(positives, order, axis=1)
        cumulative = np.cumsum(ranked, axis=1, dtype=np.float64)
        ap_chunks.append(np.sum((cumulative / ranks) * ranked, axis=1) / n_positive)
        recall_chunks.append(
            cumulative[np.arange(len(n_positive)), n_positive - 1] / n_positive
        )
    ap = np.concatenate(ap_chunks) if ap_chunks else np.array([])
    recall = np.concatenate(recall_chunks) if recall_chunks else np.array([])
    return (
        float(np.mean(ap)) if len(ap) else float("nan"),
        float(np.mean(recall)) if len(recall) else float("nan"),
    )


def _reduced(values: np.ndarray, seed: int, n_components: int = 50) -> np.ndarray:
    n = min(n_components, values.shape[0] - 1, values.shape[1] - 1)
    if n < 2:
        return np.asarray(values, dtype=np.float64)
    return TruncatedSVD(n_components=n, random_state=seed).fit_transform(values)


def clustering_metrics(prediction: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    y = LabelEncoder().fit_transform(np.asarray(labels))
    z = StandardScaler().fit_transform(_reduced(prediction, seed))
    n_clusters = len(np.unique(y))
    cluster = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed, n_init=20,
        batch_size=max(4096, min(16384, len(z))),
    ).fit_predict(z)
    sample_size = min(10000, len(z))
    asw = silhouette_score(z, y, sample_size=sample_size, random_state=seed)
    return {
        "ari": float(adjusted_rand_score(y, cluster)),
        "nmi": float(normalized_mutual_info_score(y, cluster)),
        "casw": float((asw + 1.0) / 2.0),
    }


def _macro_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    matrix = confusion_matrix(y_true, y_pred)
    total = np.sum(matrix)
    values: list[float] = []
    for i in range(len(matrix)):
        false_positive = np.sum(matrix[:, i]) - matrix[i, i]
        true_negative = total - np.sum(matrix[i, :]) - false_positive
        denominator = true_negative + false_positive
        if denominator > 0:
            values.append(float(true_negative / denominator))
    return float(np.mean(values)) if values else float("nan")


def classification_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    seed: int,
    reference_truth: np.ndarray | None = None,
    reference_labels: np.ndarray | None = None,
) -> dict[str, float]:
    pred, ref = _validated_arrays(prediction, truth)
    labels = np.asarray(labels)
    if reference_truth is None or reference_labels is None:
        encoder = LabelEncoder().fit(labels)
        y = encoder.transform(labels)
        indices = np.arange(len(y))
        train, test = train_test_split(
            indices, test_size=0.3, random_state=seed, stratify=y
        )
        train_x, train_y = ref[train], y[train]
        test_x, test_y = pred[test], y[test]
    else:
        reference_truth = np.asarray(reference_truth, dtype=np.float64)
        reference_labels = np.asarray(reference_labels)
        if reference_truth.ndim != 2 or reference_truth.shape[1] != pred.shape[1]:
            raise ValueError("reference truth feature dimension differs from recovered data")
        encoder = LabelEncoder().fit(np.concatenate([reference_labels, labels]))
        train_x, train_y = reference_truth, encoder.transform(reference_labels)
        test_x, test_y = pred, encoder.transform(labels)
    n = min(50, train_x.shape[1] - 1, len(train_x) - 1)
    reducer = TruncatedSVD(n_components=max(2, n), random_state=seed)
    train_z = reducer.fit_transform(train_x)
    test_z = reducer.transform(test_x)
    scaler = StandardScaler().fit(train_z)
    classifier = MLPClassifier(
        hidden_layer_sizes=(128,), max_iter=200, early_stopping=True,
        random_state=seed, batch_size="auto",
    )
    classifier.fit(scaler.transform(train_z), train_y)
    predicted = classifier.predict(scaler.transform(test_z))
    return {
        "oca": float(accuracy_score(test_y, predicted)),
        "aca": float(balanced_accuracy_score(test_y, predicted)),
        "sensitivity_macro": float(recall_score(test_y, predicted, average="macro", zero_division=0)),
        "specificity_macro": _macro_specificity(test_y, predicted),
        "f1_macro": float(f1_score(test_y, predicted, average="macro", zero_division=0)),
    }


def evaluate_all(
    prediction: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    modality: str,
    seed: int,
    reference_truth: np.ndarray | None = None,
    reference_labels: np.ndarray | None = None,
) -> list[MetricValue]:
    pred, ref = _validated_arrays(prediction, truth)
    modality = modality.lower()
    pred = normalize_for_evaluation(pred, modality, truth=False)
    ref = normalize_for_evaluation(ref, modality, truth=True)
    reference = (
        normalize_for_evaluation(reference_truth, modality, truth=True)
        if reference_truth is not None else None
    )
    pfcs_n = 1000 if modality == "atac" else 100
    pdes_n = 50 if modality == "atac" else 5
    pfcs_features = top_variable_features(ref, pfcs_n)
    pdes_features = top_differential_features(ref, labels, pdes_n)
    values = [
        MetricValue("structure", "smse", standardized_mse(pred, ref), False),
        MetricValue("structure", "pfcs", correlation_structure_preservation(pred, ref, pfcs_features), True),
        MetricValue("structure", "pdes", correlation_structure_preservation(pred, ref, pdes_features), True),
    ]
    if modality == "atac":
        auprc, recall = atac_ranking_metrics(pred, ref)
        values.extend([
            MetricValue("structure", "atac_auprc_macro", auprc, True),
            MetricValue("structure", "positive_peak_recall_at_truth_k", recall, True),
        ])
    values.extend(
        MetricValue("clustering", name, value, True)
        for name, value in clustering_metrics(pred, labels, seed).items()
    )
    values.extend(
        MetricValue("classification", name, value, True)
        for name, value in classification_metrics(
            pred, ref, labels, seed, reference, reference_labels
        ).items()
    )
    return values
