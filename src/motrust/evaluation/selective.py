"""Source-only calibration utilities for selective label transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


def exact_rejection_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Reject exactly the highest-scoring fraction with deterministic tie handling."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    amount = min(len(values), max(0, int(np.ceil(float(fraction) * len(values)))))
    mask = np.zeros(len(values), dtype=bool)
    if amount:
        order = np.argsort(-values, kind="stable")
        mask[order[:amount]] = True
    return mask


@dataclass
class SourceConditionalConformal:
    """Class-conditional conformal calibrator fitted without query labels."""

    neighbors: int = 15
    distance_weight: float = 0.25

    def fit(self, source: np.ndarray, labels: np.ndarray) -> "SourceConditionalConformal":
        values = np.asarray(source, dtype=np.float32)
        labels = np.asarray(labels, dtype=str)
        if len(values) != len(labels) or len(values) < 3:
            raise ValueError("source and labels must describe at least three cells")
        self.source_ = values
        self.labels_ = labels
        self.levels_ = np.unique(labels)
        self.k_ = min(max(1, int(self.neighbors)), len(values) - 1)
        self.model_ = NearestNeighbors(
            n_neighbors=self.k_ + 1, metric="euclidean", n_jobs=1
        ).fit(values)
        distances, indices = self.model_.kneighbors(values)
        distances, indices = distances[:, 1:], indices[:, 1:]
        self.distance_scale_ = max(
            float(np.median(distances[:, min(self.k_ - 1, 4)])), 1e-6
        )
        probabilities = self._probabilities(distances, indices)
        truth_index = np.searchsorted(self.levels_, labels)
        truth_support = probabilities[np.arange(len(labels)), truth_index]
        local_distance = distances[:, : min(5, self.k_)].mean(axis=1)
        nonconformity = self._nonconformity(truth_support, local_distance)
        self.class_scores_ = {
            level: np.sort(nonconformity[labels == level]) for level in self.levels_
        }
        self.global_scores_ = np.sort(nonconformity)
        return self

    def _probabilities(self, distances: np.ndarray, indices: np.ndarray) -> np.ndarray:
        weights = np.exp(-distances / self.distance_scale_)
        neighbor_labels = self.labels_[indices]
        scores = np.zeros((len(distances), len(self.levels_)), dtype=np.float64)
        for index, level in enumerate(self.levels_):
            scores[:, index] = np.sum(weights * (neighbor_labels == level), axis=1)
        return scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)

    def _nonconformity(self, support: np.ndarray, distance: np.ndarray) -> np.ndarray:
        distance_term = np.log1p(np.asarray(distance) / self.distance_scale_)
        return (1.0 - np.asarray(support)) + float(self.distance_weight) * distance_term

    def predict(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict")
        distances, indices = self.model_.kneighbors(
            np.asarray(query, dtype=np.float32), n_neighbors=self.k_, return_distance=True
        )
        probabilities = self._probabilities(distances, indices)
        best = probabilities.argmax(axis=1)
        prediction = self.levels_[best]
        confidence = probabilities[np.arange(len(best)), best]
        local_distance = distances[:, : min(5, self.k_)].mean(axis=1)
        nonconformity = self._nonconformity(confidence, local_distance)
        p_values = np.empty(len(prediction), dtype=np.float64)
        for index, (level, score) in enumerate(zip(prediction, nonconformity)):
            calibration = self.class_scores_.get(level, self.global_scores_)
            p_values[index] = (
                1.0 + np.sum(calibration >= score)
            ) / (len(calibration) + 1.0)
        novelty = 1.0 - p_values
        return prediction, confidence.astype(np.float32), novelty.astype(np.float32)
