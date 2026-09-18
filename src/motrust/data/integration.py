"""Label-safe sparse inputs for mosaic and cortical integration tasks."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from motrust.paths import workdir, resource_path


def resolve_reference(reference, root=None, index_path=None, path_base="workdir") -> Path:
    """Resolve a cache record; relative references use the work directory by default."""
    value = Path(reference).expanduser()
    if value.is_absolute():
        return value
    if path_base not in {"workdir", "index"}:
        raise ValueError("Cache path_base must be 'workdir' or 'index'")
    base = Path(index_path).parent if path_base == "index" and index_path else Path(root or workdir())
    return (base / value).resolve()


def _validate_task_id(task_id):
    if not task_id or Path(task_id).name != task_id or task_id in {".", ".."} or "/" in task_id or "\\" in task_id:
        raise ValueError("Task ID must be a single valid identifier")


def _read_names(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if frame.shape[1] != 1:
            raise ValueError(f"Expected one name column in {path}")
        return frame.iloc[:, 0].astype(str).to_numpy()
    return np.asarray(path.read_text(encoding="utf-8").splitlines(), dtype=str)


@lru_cache(maxsize=64)
def _load_matrix(matrix_path: str, feature_path: str, barcode_path: str):
    matrix = sparse.csr_matrix(mmread(matrix_path), dtype=np.float32)
    features = _read_names(feature_path)
    barcodes = _read_names(barcode_path)
    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError(
            f"Matrix/name mismatch for {matrix_path}: {matrix.shape}, "
            f"features={len(features)}, barcodes={len(barcodes)}"
        )
    return matrix, features, barcodes


class TaskData:
    """Explicit data context; installing the package never fixes an experiment path."""
    def __init__(self, domain="benchmark", root=None):
        if domain not in {"benchmark", "cortex"}:
            raise ValueError("domain must be benchmark or cortex")
        self.domain = domain
        self.root = Path(root or workdir()).expanduser().resolve()
        self.data_root = self.root / "data" / ("cortex" if domain == "cortex" else "")
        self.integration_root = self.root / "runs/integration" / ("cortex" if domain == "cortex" else "")
        self.cache_index = self.root / "data/cache/cache_index.json"
        override = self.data_root / "tasks/tasks.json"
        self.definition_path = override if override.is_file() else resource_path("integration", "cortex_tasks.json" if domain == "cortex" else "tasks.json")

    def load_task_definition(self, task_id):
        _validate_task_id(task_id)
        payload = json.loads(self.definition_path.read_text(encoding="utf-8-sig"))
        matches = [record for record in payload["tasks"] if record["task_id"] == task_id]
        if len(matches) != 1:
            raise ValueError(f"Unknown or duplicate task {task_id!r}")
        return matches[0]

    def load_task_metadata(self, task_id, evaluation=False):
        self.load_task_definition(task_id)
        filename = "metadata_evaluation.csv" if evaluation else "metadata_training.csv"
        path = self.data_root / "tasks" / task_id / filename
        frame = pd.read_csv(path, dtype=str)
        if frame["cell_id"].isna().any() or frame["cell_id"].duplicated().any():
            raise ValueError(f"Missing or duplicate cell IDs in {path}")
        if not evaluation and {"cell_type", "broad_class", "fine_cluster", "source_cell_id"}.intersection(frame.columns):
            raise ValueError("Evaluation-only metadata exposed to training")
        return frame

    def _counts(self, modality):
        if modality not in {"rna", "atac"}:
            raise ValueError("Unknown cortical modality")
        return sparse.load_npz(self.data_root / "prepared" / f"{modality}_counts.npz").tocsr()

    def _features(self, modality):
        return np.load(self.data_root / "prepared" / f"{modality}_feature_names.npy", allow_pickle=False)

    def _cortex_modalities(self, task_id):
        definition = self.load_task_definition(task_id)
        metadata = self.load_task_metadata(task_id)
        matrices, cellids, features = {}, {}, {}
        with np.load(self.data_root / "indices" / task_id / "indices.npz", allow_pickle=False) as indices:
            order = indices["task_order"]
            lookup = {int(v): i for i, v in enumerate(order)}
            for modality in definition["modalities"]:
                observed = indices[modality + "_observed"]
                matrices[modality] = self._counts(modality)[observed].copy()
                cellids[modality] = metadata.iloc[[lookup[int(i)] for i in observed]].cell_id.to_numpy()
                features[modality] = self._features(modality)
        return matrices, cellids, features

    def _cortex_aligned(self, task_id):
        definition = self.load_task_definition(task_id)
        metadata = self.load_task_metadata(task_id)
        aligned, masks = {}, {}
        with np.load(self.data_root / "indices" / task_id / "indices.npz", allow_pickle=False) as indices:
            order = indices["task_order"]
            for modality in definition["modalities"]:
                mask = np.isin(order, indices[modality + "_observed"]).astype(np.float32)
                matrix = self._counts(modality)[order]
                matrix = sparse.diags(mask, format="csr").dot(matrix).tocsr()
                matrix.eliminate_zeros()
                aligned[modality], masks[modality] = matrix, mask
        return aligned, masks, metadata

    def _cortex_blocks(self, task_id):
        definition = self.load_task_definition(task_id)
        metadata = self.load_task_metadata(task_id)
        blocks, features = [], {}
        with np.load(self.data_root / "indices" / task_id / "indices.npz", allow_pickle=False) as indices:
            for observation in definition["observations"]:
                rows = indices[observation["role"]]
                local = metadata[metadata.instance_batch == observation["instance_batch"]]
                matrices = {}
                for modality in observation["observed_modalities"]:
                    matrices[modality] = self._counts(modality)[rows].copy()
                    features[modality] = self._features(modality)
                blocks.append(dict(instance_batch=observation["instance_batch"], base_batch=observation["base_batch"], identity_policy="shared_cell_instance", cell_ids=local.cell_id.to_numpy(), matrices=matrices))
        return blocks, features

    def load_task_modalities(self, task_id: str) -> tuple[dict[str, sparse.csr_matrix], dict[str, np.ndarray], dict[str, np.ndarray]]:
        if self.domain == "cortex":
            return self._cortex_modalities(task_id)
        task = self.load_task_definition(task_id)
        index = json.loads(self.cache_index.read_text(encoding="utf-8-sig"))
        cache = index["scenarios"][task["scenario"]]
        blocks: dict[str, list[sparse.csr_matrix]] = {name: [] for name in task["modalities"]}
        cell_ids: dict[str, list[np.ndarray]] = {name: [] for name in task["modalities"]}
        feature_names: dict[str, np.ndarray] = {}

        for observation in task["observations"]:
            base = cache[observation["base_batch"]]
            for modality in observation["observed_modalities"]:
                reference = base["modalities"][modality]
                matrix, features, barcodes = _load_matrix(
                    *(str(resolve_reference(reference[key], self.root, self.cache_index, index.get("path_base", "workdir")))
                      for key in ("matrix", "features", "barcodes"))
                )
                if modality in feature_names and not np.array_equal(feature_names[modality], features):
                    raise ValueError(f"Feature order differs within {task_id}/{modality}")
                feature_names[modality] = features
                blocks[modality].append(matrix.T.tocsr())
                suffix = f"::{observation['instance_batch']}"
                cell_ids[modality].append(np.char.add(barcodes.astype(str), suffix))

        matrices: dict[str, sparse.csr_matrix] = {}
        joined_ids: dict[str, np.ndarray] = {}
        for modality in task["modalities"]:
            if not blocks[modality]:
                continue
            matrices[modality] = sparse.vstack(blocks[modality], format="csr", dtype=np.float32)
            joined_ids[modality] = np.concatenate(cell_ids[modality])
            if matrices[modality].shape[0] != len(joined_ids[modality]):
                raise RuntimeError(f"Cell order mismatch for {task_id}/{modality}")
        return matrices, joined_ids, feature_names


    def load_observation_blocks(self, task_id: str) -> tuple[list[dict], dict[str, np.ndarray]]:
        """Load each observation instance separately in task order."""

        if self.domain == "cortex":
            return self._cortex_blocks(task_id)
        task = self.load_task_definition(task_id)
        index = json.loads(self.cache_index.read_text(encoding="utf-8-sig"))
        cache = index["scenarios"][task["scenario"]]
        blocks: list[dict] = []
        feature_names: dict[str, np.ndarray] = {}
        for observation in task["observations"]:
            base = cache[observation["base_batch"]]
            matrices: dict[str, sparse.csr_matrix] = {}
            source_barcodes = None
            for modality in observation["observed_modalities"]:
                reference = base["modalities"][modality]
                matrix, features, barcodes = _load_matrix(
                    *(str(resolve_reference(reference[key], self.root, self.cache_index, index.get("path_base", "workdir")))
                      for key in ("matrix", "features", "barcodes"))
                )
                if modality in feature_names and not np.array_equal(feature_names[modality], features):
                    raise ValueError(f"Feature order differs within {task_id}/{modality}")
                feature_names[modality] = features
                if source_barcodes is not None and not np.array_equal(source_barcodes, barcodes):
                    raise ValueError(
                        f"Co-observed modalities have different cells in "
                        f"{task_id}/{observation['instance_batch']}"
                    )
                source_barcodes = barcodes
                matrices[modality] = matrix.T.tocsr()
            suffix = f"::{observation['instance_batch']}"
            cell_ids = np.char.add(source_barcodes.astype(str), suffix)
            blocks.append(
                {
                    "instance_batch": observation["instance_batch"],
                    "base_batch": observation["base_batch"],
                    "identity_policy": observation["identity_policy"],
                    "cell_ids": cell_ids,
                    "matrices": matrices,
                }
            )
        return blocks, feature_names


    def align_modalities_to_task(
        self, task_id: str,
    ) -> tuple[dict[str, sparse.csr_matrix], dict[str, np.ndarray], pd.DataFrame]:
        if self.domain == "cortex":
            return self._cortex_aligned(task_id)
        matrices, modality_ids, feature_names = self.load_task_modalities(task_id)
        metadata = self.load_task_metadata(task_id, evaluation=False)
        global_ids = metadata["cell_id"].astype(str).to_numpy()
        global_lookup = {cell_id: index for index, cell_id in enumerate(global_ids)}
        aligned: dict[str, sparse.csr_matrix] = {}
        masks: dict[str, np.ndarray] = {}
        for modality, matrix in matrices.items():
            rows = np.fromiter((global_lookup[cell] for cell in modality_ids[modality]), dtype=np.int64)
            if len(np.unique(rows)) != len(rows):
                raise ValueError(f"Repeated task rows for {task_id}/{modality}")
            coordinate = matrix.tocoo(copy=False)
            remapped = sparse.coo_matrix(
                (coordinate.data, (rows[coordinate.row], coordinate.col)),
                shape=(len(global_ids), matrix.shape[1]),
                dtype=np.float32,
            ).tocsr()
            aligned[modality] = remapped
            mask = np.zeros(len(global_ids), dtype=np.float32)
            mask[rows] = 1.0
            masks[modality] = mask
        return aligned, masks, metadata



def top_prevalent_features(matrix: sparse.csr_matrix, n_features: int) -> np.ndarray:
    if n_features <= 0 or matrix.shape[1] <= n_features:
        return np.arange(matrix.shape[1], dtype=np.int64)
    prevalence = np.asarray(matrix.getnnz(axis=0)).ravel()
    return np.argpartition(prevalence, -n_features)[-n_features:]


def log_library_normalize(matrix: sparse.csr_matrix, target_sum: float = 1e4) -> sparse.csr_matrix:
    output = matrix.astype(np.float32, copy=True).tocsr()
    library = np.asarray(output.sum(axis=1)).ravel()
    scale = np.divide(target_sum, library, out=np.zeros_like(library, dtype=np.float32), where=library > 0)
    output = sparse.diags(scale).dot(output).tocsr()
    output.data = np.log1p(output.data)
    return output
