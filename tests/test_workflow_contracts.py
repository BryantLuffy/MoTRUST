"""Small deterministic software contracts for the public workflows."""
import json
import math
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch

from motrust.workflows.integration.protection import guarded, frame_consistent_fusion
from motrust.workflows.integration.integrate import protected_fusion, run_integration
from motrust.workflows.rna_diffusion.model import project_mean
from motrust.workflows.rna_diffusion.scores import empirical_scores, feature_crps


def test_semantic_guard_identity_rotation_and_displacement_bound():
    rng = np.random.default_rng(87)
    semantic = rng.normal(size=(48, 5))
    neural = rng.normal(size=semantic.shape)
    groups = np.repeat(['RNA', 'ATAC'], 24)
    protected, diagnostics = guarded(semantic, neural, groups)
    identity, _ = guarded(semantic, semantic, groups)
    np.testing.assert_array_equal(identity, semantic)
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    rotated, _ = guarded(semantic @ rotation, neural @ rotation, groups)
    np.testing.assert_allclose(rotated, protected @ rotation, atol=1e-12, rtol=0)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        distances = NearestNeighbors(n_neighbors=21).fit(semantic[rows]).kneighbors(semantic[rows])[0]
        radii = np.median(distances[:, 1:], axis=1)
        assert np.all(np.linalg.norm(protected[rows] - semantic[rows], axis=1) <= 0.25 * radii + 1e-12)
    assert diagnostics['max_displacement_radius_ratio'] <= 0.25 + 1e-12


def test_frame_consistent_fusion_retains_equivalent_coordinates():
    rng = np.random.default_rng(18)
    raw = rng.normal(size=(80, 6))
    raw -= raw.mean(0)
    neural = np.linalg.qr(raw)[0] * np.sqrt(len(raw))
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    semantic = neural @ rotation
    reliability = rng.uniform(size=len(raw))
    for intervention in (0, 0.2, 0.8, 1):
        actual = frame_consistent_fusion(semantic, neural, reliability, intervention)
        np.testing.assert_allclose(actual, neural, atol=1e-6, rtol=0)


def test_protected_fusion_zero_gate_preserves_standardized_neural_frame():
    rng = np.random.default_rng(23)
    semantic, neural = rng.normal(size=(2, 48, 5))
    groups = np.repeat(['RNA', 'ATAC'], 24)
    actual, _ = protected_fusion(semantic, neural, groups, 0)
    np.testing.assert_allclose(actual, StandardScaler().fit_transform(neural), atol=2e-7, rtol=0)


def _prepared_reference(root):
    task = 'example_task'
    folder = root / 'data/tasks' / task
    folder.mkdir(parents=True)
    (folder.parent / 'tasks.json').write_text(json.dumps({'tasks': [{'task_id': task}]}), encoding='utf-8')
    metadata = pd.DataFrame({'cell_id': [f'cell{i}' for i in range(8)],
                             'modality': ['RNA'] * 4 + ['ATAC'] * 4,
                             'instance_batch': ['B1'] * 4 + ['B2'] * 4})
    metadata.to_csv(folder / 'metadata_training.csv', index=False)
    for representation in ('semantic', 'candidate', 'reference'):
        path = root / 'runs/integration' / representation / task
        path.mkdir(parents=True)
        metadata.to_csv(path / 'metadata.csv', index=False)
    composition = root / 'runs/integration/composition' / task
    composition.mkdir(parents=True)
    (composition / 'composition_manifest.json').write_text(json.dumps({'adaptive_mixing': {
        'semantic_intervention_fraction': 0, 'effective_strength': 1}}), encoding='utf-8')
    reference = np.arange(32, dtype=np.float32).reshape(8, 4) / 7
    np.save(root / 'runs/integration/reference' / task / 'embedding.npy', reference)
    return task, metadata, reference


def test_zero_gate_pipeline_exact_copy_and_stale_input_rejection(tmp_path):
    task, metadata, reference = _prepared_reference(tmp_path)
    from motrust.preflight import integration_check
    with patch.dict('os.environ', {'MOTRUST_WORKDIR': str(tmp_path)}):
        assert integration_check(task, evaluate=False) == []
    output = run_integration(task, evaluate=False, root=tmp_path)
    np.testing.assert_array_equal(np.load(output / 'embedding.npy'), reference)
    pd.testing.assert_frame_equal(pd.read_csv(output / 'metadata.csv'), metadata)
    assert json.loads((output / 'run_manifest.json').read_text())['diagnostics']['exact_reference_bypass']
    assert run_integration(task, evaluate=False, root=tmp_path) == output
    np.save(tmp_path / 'runs/integration/reference' / task / 'embedding.npy', reference + 1)
    with np.testing.assert_raises(ValueError):
        run_integration(task, evaluate=False, root=tmp_path)


def test_pipeline_rejects_reordered_reference_cells(tmp_path):
    task, metadata, _ = _prepared_reference(tmp_path)
    metadata.iloc[::-1].to_csv(tmp_path / 'runs/integration/reference' / task / 'metadata.csv', index=False)
    with np.testing.assert_raises(ValueError):
        run_integration(task, evaluate=False, root=tmp_path)
    assert not (tmp_path / 'results/integration/benchmark' / task / 'embedding.npy').exists()


def test_point_prediction_ignores_hidden_target_counts():
    from motrust.workflows.recovery import encoder, point
    torch.manual_seed(19)
    rng = np.random.default_rng(19)
    counts = {'rna': sparse.csr_matrix(rng.poisson(3, (8, 11)).astype(np.float32)),
              'atac': sparse.csr_matrix(rng.integers(0, 2, (8, 12)).astype(np.float32))}
    model = encoder.make_vae(counts).eval()
    with torch.no_grad():
        z = point.latent(model, {'atac': encoder.input_values(counts['atac'])}, ['atac']).cpu().numpy()
    head = point.MolecularHead(z, np.ones((8, 11), dtype=np.float32), 'rna').eval()
    expected = point.predict(model, {'rna': head}, counts, 'atac', 'rna', np.arange(8))
    poisoned = dict(counts, rna=sparse.csr_matrix(np.full((8, 11), np.nan, dtype=np.float32)))
    actual = point.predict(model, {'rna': head}, poisoned, 'atac', 'rna', np.arange(8))
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()


def test_bounded_projection_preserves_anchor_and_empirical_pairwise_scores():
    generator = torch.Generator().manual_seed(51)
    upper = math.log1p(10000)
    anchor = torch.tensor([[0.0, 1.2, upper], [2.4, 4.0, 5.0]], dtype=torch.float64)
    samples = torch.randn(7, 2, 3, generator=generator, dtype=torch.float64) * 4
    projected = project_mean(samples, anchor)
    torch.testing.assert_close(projected.mean(0), anchor, rtol=0, atol=2e-5)
    assert torch.all(projected >= 0) and torch.all(projected <= upper)
    truth = torch.rand(2, 3, generator=generator, dtype=torch.float64) * upper
    delta = projected[:, None] - projected[None, :]
    expected_features = (projected - truth).abs().mean(0) - 0.5 * delta.abs().mean((0, 1))
    expected_energy = (torch.linalg.vector_norm(projected - truth, dim=2).mean(0)
                       - 0.5 * torch.linalg.vector_norm(delta, dim=3).mean((0, 1))) / math.sqrt(3)
    crps, energy = empirical_scores(projected, truth)
    torch.testing.assert_close(crps, expected_features.mean(1))
    torch.testing.assert_close(energy, expected_energy)
    torch.testing.assert_close(feature_crps(projected, truth), expected_features)
