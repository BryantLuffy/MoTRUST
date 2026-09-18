"""Source-manifest rules must retain package data access and resources."""
import importlib.util
from pathlib import Path


def test_source_manifest_keeps_package_data_and_ignores_experiment_outputs(tmp_path):
    helper_path = Path(__file__).resolve().parents[1] / 'scripts/update_source_manifest.py'
    spec = importlib.util.spec_from_file_location('source_manifest_helper', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    keep = ['src/motrust/data/integration.py', 'src/motrust/resources/data/prepare.R',
            'configs/example.json', 'README.md']
    ignored = ['data/input.csv', 'rawdata/input.csv', 'checkpoints/model.pt',
               'results/table.csv', 'runs/task.json', 'output/table.csv', 'outputs/table.csv',
               'logs/run.txt', 'tmp/scratch.txt', '.cache/cache.txt', '.vendor/code.py',
               'manuscript/paper.md', 'presentations/deck.txt', 'build/output.py',
               'src/motrust/__pycache__/module.pyc', 'src/motrust_omics.egg-info/PKG-INFO']
    for relative in keep + ignored:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture', encoding='utf-8')
    (tmp_path / 'MANIFEST.sha256.json').write_text('{}', encoding='utf-8')
    actual = [relative for relative, _ in helper.source_files(tmp_path)]
    assert actual == sorted(keep)
