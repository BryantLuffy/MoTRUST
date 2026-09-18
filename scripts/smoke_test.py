"""Run the 30 self-contained core tests on CPU without installing pytest."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import uuid

TEST_FILES = (
    'test_model_smoke.py',
    'test_reliability_diffusion.py',
    'test_selective_transfer.py',
    'test_semantic_anchor.py',
    'test_semantic_fusion.py',
    'test_unpaired_semantic_anchor.py',
)


@contextmanager
def temporary_directory(parent: Path | None = None):
    """Create a unique temporary directory with inherited Windows permissions."""
    parent = (parent or Path(tempfile.gettempdir())).resolve()
    directory = parent / ('motrust_smoke_' + uuid.uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        if resolved.parent != parent or not resolved.name.startswith('motrust_smoke_'):
            raise RuntimeError('Unexpected temporary-directory cleanup target.')
        shutil.rmtree(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Release root (defaults to the parent of scripts/).')
    parser.add_argument('--json-out', type=Path, help='Optional machine-readable report.')
    args = parser.parse_args()
    if not __debug__:
        parser.error('Assertions must be enabled; do not use Python -O.')
    root = args.root.resolve()
    project = root
    package_dir = (project / 'src/motrust').resolve()
    if not package_dir.is_dir() or not package_dir.is_relative_to(root):
        parser.error('The selected repository does not contain its own package source.')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path.insert(0, str(project / 'src'))
    import motrust
    import torch
    imported = Path(motrust.__file__).resolve()
    if not imported.is_relative_to(package_dir):
        raise RuntimeError('Imported package is outside this repository.')
    torch.set_num_threads(1)
    records = []
    with temporary_directory() as work:
        for filename in TEST_FILES:
            test_path = (project / 'tests' / filename).resolve()
            if not test_path.is_relative_to(root) or not test_path.is_file():
                records.append({'file': filename, 'test': 'module_collection',
                                'status': 'failed', 'error_type': 'MissingRepositoryTest'})
                continue
            module_name = '_motrust_smoke_' + test_path.stem
            spec = importlib.util.spec_from_file_location(module_name, test_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception as error:
                records.append({'file': filename, 'test': 'module_collection',
                                'status': 'failed', 'error_type': type(error).__name__})
                print('FAIL', filename, 'module_collection', type(error).__name__, flush=True)
                continue
            tests = [(name, fn) for name, fn in vars(module).items()
                     if name.startswith('test_') and inspect.isfunction(fn)
                     and fn.__module__ == module_name]
            for name, fn in tests:
                start = time.monotonic()
                record = {'file': filename, 'test': name}
                try:
                    parameters = list(inspect.signature(fn).parameters)
                    if parameters == ['tmp_path']:
                        with temporary_directory(work) as fixture:
                            fn(tmp_path=fixture)
                    elif not parameters:
                        fn()
                    else:
                        raise RuntimeError('Unsupported test fixture signature.')
                    record['status'] = 'passed'
                except Exception as error:
                    record.update(status='failed', error_type=type(error).__name__)
                record['seconds'] = round(time.monotonic() - start, 4)
                records.append(record)
                print(record['status'].upper(), filename, name, flush=True)
    foreign_modules = sorted(name for name, module in sys.modules.items()
                             if (name == 'motrust' or name.startswith('motrust.'))
                             and getattr(module, '__file__', None)
                             and not Path(module.__file__).resolve().is_relative_to(package_dir))
    if foreign_modules:
        raise RuntimeError('A package submodule was imported from outside this repository.')
    failed = sum(item['status'] != 'passed' for item in records)
    summary = {'schema_version': 1, 'device': 'cpu', 'package_version': motrust.__version__,
               'package_origin': imported.relative_to(root).as_posix(),
               'imports_verified_within_repository': True,
               'expected': 30, 'executed': len(records),
               'passed': len(records) - failed, 'failed': failed, 'tests': records}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f"Core smoke tests: {summary['passed']}/{summary['executed']} passed (expected 30); CPU.")
    return 0 if len(records) == 30 and failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
