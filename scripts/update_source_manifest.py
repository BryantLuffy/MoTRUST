"""Create or check the deterministic SHA256 manifest for this source release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

MANIFEST_NAME = 'MANIFEST.sha256.json'
IGNORED_PARTS = {'.git', '.pytest_cache', '.ruff_cache', '.mypy_cache', '__pycache__',
                 '.venv', 'venv', 'build', 'dist', '.tmp'}
IGNORED_ROOT_DIRS = {'data', 'rawdata', 'checkpoints', 'results', 'runs', 'output',
                     'outputs', 'logs', 'tmp', '.cache', '.vendor', 'manuscript', 'presentations'}


def source_files(root: Path):
    root = root.resolve()
    collected = []
    for folder, directories, filenames in os.walk(root):
        directory = Path(folder)
        directories[:] = sorted(name for name in directories
                                if name not in IGNORED_PARTS and not name.endswith('.egg-info')
                                and not (directory == root and name in IGNORED_ROOT_DIRS))
        for filename in filenames:
            path = directory / filename
            relative = path.relative_to(root)
            if relative.as_posix() == MANIFEST_NAME or path.suffix in {'.pyc', '.pyo'}:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('Linked or external files are not allowed in the source manifest.')
            collected.append((relative.as_posix(), path))
    yield from sorted(collected)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Release root (defaults to the parent of scripts/).')
    parser.add_argument('--check', action='store_true', help='Check completeness and hashes without writing.')
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / 'src/motrust').is_dir():
        parser.error('The selected repository must contain src/motrust/.')
    expected = {'schema_version': 1, 'files': {relative: file_sha256(path)
                                            for relative, path in source_files(root)}}
    manifest = root / MANIFEST_NAME
    if args.check:
        try:
            current = json.loads(manifest.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            print('FAIL: source manifest is missing or invalid.')
            return 1
        if current != expected:
            print('FAIL: source manifest is incomplete or differs from current source files.')
            return 1
        print(f"PASS: source manifest matches all {len(expected['files'])} source files.")
        return 0
    manifest.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"Updated source manifest: {len(expected['files'])} files.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
