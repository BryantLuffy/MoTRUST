"""Verify the source files listed in MANIFEST.sha256.json without data or network access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re


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
    parser.add_argument('--json-out', type=Path, help='Optional machine-readable report.')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / 'MANIFEST.sha256.json'
    try:
        data = json.loads(manifest.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        print('FAIL: MANIFEST.sha256.json is missing, unreadable or invalid JSON.')
        return 1
    if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('files'), dict):
        print('FAIL: expected schema_version 1 and a files object.')
        return 1
    if not data['files']:
        print('FAIL: the source manifest contains no files.')
        return 1
    missing, changed, unreadable, invalid = [], [], [], []
    verified = 0
    for index, (relative, expected) in enumerate(data['files'].items(), 1):
        valid_path = (isinstance(relative, str) and bool(relative)
                      and not PurePosixPath(relative).is_absolute()
                      and not PureWindowsPath(relative).drive
                      and '\\' not in relative
                      and '..' not in PurePosixPath(relative).parts)
        if not valid_path or not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
            invalid.append(index)
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            invalid.append(index)
        elif not path.is_file():
            missing.append(relative)
        else:
            try:
                actual = file_sha256(path)
            except OSError:
                unreadable.append(relative)
                continue
            if actual != expected.lower():
                changed.append(relative)
            else:
                verified += 1
    passed = not (missing or changed or unreadable or invalid)
    report = {'schema_version': 1, 'status': 'passed' if passed else 'failed',
              'listed_files': len(data['files']), 'verified_files': verified,
              'missing': missing, 'changed': changed, 'unreadable': unreadable,
              'invalid_entry_numbers': invalid}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for label, entries in [('MISSING', missing), ('CHANGED', changed), ('UNREADABLE', unreadable)]:
        for relative in entries:
            print(label, relative)
    for index in invalid:
        print('INVALID manifest entry', index)
    print(f"{'PASS' if passed else 'FAIL'}: verified {verified}/{len(data['files'])} listed source files.")
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
