"""Locations of user-owned data and read-only package resources."""
from __future__ import annotations

import os
from pathlib import Path


def workdir() -> Path:
    """Return the experiment directory, independently of the installation path."""
    return Path(os.environ.get("MOTRUST_WORKDIR", Path.cwd())).expanduser().resolve()


def resource_path(*parts: str) -> Path:
    """Resolve a bundled protocol or helper without depending on a source checkout."""
    base = Path(__file__).resolve().parent / "resources"
    target = base.joinpath(*parts).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError("Package resource paths must remain within resources/.")
    return target
