"""Shared fixtures for repository-level E2E smoke tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _find_repo_root(start: Path) -> Path:
    """Walk up from *start* until we find a directory that looks like the repo root."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "index.php").is_file() and (candidate / "lib" / "pkp").is_dir():
            return candidate
    raise RuntimeError(
        f"Could not locate repository root starting from {start}; "
        "expected to find index.php and lib/pkp."
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return _find_repo_root(Path(__file__).parent)


@pytest.fixture(scope="session", autouse=True)
def _chdir_repo_root(repo_root: Path):
    """Run E2E tests from the repository root for stable relative paths."""
    previous = Path.cwd()
    os.chdir(repo_root)
    try:
        yield
    finally:
        os.chdir(previous)
