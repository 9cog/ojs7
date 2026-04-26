"""Smoke tests for GitHub Actions workflow YAML files.

We don't run the workflows here — that's GitHub's job — but we do enforce
basic invariants so a malformed YAML file or a regression on important
trigger/permission settings is caught at PR time.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# PyYAML ships with most CI Python images; if it is missing we skip rather
# than fail, so that local environments without it don't block the suite.
yaml = pytest.importorskip("yaml")


WORKFLOWS_DIR = Path(".github/workflows")


def _workflow_files(repo_root: Path) -> list[Path]:
    return sorted((repo_root / WORKFLOWS_DIR).glob("*.yml")) + sorted(
        (repo_root / WORKFLOWS_DIR).glob("*.yaml")
    )


def test_workflows_directory_present(repo_root: Path) -> None:
    assert (repo_root / WORKFLOWS_DIR).is_dir(), (
        f"Expected workflows directory at {WORKFLOWS_DIR}"
    )


def test_workflows_are_valid_yaml(repo_root: Path) -> None:
    files = _workflow_files(repo_root)
    assert files, "No workflow files found under .github/workflows"
    for path in files:
        with path.open("r", encoding="utf-8") as fh:
            try:
                doc = yaml.safe_load(fh)
            except yaml.YAMLError as exc:  # pragma: no cover - explicit failure
                pytest.fail(f"{path} is not valid YAML: {exc}")
        assert isinstance(doc, dict), f"{path} must parse to a mapping"
        # PyYAML parses bare `on:` as boolean True; tolerate both spellings.
        assert "name" in doc, f"{path} is missing a top-level 'name'"
        assert (
            "on" in doc or True in doc
        ), f"{path} is missing a top-level 'on' trigger"
        assert "jobs" in doc and isinstance(doc["jobs"], dict) and doc["jobs"], (
            f"{path} must define at least one job"
        )


def test_build_and_test_workflow_has_expected_jobs(repo_root: Path) -> None:
    path = repo_root / WORKFLOWS_DIR / "build-and-test.yml"
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    expected_jobs = {
        "php-lint",
        "php-structure",
        "python-agents-tests",
        "python-agents-integration",
        "node-dashboards",
        "shell-scripts",
        "e2e-smoke",
        "build-and-test",
    }
    actual = set(doc["jobs"].keys())
    missing = expected_jobs - actual
    assert not missing, f"build-and-test.yml is missing jobs: {sorted(missing)}"


def test_workflows_pin_actions_by_major_or_sha(repo_root: Path) -> None:
    """Every `uses:` reference must be pinned (not @main / @master / unpinned)."""
    forbidden_refs = {"main", "master", "develop", "latest"}
    offenders: list[str] = []
    for path in _workflow_files(repo_root):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith("- uses:") and not stripped.startswith("uses:"):
                continue
            ref = stripped.split("uses:", 1)[1].strip()
            if "@" not in ref:
                offenders.append(f"{path}:{lineno}: '{ref}' is not pinned")
                continue
            tag = ref.rsplit("@", 1)[1].strip().strip("'\"")
            if tag.lower() in forbidden_refs:
                offenders.append(f"{path}:{lineno}: '{ref}' uses floating ref '{tag}'")
    assert not offenders, "Unpinned action references found:\n" + "\n".join(offenders)
