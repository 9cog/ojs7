"""Repository-structure smoke tests.

These verify that the OJS + SKZ tree contains the directories and files that
the rest of the build pipeline depends on. If a refactor moves or removes one
of these paths, this test catches it before it breaks downstream consumers
(deploy scripts, the AI Engine Enforcement workflow, the dashboards, etc.).
"""
from __future__ import annotations

from pathlib import Path

import pytest

REQUIRED_TOP_LEVEL = [
    "index.php",
    "config.TEMPLATE.inc.php",
    "classes",
    "controllers",
    "pages",
    "api",
    "plugins",
    "schemas",
    "tools",
    "lib/pkp",
    "skz-integration",
]

REQUIRED_SKZ_PATHS = [
    "skz-integration/autonomous-agents-framework",
    "skz-integration/autonomous-agents-framework/requirements.txt",
    "skz-integration/autonomous-agents-framework/pytest.ini",
    "skz-integration/autonomous-agents-framework/tests",
    "skz-integration/workflow-visualization-dashboard/package.json",
    "skz-integration/simulation-dashboard/package.json",
    "skz-integration/scripts/health-check.sh",
]

REQUIRED_WORKFLOWS = [
    ".github/workflows/ai-validation.yml",
    ".github/workflows/build-and-test.yml",
]


@pytest.mark.parametrize("relpath", REQUIRED_TOP_LEVEL)
def test_required_top_level_path_exists(repo_root: Path, relpath: str) -> None:
    assert (repo_root / relpath).exists(), f"Missing required path: {relpath}"


@pytest.mark.parametrize("relpath", REQUIRED_SKZ_PATHS)
def test_required_skz_path_exists(repo_root: Path, relpath: str) -> None:
    assert (repo_root / relpath).exists(), f"Missing required SKZ path: {relpath}"


@pytest.mark.parametrize("relpath", REQUIRED_WORKFLOWS)
def test_required_workflow_exists(repo_root: Path, relpath: str) -> None:
    p = repo_root / relpath
    assert p.is_file(), f"Missing required GitHub Actions workflow: {relpath}"
    # Workflows must not be empty stubs.
    assert p.stat().st_size > 100, f"Workflow {relpath} looks suspiciously small"


def test_index_php_has_php_open_tag(repo_root: Path) -> None:
    """index.php must start with a PHP open tag — otherwise OJS will not boot."""
    head = (repo_root / "index.php").read_text(encoding="utf-8", errors="replace")[:64]
    assert head.lstrip().startswith("<?php"), (
        "index.php must start with '<?php'; otherwise the OJS dispatcher will "
        f"not run. Got: {head!r}"
    )


def test_config_template_has_database_section(repo_root: Path) -> None:
    """The config template must continue to expose a [database] section."""
    text = (repo_root / "config.TEMPLATE.inc.php").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "[database]" in text, "config.TEMPLATE.inc.php is missing [database]"
    assert "[general]" in text, "config.TEMPLATE.inc.php is missing [general]"
