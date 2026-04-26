"""Smoke tests for the SKZ frontend dashboards.

These tests ensure the dashboards remain buildable from a clean checkout: the
package manifests must declare the build/lint/dev scripts the CI workflow and
deployment scripts rely on, and the entry HTML / source roots must exist.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

DASHBOARDS = [
    "skz-integration/workflow-visualization-dashboard",
    "skz-integration/simulation-dashboard",
]

REQUIRED_SCRIPTS = ("dev", "build", "lint")


@pytest.mark.parametrize("dashboard", DASHBOARDS)
def test_dashboard_package_json_is_valid(repo_root: Path, dashboard: str) -> None:
    pkg_path = repo_root / dashboard / "package.json"
    assert pkg_path.is_file(), f"{dashboard}/package.json is missing"
    pkg = json.loads(pkg_path.read_text(encoding="utf-8"))

    assert pkg.get("name"), f"{dashboard}/package.json is missing 'name'"
    scripts = pkg.get("scripts", {})
    for script in REQUIRED_SCRIPTS:
        assert script in scripts, (
            f"{dashboard}/package.json is missing required script '{script}'"
        )

    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    # Every dashboard is a Vite + React app; these must remain present.
    assert any(k.startswith("vite") for k in deps), (
        f"{dashboard} must depend on vite"
    )
    assert "react" in deps, f"{dashboard} must depend on react"


@pytest.mark.parametrize("dashboard", DASHBOARDS)
def test_dashboard_entry_files_present(repo_root: Path, dashboard: str) -> None:
    base = repo_root / dashboard
    assert (base / "index.html").is_file(), f"{dashboard}/index.html missing"
    assert (base / "src").is_dir(), f"{dashboard}/src missing"
    # At least one source file must exist.
    sources = list((base / "src").rglob("*.*"))
    assert sources, f"{dashboard}/src appears to be empty"
