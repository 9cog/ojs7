"""Smoke tests for the SKZ autonomous agents framework layout.

These tests guarantee that the directory structure CI relies on remains
intact and that pytest can at least collect the unit-test suite.
"""
from __future__ import annotations

import configparser
from pathlib import Path

FRAMEWORK = Path("skz-integration/autonomous-agents-framework")


def test_framework_layout(repo_root: Path) -> None:
    base = repo_root / FRAMEWORK
    for sub in ("src", "tests", "tests/unit", "requirements.txt", "pytest.ini"):
        assert (base / sub).exists(), f"Missing {FRAMEWORK / sub}"


def test_pytest_ini_has_expected_markers(repo_root: Path) -> None:
    ini_path = repo_root / FRAMEWORK / "pytest.ini"
    parser = configparser.ConfigParser(allow_no_value=True)
    parser.read(ini_path)
    section = "tool:pytest"
    assert parser.has_section(section), f"{ini_path} missing [{section}]"
    markers = parser.get(section, "markers", fallback="")
    for required in ("unit", "integration", "e2e", "slow", "agent", "critical"):
        assert required in markers, (
            f"pytest.ini is missing marker '{required}' relied on by CI"
        )


def test_unit_tests_directory_is_populated(repo_root: Path) -> None:
    unit_dir = repo_root / FRAMEWORK / "tests" / "unit"
    files = list(unit_dir.glob("test_*.py"))
    assert files, f"No unit tests found under {unit_dir}"


def test_requirements_txt_lists_core_runtime_deps(repo_root: Path) -> None:
    req = (repo_root / FRAMEWORK / "requirements.txt").read_text(encoding="utf-8")
    # These are the dependency families the AI Engine Enforcement workflow
    # advertises as required for production AI inference. They should remain
    # declared even when alternative inference engines are added.
    for needle in ("torch", "transformers", "llama-cpp-python"):
        assert needle in req, (
            f"requirements.txt no longer declares '{needle}'; update both this "
            "test and .github/workflows/ai-validation.yml if this is intentional."
        )
