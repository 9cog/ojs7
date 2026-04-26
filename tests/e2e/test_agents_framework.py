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
    req_path = repo_root / FRAMEWORK / "requirements.txt"
    # Parse package names line-by-line so substrings like 'torch' don't
    # accidentally match 'pytorch' or 'torchvision'.
    declared: set[str] = set()
    for raw_line in req_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        # Strip any version specifier / extras to get the bare package name.
        name = line
        for sep in ("[", "=", "<", ">", "!", "~", " ", ";"):
            idx = name.find(sep)
            if idx != -1:
                name = name[:idx]
        name = name.strip().lower()
        if name:
            declared.add(name)

    # These are the dependency families the AI Engine Enforcement workflow
    # advertises as required for production AI inference. They should remain
    # declared even when alternative inference engines are added.
    for required in ("torch", "transformers", "llama-cpp-python"):
        assert required in declared, (
            f"requirements.txt no longer declares '{required}'; update both this "
            "test and .github/workflows/ai-validation.yml if this is intentional. "
            f"Declared packages: {sorted(declared)}"
        )
