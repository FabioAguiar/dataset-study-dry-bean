"""Common Dataset Study environment contract: .python-version, pyproject.toml, pylock.toml."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

from scripts.canonical_run import canonical_python_version, locked_package_versions


ROOT = Path(__file__).resolve().parents[1]
LEGACY_REQUIREMENT_FILES = (
    "requirements.txt",
    "requirements",
    "environment.yml",
    "reproducibility/requirements-reference.txt",
)


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def test_structural_contract_files_exist() -> None:
    for name in (".python-version", "pyproject.toml", "pylock.toml", "notebooks", "scripts", "tests", "data", "artifacts"):
        assert (ROOT / name).exists(), name


def test_python_version_is_exact_and_allowed_by_requires_python() -> None:
    version = canonical_python_version(ROOT)
    requires = _pyproject()["project"]["requires-python"]
    match = re.fullmatch(r">=\s*(\d+(?:\.\d+)*)", requires)
    assert match, f"Unsupported requires-python form: {requires}"
    assert _version_tuple(version) >= _version_tuple(match.group(1))


def test_pylock_is_a_generated_pep751_lock_for_the_canonical_python() -> None:
    lock = tomllib.loads((ROOT / "pylock.toml").read_text(encoding="utf-8"))
    assert lock["lock-version"] == "1.0"
    assert lock["created-by"]
    required = lock.get("requires-python", "")
    assert required.lstrip(">=") == canonical_python_version(ROOT)
    for package in lock["packages"]:
        artifacts = [*package.get("wheels", []), *([package["sdist"]] if "sdist" in package else [])]
        assert artifacts, package["name"]
        assert all(artifact["hashes"].get("sha256") for artifact in artifacts), package["name"]


def test_lock_covers_declared_dependencies_and_groups() -> None:
    project = _pyproject()
    locked = locked_package_versions(ROOT)
    declared = list(project["project"]["dependencies"])
    groups = project["dependency-groups"]
    declared += [item for name in ("notebook", "test") for item in groups[name] if isinstance(item, str)]
    for requirement in declared:
        name = re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0]
        assert re.sub(r"[-_.]+", "-", name).lower() in locked, requirement


def test_no_competing_dependency_sources() -> None:
    for name in LEGACY_REQUIREMENT_FILES:
        assert not (ROOT / name).exists(), name
    assert "optional-dependencies" not in _pyproject()["project"]


def test_no_normative_references_to_legacy_environment_files() -> None:
    paths = [ROOT / "README.md", *ROOT.glob("scripts/*.py"), *ROOT.glob("notebooks/*.ipynb")]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "requirements-reference" not in text, path.name
        assert ".[notebook,test]" not in text, path.name
