"""Contracts for the notebook runner that preserves clean source notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_notebooks
from scripts.run_notebooks import NotebookRunError, execute_notebook, notebook_is_clean


ROOT = Path(__file__).resolve().parents[1]


def _write_notebook(path: Path, sources: list[str]) -> None:
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_code_cell(source) for source in sources]
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.write(notebook, path)


def test_runner_covers_exactly_the_official_notebooks() -> None:
    official = run_notebooks.OFFICIAL_NOTEBOOKS
    present = tuple(sorted(path.name for path in (ROOT / "notebooks").glob("*.ipynb")))
    assert present == official
    assert [name[:3] for name in official] == ["01_", "02_", "03_", "04_", "05_"]


def test_notebook_is_clean_detects_outputs_and_counts() -> None:
    clean = {"cells": [{"cell_type": "code", "execution_count": None, "outputs": []}]}
    assert notebook_is_clean(clean)
    assert not notebook_is_clean({"cells": [{"cell_type": "code", "execution_count": 1, "outputs": []}]})
    assert not notebook_is_clean({"cells": [{"cell_type": "code", "execution_count": None, "outputs": [{}]}]})


def test_execution_never_writes_the_source_notebook(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("nbclient")
    working_dir = tmp_path / "notebooks"
    working_dir.mkdir()
    source = working_dir / "demo.ipynb"
    _write_notebook(source, ["import os\nprint(os.getcwd())", "open('produced.txt', 'w').write('ok')"])
    before = source.read_bytes()

    scratch = tmp_path / "scratch"
    run_notebooks._write_kernelspec(scratch)
    monkeypatch.setenv("JUPYTER_PATH", str(scratch))
    executed = tmp_path / "executed" / "demo.ipynb"
    execute_notebook(source, working_dir=working_dir, timeout=120, executed_path=executed)

    assert source.read_bytes() == before
    assert notebook_is_clean(json.loads(source.read_text(encoding="utf-8")))
    copy = json.loads(executed.read_text(encoding="utf-8"))
    assert copy["cells"][0]["execution_count"] == 1
    assert "".join(copy["cells"][0]["outputs"][0]["text"]).strip() == str(working_dir)
    assert (working_dir / "produced.txt").read_text() == "ok"


def test_execution_refuses_a_dirty_source(tmp_path: Path) -> None:
    source = tmp_path / "dirty.ipynb"
    _write_notebook(source, ["1"])
    notebook = json.loads(source.read_text(encoding="utf-8"))
    notebook["cells"][0]["execution_count"] = 3
    source.write_text(json.dumps(notebook), encoding="utf-8")
    with pytest.raises(NotebookRunError, match="not a clean source"):
        execute_notebook(source, working_dir=tmp_path, timeout=10, executed_path=None)


def test_executed_copies_cannot_target_versionable_locations() -> None:
    with pytest.raises(NotebookRunError, match="inside notebooks/"):
        run_notebooks._require_ignored(ROOT, ROOT / "notebooks" / "executed")
    if (ROOT / ".git").exists():
        with pytest.raises(NotebookRunError, match="not Git-ignored"):
            run_notebooks._require_ignored(ROOT, ROOT / "docs" / "executed")
        run_notebooks._require_ignored(ROOT, ROOT / "build" / "executed")


def test_fresh_only_targets_ignored_runtime_outputs() -> None:
    for relative in run_notebooks.RUNTIME_OUTPUT_DIRS:
        assert relative.parts[0] in {"artifacts", "data"}
        assert relative.parts[-1] == "dry-bean"
        assert "raw" not in relative.parts


def test_readme_documents_the_runner_and_never_recommends_inplace() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python -m scripts.run_notebooks" in readme
    in_code = False
    for line in readme.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        elif in_code:
            assert "--inplace" not in line, line
