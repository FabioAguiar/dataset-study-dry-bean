"""Execute the official Dry Bean notebooks without modifying them.

The versioned notebooks in ``notebooks/`` are clean sources: no outputs and no
execution counts (enforced by ``tests/test_notebooks_clean.py``). This runner
reads each notebook into memory, executes that copy in a fresh kernel whose
working directory is ``notebooks/``, and never writes back to the source file.
Scientific outputs (data, artifacts, figures) are written by the notebook code
to their usual project locations.

The kernel is pinned to the interpreter running this script through a
temporary kernelspec, so a user-level ``python3`` kernelspec cannot silently
redirect execution to a different environment. Executed copies are discarded
unless ``--keep-executed DIR`` is given; ``DIR`` must be Git-ignored.

Usage::

    python -m scripts.run_notebooks                     # all five, in order
    python -m scripts.run_notebooks --fresh             # delete runtime outputs first
    python -m scripts.run_notebooks --keep-executed build/executed
    python -m scripts.run_notebooks --only 05_inference_demo.ipynb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence


OFFICIAL_NOTEBOOKS = (
    "01_data_understanding_and_exploration.ipynb",
    "02_data_preparation.ipynb",
    "03_model_selection_and_evaluation.ipynb",
    "04_final_model_and_bundle.ipynb",
    "05_inference_demo.ipynb",
)
NOTEBOOK_DIR = Path("notebooks")
# Ignored, regenerable runtime outputs removed by --fresh. Raw source data is kept.
RUNTIME_OUTPUT_DIRS = (
    Path("artifacts/exploration/dry-bean"),
    Path("artifacts/preparation/dry-bean"),
    Path("artifacts/model-selection/dry-bean"),
    Path("artifacts/models/dry-bean"),
    Path("data/interim/dry-bean"),
    Path("data/processed/dry-bean"),
)
KERNEL_NAME = "dataset-study-runner"
DEFAULT_CELL_TIMEOUT = 3600
PROJECT_ROOT_ENV = "DATASET_STUDY_ROOT"


class NotebookRunError(RuntimeError):
    """Raised when notebook execution would violate the clean-source policy."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def notebook_is_clean(notebook: dict) -> bool:
    """Return True when no code cell carries outputs or an execution count."""
    return all(
        cell.get("execution_count") is None and cell.get("outputs", []) == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    if shutil.which("git") is None or not (root / ".git").exists():
        return None
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )


def versionable_changes(root: Path) -> set[str]:
    """Return `git status --porcelain` entries (tracked changes and non-ignored files)."""
    result = _git(root, "status", "--porcelain", "--untracked-files=all")
    if result is None or result.returncode != 0:
        return set()
    return {line for line in result.stdout.splitlines() if line}


def _require_ignored(root: Path, directory: Path) -> None:
    resolved = directory.resolve()
    if resolved == (root / NOTEBOOK_DIR).resolve() or (root / NOTEBOOK_DIR).resolve() in resolved.parents:
        raise NotebookRunError("Executed copies must never be written inside notebooks/.")
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return  # outside the repository
    probe = (relative / "probe.ipynb").as_posix()
    result = _git(root, "check-ignore", "-q", probe)
    if result is not None and result.returncode != 0:
        raise NotebookRunError(
            f"{relative.as_posix()}/ is not Git-ignored; executed copies would be versionable."
        )


def _write_kernelspec(directory: Path) -> None:
    spec_dir = directory / "kernels" / KERNEL_NAME
    spec_dir.mkdir(parents=True)
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": f"Python {sys.version.split()[0]} (runner)",
        "language": "python",
    }
    (spec_dir / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")


def remove_runtime_outputs(root: Path) -> list[str]:
    removed = []
    for relative in RUNTIME_OUTPUT_DIRS:
        target = root / relative
        if target.exists():
            shutil.rmtree(target)
            removed.append(relative.as_posix())
    return removed


def execute_notebook(
    source: Path, *, working_dir: Path, timeout: int, executed_path: Path | None
) -> float:
    """Execute an in-memory copy of ``source``; the source file is never written."""
    import nbformat
    from nbclient import NotebookClient

    before = _sha256(source)
    notebook = nbformat.read(source, as_version=4)
    if not notebook_is_clean(notebook):
        raise NotebookRunError(f"{source.name} is not a clean source notebook.")

    started = time.perf_counter()
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=KERNEL_NAME,
        resources={"metadata": {"path": str(working_dir)}},
    )
    try:
        client.execute()
    finally:
        if executed_path is not None:
            executed_path.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, executed_path)
        if _sha256(source) != before:
            raise NotebookRunError(f"{source.name} was modified during execution.")
    return time.perf_counter() - started


def run(
    root: Path,
    notebooks: Sequence[str] = OFFICIAL_NOTEBOOKS,
    *,
    timeout: int = DEFAULT_CELL_TIMEOUT,
    keep_executed: Path | None = None,
    fresh: bool = False,
) -> None:
    root = root.resolve()
    notebook_dir = root / NOTEBOOK_DIR
    if keep_executed is not None:
        keep_executed = (root / keep_executed).resolve()
        _require_ignored(root, keep_executed)

    if fresh:
        for relative in remove_runtime_outputs(root):
            print(f"removed runtime output: {relative}")

    changes_before = versionable_changes(root)
    with tempfile.TemporaryDirectory(prefix="dataset-study-run-") as scratch:
        scratch_path = Path(scratch)
        _write_kernelspec(scratch_path)
        previous = {key: os.environ.get(key) for key in ("JUPYTER_PATH", PROJECT_ROOT_ENV)}
        os.environ["JUPYTER_PATH"] = os.pathsep.join(
            filter(None, [str(scratch_path), previous["JUPYTER_PATH"]])
        )
        os.environ[PROJECT_ROOT_ENV] = str(root)
        try:
            for name in notebooks:
                source = notebook_dir / name
                executed = (keep_executed or scratch_path / "executed") / name
                print(f"executing {NOTEBOOK_DIR.as_posix()}/{name} ...", flush=True)
                elapsed = execute_notebook(
                    source, working_dir=notebook_dir, timeout=timeout, executed_path=executed
                )
                print(f"  ok in {elapsed:.1f}s", flush=True)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    touched = {
        entry
        for entry in versionable_changes(root) - changes_before
        if entry[3:].startswith(f"{NOTEBOOK_DIR.as_posix()}/")
    }
    if touched:
        raise NotebookRunError(f"Execution left notebook changes: {sorted(touched)}")
    for name in notebooks:
        notebook = json.loads((notebook_dir / name).read_text(encoding="utf-8"))
        if not notebook_is_clean(notebook):
            raise NotebookRunError(f"{name} is no longer clean after execution.")
    print(f"{len(notebooks)} notebook(s) executed; official notebooks unchanged and clean.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=OFFICIAL_NOTEBOOKS,
        help="Execute only these notebooks (still in official order).",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_CELL_TIMEOUT, help="Per-cell timeout in seconds.")
    parser.add_argument(
        "--keep-executed",
        type=Path,
        help="Git-ignored directory (or one outside the repository) for executed copies.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete ignored runtime outputs (artifacts/*/dry-bean, data/{interim,processed}/dry-bean) first.",
    )
    args = parser.parse_args(argv)

    selected = [name for name in OFFICIAL_NOTEBOOKS if not args.only or name in args.only]
    run(
        Path(args.project_root),
        selected,
        timeout=args.timeout,
        keep_executed=args.keep_executed,
        fresh=args.fresh,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
