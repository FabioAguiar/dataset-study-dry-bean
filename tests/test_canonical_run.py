"""Contracts for the versioned canonical reference-run manifest."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from scripts.canonical_run import (
    MANIFEST_SCHEMA_VERSION,
    CanonicalRunError,
    canonical_python_version,
    compare_with_canonical,
    environment_contract_report,
    locked_package_versions,
    validate_manifest_structure,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "reproducibility" / "canonical-run.json"
README_PATH = ROOT / "README.md"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_committed_manifest_is_internally_consistent(manifest: dict) -> None:
    validate_manifest_structure(manifest)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["dataset"]["uci_dataset_id"] == 602
    assert len(manifest["dataset"]["source_sha256"]) == 64


def test_committed_manifest_encodes_split_before_train_only_eda(manifest: dict) -> None:
    protocol = manifest["protocol"]
    assert protocol["split_frozen_before_distribution_aware_eda"] is True
    assert protocol["eda_partition"] == "train"
    assert protocol["held_out_partitions_used_for_eda"] == []
    assert protocol["feature_policy_evidence_scope"] == "training_partition_exploration_only"
    assert protocol["test_opened_after_freeze_and_final_fit"] is True
    assert protocol["test_evaluation_count"] == 1
    assert protocol["test_used_for_adjustment"] is False
    assert manifest["superseded_reference"]["status"] == "superseded_not_canonical"


def test_selected_model_is_consistent_across_manifest_sections(manifest: dict) -> None:
    selection = manifest["model_selection"]
    policies = {row["model_id"]: row for row in selection["feature_policy_candidates"]}
    selected = policies[selection["selected_model_id"]]
    assert selected["feature_policy"] == selection["selected_feature_policy"]
    assert selected["feature_count"] == manifest["features"]["count"]
    assert selected["cv_macro_f1_mean"] == manifest["selected_cv_metrics"]["cv_macro_f1_mean"]
    assert selected["validation_metrics"]["macro_f1"] == manifest["validation_metrics"]["macro_f1"]
    families = {row["model_id"]: row for row in selection["family_search"]}
    base_model_id = selection["selected_model_id"].split("__")[0]
    assert families[base_model_id]["best_parameters"] == selection["selected_hyperparameters"]


def _mutated(manifest: dict, path: tuple[str, ...], value) -> dict:
    observed = copy.deepcopy(manifest)
    target = observed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return observed


def test_comparison_accepts_identical_run(manifest: dict) -> None:
    report = compare_with_canonical(manifest, copy.deepcopy(manifest))
    assert report["matches_canonical"] is True
    assert report["same_runtime_as_reference"] is True


@pytest.mark.parametrize(
    "path, value",
    [
        (("final_test", "metrics", "macro_f1"), 0.5),
        (("validation_metrics", "balanced_accuracy"), 0.5),
        (("model_selection", "selected_model_id"), "other"),
        (("split", "partition_sha256", "test"), "0" * 64),
        (("protocol", "eda_partition"), "validation"),
    ],
)
def test_comparison_detects_scientific_divergence(manifest: dict, path, value) -> None:
    report = compare_with_canonical(manifest, _mutated(manifest, path, value))
    assert report["matches_canonical"] is False
    assert any(key.startswith(".".join(path[:2])) for key in report["mismatches"])


def test_runtime_dependent_hashes_only_block_same_runtime(manifest: dict) -> None:
    observed = _mutated(manifest, ("final_model", "artifact_sha256"), "f" * 64)
    assert compare_with_canonical(manifest, observed)["matches_canonical"] is False

    observed["runtime"]["packages"] = {**observed["runtime"]["packages"], "numpy": "0.0"}
    report = compare_with_canonical(manifest, observed)
    assert report["matches_canonical"] is True
    assert report["same_runtime_as_reference"] is False
    assert report["runtime_dependent_equalities"]["final_model.artifact_sha256"] is False


def test_structure_validation_rejects_test_exposure(manifest: dict) -> None:
    with pytest.raises(CanonicalRunError, match="training partition only"):
        validate_manifest_structure(
            _mutated(manifest, ("protocol", "held_out_partitions_used_for_eda"), ["test"])
        )
    with pytest.raises(CanonicalRunError, match="evaluated once"):
        validate_manifest_structure(_mutated(manifest, ("protocol", "test_evaluation_count"), 2))


def test_manifest_runtime_was_observed_in_the_locked_environment(manifest: dict) -> None:
    runtime = manifest["runtime"]
    assert runtime["python_version_file"] == ".python-version"
    assert runtime["lock_file"] == "pylock.toml"
    report = environment_contract_report(ROOT, runtime)
    assert report["matches_locked_environment"] is True, report


def test_environment_contract_detects_drift(manifest: dict, tmp_path: Path) -> None:
    runtime = copy.deepcopy(manifest["runtime"])
    runtime["python"] = "0.0.0"
    runtime["packages"]["pandas"] = "0.0"
    report = environment_contract_report(ROOT, runtime)
    assert report["python_matches"] is False
    assert set(report["package_mismatches"]) == {"pandas"}

    (tmp_path / ".python-version").write_text("3.13\n", encoding="utf-8")
    with pytest.raises(CanonicalRunError, match="X.Y.Z"):
        canonical_python_version(tmp_path)
    with pytest.raises(CanonicalRunError, match="pylock.toml"):
        locked_package_versions(tmp_path)


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def test_readme_reports_canonical_headline_results(manifest: dict) -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    test_metrics = manifest["final_test"]["metrics"]
    for key in ("macro_f1", "balanced_accuracy", "accuracy", "weighted_f1", "log_loss"):
        assert _fmt(test_metrics[key]) in readme, key
    assert _fmt(manifest["validation_metrics"]["macro_f1"]) in readme
    assert _fmt(manifest["selected_cv_metrics"]["cv_macro_f1_mean"]) in readme
    for partition, rows in manifest["split"]["row_counts"].items():
        assert f"{rows:,}" in readme, partition
    for name, value in manifest["model_selection"]["selected_hyperparameters"].items():
        assert f"`{name.removeprefix('model__')}`" in readme
    assert "`.python-version`" in readme
    assert "`pylock.toml`" in readme


def test_readme_figures_exist_and_no_stale_figures_are_versioned() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    referenced = set(re.findall(r"\]\((docs/images/[^)]+\.png)\)", readme))
    assert referenced
    for relative in referenced:
        assert (ROOT / relative).is_file(), relative
    present = {f"docs/images/{path.name}" for path in (ROOT / "docs/images").glob("*.png")}
    assert present == referenced
