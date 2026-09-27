"""Canonical reference-run manifest for the Dry Bean dataset study.

The manifest is a small, versioned, machine-readable summary of the scientific
reference execution. It is derived exclusively from persisted, hash-validated
runtime artifacts (exploration, preparation, model-selection, and final-model
handoffs) and never stores rows, probabilities, or model binaries.

The environment contract is not duplicated here: the exact interpreter is read
from ``.python-version`` and exact package versions from ``pylock.toml``. The
runtime recorded in the manifest is observed evidence of one execution.

Usage::

    python -m scripts.canonical_run build    # write reproducibility/canonical-run.json
    python -m scripts.canonical_run verify   # compare a fresh run with the manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "canonical-run.v1"
MANIFEST_ARTIFACT_TYPE = "canonical_reference_run"
DEFAULT_MANIFEST_PATH = Path("reproducibility/canonical-run.json")
PYTHON_VERSION_FILE = Path(".python-version")
LOCK_FILE = Path("pylock.toml")
DATASET_SLUG = "dry-bean"
METRIC_TOLERANCE = 1e-9

ARTIFACT_PATHS = {
    "exploration_handoff": Path("artifacts/exploration/dry-bean/exploration-handoff.json"),
    "preparation_handoff": Path("artifacts/preparation/dry-bean/preparation-handoff.json"),
    "preparation_manifest": Path("artifacts/preparation/dry-bean/preparation-manifest.json"),
    "feature_manifest": Path("artifacts/preparation/dry-bean/feature-manifest.json"),
    "split_manifest": Path("artifacts/preparation/dry-bean/split-manifest.json"),
    "model_selection_handoff": Path("artifacts/model-selection/dry-bean/model-selection-handoff.json"),
    "model_selection_manifest": Path("artifacts/model-selection/dry-bean/model-selection-manifest.json"),
    "candidate_results": Path("artifacts/model-selection/dry-bean/candidate-results.json"),
    "final_model_handoff": Path("artifacts/models/dry-bean/final-model-handoff.json"),
    "final_model_manifest": Path("artifacts/models/dry-bean/final-model-manifest.json"),
    "final_test_evidence": Path("artifacts/models/dry-bean/final-test-evidence.json"),
    "inference_bundle": Path("artifacts/models/dry-bean/inference-bundle.json"),
}
RUNTIME_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "joblib",
    "scipy",
    "matplotlib",
    "ucimlrepo",
)

# Headline results of the execution that preceded the split-before-EDA
# correction. Kept only as a traceable historical reference; not canonical.
SUPERSEDED_REFERENCE = {
    "status": "superseded_not_canonical",
    "reason": (
        "Notebook 01 performed target-aware and distribution-aware exploration on "
        "the complete source before the split, so final-test rows contributed to "
        "the exploratory evidence behind feature-policy design. The corrected "
        "protocol freezes the split first and restricts exploration to train."
    ),
    "recorded_in": "README.md at commit 49cb600 (pre-correction documentation)",
    "selected_model_id": "hist_gradient_boosting__all_features",
    "cv_macro_f1_mean": 0.937012,
    "validation_macro_f1": 0.937881,
    "final_test_metrics": {
        "macro_f1": 0.941835,
        "balanced_accuracy": 0.939897,
        "accuracy": 0.932419,
        "weighted_f1": 0.932187,
        "log_loss": 0.181024,
    },
    # Historical evidence copied from the superseded README; not a contract.
    "recorded_runtime": {
        "python": "3.13.13",
        "pandas": "3.0.5",
        "scikit_learn": "1.9.0",
        "joblib": "1.5.3",
    },
}


class CanonicalRunError(RuntimeError):
    """Raised when artifacts cannot produce or satisfy the canonical manifest."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise CanonicalRunError(f"Required artifact is missing: {relative.as_posix()}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifacts(project_root: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(project_root).resolve()
    return {name: _load_json(root, path) for name, path in ARTIFACT_PATHS.items()}


def runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "packages": packages,
    }


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def canonical_python_version(project_root: str | Path) -> str:
    """Return the exact canonical CPython version declared in .python-version."""
    path = Path(project_root) / PYTHON_VERSION_FILE
    if not path.is_file():
        raise CanonicalRunError(f"Missing {PYTHON_VERSION_FILE.as_posix()}.")
    version = path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise CanonicalRunError(f"{PYTHON_VERSION_FILE.as_posix()} must contain X.Y.Z, got {version!r}.")
    return version


def locked_package_versions(project_root: str | Path) -> dict[str, str]:
    """Return {normalized name: version} from the canonical PEP 751 lock."""
    try:
        import tomllib
    except ModuleNotFoundError as exc:  # Python 3.10
        raise CanonicalRunError("Reading pylock.toml requires Python >= 3.11.") from exc
    path = Path(project_root) / LOCK_FILE
    if not path.is_file():
        raise CanonicalRunError(f"Missing {LOCK_FILE.as_posix()}.")
    lock = tomllib.loads(path.read_text(encoding="utf-8"))
    if lock.get("lock-version") != "1.0":
        raise CanonicalRunError("pylock.toml must use lock-version 1.0 (PEP 751).")
    return {_normalize_name(item["name"]): item["version"] for item in lock["packages"]}


def environment_contract_report(
    project_root: str | Path, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare an observed runtime with .python-version and pylock.toml."""
    expected_python = canonical_python_version(project_root)
    locked = locked_package_versions(project_root)
    package_mismatches = {
        name: {"locked": locked.get(_normalize_name(name)), "observed": version}
        for name, version in runtime["packages"].items()
        if locked.get(_normalize_name(name)) != version
    }
    return {
        "python_version_file": PYTHON_VERSION_FILE.as_posix(),
        "lock_file": LOCK_FILE.as_posix(),
        "expected_python": expected_python,
        "observed_python": runtime["python"],
        "python_matches": runtime["python"] == expected_python,
        "package_mismatches": package_mismatches,
        "matches_locked_environment": runtime["python"] == expected_python
        and not package_mismatches,
    }


def _round(value: Any) -> Any:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value


def _metric_subset(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "macro_f1",
        "balanced_accuracy",
        "macro_recall",
        "weighted_f1",
        "accuracy",
        "log_loss",
        "minimum_per_class_recall",
    )
    return {key: _round(metrics[key]) for key in keys if key in metrics}


def _per_class(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "class": row["class"],
            "precision": _round(row["precision"]),
            "recall": _round(row["recall"]),
            "f1": _round(row["f1"]),
            "support": int(row["support"]),
        }
        for row in rows
    ]


def build_canonical_manifest(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    source_sha256: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the canonical manifest from already-validated artifacts."""
    exploration = artifacts["exploration_handoff"]
    feature = artifacts["feature_manifest"]
    split = artifacts["split_manifest"]
    selection = artifacts["model_selection_handoff"]
    selection_manifest = artifacts["model_selection_manifest"]
    candidates = artifacts["candidate_results"]
    final_handoff = artifacts["final_model_handoff"]
    final_manifest = artifacts["final_model_manifest"]
    test_evidence = artifacts["final_test_evidence"]
    bundle = artifacts["inference_bundle"]

    if source_sha256 != exploration["source"]["sha256"]:
        raise CanonicalRunError("Source bytes differ from the exploration handoff.")
    development_split = exploration["development_split"]
    if split["partition_sha256"] != development_split["partition_sha256"]:
        raise CanonicalRunError("Materialized split differs from the split frozen before EDA.")
    if bundle["model_state_fingerprint"] != final_handoff["model_state_fingerprint"]:
        raise CanonicalRunError("Bundle and final handoff model fingerprints differ.")
    if bundle["selected_hyperparameters"] != selection["selected_hyperparameters"]:
        raise CanonicalRunError("Bundle hyperparameters differ from the model-selection freeze.")
    if bundle["feature_columns"] != selection["selected_feature_columns"]:
        raise CanonicalRunError("Bundle feature order differs from the model-selection freeze.")
    if test_evidence["test_partition_evaluation_count"] != 1:
        raise CanonicalRunError("Final test must be evaluated exactly once.")

    selected_cv = selection["selected_cv_evidence"]
    selected_validation = selection["selected_validation_evidence"]
    family_rows = sorted(
        (
            {
                "model_id": row["model_id"],
                "family": row["family"],
                "search_strategy": row["search_strategy"],
                "candidate_count": int(row["candidate_count"]),
                "cv_macro_f1_mean": _round(row["cv_macro_f1_mean"]),
                "cv_macro_f1_std": _round(row["cv_macro_f1_std"]),
                "best_parameters": row["best_parameters"],
            }
            for row in candidates["family_searches"]
        ),
        key=lambda row: (-row["cv_macro_f1_mean"], row["model_id"]),
    )
    policy_rows = sorted(
        (
            {
                "model_id": row["model_id"],
                "feature_policy": row["feature_policy"],
                "feature_count": int(row["feature_count"]),
                "cv_macro_f1_mean": _round(row["cv_macro_f1_mean"]),
                "cv_macro_f1_std": _round(row["cv_macro_f1_std"]),
                "validation_metrics": _metric_subset(row["validation"]["metrics"]),
            }
            for row in candidates["policy_candidates"]
        ),
        key=lambda row: row["model_id"],
    )
    confusion = test_evidence["confusion_matrix"]
    pair_comparison = test_evidence["confusion_pair_comparison"]
    repeated = test_evidence["repeated_profile_sensitivity"]
    split_repeated = split.get("repeated_profile_evidence", {})

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "dataset": {
            "slug": exploration["dataset_slug"],
            "name": "Dry Bean",
            "repository": exploration["source"]["repository"],
            "uci_dataset_id": int(exploration["source"]["dataset_id"]),
            "dataset_doi": "10.24432/C50S4B",
            "source_file": exploration["source"]["path"],
            "source_sha256": source_sha256,
            "source_row_count": int(exploration["source"]["row_count"]),
            "source_column_count": int(exploration["source"]["column_count"]),
        },
        "target": {
            "column": feature["target_column"],
            "problem_type": feature["problem_type"],
            "semantics": feature["target_contract"]["semantics"],
            "public_class_order": list(bundle["output_class_order"]),
            "estimator_class_order": list(bundle["estimator_class_order"]),
            "decision_rule": bundle["decision_rule"],
            "positive_class": None,
        },
        "features": {
            "ordered_columns": list(feature["feature_columns"]),
            "count": len(feature["feature_columns"]),
            "numerical": list(feature["numerical_features"]),
            "categorical": list(feature["categorical_features"]),
            "identifiers": list(feature["identifier_columns"]),
        },
        "protocol": {
            "split_frozen_before_distribution_aware_eda": bool(
                development_split["split_frozen_before_distribution_aware_analysis"]
            ),
            "eda_partition": development_split["eda_partition"],
            "eda_row_count": int(development_split["eda_row_count"]),
            "held_out_partitions_used_for_eda": list(
                development_split["held_out_partitions_used_for_eda"]
            ),
            "pre_split_full_source_checks": [
                "source identity and schema",
                "declared target class set",
                "row-wise domain rules",
                "missing and invalid values",
                "exact-row structure (no source identifier)",
            ],
            "feature_policy_evidence_scope": selection_manifest["search_contract"].get(
                "feature_policy_evidence_scope"
            ),
            "validation_in_search": selection_manifest["search_contract"]["validation_in_search"],
            "test_in_search": selection_manifest["search_contract"]["test_in_search"],
            "final_fit_partitions": list(final_manifest["final_training_partitions"]),
            "test_opened_after_freeze_and_final_fit": bool(
                test_evidence["test_loaded_only_after_final_fit"]
            ),
            "test_evaluation_count": int(test_evidence["test_partition_evaluation_count"]),
            "test_used_for_adjustment": not bool(test_evidence["no_post_test_adjustment"]),
            "test_partition_prior_exposure": (
                "The seed-42 test partition was also evaluated once by the superseded "
                "pre-correction execution. The corrected run re-derived every "
                "development decision from training-only evidence with the "
                "pre-declared seed, candidate space, metric, and tie rules; none of "
                "them was changed after that historical test result."
            ),
        },
        "split": {
            "policy": {
                "evaluation_mode": split["evaluation_mode"],
                "train_fraction": split["train_fraction"],
                "validation_fraction": split["validation_fraction"],
                "test_fraction": split["test_fraction"],
                "stratify_by": split["stratify_by"],
                "shuffle": split["shuffle"],
            },
            "split_method": split["split_method"],
            "membership_kind": split["membership_kind"],
            "row_counts": dict(split["row_counts"]),
            "class_counts": dict(split["class_counts"]),
            "partition_sha256": dict(split["partition_sha256"]),
            "membership_sha256": dict(development_split["membership_sha256"]),
            "repeated_profiles": {
                "source_exact_row_equality_groups": split_repeated.get(
                    "source_exact_row_equality_group_count"
                ),
                "cross_partition_repeated_profile_groups": split_repeated.get(
                    "cross_partition_repeated_feature_profile_group_count"
                ),
                "target_conflicting_profile_groups": split_repeated.get(
                    "target_conflicting_feature_profile_group_count"
                ),
                "proven_duplicate_identity": False,
            },
        },
        "seeds": {
            "split_first_stage": split["stage_seeds"]["train_vs_temporary"],
            "split_second_stage": split["stage_seeds"]["validation_vs_test"],
            **{key: value for key, value in selection["random_seeds"].items() if key != "split_reference"},
        },
        "cross_validation": {
            key: selection["cv_contract"][key]
            for key in ("strategy", "n_splits", "shuffle", "random_state", "fit_partition")
        },
        "model_selection": {
            "primary_metric": selection["primary_metric"],
            "dummy_validation_macro_f1": _round(candidates["selection"]["dummy_macro_f1"]),
            "dummy_macro_f1_margin": _round(candidates["selection"]["required_margin"]),
            "practical_tie_tolerance": _round(candidates["selection"]["practical_tie_tolerance"]),
            "family_search": family_rows,
            "feature_policy_candidates": policy_rows,
            "selected_model_id": selection["selected_model_id"],
            "selected_model_family": selection["selected_model_family"],
            "selected_feature_policy": selection["selected_feature_policy"],
            "selected_hyperparameters": dict(selection["selected_hyperparameters"]),
            "selected_preprocessing": dict(selection["selected_preprocessing_contract"]),
            "selected_imbalance_policy": dict(selection["selected_imbalance_policy"]),
            "practical_tie": candidates["selection"]["practical_tie"],
            "selection_rationale": selection["selection_rationale"],
        },
        "selected_cv_metrics": {
            "cv_macro_f1_mean": _round(selected_cv["cv_macro_f1_mean"]),
            "cv_macro_f1_std": _round(selected_cv["cv_macro_f1_std"]),
            "cv_balanced_accuracy_mean": _round(selected_cv["cv_balanced_accuracy_mean"]),
            "cv_weighted_f1_mean": _round(selected_cv["cv_weighted_f1_mean"]),
            "cv_log_loss_mean": _round(selected_cv["cv_log_loss_mean"]),
        },
        "validation_metrics": _metric_subset(selected_validation["metrics"]),
        "validation_per_class": _per_class(selected_validation["per_class"]),
        "final_test": {
            "row_count": int(test_evidence["row_count"]),
            "metrics": _metric_subset(test_evidence["metrics"]),
            "per_class": _per_class(test_evidence["per_class"]),
            "confusion_matrix": {
                "class_order": list(confusion["class_order"]),
                "counts": [list(map(int, row)) for row in confusion["counts"]],
            },
            "focal_confusion_pairs": [
                {
                    "class_pair": row["class_pair"],
                    "validation_mutual_errors": int(row["validation"]["mutual_confusion_count"]),
                    "test_mutual_errors": int(row["test"]["mutual_confusion_count"]),
                }
                for row in pair_comparison["focal_pair_comparisons"]
            ],
            "top_test_confusion_pairs": [
                {
                    "class_pair": row["class_pair"],
                    "mutual_errors": int(row["mutual_confusion_count"]),
                }
                for row in pair_comparison["ranked_test_pairs"][:3]
            ],
            "repeated_profile_sensitivity": {
                key: _round(value)
                for key, value in repeated.items()
                if not isinstance(value, (dict, list))
            },
            "probability_matrix_sha256": test_evidence["probability_matrix_sha256_aggregate_only"],
        },
        "final_model": {
            "artifact_path": bundle["model_artifact_path"],
            "artifact_format": bundle["model_artifact_format"],
            "artifact_sha256": bundle["model_artifact_sha256"],
            "model_state_fingerprint": bundle["model_state_fingerprint"],
            "frozen_finalization_contract_fingerprint": final_manifest[
                "frozen_finalization_contract_fingerprint"
            ],
            "fit_row_count": int(final_manifest["final_training_row_count"]),
            "fit_class_counts": dict(final_manifest["final_training_class_counts"]),
        },
        "runtime": {
            **dict(runtime),
            "recorded_by_final_bundle": dict(bundle["runtime_version_requirements"]),
            "python_version_file": PYTHON_VERSION_FILE.as_posix(),
            "lock_file": LOCK_FILE.as_posix(),
        },
        "superseded_reference": SUPERSEDED_REFERENCE,
    }


def build_from_project(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    artifacts = load_artifacts(root)
    source = root / artifacts["exploration_handoff"]["source"]["path"]
    if not source.is_file():
        raise CanonicalRunError("Source dataset file is missing.")
    return build_canonical_manifest(
        artifacts,
        source_sha256=_sha256_file(source),
        runtime=runtime_environment(),
    )


def validate_manifest_structure(manifest: Mapping[str, Any]) -> None:
    """Check internal consistency of a canonical manifest without artifacts."""
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CanonicalRunError("Unexpected canonical manifest schema version.")
    if manifest.get("artifact_type") != MANIFEST_ARTIFACT_TYPE:
        raise CanonicalRunError("Unexpected canonical manifest artifact type.")
    protocol = manifest["protocol"]
    if not protocol["split_frozen_before_distribution_aware_eda"]:
        raise CanonicalRunError("Canonical protocol must freeze the split before EDA.")
    if protocol["eda_partition"] != "train" or protocol["held_out_partitions_used_for_eda"]:
        raise CanonicalRunError("Canonical exploration must use the training partition only.")
    if protocol["test_in_search"] or protocol["validation_in_search"]:
        raise CanonicalRunError("Held-out partitions must not enter the search.")
    if protocol["test_evaluation_count"] != 1 or protocol["test_used_for_adjustment"]:
        raise CanonicalRunError("Final test must be evaluated once without adjustment.")
    split = manifest["split"]
    if sum(split["row_counts"].values()) != manifest["dataset"]["source_row_count"]:
        raise CanonicalRunError("Partition row counts do not cover the source.")
    if split["row_counts"]["train"] != protocol["eda_row_count"]:
        raise CanonicalRunError("EDA row count differs from the training partition.")
    classes = manifest["target"]["public_class_order"]
    if sorted(classes) != sorted(manifest["target"]["estimator_class_order"]):
        raise CanonicalRunError("Public and estimator class sets differ.")
    test = manifest["final_test"]
    counts = test["confusion_matrix"]["counts"]
    if test["confusion_matrix"]["class_order"] != classes:
        raise CanonicalRunError("Confusion matrix does not follow the public class order.")
    if sum(map(sum, counts)) != test["row_count"] or test["row_count"] != split["row_counts"]["test"]:
        raise CanonicalRunError("Confusion matrix total differs from the test partition.")
    supports = [row["support"] for row in test["per_class"]]
    if supports != [sum(row) for row in counts]:
        raise CanonicalRunError("Per-class support differs from confusion-matrix rows.")
    if [row["class"] for row in test["per_class"]] != classes:
        raise CanonicalRunError("Per-class rows do not follow the public class order.")
    if manifest["final_model"]["fit_row_count"] != (
        split["row_counts"]["train"] + split["row_counts"]["validation"]
    ):
        raise CanonicalRunError("Final fit must use exactly train + validation.")
    features = manifest["features"]["ordered_columns"]
    if manifest["features"]["count"] != len(features) or manifest["target"]["column"] in features:
        raise CanonicalRunError("Feature contract is inconsistent.")


_EXACT_PATHS = (
    ("dataset", "source_sha256"),
    ("dataset", "source_row_count"),
    ("target", "public_class_order"),
    ("target", "estimator_class_order"),
    ("features", "ordered_columns"),
    ("protocol", "eda_partition"),
    ("protocol", "held_out_partitions_used_for_eda"),
    ("split", "row_counts"),
    ("split", "class_counts"),
    ("split", "partition_sha256"),
    ("split", "membership_sha256"),
    ("seeds",),
    ("cross_validation",),
    ("model_selection", "selected_model_id"),
    ("model_selection", "selected_model_family"),
    ("model_selection", "selected_feature_policy"),
    ("model_selection", "selected_hyperparameters"),
    ("model_selection", "selected_preprocessing"),
    ("model_selection", "selected_imbalance_policy"),
    ("final_test", "confusion_matrix"),
    ("final_model", "fit_row_count"),
)
_NUMERIC_PATHS = (
    ("selected_cv_metrics",),
    ("validation_metrics",),
    ("final_test", "metrics"),
)
_RUNTIME_DEPENDENT_PATHS = (
    ("final_model", "model_state_fingerprint"),
    ("final_model", "artifact_sha256"),
    ("final_test", "probability_matrix_sha256"),
)


def _get(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        value = value[key]
    return value


def _numeric_differences(
    expected: Mapping[str, Any], observed: Mapping[str, Any], tolerance: float
) -> dict[str, tuple[Any, Any]]:
    differences = {}
    for key in sorted(set(expected) | set(observed)):
        left, right = expected.get(key), observed.get(key)
        if left is None or right is None or not math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        ):
            differences[key] = (left, right)
    return differences


def compare_with_canonical(
    canonical: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    tolerance: float = METRIC_TOLERANCE,
) -> dict[str, Any]:
    """Compare a freshly built manifest with the canonical reference."""
    mismatches: dict[str, Any] = {}
    for path in _EXACT_PATHS:
        if _get(canonical, path) != _get(observed, path):
            mismatches[".".join(path)] = {
                "expected": _get(canonical, path),
                "observed": _get(observed, path),
            }
    for path in _NUMERIC_PATHS:
        differences = _numeric_differences(_get(canonical, path), _get(observed, path), tolerance)
        for key, (left, right) in differences.items():
            mismatches[".".join((*path, key))] = {"expected": left, "observed": right}

    same_runtime = (
        canonical["runtime"]["python"] == observed["runtime"]["python"]
        and canonical["runtime"]["packages"] == observed["runtime"]["packages"]
    )
    runtime_dependent = {}
    for path in _RUNTIME_DEPENDENT_PATHS:
        equal = _get(canonical, path) == _get(observed, path)
        runtime_dependent[".".join(path)] = equal
        if same_runtime and not equal:
            mismatches[".".join(path)] = {
                "expected": _get(canonical, path),
                "observed": _get(observed, path),
            }
    return {
        "matches_canonical": not mismatches,
        "same_runtime_as_reference": same_runtime,
        "metric_tolerance": tolerance,
        "mismatches": mismatches,
        "runtime_dependent_equalities": runtime_dependent,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH.as_posix())
    parser.add_argument("--tolerance", type=float, default=METRIC_TOLERANCE)
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    manifest_path = root / args.manifest
    observed = build_from_project(root)
    validate_manifest_structure(observed)
    environment = environment_contract_report(root, observed["runtime"])
    if args.command == "build":
        if not environment["matches_locked_environment"]:
            print(json.dumps(environment, indent=2, sort_keys=True))
            raise CanonicalRunError(
                "A canonical manifest can only be built in the environment declared "
                "by .python-version and pylock.toml."
            )
        _write_json(manifest_path, observed)
        print(f"Canonical manifest written: {args.manifest}")
        return 0

    canonical = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest_structure(canonical)
    report = compare_with_canonical(canonical, observed, tolerance=args.tolerance)
    report["environment_contract"] = environment
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["matches_canonical"] else 1


if __name__ == "__main__":
    sys.exit(main())
