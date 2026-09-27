"""Tests for the portable Notebook-01 exploration handoff."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.build_exploration_handoff import (
    ExplorationHandoffError,
    build_static_multiclass_exploration_handoff,
    development_feature_evidence,
    load_and_validate_exploration_handoff,
)


FEATURES = ("Area", "Perimeter", "Compactness")
CLASSES = ("A", "B", "C")


class TargetReport:
    has_issues = False
    class_count = 3
    imbalance_ratio = 2.0
    normalized_class_entropy = 0.9
    majority_classes = ("A",)
    minority_classes = ("C",)

    def distribution_frame(self, *, format_percentages: bool = False) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Class": CLASSES,
                "Count": [6, 4, 3],
                "Proportion": [6 / 13, 4 / 13, 3 / 13],
                "Role": ["Majority", "Intermediate", "Minority"],
            }
        )


class DuplicateReport:
    has_source_identifiers = False
    exact_duplicate_group_count = 1
    exact_duplicate_row_count = 2
    target_conflict_group_count = 0


class LeakageReport:
    is_structurally_valid = True
    has_direct_target_leakage = False
    confirmed_derived_dependency_count = 1

    def target_proxy_candidates_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["Feature"])

    def dependency_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Derived feature": "Compactness",
                    "Dependency status": "Confirmed from retained columns",
                    "Target-derived": False,
                },
                {
                    "Derived feature": "Perimeter",
                    "Dependency status": "Declared dependency not confirmed",
                    "Target-derived": False,
                },
            ]
        )


class QualityReport:
    is_structurally_valid = True

    def findings_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Finding ID": "DQ-001",
                    "Title": "Review exact matches",
                    "Disposition": "Review",
                }
            ]
        )

    def validated_non_issues_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Non-issue ID": "NI-001",
                    "Title": "No missing values",
                    "Disposition": "No action",
                }
            ]
        )


class InsightsReport:
    is_structurally_valid = True

    def key_insights_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Insight ID": "INS-001",
                    "Theme": "Class separation",
                    "Title": "Overlap remains",
                    "Relevance": "High",
                    "Status": "Observed",
                    "Summary": "Some class profiles overlap.",
                    "Modeling implication": "Inspect confusion.",
                    "Interpretation boundary": "EDA only.",
                }
            ]
        )

    def hypotheses_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Hypothesis ID": "HYP-001",
                    "Title": "Overlap predicts confusion",
                    "Hypothesis": "Overlapping classes may be confused.",
                    "Required validation": "Inspect validation confusion matrices.",
                }
            ]
        )

    def limitations_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Limitation ID": "LIM-001",
                    "Title": "EDA is not performance",
                    "Limitation type": "Modeling",
                }
            ]
        )


class PreparationReport:
    is_structurally_valid = True
    is_ready_for_deterministic_preparation = True
    is_ready_for_split_execution = True
    is_ready_for_modeling = False

    def decisions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Decision ID": "PREP-001",
                    "Domain": "Cleaning",
                    "Title": "Preserve source",
                    "Status": "Approved",
                }
            ]
        )

    def execution_plan_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Step ID": "STEP-001",
                    "Sequence": 1,
                    "Action": "Create prepared copy",
                    "Status": "Planned",
                }
            ]
        )

    def guardrails_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Guardrail ID": "GRD-001",
                    "Title": "Protect target",
                    "Status": "Active",
                }
            ]
        )

    def split_policy_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Policy item": "Train fraction", "Value": 0.70},
                {"Policy item": "Validation fraction", "Value": 0.15},
                {"Policy item": "Test fraction", "Value": 0.15},
                {"Policy item": "Stratification field", "Value": "Class"},
                {"Policy item": "Random seed", "Value": 42},
                {"Policy item": "Final test holdout", "Value": True},
                {"Policy item": "Disjoint partitions", "Value": True},
                {"Policy item": "Identifier grouping", "Value": ()},
                {
                    "Policy item": "Temporal policy status",
                    "Value": "Resolved snapshot fallback",
                },
            ]
        )


class RelationshipReport:
    numerical_relationships = pd.DataFrame(
        [
            {
                "Feature A": "Area",
                "Feature B": "Perimeter",
                "Potential redundancy": True,
            }
        ]
    )


class ClassProfileReport:
    row_count = 13

    def pairwise_overlap_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Class A": "B",
                    "Class B": "C",
                    "Mean IQR overlap coefficient": 0.4,
                    "RMS robust median gap": 0.5,
                },
                {
                    "Class A": "A",
                    "Class B": "B",
                    "Mean IQR overlap coefficient": 0.1,
                    "RMS robust median gap": 1.5,
                },
            ]
        )


SOURCE_ROWS = 19


def source_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Area": [float(index + 1) for index in range(SOURCE_ROWS)],
            "Perimeter": [float(index + 2) for index in range(SOURCE_ROWS)],
            "Compactness": [0.7] * SOURCE_ROWS,
            "Class": [CLASSES[index % 3] for index in range(SOURCE_ROWS)],
        }
    )


def development_split(**overrides) -> dict:
    payload = {
        "schema_version": "development-split.v1",
        "split_frozen_before_distribution_aware_analysis": True,
        "policy": {"random_seed": 42},
        "row_counts": {"train": 13, "validation": 3, "test": 3},
        "partition_sha256": {"train": "a" * 64, "validation": "b" * 64, "test": "c" * 64},
        "membership_sha256": {"train": "d" * 64, "validation": "e" * 64, "test": "f" * 64},
        "eda_partition": "train",
        "eda_row_count": 13,
        "held_out_partitions_used_for_eda": [],
    }
    payload.update(overrides)
    return payload


def build_report(tmp_path: Path, **overrides):
    source = tmp_path / "dataset.csv"
    source.write_text(
        "Area,Perimeter,Compactness,Class\n1,2,0.7,A\n",
        encoding="utf-8",
    )
    params = {
        "dataset_slug": "dry-bean",
        "source_repository": "UCI Machine Learning Repository",
        "source_dataset_id": 602,
        "source_file": source,
        "project_root": tmp_path,
        "source_dataframe": source_dataframe(),
        "target_contract": SimpleNamespace(
            target="Class",
            expected_classes=CLASSES,
            problem_type="multiclass_classification",
            class_semantics="Nominal / unordered",
        ),
        "feature_columns": FEATURES,
        "numerical_features": FEATURES,
        "identifier_columns": (),
        "target_report": TargetReport(),
        "duplicate_report": DuplicateReport(),
        "feature_relationship_report": RelationshipReport(),
        "leakage_report": LeakageReport(),
        "quality_report": QualityReport(),
        "insights_report": InsightsReport(),
        "preparation_report": PreparationReport(),
        "class_profile_report": ClassProfileReport(),
        "development_split": development_split(),
    }
    params.update(overrides)
    return build_static_multiclass_exploration_handoff(**params)


def test_builds_ready_portable_handoff(tmp_path: Path) -> None:
    report = build_report(tmp_path)

    assert report.is_structurally_valid
    assert report.is_handoff_ready
    assert report.payload["schema_version"] == "exploration-handoff.v2"
    assert report.payload["source"]["dataset_id"] == 602
    assert report.payload["source"]["path"] == "dataset.csv"
    assert report.payload["prediction_contract"]["positive_class"] is None
    assert report.payload["feature_contract"]["feature_columns"] == list(FEATURES)
    assert report.payload["preparation_contract"]["split_policy"]["random_seed"] == 42
    assert report.payload["readiness"]["model_selection_ready"] is False


def test_open_reviews_preserve_nonblocking_evidence(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    reviews = report.open_reviews_frame()

    assert not reviews.empty
    assert reviews["blocking"].eq(False).all()
    assert "Duplicate identity" in set(reviews["theme"])
    assert "Derived-feature dependency" in set(reviews["theme"])
    assert "Feature redundancy" in set(reviews["theme"])
    assert "Class support" in set(reviews["theme"])
    assert "Exploratory hypothesis" in set(reviews["theme"])


def test_next_steps_keep_notebook_03_waiting(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    steps = report.next_steps_frame()

    notebook_02 = steps.loc[steps["Notebook"].eq("02_data_preparation.ipynb")]
    notebook_03 = steps.loc[
        steps["Notebook"].eq("03_model_selection_and_evaluation.ipynb")
    ]

    assert notebook_02["Status"].eq("Ready").all()
    assert notebook_03["Status"].eq("Waiting on Notebook 02").all()


def test_write_is_atomic_and_reloadable(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    destination = tmp_path / "artifacts/exploration/dry-bean/exploration-handoff.json"

    persisted = report.write(destination)

    assert persisted.path == destination.resolve()
    assert persisted.size_bytes > 0
    assert len(persisted.sha256) == 64

    payload = load_and_validate_exploration_handoff(
        destination,
        expected_dataset_slug="dry-bean",
        expected_source_dataset_id=602,
    )
    assert payload["readiness"]["split_execution_ready"] is True
    assert payload["source"]["sha256"]


def test_write_is_deterministic_for_same_payload(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    destination = tmp_path / "handoff.json"

    first = report.write(destination)
    second = report.write(destination)

    assert first.sha256 == second.sha256
    assert first.size_bytes == second.size_bytes


def test_target_cannot_be_predictor(tmp_path: Path) -> None:
    report = build_report(
        tmp_path,
        feature_columns=(*FEATURES, "Class"),
        numerical_features=(*FEATURES, "Class"),
    )

    assert not report.is_structurally_valid
    with pytest.raises(ExplorationHandoffError, match="not ready"):
        report.raise_if_invalid()


def test_static_dry_bean_handoff_requires_all_numeric_features(tmp_path: Path) -> None:
    report = build_report(
        tmp_path,
        numerical_features=("Area", "Perimeter"),
    )

    assert not report.is_structurally_valid
    issues = report.issues_frame()
    assert issues["Issue"].str.contains("entirely numerical").any()


def test_upstream_leakage_failure_blocks_handoff(tmp_path: Path) -> None:
    leakage = LeakageReport()
    leakage.has_direct_target_leakage = True  # type: ignore[attr-defined]

    report = build_report(tmp_path, leakage_report=leakage)

    assert not report.is_handoff_ready
    assert report.issues_frame()["Issue"].str.contains("Leakage audit").any()


def test_split_readiness_is_required(tmp_path: Path) -> None:
    preparation = PreparationReport()
    preparation.is_ready_for_split_execution = False  # type: ignore[attr-defined]

    report = build_report(tmp_path, preparation_report=preparation)

    assert not report.is_handoff_ready
    assert report.issues_frame()["Issue"].str.contains("Split execution").any()


def test_loader_rejects_tampered_contract(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    destination = tmp_path / "handoff.json"
    report.write(destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["prediction_contract"]["target_classes"] = ["A", "B"]
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ExplorationHandoffError,
        match="multiclass target contract",
    ):
        load_and_validate_exploration_handoff(destination)


def test_loader_rejects_wrong_source_dataset_id(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    destination = tmp_path / "handoff.json"
    report.write(destination)

    with pytest.raises(
        ExplorationHandoffError,
        match="source dataset ID mismatch",
    ):
        load_and_validate_exploration_handoff(
            destination,
            expected_source_dataset_id=999,
        )


def test_handoff_records_split_frozen_before_train_only_eda(tmp_path: Path) -> None:
    report = build_report(tmp_path)

    split = report.payload["development_split"]
    assert split["split_frozen_before_distribution_aware_analysis"] is True
    assert split["eda_partition"] == "train"
    assert split["eda_row_count"] == 13
    assert split["held_out_partitions_used_for_eda"] == []
    assert split["distribution_aware_reports_scope"] == "train"
    assert report.payload["class_overlap"]["highest_overlap_pair"] == ["B", "C"]


@pytest.mark.parametrize(
    ("override", "issue"),
    [
        ({"eda_partition": "test", "eda_row_count": 3}, "Unexpected EDA partition"),
        ({"held_out_partitions_used_for_eda": ["test"]}, "Held-out data used in EDA"),
        ({"split_frozen_before_distribution_aware_analysis": False}, "Split not frozen"),
        ({"row_counts": {"train": 13, "validation": 3, "test": 4}}, "do not cover the source"),
    ],
)
def test_development_split_violations_block_handoff(
    tmp_path: Path, override: dict, issue: str
) -> None:
    report = build_report(tmp_path, development_split=development_split(**override))

    assert not report.is_handoff_ready
    assert report.issues_frame()["Issue"].str.contains(issue).any()


def test_full_source_target_exploration_is_rejected(tmp_path: Path) -> None:
    class FullSourceTargetReport(TargetReport):
        def distribution_frame(self, *, format_percentages: bool = False) -> pd.DataFrame:
            frame = super().distribution_frame(format_percentages=format_percentages)
            frame["Count"] = [7, 6, 6]
            return frame

    report = build_report(tmp_path, target_report=FullSourceTargetReport())

    assert not report.is_handoff_ready
    assert report.issues_frame()["Issue"].str.contains(
        "Target exploration scope differs"
    ).any()


def test_distribution_aware_report_row_count_must_match_eda_partition(
    tmp_path: Path,
) -> None:
    profile = ClassProfileReport()
    profile.row_count = SOURCE_ROWS  # type: ignore[misc]

    report = build_report(tmp_path, class_profile_report=profile)

    assert not report.is_handoff_ready
    assert report.issues_frame()["Details"].str.contains("class_profile_report").any()


def test_loader_requires_train_only_development_split(tmp_path: Path) -> None:
    report = build_report(tmp_path)
    destination = tmp_path / "handoff.json"
    report.write(destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["development_split"]["eda_partition"] = "validation"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExplorationHandoffError, match="training partition only"):
        load_and_validate_exploration_handoff(destination)

    del payload["development_split"]
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExplorationHandoffError, match="development_split"):
        load_and_validate_exploration_handoff(destination)


def test_development_feature_evidence_summarizes_train_only_governance(
    tmp_path: Path,
) -> None:
    report = build_report(tmp_path)

    evidence = development_feature_evidence(report.payload)

    assert evidence["eda_partition"] == "train"
    assert evidence["eda_row_count"] == 13
    assert evidence["held_out_partitions_used_for_eda"] == []
    assert evidence["confirmed_derived_features"] == ["Compactness"]
    assert evidence["unresolved_provenance_features"] == ["Perimeter"]
    assert evidence["highest_overlap_class_pair"] == ["B", "C"]
