"""Structural guards for the split-before-EDA protocol in the official notebooks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTION_AWARE_CALLS = (
    "analyze_target_distribution(",
    "analyze_numerical_features(",
    "analyze_numerical_feature_relationships(",
    "analyze_multiclass_numerical_target_relationships(",
    "analyze_multiclass_class_profiles(",
    "analyze_static_classification_leakage(",
)


def _code_cells(name: str) -> list[str]:
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _first_index(cells: list[str], token: str) -> int:
    return next(index for index, source in enumerate(cells) if token in source)


def test_notebook_01_freezes_split_before_distribution_aware_exploration() -> None:
    cells = _code_cells("01_data_understanding_and_exploration.ipynb")
    split_index = _first_index(cells, "build_development_split_reference(")
    assert "split_classification_dataset(" in cells[split_index]
    assert "development_df = development_partitions.train" in cells[split_index]
    assert "del development_partitions" in cells[split_index]
    for call in DISTRIBUTION_AWARE_CALLS:
        assert _first_index(cells, call) > split_index, call


@pytest.mark.parametrize("call", DISTRIBUTION_AWARE_CALLS)
def test_notebook_01_explores_training_partition_only(call: str) -> None:
    cells = _code_cells("01_data_understanding_and_exploration.ipynb")
    source = cells[_first_index(cells, call)]
    arguments = source.split(call, 1)[1]
    assert re.match(r"\s*development_df,", arguments), call


def test_notebook_01_never_retains_held_out_partitions() -> None:
    code = "\n".join(_code_cells("01_data_understanding_and_exploration.ipynb"))
    assert not re.search(r"\.(validation|test)\b", code)
    assert "development_split=DEVELOPMENT_SPLIT" in code


def test_notebook_02_reproduces_the_split_frozen_before_eda() -> None:
    code = "\n".join(_code_cells("02_data_preparation.ipynb"))
    assert "validate_partitions_against_development_split(" in code
    assert 'development_split["policy"]' in code
    assert "development_evidence=DEVELOPMENT_EVIDENCE" in code


def test_notebook_03_feature_policies_come_from_training_only_evidence() -> None:
    code = "\n".join(_code_cells("03_model_selection_and_evaluation.ipynb"))
    assert 'feature_manifest["development_evidence"]' in code
    assert '"AspectRatio", "Eccentricity"' not in code
    assert '("BARBUNYA", "CALI")' not in code
    assert 'DEVELOPMENT_EVIDENCE["highest_overlap_class_pair"]' in code


def test_notebook_04_focal_pairs_come_from_frozen_selection_evidence() -> None:
    code = "\n".join(_code_cells("04_final_model_and_bundle.ipynb"))
    assert "focal_pairs=FOCAL_CONFUSION_PAIRS" in code
    assert '("DERMASON", "SIRA")' not in code


def test_notebook_markdown_has_no_obsolete_stage_claims() -> None:
    for path in (ROOT / "notebooks").glob("0*.ipynb"):
        text = path.read_text(encoding="utf-8")
        assert "Notebook 05 is not implemented" not in text, path.name
        assert "no inference demo was implemented" not in text, path.name
        assert "must be generalized for the multiclass" not in text, path.name
