from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_public_result_snapshot import (  # noqa: E402
    build_anchor_points,
    build_headline_summary,
    validate_analysis_record,
)
from verify_public_results import verify  # noqa: E402


def test_tracked_public_results_are_complete_and_self_consistent():
    assert verify(ROOT, ROOT / "results", None) == []


def test_headline_values_are_derived_from_complete_results():
    composition = ROOT / "results/composition"
    headline = build_headline_summary(
        composition,
        ROOT / "data/norman/regime_manifest.csv",
    ).set_index("metric_name")
    relative = headline.loc["relative_l2_response_error"]
    spearman = headline.loc["gene_wise_spearman"]
    assert relative["full_test_selected_model_id"] == "pb_latent_additive"
    assert relative["two_regime_switch_model_id"] == "pb_cpa"
    assert relative["analytical_switch_lambda_high_av"] == pytest.approx(0.8190334837359287)
    assert relative["first_grid_switch_lambda_high_av"] == pytest.approx(0.82)
    assert relative["interior_pairwise_crossover_count"] == 6
    assert spearman["analytical_switch_lambda_high_av"] == pytest.approx(0.7039024366025521)
    assert spearman["first_grid_switch_lambda_high_av"] == pytest.approx(0.704)
    assert spearman["interior_pairwise_crossover_count"] == 1
    assert set(headline["n_two_regime_conditions"]) == {28}


def test_anchor_table_contains_all_models_at_declared_weights():
    anchors = build_anchor_points(ROOT / "results/composition")
    assert anchors.shape[0] == 2 * 5 * 6
    assert set(anchors["lambda_high_av"]) == {0.0, 0.25, 0.5, 0.75, 1.0}
    assert anchors.groupby(["metric_name", "lambda_high_av"])["model_id"].nunique().eq(6).all()


def test_analysis_record_is_semantically_valid():
    record = validate_analysis_record(ROOT / "results/composition/analysis_record.json")
    assert record["n_conditions"] == 46
    assert record["n_models"] == 6
    assert record["population_confidence_interval"] is False


def test_unexpected_result_artifact_is_rejected(tmp_path):
    results = tmp_path / "results"
    composition = results / "composition"
    composition.mkdir(parents=True)
    for source in (ROOT / "results").rglob("*"):
        if source.is_file():
            target = results / source.relative_to(ROOT / "results")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    (composition / "undeclared.tsv").write_text("unexpected\n", encoding="utf-8")
    assert "result tree differs from the closed public allowlist" in verify(
        ROOT, results, None
    )
