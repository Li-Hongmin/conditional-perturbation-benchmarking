from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crg.benchmark_composition import (
    BenchmarkCompositionError,
    OUTPUT_NAMES,
    build_analysis,
    build_from_policy,
    collapse_model_metrics,
    sha256_file,
    write_analysis,
)


COMMIT = "a" * 40
CONDITIONS = [f"c{index}" for index in range(8)]
RUNS = [
    ("RUN-A-1", "model_a", "family_a", 1),
    ("RUN-A-2", "model_a", "family_a", 2),
    ("RUN-B-1", "model_b", "family_b", 1),
]


def _policy(*, draws: int = 40, permutations: int = 40) -> dict[str, object]:
    return {
        "schema_version": "crg_benchmark_composition_policy_v1",
        "status": "FROZEN_BEFORE_SECONDARY_CALCULATION",
        "inputs": {},
        "registered_run_ids": [record[0] for record in RUNS],
        "model_level_collapse": {
            "group_by": [
                "dataset_id", "split_id", "condition_id", "model_id",
                "metric_name", "metric_direction",
            ],
            "seed_aggregation": "arithmetic_mean",
            "seeds_are_sampling_units": False,
            "models_receive_equal_weight": True,
        },
        "focal_seed_run_ids": [record[0] for record in RUNS],
        "primary_metric": "relative_l2_response_error",
        "secondary_metric": "gene_wise_spearman",
        "primary_arms": [
            "high_additivity_violation_only", "high_effect_only",
        ],
        "composition_grid": {"start": 0.0, "stop": 1.0, "step": 0.001},
        "tie_absolute_tolerance": 1e-12,
        "resampling": {
            "generator": "PCG64", "draws": draws, "seed": 101,
            "quantiles": [0.025, 0.975],
            "sampling_unit": "condition_within_primary_arm",
        },
        "permutation": {
            "generator": "PCG64", "draws": permutations, "seed": 202,
            "preserve_category_sizes": True,
        },
        "claim_boundaries": {"population_inference": False},
    }


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    # In native lower-is-better units A wins in HE and B wins in HA.  The
    # seed-mean A crossover with B is exactly lambda=0.5.
    arms = [
        "high_additivity_violation_only", "high_additivity_violation_only",
        "high_effect_only", "high_effect_only", "neither_high", "neither_high",
        "both_high", "both_high",
    ]
    metrics: list[dict[str, object]] = []
    for run_id, model, family, seed in RUNS:
        for index, condition in enumerate(CONDITIONS):
            arm = arms[index]
            if model == "model_a":
                base_error = 3.0 if arm == "high_additivity_violation_only" else 1.0
                base_rho = 0.2 if arm == "high_additivity_violation_only" else 0.8
                seed_offset = -0.1 if seed == 1 else 0.1
            else:
                base_error = 1.0 if arm == "high_additivity_violation_only" else 3.0
                base_rho = 0.8 if arm == "high_additivity_violation_only" else 0.2
                seed_offset = 0.0
            for metric_name, value, direction in (
                ("relative_l2_response_error", base_error + seed_offset, "lower_is_better"),
                ("gene_wise_spearman", base_rho - seed_offset / 10, "higher_is_better"),
            ):
                metrics.append({
                    "run_id": run_id, "dataset_id": "synthetic", "model_id": model,
                    "family": family, "seed": seed, "split_id": "test",
                    "condition_id": condition, "metric_name": metric_name,
                    "metric_value": value, "metric_direction": direction,
                    "metric_status": "estimated", "not_evaluable_reason": "",
                })
    regimes = pd.DataFrame({
        "dataset_id": ["synthetic"] * 8, "split_id": ["test"] * 8,
        "condition_id": CONDITIONS, "primary_arm": arms,
        "high_additivity_violation": [True, True, False, False, False, False, True, True],
        "high_effect": [False, False, True, True, False, False, True, True],
        "additivity_violation_score": [0.9, 0.8, 0.2, 0.3, 0.1, 0.4, 0.7, 0.6],
        "effect_size_covariate": [0.2, 0.3, 0.8, 0.9, 0.1, 0.4, 0.7, 0.6],
    })
    return pd.DataFrame(metrics), regimes


def _build(draws: int = 40, permutations: int = 40):
    metrics, regimes = _inputs()
    return build_analysis(
        metrics, regimes, _policy(draws=draws, permutations=permutations),
        producer_commit=COMMIT,
        input_sha256={"condition_metrics": "b" * 64, "regime_manifest": "c" * 64},
    )


def test_seed_collapse_is_arithmetic_mean_and_models_receive_equal_rows():
    metrics, _ = _inputs()
    collapsed = collapse_model_metrics(metrics, _policy())
    selected = collapsed.loc[
        collapsed["model_id"].eq("model_a")
        & collapsed["condition_id"].eq("c0")
        & collapsed["metric_name"].eq("relative_l2_response_error")
    ].iloc[0]
    assert selected["metric_value"] == pytest.approx(3.0)
    assert selected["n_registered_runs_collapsed"] == 2
    assert selected["seed_ids"] == "1,2"
    assert collapsed.groupby(["model_id", "metric_name"]).size().nunique() == 1


def test_composition_grid_crossovers_ties_metric_direction_and_regret():
    artifacts = _build()
    assert set(artifacts.tables) == set(OUTPUT_NAMES)
    curves = artifacts.tables["composition_risk_curves.tsv"]
    assert curves["composition_id"].nunique() == 1001
    midpoint = curves.loc[
        curves["metric_name"].eq("relative_l2_response_error")
        & curves["composition_id"].eq("lambda_0500")
    ]
    assert set(midpoint.loc[midpoint["is_winner"], "model_id"]) == {"model_a", "model_b"}
    assert set(midpoint["rank"]) == {1.5}
    crossovers = artifacts.tables["pairwise_crossovers.tsv"]
    lower = crossovers.loc[crossovers["metric_name"].eq("relative_l2_response_error")].iloc[0]
    assert lower["crossover_lambda"] == pytest.approx(0.5)
    assert lower["point_status"] == "interior_or_endpoint_crossover"
    winners = artifacts.tables["composition_winners.tsv"]
    rho_start = winners.loc[
        winners["metric_name"].eq("gene_wise_spearman")
        & winners["composition_id"].eq("lambda_0000"), "winner_model_ids"
    ].iloc[0]
    assert rho_start == "model_a"  # higher Spearman is correctly better
    regret = artifacts.tables["selection_regret.tsv"]
    assert (regret["selection_regret"] >= -1e-14).all()
    assert artifacts.record["population_confidence_interval"] is False
    assert artifacts.record["p_values_calculated"] is False


def test_resampling_geometry_and_permutation_outputs_are_deterministic():
    first = _build(draws=30, permutations=30)
    second = _build(draws=30, permutations=30)
    for name in OUTPUT_NAMES:
        pd.testing.assert_frame_equal(first.tables[name], second.tables[name])
    intervals = first.tables["composition_resampling_intervals.tsv"]
    assert set(intervals["quantity"]) == {
        "model_oriented_risk", "aggregate_winner_selection_regret",
    }
    assert intervals["interval_method"].str.contains("not_population_ci").all()
    summary = first.tables["geometry_ablation_summary.tsv"]
    assert set(summary["analysis_type"]) == {
        "continuous_pairwise", "hard_partition", "permutation_reference",
    }
    assert {
        "continuous_intercept", "continuous_effect_rank",
        "continuous_additivity_violation_rank", "continuous_additive_axes",
        "continuous_axis_interaction",
    }.issubset(set(summary["continuous_specification"]))
    permuted = summary.loc[summary["analysis_type"].eq("permutation_reference")]
    assert set(permuted["permutation_draws"]) == {30}
    assert permuted["permutation_exceedance_fraction"].between(0, 1).all()
    float_columns = summary.select_dtypes(include=[np.floating]).columns
    pd.testing.assert_frame_equal(
        summary.loc[:, float_columns],
        summary.loc[:, float_columns].round(12),
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_path_entrypoint_fails_closed_on_hash_drift_and_writer_is_deterministic(tmp_path):
    metrics, regimes = _inputs()
    metrics_path = tmp_path / "metrics.tsv"
    regimes_path = tmp_path / "regimes.csv"
    registry_path = tmp_path / "registry.yaml"
    policy_path = tmp_path / "policy.json"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    regimes.to_csv(regimes_path, index=False)
    registry_path.write_text("frozen: true\n", encoding="utf-8")
    policy = _policy(draws=10, permutations=10)
    policy["inputs"] = {
        "condition_metrics": {"path": metrics_path.name, "sha256": _hash(metrics_path)},
        "regime_manifest": {"path": regimes_path.name, "sha256": _hash(regimes_path)},
        "run_registry": {"path": registry_path.name, "sha256": _hash(registry_path)},
    }
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    artifacts = build_from_policy(tmp_path, policy_path, producer_commit=COMMIT)
    first = write_analysis(tmp_path / "out-a", artifacts)
    second = write_analysis(tmp_path / "out-b", artifacts)
    for name in (*OUTPUT_NAMES, "analysis_record.json"):
        assert sha256_file(first / name) == sha256_file(second / name)
    metrics_path.write_text(metrics_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkCompositionError, match="hash mismatch: condition_metrics"):
        build_from_policy(tmp_path, policy_path, producer_commit=COMMIT)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda m, r: (m.iloc[:-1], r), "complete condition universe"),
        (lambda m, r: (pd.concat([m, m.iloc[[0]]], ignore_index=True), r), "duplicate"),
        (lambda m, r: (m.assign(metric_value=np.inf), r), "finite"),
        (lambda m, r: (m, r.iloc[:-1]), "condition universes differ"),
    ],
)
def test_input_contract_fails_closed(mutation, message):
    metrics, regimes = _inputs()
    changed_metrics, changed_regimes = mutation(metrics, regimes)
    with pytest.raises(BenchmarkCompositionError, match=message):
        build_analysis(
            changed_metrics, changed_regimes, _policy(), producer_commit=COMMIT,
            input_sha256={"condition_metrics": "b" * 64, "regime_manifest": "c" * 64},
        )
