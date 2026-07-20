#!/usr/bin/env python3
"""Build the tracked, low-compute result snapshot from a verified replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_RECORD = "analysis_record.json"
SCIENTIFIC_ARTIFACTS = (
    "composition_resampling_intervals.tsv",
    "composition_risk_curves.tsv",
    "composition_winners.tsv",
    "focal_seed_sensitivity.tsv",
    "geometry_ablation_model_strata.tsv",
    "geometry_ablation_summary.tsv",
    "metric_sensitivity.tsv",
    "model_condition_metrics_collapsed.tsv",
    "pairwise_crossovers.tsv",
    "ranking_stability.tsv",
    "selection_regret.tsv",
)
SNAPSHOT_ARTIFACTS = (ANALYSIS_RECORD, *SCIENTIFIC_ARTIFACTS)
TOP_LEVEL_ARTIFACTS = (
    "headline_summary.tsv",
    "composition_anchor_points.tsv",
    "RESULT_CHECKSUMS.tsv",
)
ANCHOR_COMPOSITION_IDS = (
    "lambda_0000",
    "lambda_0250",
    "lambda_0500",
    "lambda_0750",
    "lambda_1000",
)


class PublicResultError(ValueError):
    """Raised when a source replay cannot support the public snapshot."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_scientific_hashes(repository_root: Path) -> dict[str, str]:
    ledger = repository_root / "expected/EXPECTED_OUTPUT_CHECKSUMS.tsv"
    with ledger.open(newline="", encoding="utf-8") as handle:
        records = {
            row["artifact"]: row["sha256"]
            for row in csv.DictReader(handle, delimiter="\t")
            if row["artifact"] != ANALYSIS_RECORD
        }
    if set(records) != set(SCIENTIFIC_ARTIFACTS):
        raise PublicResultError("expected-output ledger does not name the 11 scientific tables")
    return records


def validate_analysis_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise PublicResultError(f"missing analysis record: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    valid = (
        record.get("schema_version") == "crg_benchmark_composition_analysis_v1"
        and record.get("status") == "CALCULATED_POST_RESULT_SECONDARY_ANALYSIS"
        and record.get("n_conditions") == 46
        and record.get("n_models") == 6
        and record.get("composition_grid_points") == 1001
        and record.get("resampling", {}).get("draws") == 10000
        and record.get("permutation", {}).get("draws") == 10000
        and record.get("population_confidence_interval") is False
        and record.get("p_values_calculated") is False
    )
    if not valid:
        raise PublicResultError("analysis record does not satisfy the public semantic contract")
    return record


def validate_replay(repository_root: Path, replay_output_dir: Path) -> None:
    expected = expected_scientific_hashes(repository_root)
    errors: list[str] = []
    for artifact, wanted in expected.items():
        path = replay_output_dir / artifact
        observed = sha256(path) if path.is_file() else "MISSING"
        if observed != wanted:
            errors.append(f"{artifact}: expected {wanted}, observed {observed}")
    validate_analysis_record(replay_output_dir / ANALYSIS_RECORD)
    if errors:
        raise PublicResultError("scientific replay differs from the frozen ledger:\n" + "\n".join(errors))


def _read_tables(composition_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "risk": pd.read_csv(composition_dir / "composition_risk_curves.tsv", sep="\t"),
        "winners": pd.read_csv(composition_dir / "composition_winners.tsv", sep="\t"),
        "regret": pd.read_csv(composition_dir / "selection_regret.tsv", sep="\t"),
        "ranking": pd.read_csv(composition_dir / "ranking_stability.tsv", sep="\t"),
        "crossovers": pd.read_csv(composition_dir / "pairwise_crossovers.tsv", sep="\t"),
        "collapsed": pd.read_csv(
            composition_dir / "model_condition_metrics_collapsed.tsv", sep="\t"
        ),
    }


def build_headline_summary(
    composition_dir: Path,
    regime_manifest_path: Path,
) -> pd.DataFrame:
    tables = _read_tables(composition_dir)
    regimes = pd.read_csv(regime_manifest_path)
    arm_counts = regimes["primary_arm"].value_counts().to_dict()
    rows: list[dict[str, object]] = []

    for metric_name in sorted(tables["risk"]["metric_name"].unique()):
        winners = (
            tables["winners"].loc[
                tables["winners"]["metric_name"].eq(metric_name)
            ].sort_values("lambda_high_av")
        )
        regret = tables["regret"].loc[tables["regret"]["metric_name"].eq(metric_name)]
        crossovers = tables["crossovers"].loc[
            tables["crossovers"]["metric_name"].eq(metric_name)
        ]
        collapsed = tables["collapsed"].loc[
            tables["collapsed"]["metric_name"].eq(metric_name)
        ]
        direction = str(winners["metric_direction"].iloc[0])
        aggregate_winner = str(regret["aggregate_winner_model_id"].iloc[0])
        aggregate_value = float(
            collapsed.loc[collapsed["model_id"].eq(aggregate_winner), "metric_value"].mean()
        )
        first_model = str(winners["winner_model_ids"].iloc[0])
        changed = winners.loc[winners["winner_model_ids"].ne(first_model)]
        if changed.empty:
            switch_model = ""
            first_grid_switch = float("nan")
            analytical_switch = float("nan")
        else:
            switch_model = str(changed["winner_model_ids"].iloc[0])
            first_grid_switch = float(changed["lambda_high_av"].iloc[0])
            pair = crossovers.loc[
                (
                    crossovers["model_a"].eq(first_model)
                    & crossovers["model_b"].eq(switch_model)
                )
                | (
                    crossovers["model_a"].eq(switch_model)
                    & crossovers["model_b"].eq(first_model)
                )
            ]
            finite = pair.loc[pair["crossover_lambda"].notna()]
            analytical_switch = (
                float(finite["crossover_lambda"].iloc[0]) if len(finite) == 1 else float("nan")
            )

        rows.append({
            "metric_name": metric_name,
            "metric_direction": direction,
            "full_test_selected_model_id": aggregate_winner,
            "full_test_selected_metric_value": aggregate_value,
            "two_regime_start_model_id": first_model,
            "two_regime_switch_model_id": switch_model,
            "analytical_switch_lambda_high_av": analytical_switch,
            "first_grid_switch_lambda_high_av": first_grid_switch,
            "maximum_selection_regret": float(regret["selection_regret"].max()),
            "interior_pairwise_crossover_count": int(
                crossovers["crossover_lambda"].notna().sum()
            ),
            "n_models": int(tables["risk"]["model_id"].nunique()),
            "n_full_test_conditions": int(collapsed["condition_id"].nunique()),
            "n_high_additivity_violation_only": int(
                arm_counts.get("high_additivity_violation_only", 0)
            ),
            "n_high_effect_only": int(arm_counts.get("high_effect_only", 0)),
            "n_two_regime_conditions": int(
                arm_counts.get("high_additivity_violation_only", 0)
                + arm_counts.get("high_effect_only", 0)
            ),
            "composition_grid_points": int(winners["composition_id"].nunique()),
        })
    return pd.DataFrame(rows)


def build_anchor_points(composition_dir: Path) -> pd.DataFrame:
    tables = _read_tables(composition_dir)
    risk = tables["risk"].loc[
        tables["risk"]["composition_id"].isin(ANCHOR_COMPOSITION_IDS)
    ].copy()
    regret = tables["regret"][
        [
            "metric_name",
            "metric_direction",
            "composition_id",
            "lambda_high_av",
            "aggregate_winner_model_id",
            "conditional_winner_model_ids",
            "selection_regret",
            "aggregate_winner_is_optimal",
        ]
    ]
    ranking = tables["ranking"][
        [
            "metric_name",
            "metric_direction",
            "composition_id",
            "lambda_high_av",
            "kendall_tau_b_vs_aggregate",
        ]
    ]
    keys = ["metric_name", "metric_direction", "composition_id", "lambda_high_av"]
    result = risk.merge(regret, on=keys, how="left", validate="many_to_one")
    result = result.merge(ranking, on=keys, how="left", validate="many_to_one")
    columns = [
        "metric_name",
        "metric_direction",
        "composition_id",
        "lambda_high_av",
        "weight_high_effect",
        "model_id",
        "metric_value",
        "oriented_risk",
        "rank",
        "is_winner",
        "regret",
        "aggregate_winner_model_id",
        "conditional_winner_model_ids",
        "selection_regret",
        "aggregate_winner_is_optimal",
        "kendall_tau_b_vs_aggregate",
    ]
    return result[columns].sort_values(
        ["metric_name", "lambda_high_av", "rank", "model_id"]
    ).reset_index(drop=True)


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n", float_format="%.17g")


def write_checksum_ledger(results_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    ledger = results_dir / "RESULT_CHECKSUMS.tsv"
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file() or path == ledger:
            continue
        rows.append({
            "artifact": path.relative_to(results_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    _write_table(pd.DataFrame(rows), ledger)


def build_snapshot(
    repository_root: Path,
    replay_output_dir: Path,
    results_dir: Path,
) -> None:
    repository_root = repository_root.resolve()
    replay_output_dir = replay_output_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    composition_dir = results_dir / "composition"
    composition_dir.mkdir(parents=True, exist_ok=True)

    allowed = {
        *(Path("composition") / artifact for artifact in SNAPSHOT_ARTIFACTS),
        *(Path(artifact) for artifact in TOP_LEVEL_ARTIFACTS),
    }
    unexpected = sorted(
        str(path.relative_to(results_dir))
        for path in results_dir.rglob("*")
        if path.is_file() and path.relative_to(results_dir) not in allowed
    )
    if unexpected:
        raise PublicResultError(
            "results directory contains unexpected files: " + ", ".join(unexpected)
        )

    validate_replay(repository_root, replay_output_dir)
    for artifact in SNAPSHOT_ARTIFACTS:
        shutil.copyfile(replay_output_dir / artifact, composition_dir / artifact)

    headline = build_headline_summary(
        composition_dir,
        repository_root / "data/norman/regime_manifest.csv",
    )
    anchors = build_anchor_points(composition_dir)
    _write_table(headline, results_dir / "headline_summary.tsv")
    _write_table(anchors, results_dir / "composition_anchor_points.tsv")
    write_checksum_ledger(results_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--replay-output-dir", type=Path, default=ROOT / "replay_outputs")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    try:
        build_snapshot(args.repository_root, args.replay_output_dir, args.results_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"PUBLIC_RESULT_SNAPSHOT_INVALID: {exc}")
        return 2
    print(f"PUBLIC_RESULT_SNAPSHOT_WRITTEN: {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
