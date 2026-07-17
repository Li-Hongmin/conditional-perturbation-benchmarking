"""Finite-benchmark composition analysis for perturbation predictors.

The module operates only on condition-level metrics and post-inference regime
annotations.  Its intervals describe reweighting stability in the finite
benchmark; they are not population confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy.stats import kendalltau, rankdata


SCHEMA_VERSION = "crg_benchmark_composition_analysis_v1"
POLICY_SCHEMA_VERSION = "crg_benchmark_composition_policy_v1"
PRIMARY_ARM = "high_additivity_violation_only"
SECONDARY_ARM = "high_effect_only"
JOINT_ARMS = (
    "high_additivity_violation_only",
    "high_effect_only",
    "neither_high",
    "both_high",
)
METRIC_REQUIRED = (
    "run_id", "dataset_id", "model_id", "family", "seed", "split_id",
    "condition_id", "metric_name", "metric_value", "metric_direction",
    "metric_status", "not_evaluable_reason",
)
REGIME_REQUIRED = (
    "dataset_id", "split_id", "condition_id", "primary_arm",
    "high_additivity_violation", "high_effect", "additivity_violation_score",
    "effect_size_covariate",
)
OUTPUT_NAMES = (
    "model_condition_metrics_collapsed.tsv",
    "composition_risk_curves.tsv",
    "composition_winners.tsv",
    "pairwise_crossovers.tsv",
    "ranking_stability.tsv",
    "selection_regret.tsv",
    "composition_resampling_intervals.tsv",
    "focal_seed_sensitivity.tsv",
    "metric_sensitivity.tsv",
    "geometry_ablation_summary.tsv",
    "geometry_ablation_model_strata.tsv",
)


class BenchmarkCompositionError(ValueError):
    """Raised when the frozen analysis contract cannot be evaluated."""


@dataclass(frozen=True)
class BenchmarkCompositionArtifacts:
    tables: Mapping[str, pd.DataFrame]
    record: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_policy(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise BenchmarkCompositionError("unsupported benchmark-composition policy")
    if value.get("status") != "FROZEN_BEFORE_SECONDARY_CALCULATION":
        raise BenchmarkCompositionError("policy is not frozen before calculation")
    if value.get("primary_arms") != [PRIMARY_ARM, SECONDARY_ARM]:
        raise BenchmarkCompositionError("primary arm order differs from the frozen contract")
    collapse = value.get("model_level_collapse", {})
    expected_group = [
        "dataset_id", "split_id", "condition_id", "model_id", "metric_name",
        "metric_direction",
    ]
    if collapse.get("group_by") != expected_group or collapse.get("seed_aggregation") != "arithmetic_mean":
        raise BenchmarkCompositionError("model-level collapse policy has drifted")
    grid = value.get("composition_grid", {})
    if grid != {"start": 0.0, "stop": 1.0, "step": 0.001}:
        raise BenchmarkCompositionError("composition grid has drifted")
    if value.get("resampling", {}).get("generator") != "PCG64":
        raise BenchmarkCompositionError("resampling generator must be PCG64")
    if value.get("permutation", {}).get("generator") != "PCG64":
        raise BenchmarkCompositionError("permutation generator must be PCG64")
    return value


def _text(table: pd.DataFrame, field: str, context: str) -> pd.Series:
    values = table[field]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise BenchmarkCompositionError(f"{context} {field} contains null or blank values")
    return values.astype(str)


def _validate_inputs(
    metrics: pd.DataFrame, regimes: pd.DataFrame, policy: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        raise BenchmarkCompositionError("condition metrics must be non-empty")
    if not isinstance(regimes, pd.DataFrame) or regimes.empty:
        raise BenchmarkCompositionError("regime manifest must be non-empty")
    missing = [field for field in METRIC_REQUIRED if field not in metrics]
    if missing:
        raise BenchmarkCompositionError("condition metrics missing fields: " + ", ".join(missing))
    missing = [field for field in REGIME_REQUIRED if field not in regimes]
    if missing:
        raise BenchmarkCompositionError("regime manifest missing fields: " + ", ".join(missing))
    metrics = metrics.loc[:, METRIC_REQUIRED].copy()
    regimes = regimes.loc[:, REGIME_REQUIRED].copy()
    for field in ("run_id", "dataset_id", "model_id", "family", "split_id", "condition_id", "metric_name", "metric_direction"):
        metrics[field] = _text(metrics, field, "condition metrics")
    for field in ("dataset_id", "split_id", "condition_id", "primary_arm"):
        regimes[field] = _text(regimes, field, "regime manifest")
    if set(metrics["run_id"]) != set(map(str, policy["registered_run_ids"])):
        raise BenchmarkCompositionError("registered run identity differs from policy")
    if metrics["metric_status"].astype(str).ne("estimated").any():
        raise BenchmarkCompositionError("all frozen metric rows must be estimated")
    numeric = pd.to_numeric(metrics["metric_value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise BenchmarkCompositionError("metric values must be finite")
    metrics["metric_value"] = numeric
    if metrics.duplicated(["run_id", "condition_id", "metric_name"]).any():
        raise BenchmarkCompositionError("condition metrics contain duplicate run-condition-metric rows")
    directions = metrics.groupby("metric_name", observed=True)["metric_direction"].nunique()
    if directions.ne(1).any() or not set(metrics["metric_direction"]).issubset({"lower_is_better", "higher_is_better"}):
        raise BenchmarkCompositionError("each metric requires one supported direction")
    expected_metrics = {str(policy["primary_metric"]), str(policy["secondary_metric"])}
    if set(metrics["metric_name"]) != expected_metrics:
        raise BenchmarkCompositionError("metric set differs from policy")
    if regimes.duplicated(["dataset_id", "split_id", "condition_id"]).any():
        raise BenchmarkCompositionError("regime manifest contains duplicate conditions")
    for field in ("additivity_violation_score", "effect_size_covariate"):
        regimes[field] = pd.to_numeric(regimes[field], errors="coerce")
        if not np.isfinite(regimes[field].to_numpy(dtype=float)).all():
            raise BenchmarkCompositionError(f"regime {field} must be finite")
    for field in ("high_additivity_violation", "high_effect"):
        values = regimes[field]
        if values.dtype != bool:
            normalized = values.astype(str).str.lower().map({"true": True, "false": False})
            if normalized.isna().any():
                raise BenchmarkCompositionError(f"regime {field} must be boolean")
            regimes[field] = normalized.astype(bool)
    if not set(regimes["primary_arm"]).issubset(set(JOINT_ARMS)):
        raise BenchmarkCompositionError("regime manifest contains unsupported joint arms")
    if metrics["dataset_id"].nunique() != 1 or metrics["split_id"].nunique() != 1:
        raise BenchmarkCompositionError("condition metrics require one dataset and split identity")
    if regimes["dataset_id"].nunique() != 1 or regimes["split_id"].nunique() != 1:
        raise BenchmarkCompositionError("regime manifest requires one dataset and split identity")
    if set(metrics["condition_id"]) != set(regimes["condition_id"]):
        raise BenchmarkCompositionError("metric and regime condition universes differ")
    counts = metrics.groupby(["run_id", "metric_name"], observed=True)["condition_id"].nunique()
    if counts.nunique() != 1 or int(counts.iloc[0]) != len(regimes):
        raise BenchmarkCompositionError("every registered run and metric must cover the complete condition universe")
    return metrics, regimes


def collapse_model_metrics(metrics: pd.DataFrame, policy: Mapping[str, Any]) -> pd.DataFrame:
    group = list(policy["model_level_collapse"]["group_by"])
    families = metrics.groupby(group, observed=True)["family"].nunique()
    if families.gt(1).any():
        raise BenchmarkCompositionError("one model-level identity maps to multiple families")
    collapsed = metrics.groupby(group, sort=True, observed=True).agg(
        family=("family", "first"),
        metric_value=("metric_value", "mean"),
        n_registered_runs_collapsed=("run_id", "nunique"),
        seed_ids=("seed", lambda values: ",".join(map(str, sorted(set(map(int, values)))))),
        run_ids=("run_id", lambda values: ",".join(sorted(set(map(str, values))))),
    ).reset_index()
    collapsed = collapsed.sort_values(group, kind="stable").reset_index(drop=True)
    return collapsed


def _oriented(values: np.ndarray, direction: str) -> np.ndarray:
    return values if direction == "lower_is_better" else -values


def _rank_with_ties(values: Sequence[float], tolerance: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    group_start = 0
    while group_start < values.size:
        group_end = group_start + 1
        base = values[order[group_start]]
        while group_end < values.size and abs(values[order[group_end]] - base) <= tolerance:
            group_end += 1
        rank_value = (group_start + 1 + group_end) / 2.0
        ranks[order[group_start:group_end]] = rank_value
        group_start = group_end
    return ranks


def _winner_set(values: np.ndarray, models: Sequence[str], tolerance: float) -> tuple[list[str], np.ndarray]:
    best = float(np.min(values))
    mask = np.abs(values - best) <= tolerance
    return [str(model) for model, tied in zip(models, mask, strict=True) if tied], mask


def _grid(policy: Mapping[str, Any]) -> np.ndarray:
    spec = policy["composition_grid"]
    count = int(round((float(spec["stop"]) - float(spec["start"])) / float(spec["step"]))) + 1
    grid = float(spec["start"]) + np.arange(count, dtype=float) * float(spec["step"])
    grid[-1] = float(spec["stop"])
    return grid


def _arm_matrices(
    joined: pd.DataFrame, metric_name: str
) -> tuple[list[str], str, np.ndarray, np.ndarray, list[str], list[str]]:
    frame = joined.loc[joined["metric_name"].eq(metric_name)].copy()
    direction_values = frame["metric_direction"].unique()
    if len(direction_values) != 1:
        raise BenchmarkCompositionError("metric direction is not unique")
    models = sorted(frame["model_id"].unique())
    matrices: list[np.ndarray] = []
    conditions: list[list[str]] = []
    for arm in (PRIMARY_ARM, SECONDARY_ARM):
        subset = frame.loc[frame["primary_arm"].eq(arm)]
        matrix = subset.pivot(index="condition_id", columns="model_id", values="metric_value")
        matrix = matrix.reindex(columns=models).sort_index()
        if matrix.isna().any().any() or len(matrix) < 2:
            raise BenchmarkCompositionError(f"primary arm {arm} lacks a complete model matrix")
        matrices.append(matrix.to_numpy(dtype=float))
        conditions.append(list(map(str, matrix.index)))
    return models, str(direction_values[0]), matrices[0], matrices[1], conditions[0], conditions[1]


def _composition_tables(
    joined: pd.DataFrame, metrics: pd.DataFrame, policy: Mapping[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    grid = _grid(policy)
    tolerance = float(policy["tie_absolute_tolerance"])
    curve_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    crossover_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    metric_cache: dict[str, dict[str, Any]] = {}

    for metric_name in sorted(joined["metric_name"].unique()):
        models, direction, ha, he, ha_ids, he_ids = _arm_matrices(joined, metric_name)
        ha_mean = ha.mean(axis=0)
        he_mean = he.mean(axis=0)
        oriented_ha = _oriented(ha_mean, direction)
        oriented_he = _oriented(he_mean, direction)
        oriented_curve = grid[:, None] * oriented_ha + (1.0 - grid[:, None]) * oriented_he
        raw_curve = grid[:, None] * ha_mean + (1.0 - grid[:, None]) * he_mean
        aggregate = joined.loc[joined["metric_name"].eq(metric_name)].groupby(
            "model_id", observed=True
        )["metric_value"].mean().reindex(models).to_numpy(dtype=float)
        oriented_aggregate = _oriented(aggregate, direction)
        aggregate_winners, aggregate_mask = _winner_set(oriented_aggregate, models, tolerance)
        aggregate_ranks = _rank_with_ties(oriented_aggregate, tolerance)

        for grid_index, lam in enumerate(grid):
            winners, winner_mask = _winner_set(oriented_curve[grid_index], models, tolerance)
            ranks = _rank_with_ties(oriented_curve[grid_index], tolerance)
            winner_rows.append({
                "metric_name": metric_name, "metric_direction": direction,
                "composition_id": f"lambda_{grid_index:04d}", "lambda_high_av": lam,
                "weight_high_effect": 1.0 - lam, "winner_model_ids": ";".join(winners),
                "n_tied_winners": len(winners), "status": "estimated",
            })
            tau = kendalltau(aggregate_ranks, ranks, variant="b", nan_policy="raise").statistic
            ranking_rows.append({
                "metric_name": metric_name, "metric_direction": direction,
                "composition_id": f"lambda_{grid_index:04d}", "lambda_high_av": lam,
                "kendall_tau_b_vs_aggregate": float(tau) if np.isfinite(tau) else np.nan,
                "aggregate_winner_model_ids": ";".join(aggregate_winners),
                "conditional_winner_model_ids": ";".join(winners),
                "winner_set_concordant": set(winners) == set(aggregate_winners),
                "status": "estimated" if np.isfinite(tau) else "not_estimable",
                "not_estimable_reason": "" if np.isfinite(tau) else "kendall_tau_b_undefined",
            })
            best = float(np.min(oriented_curve[grid_index]))
            for model_index, model in enumerate(models):
                curve_rows.append({
                    "metric_name": metric_name, "metric_direction": direction,
                    "composition_id": f"lambda_{grid_index:04d}", "lambda_high_av": lam,
                    "weight_high_effect": 1.0 - lam, "model_id": model,
                    "metric_value": float(raw_curve[grid_index, model_index]),
                    "oriented_risk": float(oriented_curve[grid_index, model_index]),
                    "rank": float(ranks[model_index]), "is_winner": bool(winner_mask[model_index]),
                    "regret": float(oriented_curve[grid_index, model_index] - best),
                })
            for aggregate_winner in aggregate_winners:
                model_index = models.index(aggregate_winner)
                regret_rows.append({
                    "metric_name": metric_name, "metric_direction": direction,
                    "composition_id": f"lambda_{grid_index:04d}", "lambda_high_av": lam,
                    "aggregate_winner_model_id": aggregate_winner,
                    "conditional_winner_model_ids": ";".join(winners),
                    "selection_regret": float(oriented_curve[grid_index, model_index] - best),
                    "aggregate_winner_is_optimal": bool(winner_mask[model_index]),
                })

        pair_point: dict[tuple[str, str], tuple[float, str]] = {}
        for left, model_a in enumerate(models):
            for right in range(left + 1, len(models)):
                model_b = models[right]
                d_he = float(oriented_he[left] - oriented_he[right])
                d_ha = float(oriented_ha[left] - oriented_ha[right])
                slope = d_ha - d_he
                if abs(d_he) <= tolerance and abs(d_ha) <= tolerance:
                    crossover, status = math.nan, "coincident_all_compositions"
                elif abs(slope) <= tolerance:
                    crossover, status = math.nan, "parallel_no_crossover"
                else:
                    candidate = -d_he / slope
                    if -tolerance <= candidate <= 1.0 + tolerance:
                        crossover, status = float(min(1.0, max(0.0, candidate))), "interior_or_endpoint_crossover"
                    else:
                        crossover, status = math.nan, "no_crossover_in_unit_interval"
                pair_point[(model_a, model_b)] = (crossover, status)
                crossover_rows.append({
                    "metric_name": metric_name, "metric_direction": direction,
                    "model_a": model_a, "model_b": model_b,
                    "oriented_difference_at_high_effect": d_he,
                    "oriented_difference_at_high_av": d_ha, "slope": slope,
                    "crossover_lambda": crossover, "point_status": status,
                    "resampling_draws": int(policy["resampling"]["draws"]),
                    "resampling_valid_fraction": np.nan,
                    "crossover_stability_low": np.nan, "crossover_stability_high": np.nan,
                    "interval_conditioning": "finite_crossover_draws_only_no_redraw",
                })

        # Fixed-benchmark condition resampling.  Draws are shared across models.
        resampling = policy["resampling"]
        draws = int(resampling["draws"])
        rng = np.random.Generator(np.random.PCG64(int(resampling["seed"])))
        ha_boot = np.empty((draws, len(models)), dtype=float)
        he_boot = np.empty_like(ha_boot)
        chunk = 500
        for start in range(0, draws, chunk):
            stop = min(draws, start + chunk)
            ha_index = rng.integers(0, len(ha), size=(stop - start, len(ha)))
            he_index = rng.integers(0, len(he), size=(stop - start, len(he)))
            ha_boot[start:stop] = ha[ha_index].mean(axis=1)
            he_boot[start:stop] = he[he_index].mean(axis=1)
        oriented_ha_boot = _oriented(ha_boot, direction)
        oriented_he_boot = _oriented(he_boot, direction)
        q_low, q_high = map(float, resampling["quantiles"])
        for model_index, model in enumerate(models):
            boot_curve = (
                grid[None, :] * oriented_ha_boot[:, model_index, None]
                + (1.0 - grid[None, :]) * oriented_he_boot[:, model_index, None]
            )
            lows, highs = np.quantile(boot_curve, [q_low, q_high], axis=0, method="linear")
            for grid_index, lam in enumerate(grid):
                interval_rows.append({
                    "quantity": "model_oriented_risk", "metric_name": metric_name,
                    "metric_direction": direction, "model_id": model,
                    "aggregate_winner_model_id": "", "composition_id": f"lambda_{grid_index:04d}",
                    "lambda_high_av": lam, "estimate": float(oriented_curve[grid_index, model_index]),
                    "stability_interval_low": float(lows[grid_index]),
                    "stability_interval_high": float(highs[grid_index]),
                    "draws": draws, "interval_method": "finite_benchmark_condition_resampling_percentile_not_population_ci",
                })
        # Point-estimate aggregate winners are kept fixed for this stability question.
        for aggregate_winner in aggregate_winners:
            winner_index = models.index(aggregate_winner)
            regret_boot = np.empty((draws, len(grid)), dtype=float)
            for start in range(0, draws, chunk):
                stop = min(draws, start + chunk)
                risk = (
                    grid[None, :, None] * oriented_ha_boot[start:stop, None, :]
                    + (1.0 - grid[None, :, None]) * oriented_he_boot[start:stop, None, :]
                )
                regret_boot[start:stop] = risk[:, :, winner_index] - risk.min(axis=2)
            lows, highs = np.quantile(regret_boot, [q_low, q_high], axis=0, method="linear")
            point = np.asarray([
                row["selection_regret"] for row in regret_rows
                if row["metric_name"] == metric_name and row["aggregate_winner_model_id"] == aggregate_winner
            ])
            for grid_index, lam in enumerate(grid):
                interval_rows.append({
                    "quantity": "aggregate_winner_selection_regret", "metric_name": metric_name,
                    "metric_direction": direction, "model_id": "",
                    "aggregate_winner_model_id": aggregate_winner,
                    "composition_id": f"lambda_{grid_index:04d}", "lambda_high_av": lam,
                    "estimate": float(point[grid_index]),
                    "stability_interval_low": float(lows[grid_index]),
                    "stability_interval_high": float(highs[grid_index]), "draws": draws,
                    "interval_method": "finite_benchmark_condition_resampling_percentile_not_population_ci",
                })
        # Conditional crossover intervals, retaining the non-crossing fraction.
        for row in crossover_rows:
            if row["metric_name"] != metric_name:
                continue
            left = models.index(str(row["model_a"])); right = models.index(str(row["model_b"]))
            d_he_boot = oriented_he_boot[:, left] - oriented_he_boot[:, right]
            d_ha_boot = oriented_ha_boot[:, left] - oriented_ha_boot[:, right]
            denominator = d_ha_boot - d_he_boot
            finite_denominator = np.abs(denominator) > tolerance
            candidate = np.full(draws, np.nan)
            candidate[finite_denominator] = -d_he_boot[finite_denominator] / denominator[finite_denominator]
            valid = np.isfinite(candidate) & (candidate >= 0.0) & (candidate <= 1.0)
            row["resampling_valid_fraction"] = float(valid.mean())
            if valid.any():
                row["crossover_stability_low"], row["crossover_stability_high"] = map(
                    float, np.quantile(candidate[valid], [q_low, q_high], method="linear")
                )
        metric_cache[metric_name] = {
            "models": models, "direction": direction, "curve": oriented_curve,
            "ranks": np.stack([_rank_with_ties(row, tolerance) for row in oriented_curve]),
            "aggregate_winners": aggregate_winners,
        }

    # Native-direction metric sensitivity, represented through oriented ranks.
    metric_names = sorted(metric_cache)
    sensitivity_rows: list[dict[str, Any]] = []
    if len(metric_names) == 2:
        first, second = metric_names
        if metric_cache[first]["models"] != metric_cache[second]["models"]:
            raise BenchmarkCompositionError("metric sensitivity requires the same model set")
        for index, lam in enumerate(grid):
            first_ranks = metric_cache[first]["ranks"][index]
            second_ranks = metric_cache[second]["ranks"][index]
            tau = kendalltau(first_ranks, second_ranks, variant="b", nan_policy="raise").statistic
            first_winners, _ = _winner_set(metric_cache[first]["curve"][index], metric_cache[first]["models"], tolerance)
            second_winners, _ = _winner_set(metric_cache[second]["curve"][index], metric_cache[second]["models"], tolerance)
            sensitivity_rows.append({
                "composition_id": f"lambda_{index:04d}", "lambda_high_av": lam,
                "metric_a": first, "metric_b": second,
                "kendall_tau_b_between_metric_orderings": float(tau) if np.isfinite(tau) else np.nan,
                "metric_a_winner_model_ids": ";".join(first_winners),
                "metric_b_winner_model_ids": ";".join(second_winners),
                "winner_set_concordant": set(first_winners) == set(second_winners),
                "status": "estimated" if np.isfinite(tau) else "not_estimable",
            })

    tables = {
        "composition_risk_curves.tsv": pd.DataFrame(curve_rows),
        "composition_winners.tsv": pd.DataFrame(winner_rows),
        "pairwise_crossovers.tsv": pd.DataFrame(crossover_rows),
        "ranking_stability.tsv": pd.DataFrame(ranking_rows),
        "selection_regret.tsv": pd.DataFrame(regret_rows),
        "composition_resampling_intervals.tsv": pd.DataFrame(interval_rows),
        "metric_sensitivity.tsv": pd.DataFrame(sensitivity_rows),
    }
    metadata = {"grid_points": len(grid), "metric_names": sorted(metric_cache)}
    return tables, metadata


def _focal_seed_sensitivity(
    metrics: pd.DataFrame, regimes: pd.DataFrame, policy: Mapping[str, Any]
) -> pd.DataFrame:
    focal = metrics.loc[metrics["run_id"].isin(set(policy["focal_seed_run_ids"]))].merge(
        regimes[["condition_id", "primary_arm"]],
        on="condition_id", how="left", validate="many_to_one",
    )
    grid = _grid(policy)
    rows: list[dict[str, Any]] = []
    identity = ["run_id", "model_id", "family", "seed", "metric_name", "metric_direction"]
    for key, group in focal.groupby(identity, sort=True, observed=True):
        base = dict(zip(identity, key, strict=True))
        ha = group.loc[group["primary_arm"].eq(PRIMARY_ARM), "metric_value"].mean()
        he = group.loc[group["primary_arm"].eq(SECONDARY_ARM), "metric_value"].mean()
        for index, lam in enumerate(grid):
            rows.append({
                **base, "composition_id": f"lambda_{index:04d}", "lambda_high_av": lam,
                "metric_value": float(lam * ha + (1.0 - lam) * he),
            })
    return pd.DataFrame(rows)


def _loo_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    predictions = np.empty(len(y), dtype=float)
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        coefficients = np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]
        predictions[index] = float(x[index] @ coefficients)
    residual = predictions - y
    return float(np.mean(np.abs(residual))), float(np.mean(residual**2))


def _interaction_dispersion(matrix: np.ndarray) -> float:
    """RMS of the double-centred model-by-stratum oriented-risk matrix."""
    centred = matrix - matrix.mean(axis=0, keepdims=True) - matrix.mean(axis=1, keepdims=True) + matrix.mean()
    return float(np.sqrt(np.mean(centred**2)))


def _hard_partition_summary(
    frame: pd.DataFrame,
    labels: pd.Series,
    partition_id: str,
    metric_name: str,
    direction: str,
    tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label_map = labels.astype(str)
    if label_map.index.duplicated().any():
        raise BenchmarkCompositionError("hard-partition labels contain duplicate conditions")
    work = frame.copy()
    work["stratum"] = work["condition_id"].map(label_map)
    if work["stratum"].isna().any():
        raise BenchmarkCompositionError("hard-partition labels do not cover all conditions")
    models = sorted(work["model_id"].unique())
    aggregate = work.groupby("model_id", observed=True)["metric_value"].mean().reindex(models).to_numpy(float)
    aggregate_oriented = _oriented(aggregate, direction)
    aggregate_winners, _ = _winner_set(aggregate_oriented, models, tolerance)
    aggregate_ranks = _rank_with_ties(aggregate_oriented, tolerance)
    matrix = work.groupby(["stratum", "model_id"], observed=True)["metric_value"].mean().unstack().reindex(columns=models)
    oriented_matrix = _oriented(matrix.to_numpy(float), direction)
    strata_rows: list[dict[str, Any]] = []
    all_winners: set[str] = set()
    taus: list[float] = []
    regrets: list[float] = []
    for stratum_index, stratum in enumerate(matrix.index.astype(str)):
        values = oriented_matrix[stratum_index]
        winners, mask = _winner_set(values, models, tolerance)
        all_winners.update(winners)
        ranks = _rank_with_ties(values, tolerance)
        tau = kendalltau(aggregate_ranks, ranks, variant="b", nan_policy="raise").statistic
        if np.isfinite(tau):
            taus.append(float(tau))
        best = float(values.min())
        for model_index, model in enumerate(models):
            strata_rows.append({
                "analysis_type": "hard_partition", "partition_id": partition_id,
                "metric_name": metric_name, "metric_direction": direction,
                "stratum": stratum, "model_id": model,
                "n_conditions": int(work.loc[work["stratum"].eq(stratum), "condition_id"].nunique()),
                "mean_metric": float(matrix.iloc[stratum_index, model_index]),
                "oriented_risk": float(values[model_index]), "is_winner": bool(mask[model_index]),
                "aggregate_winner_regret": float(values[models.index(aggregate_winners[0])] - best),
            })
        regrets.append(max(float(values[models.index(winner)] - best) for winner in aggregate_winners))
    summary = {
        "analysis_type": "hard_partition", "partition_id": partition_id,
        "metric_name": metric_name, "metric_direction": direction,
        "model_a": "", "model_b": "", "continuous_specification": "",
        "n_conditions": int(work["condition_id"].nunique()), "n_strata": len(matrix),
        "unique_winner_count": len(all_winners),
        "minimum_kendall_tau_b_vs_aggregate": min(taus) if taus else np.nan,
        "maximum_aggregate_winner_regret": max(regrets),
        "interaction_dispersion": _interaction_dispersion(oriented_matrix),
        "loo_mae": np.nan, "loo_mse": np.nan, "delta_loo_mae_vs_intercept": np.nan,
        "delta_loo_mae_vs_av_only": np.nan, "permutation_draws": 0,
        "permutation_interval_low": np.nan, "permutation_interval_high": np.nan,
        "permutation_exceedance_fraction": np.nan, "status": "estimated",
    }
    return summary, strata_rows


def _geometry_ablations(
    joined: pd.DataFrame, policy: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tolerance = float(policy["tie_absolute_tolerance"])
    summaries: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    for metric_name in sorted(joined["metric_name"].unique()):
        frame = joined.loc[joined["metric_name"].eq(metric_name)].copy()
        direction = str(frame["metric_direction"].iloc[0])
        models = sorted(frame["model_id"].unique())
        conditions = sorted(frame["condition_id"].unique())
        covariates = frame.drop_duplicates("condition_id").set_index("condition_id").reindex(conditions)
        av = (rankdata(covariates["additivity_violation_score"], method="average") - 1) / (len(conditions) - 1) - 0.5
        effect = (rankdata(covariates["effect_size_covariate"], method="average") - 1) / (len(conditions) - 1) - 0.5
        designs = {
            "continuous_intercept": np.ones((len(conditions), 1)),
            "continuous_effect_rank": np.column_stack([np.ones(len(conditions)), effect]),
            "continuous_additivity_violation_rank": np.column_stack([np.ones(len(conditions)), av]),
            "continuous_additive_axes": np.column_stack([np.ones(len(conditions)), effect, av]),
            "continuous_axis_interaction": np.column_stack([np.ones(len(conditions)), effect, av, effect * av]),
        }
        matrix = frame.pivot(index="condition_id", columns="model_id", values="metric_value").reindex(index=conditions, columns=models)
        oriented = _oriented(matrix.to_numpy(float), direction)
        for left, model_a in enumerate(models):
            for right in range(left + 1, len(models)):
                model_b = models[right]
                y = oriented[:, left] - oriented[:, right]
                results = {name: _loo_linear(x, y) for name, x in designs.items()}
                for name, (mae, mse) in results.items():
                    summaries.append({
                        "analysis_type": "continuous_pairwise", "partition_id": "",
                        "metric_name": metric_name, "metric_direction": direction,
                        "model_a": model_a, "model_b": model_b,
                        "continuous_specification": name, "n_conditions": len(conditions),
                        "n_strata": 0, "unique_winner_count": np.nan,
                        "minimum_kendall_tau_b_vs_aggregate": np.nan,
                        "maximum_aggregate_winner_regret": np.nan,
                        "interaction_dispersion": np.nan, "loo_mae": mae, "loo_mse": mse,
                        "delta_loo_mae_vs_intercept": mae - results["continuous_intercept"][0],
                        "delta_loo_mae_vs_av_only": mae - results["continuous_additivity_violation_rank"][0],
                        "permutation_draws": 0, "permutation_interval_low": np.nan,
                        "permutation_interval_high": np.nan,
                        "permutation_exceedance_fraction": np.nan, "status": "estimated",
                    })
        condition_covariates = covariates.reset_index()
        condition_covariates = condition_covariates.set_index("condition_id", drop=False)
        partitions = {
            "magnitude_only_binary": condition_covariates["high_effect"].map({True: "high_effect", False: "not_high_effect"}),
            "additivity_violation_only_binary": condition_covariates["high_additivity_violation"].map({True: "high_av", False: "not_high_av"}),
            "frozen_two_axis_joint": condition_covariates["primary_arm"],
        }
        observed_joint: dict[str, Any] | None = None
        for partition_id, labels in partitions.items():
            summary, rows = _hard_partition_summary(
                frame, labels, partition_id, metric_name, direction, tolerance
            )
            summaries.append(summary); strata_rows.extend(rows)
            if partition_id == "frozen_two_axis_joint":
                observed_joint = summary
        assert observed_joint is not None
        permutation = policy["permutation"]
        draws = int(permutation["draws"])
        rng = np.random.Generator(np.random.PCG64(int(permutation["seed"])))
        base_labels = condition_covariates["primary_arm"].astype(str).to_numpy()
        dispersions = np.empty(draws, dtype=float)
        for draw in range(draws):
            shuffled = rng.permutation(base_labels)
            permuted = frame.merge(
                pd.DataFrame({"condition_id": conditions, "permuted": shuffled}),
                on="condition_id", how="left", validate="many_to_one",
            )
            perm_matrix = permuted.groupby(["permuted", "model_id"], observed=True)["metric_value"].mean().unstack().reindex(columns=models)
            dispersions[draw] = _interaction_dispersion(_oriented(perm_matrix.to_numpy(float), direction))
        low, high = np.quantile(
            dispersions, policy["resampling"]["quantiles"], method="linear"
        )
        summaries.append({
            "analysis_type": "permutation_reference", "partition_id": "permuted_joint_labels",
            "metric_name": metric_name, "metric_direction": direction, "model_a": "", "model_b": "",
            "continuous_specification": "", "n_conditions": len(conditions),
            "n_strata": len(set(base_labels)), "unique_winner_count": np.nan,
            "minimum_kendall_tau_b_vs_aggregate": np.nan,
            "maximum_aggregate_winner_regret": np.nan,
            "interaction_dispersion": float(dispersions.mean()), "loo_mae": np.nan, "loo_mse": np.nan,
            "delta_loo_mae_vs_intercept": np.nan, "delta_loo_mae_vs_av_only": np.nan,
            "permutation_draws": draws, "permutation_interval_low": float(low),
            "permutation_interval_high": float(high),
            "permutation_exceedance_fraction": float(
                np.mean(dispersions >= float(observed_joint["interaction_dispersion"]))
            ), "status": "estimated_descriptive_reference",
        })
    return pd.DataFrame(summaries), pd.DataFrame(strata_rows)


def build_analysis(
    condition_metrics: pd.DataFrame,
    regime_manifest: pd.DataFrame,
    policy: Mapping[str, Any],
    *,
    producer_commit: str,
    input_sha256: Mapping[str, str],
) -> BenchmarkCompositionArtifacts:
    if len(producer_commit) != 40 or any(c not in "0123456789abcdef" for c in producer_commit):
        raise BenchmarkCompositionError("producer_commit must be a full lowercase Git SHA")
    metrics, regimes = _validate_inputs(condition_metrics, regime_manifest, policy)
    collapsed = collapse_model_metrics(metrics, policy)
    regime_annotations = regimes.drop(columns=["dataset_id", "split_id"])
    joined = collapsed.merge(
        regime_annotations, on="condition_id", how="left", validate="many_to_one"
    )
    composition_tables, metadata = _composition_tables(joined, metrics, policy)
    geometry_summary, geometry_strata = _geometry_ablations(joined, policy)
    tables: dict[str, pd.DataFrame] = {
        "model_condition_metrics_collapsed.tsv": collapsed,
        **composition_tables,
        "focal_seed_sensitivity.tsv": _focal_seed_sensitivity(metrics, regimes, policy),
        "geometry_ablation_summary.tsv": geometry_summary,
        "geometry_ablation_model_strata.tsv": geometry_strata,
    }
    if set(tables) != set(OUTPUT_NAMES):
        raise BenchmarkCompositionError("expected output table contract is incomplete")
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "CALCULATED_POST_RESULT_SECONDARY_ANALYSIS",
        "producer_commit": producer_commit,
        "input_sha256": dict(sorted(map(lambda item: (str(item[0]), str(item[1])), input_sha256.items()))),
        "policy_schema_version": policy["schema_version"],
        "n_conditions": int(regimes["condition_id"].nunique()),
        "n_models": int(collapsed["model_id"].nunique()),
        "n_registered_runs": int(metrics["run_id"].nunique()),
        "metric_names": sorted(collapsed["metric_name"].unique()),
        "composition_grid_points": metadata["grid_points"],
        "formulas": {
            "composition_risk": "R_m(lambda)=lambda*mean_HA(m)+(1-lambda)*mean_HE(m)",
            "selection_regret_lower": "R_aggregate_winner(lambda)-min_m R_m(lambda)",
            "selection_regret_higher": "max_m R_m(lambda)-R_aggregate_winner(lambda)",
            "interaction_dispersion": "RMS of double-centred model-by-stratum oriented-risk matrix",
            "continuous_covariates": "midrank/(n_conditions-1)-0.5",
        },
        "resampling": policy["resampling"],
        "permutation": policy["permutation"],
        "claim_boundaries": policy["claim_boundaries"],
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "population_confidence_interval": False,
        "p_values_calculated": False,
        "models_retrained": False,
    }
    return BenchmarkCompositionArtifacts(tables=tables, record=record)


def build_from_policy(
    repository_root: Path,
    policy_path: Path,
    *,
    producer_commit: str,
) -> BenchmarkCompositionArtifacts:
    root = Path(repository_root).resolve()
    policy_path = Path(policy_path).resolve()
    policy = load_policy(policy_path)
    paths = {name: root / record["path"] for name, record in policy["inputs"].items()}
    for name, path in paths.items():
        if not path.is_file():
            raise BenchmarkCompositionError(f"frozen input is missing: {name}")
        observed = sha256_file(path)
        if observed != policy["inputs"][name]["sha256"]:
            raise BenchmarkCompositionError(f"frozen input hash mismatch: {name}")
    metrics = pd.read_csv(paths["condition_metrics"], sep="\t", keep_default_na=False)
    regimes = pd.read_csv(paths["regime_manifest"])
    hashes = {name: policy["inputs"][name]["sha256"] for name in sorted(paths)}
    hashes["policy"] = sha256_file(policy_path)
    return build_analysis(
        metrics, regimes, policy, producer_commit=producer_commit, input_sha256=hashes
    )


def _write_tsv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, sep="\t", index=False, lineterminator="\n", float_format="%.17g")


def write_analysis(output_dir: Path, artifacts: BenchmarkCompositionArtifacts) -> Path:
    destination = Path(output_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name in OUTPUT_NAMES:
            _write_tsv(temporary / name, artifacts.tables[name])
        output_hashes = {name: sha256_file(temporary / name) for name in OUTPUT_NAMES}
        record = {**artifacts.record, "output_sha256": output_hashes}
        (temporary / "analysis_record.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = [
    "BenchmarkCompositionArtifacts", "BenchmarkCompositionError", "OUTPUT_NAMES",
    "build_analysis", "build_from_policy", "collapse_model_metrics", "load_policy",
    "sha256_file", "write_analysis",
]
