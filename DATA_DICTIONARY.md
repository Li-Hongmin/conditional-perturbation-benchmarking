# Data dictionary

The distributed tables contain derived, condition-level benchmark quantities.
They contain no cells, expression matrices, checkpoints or personal data.

## Keys and joins

- A registered run is identified by `run_id` and maps to one `model_id` and
  `seed` in `configs/public_run_registry_v1.yaml`.
- A prediction metric row is identified by `run_id`, `condition_id` and
  `metric_name`.
- Regime annotations join to metric rows by `condition_id`. Dataset and split
  identifiers are retained to prevent accidental cross-task joins.

## `data/norman/condition_metrics.tsv`

| Column | Type | Definition |
| --- | --- | --- |
| `run_id` | string | Frozen run identity. Historical `NM-` prefixes are opaque identifiers. |
| `dataset_id` | string | `norman_2019_matched_combination_prediction`. |
| `model_id` | string | Predictor implementation identifier. |
| `family` | string | Operational predictor family. |
| `seed` | integer | Registered training seed. |
| `split_id` | string | Frozen 46-condition Norman test split. |
| `condition_id` | string | Double-perturbation condition, represented as `geneA+geneB`. |
| `metric_name` | enum | `relative_l2_response_error` or `gene_wise_spearman`. |
| `metric_value` | number | Condition-level metric on the frozen 1,931-gene feature space. Relative-L2 is unitless and lower is better; Spearman correlation is unitless and higher is better. |
| `metric_direction` | enum | `lower_is_better` or `higher_is_better`. |
| `n_features_evaluated` | integer | Number of common features used, expected to be 1,931. |
| `metric_status` | enum | `estimated` for an evaluable row; other values must be accompanied by a reason. |
| `not_evaluable_reason` | nullable string | Empty for estimated rows; machine-readable explanation otherwise. |

Relative-L2 error is the Euclidean norm of the prediction residual divided by
the Euclidean norm of the observed response vector, with the frozen evaluator's
documented numerical guard. Gene-wise Spearman is the correlation between the
predicted and observed condition-response vectors.

## `data/norman/regime_manifest.csv`

| Column | Type | Definition |
| --- | --- | --- |
| `dataset_id` | string | Source dataset identity. |
| `split_id` | string | Frozen PerturBench formal test split. |
| `condition_id` | string | Join key for a double-perturbation condition. |
| `primary_arm` | enum | `high_additivity_violation_only`, `high_effect_only`, `both` or `neither`. |
| `high_additivity_violation` | boolean | Whether normalized additivity violation is in the calibrated high group. |
| `high_effect` | boolean | Whether response magnitude is in the calibrated high group. |
| `additivity_violation_score` | number | Unitless normalized distance between the observed double response and the sum of its parent-single responses. |
| `effect_size_covariate` | number | Euclidean norm of the observed double-perturbation response vector in the frozen common-feature space. |

Regime annotations were calculated from observed responses after predictions
were fixed. They are evaluation labels, not model inputs or mechanistic claims.

## Missingness and eligibility

No distributed row is silently imputed. The replay fails if a required model,
condition, metric, direction or regime label is absent. Missing predictions are
not replaced and mixture weights are not renormalized over unsupported arms.
