# Results available without recomputation

The `results/` directory is a tracked, checksum-protected snapshot of the
complete benchmark-composition replay. It lets readers inspect every reported
secondary-analysis output without installing Python or repeating model
training.

## Headline results

The full 46-condition benchmark selected `pb_latent_additive` under both
reported metrics. The composition analysis then focused on 28 conditions: 14
high-additivity-violation-only responses and 14 high-effect-only responses.

| Metric | Analytical LatentAdditive--CPA crossover | First switch on the 0.001 grid | Maximum selection regret | Interior pairwise crossovers |
| --- | ---: | ---: | ---: | ---: |
| Relative-L2 response error | 0.819033 | 0.820 | 0.017440 | 6 of 15 model pairs |
| Gene-wise Spearman correlation | 0.703902 | 0.704 | 0.007924 | 1 of 15 model pairs |

Here the mixture coordinate is the weight assigned to the
high-additivity-violation-only arm. These are point estimates on a finite,
frozen benchmark. Resampling intervals describe sensitivity to condition
composition within that benchmark; they are not population confidence
intervals. No p-values were calculated.

The machine-readable version of these quantities is
[`results/headline_summary.tsv`](results/headline_summary.tsv). Results for all
six models at five representative mixture weights are in
[`results/composition_anchor_points.tsv`](results/composition_anchor_points.tsv).

## Complete output tables

The following files are retained exactly as produced by the verified replay:

| File | Contents |
| --- | --- |
| `composition_risk_curves.tsv` | Model-level conditional risk, rank, winner status and regret at all 1,001 mixture weights for both metrics. |
| `composition_winners.tsv` | Minimum-risk model set at each mixture weight. |
| `pairwise_crossovers.tsv` | Analytical crossover locations and finite-benchmark resampling stability for all 15 model pairs. |
| `selection_regret.tsv` | Regret from retaining the model selected on the full 46-condition test set. |
| `ranking_stability.tsv` | Kendall rank agreement and winner-set concordance against the full-test ordering. |
| `composition_resampling_intervals.tsv` | Condition-resampling intervals for model risks and selection regret. |
| `metric_sensitivity.tsv` | Agreement between relative-L2 and gene-wise Spearman orderings. |
| `focal_seed_sensitivity.tsv` | Seed-resolved trajectories for the two multi-seed focal models. |
| `geometry_ablation_model_strata.tsv` | Model-by-stratum results for the hard-partition coordinate ablations. |
| `geometry_ablation_summary.tsv` | Continuous-coordinate, hard-partition and permutation-reference summaries. |
| `model_condition_metrics_collapsed.tsv` | Six-model, 46-condition metric table after registered seed collapse. |
| `analysis_record.json` | Analysis formulas, input identities, software environment and claim boundaries. |

All paths above are under [`results/composition/`](results/composition/).
[`results/RESULT_CHECKSUMS.tsv`](results/RESULT_CHECKSUMS.tsv) records the size
and SHA-256 of every tracked result artifact. The source condition-level inputs
remain under `data/norman/`; the repository does not contain raw expression
matrices, checkpoints or model implementations.

## How the snapshot is maintained

`scripts/build_public_result_snapshot.py` accepts only a replay whose 11
scientific tables match the frozen output ledger. It copies those tables,
retains a semantically validated analysis record, derives the two compact
summary tables and regenerates the result checksum ledger.

`scripts/verify_public_results.py` independently checks the tracked result
tree, re-derives both compact tables and can compare a new replay against the
tracked snapshot. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the
corresponding commands and evidence tiers.
