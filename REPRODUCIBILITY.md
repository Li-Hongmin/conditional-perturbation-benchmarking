# Reproducibility guide

This repository separates inspection, secondary-analysis replay and upstream
model retraining. These levels answer different questions and have different
resource requirements.

## Level 0: inspect the frozen results

No software installation is required. Start with:

- [`RESULTS.md`](RESULTS.md) for the scientific summary and table map;
- `results/headline_summary.tsv` for the headline quantities;
- `results/composition_anchor_points.tsv` for all six models at five mixture
  weights;
- `results/composition/` for the complete 11-table replay and analysis record;
- `results/RESULT_CHECKSUMS.tsv` for file integrity.

This level establishes what the verified replay produced. It does not
independently recalculate the results.

## Level 1: verify the distributed snapshot

After installing the locked environment:

```bash
uv sync --frozen --all-groups
uv run python scripts/verify_public_results.py
uv run python scripts/verify_public_release.py
uv run python scripts/verify_release_manifest.py
```

The first command checks all result checksums, the 11 frozen scientific-output
hashes, the analysis-record semantics and the derivation of both compact result
tables.

## Level 2: recalculate the secondary analysis

This is the complete public, low-compute replay. It starts from the distributed
condition-level metrics, not from expression matrices or checkpoints.

```bash
uv run python scripts/run_benchmark_composition.py \
  --repository-root . \
  --output-dir replay_outputs
uv run python scripts/verify_reproduction.py \
  --output-dir replay_outputs
uv run python scripts/verify_public_results.py \
  --replay-output-dir replay_outputs
uv run python scripts/plot_benchmark_composition.py \
  --input-dir replay_outputs \
  --output-dir replay_figure
cmp replay_figure/benchmark_composition_landscape.png \
  docs/benchmark_composition_landscape.png
```

The 11 scientific tables must match byte for byte. `analysis_record.json`
contains the producing commit and platform, so its environment-bearing fields
are validated semantically rather than required to be identical across
machines.

To rebuild the tracked snapshot after an intentional scientific change:

```bash
uv run python scripts/build_public_result_snapshot.py \
  --replay-output-dir replay_outputs \
  --results-dir results
```

This maintenance command is not needed for ordinary verification.

## Level 3: model training and prediction provenance

The condition-level predictions were generated before this secondary replay.
`configs/training_run_matrix.tsv` identifies all 10 completed run identities,
model families, seeds, the exact PerturBench commit and experiment-config
hashes, the processed-matrix and split hashes, and the public output schema.
`UPSTREAM_PROVENANCE.md` provides the corresponding source locations.

This repository deliberately does not redistribute raw expression matrices,
third-party model implementations, trained checkpoints or cloud execution
records. Consequently, Level 3 is a provenance-supported best-effort rerun,
not a one-command or byte-deterministic reproduction. A training rerun also
depends on upstream data preparation, accelerator hardware and stochastic
software behavior. The public condition-level tables and exact expected
secondary outputs allow readers to inspect and replay the claim-bearing
composition analysis without bearing that training cost.

## Failure interpretation

- A Level 1 failure indicates a damaged or internally inconsistent release
  tree.
- A Level 2 scientific-table mismatch indicates a change in inputs,
  implementation, dependencies or numerical behavior and must not be silently
  accepted.
- A different training trajectory at Level 3 does not by itself invalidate the
  distributed secondary result; it must be interpreted against the documented
  upstream data, configuration, seed, metric and split identities.
