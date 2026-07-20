# Conditional perturbation benchmarking v0.1.1

This release makes the verified secondary-analysis evidence directly
inspectable without repeating model training.

## Added

- Complete tracked snapshot of all 11 scientific replay tables.
- Compact headline and five-weight composition-anchor tables for GitHub review.
- Tiered reproducibility guide from zero-compute inspection to full secondary
  replay and upstream training provenance.
- Measured CPU, memory, disk and wall-time guidance for the public replay.
- Ten-run training provenance matrix with immutable upstream commit,
  configuration, dataset and split identities.
- Closed result allowlist, checksum ledger and fresh-replay comparison tests.

## Verification

- 13 focused tests pass in the locked Python 3.11 environment.
- All 11 scientific tables reproduce byte for byte.
- The analysis record passes its semantic contract.
- The public PNG reproduces byte for byte.
- Public-boundary and 53-file release-manifest audits pass.

## Scope

This release does not retrain models or redistribute expression matrices,
third-party implementations, checkpoints or private infrastructure records.
The distributed results are finite-benchmark secondary analyses, not population
confidence intervals or a universal model ranking.
