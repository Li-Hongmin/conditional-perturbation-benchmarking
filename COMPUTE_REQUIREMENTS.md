# Compute requirements

## Public secondary-analysis replay

The public replay is CPU-only and does not load a single-cell expression
matrix. A local reference run on Apple silicon with Python 3.11.14 used the
following approximate resources:

| Task | Wall time | Peak resident memory |
| --- | ---: | ---: |
| 10,000-draw analysis and 10,000-draw permutation reference | 24 s | 566 MiB |
| Figure generation | 3 s | 200 MiB |
| Focused test suite | 4 s | 165 MiB |

The locked environment occupies approximately 235 MiB, and the complete
tracked result snapshot occupies approximately 8 MiB. A practical minimum for
the public replay is one CPU core, 1 GiB available RAM and 1 GiB free disk.
Allow up to two minutes on an ordinary laptop or a small continuous-integration
worker. Times are reference measurements, not performance guarantees.

## Upstream model training

The six-model, 10-run training panel is not executed by this repository and
does not share the resource envelope above. It requires the upstream Norman
data preparation, PerturBench model implementations and accelerator-dependent
training. No universal time or memory estimate is asserted here because the
repository does not distribute the checkpoints or a cross-platform training
benchmark.

The exact run identities and configuration provenance are provided in
`configs/training_run_matrix.tsv`. Readers who do not need to repeat training
can inspect every downstream condition-level metric and composition result in
`data/norman/` and `results/`.
