# Contributing

Bug reports and narrowly scoped reproducibility improvements are welcome
through GitHub issues or pull requests.

Contributions should preserve the scientific contract:

- do not modify frozen inputs, run identities or expected scientific checksums
  without documenting the provenance and scientific reason;
- keep population inference, biological mechanism and prospective utility
  outside the claims supported by this finite-benchmark replay;
- add or update tests for any change to the evaluator;
- run the full replay and `scripts/verify_public_release.py` before opening a
  pull request;
- do not commit raw single-cell matrices, model checkpoints, credentials,
  cloud locations, machine-specific paths or non-public editorial material.

By submitting a contribution, you agree that it may be distributed under the
repository's Apache-2.0 license.
