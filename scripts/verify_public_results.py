#!/usr/bin/env python3
"""Verify the tracked result snapshot and an optional fresh replay."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_public_result_snapshot import (  # noqa: E402
    ANALYSIS_RECORD,
    SCIENTIFIC_ARTIFACTS,
    SNAPSHOT_ARTIFACTS,
    build_anchor_points,
    build_headline_summary,
    expected_scientific_hashes,
    sha256,
    validate_analysis_record,
)


def _read_ledger(path: Path) -> dict[str, tuple[int, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
        records = {
            row["artifact"]: (int(row["size_bytes"]), row["sha256"])
            for row in rows
        }
    if len(records) != len(rows):
        raise ValueError("result checksum ledger contains duplicate artifact names")
    return records


def _same_table(observed: Path, expected: pd.DataFrame) -> bool:
    buffer = io.StringIO()
    expected.to_csv(
        buffer,
        sep="\t",
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return observed.read_text(encoding="utf-8") == buffer.getvalue()


def verify(
    repository_root: Path,
    results_dir: Path,
    replay_output_dir: Path | None,
) -> list[str]:
    errors: list[str] = []
    composition = results_dir / "composition"
    ledger_path = results_dir / "RESULT_CHECKSUMS.tsv"
    if not ledger_path.is_file():
        return ["results/RESULT_CHECKSUMS.tsv is missing"]

    ledger = _read_ledger(ledger_path)
    observed_files = {
        path.relative_to(results_dir).as_posix(): path
        for path in results_dir.rglob("*")
        if path.is_file() and path != ledger_path
    }
    expected_files = {
        *(f"composition/{artifact}" for artifact in SNAPSHOT_ARTIFACTS),
        "headline_summary.tsv",
        "composition_anchor_points.tsv",
    }
    if set(observed_files) != expected_files:
        errors.append("result tree differs from the closed public allowlist")
    if set(ledger) != set(observed_files):
        errors.append("result checksum ledger and result tree name different artifacts")
    for artifact, path in observed_files.items():
        wanted = ledger.get(artifact)
        if wanted and wanted != (path.stat().st_size, sha256(path)):
            errors.append(f"checksum or size mismatch: results/{artifact}")

    expected_hashes = expected_scientific_hashes(repository_root)
    for artifact in SCIENTIFIC_ARTIFACTS:
        path = composition / artifact
        if not path.is_file() or sha256(path) != expected_hashes[artifact]:
            errors.append(f"tracked scientific table differs from frozen output: {artifact}")
    try:
        validate_analysis_record(composition / ANALYSIS_RECORD)
    except ValueError as exc:
        errors.append(str(exc))

    headline = build_headline_summary(
        composition,
        repository_root / "data/norman/regime_manifest.csv",
    )
    anchors = build_anchor_points(composition)
    if not _same_table(results_dir / "headline_summary.tsv", headline):
        errors.append("headline_summary.tsv is not derivable from the tracked composition tables")
    if not _same_table(results_dir / "composition_anchor_points.tsv", anchors):
        errors.append("composition_anchor_points.tsv is not derivable from the tracked composition tables")

    if replay_output_dir is not None:
        try:
            validate_analysis_record(replay_output_dir / ANALYSIS_RECORD)
        except ValueError as exc:
            errors.append(f"fresh replay: {exc}")
        for artifact in SCIENTIFIC_ARTIFACTS:
            tracked = composition / artifact
            replayed = replay_output_dir / artifact
            if not replayed.is_file() or sha256(replayed) != sha256(tracked):
                errors.append(f"fresh replay differs from tracked result: {artifact}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--replay-output-dir", type=Path)
    args = parser.parse_args()
    try:
        errors = verify(args.repository_root, args.results_dir, args.replay_output_dir)
    except (OSError, ValueError, KeyError) as exc:
        errors = [str(exc)]
    if errors:
        print("PUBLIC_RESULTS_INVALID")
        for error in errors:
            print(f"- {error}")
        return 2
    print("PUBLIC_RESULTS_PASS")
    print(f"scientific_tables={len(SCIENTIFIC_ARTIFACTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
