#!/usr/bin/env python3
"""Verify replay outputs against the frozen expected SHA-256 ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path("expected/EXPECTED_OUTPUT_CHECKSUMS.tsv"),
    )
    parser.add_argument(
        "--require-analysis-record-byte-match",
        action="store_true",
        help="Fail if environment-bearing analysis_record.json differs byte-for-byte.",
    )
    args = parser.parse_args()

    with args.expected.open(newline="", encoding="utf-8") as handle:
        expected = {row["artifact"]: row["sha256"] for row in csv.DictReader(handle, delimiter="\t")}

    rows = []
    scientific_ok = True
    record_ok = True
    for artifact, wanted in expected.items():
        path = args.output_dir / artifact
        observed = sha256(path) if path.is_file() else "MISSING"
        matches = observed == wanted
        rows.append({"artifact": artifact, "expected": wanted, "observed": observed, "matches": matches})
        if artifact == "analysis_record.json":
            record_ok = matches
        else:
            scientific_ok = scientific_ok and matches

    record_path = args.output_dir / "analysis_record.json"
    record_semantics_ok = False
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record_semantics_ok = (
            isinstance(record.get("producer_commit"), str)
            and len(record["producer_commit"]) == 40
            and record.get("n_conditions") == 46
            and record.get("n_models") == 6
            and record.get("composition_grid_points") == 1001
            and record.get("resampling", {}).get("draws") == 10000
            and record.get("permutation", {}).get("draws") == 10000
            and record.get("population_confidence_interval") is False
            and record.get("p_values_calculated") is False
        )

    report = {
        "scientific_tables_match": scientific_ok,
        "analysis_record_byte_match": record_ok,
        "analysis_record_semantics_match": record_semantics_ok,
        "artifacts": rows,
    }
    print(json.dumps(report, indent=2))
    ok = scientific_ok and record_semantics_ok
    if args.require_analysis_record_byte_match:
        ok = ok and record_ok
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
