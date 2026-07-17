#!/usr/bin/env python3
"""Write or verify the deterministic public-release file manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_FILE_MANIFEST.tsv"
ANCHOR = ROOT / "RELEASE_FILE_MANIFEST.sha256"
EXCLUDED_PARTS = {
    ".git", ".pytest_cache", ".venv", "__pycache__", "build", "dist",
    "replay_outputs", "replay_figure",
}
EXCLUDED_NAMES = {MANIFEST.name, ANCHOR.name, ".DS_Store"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows() -> list[dict[str, str | int]]:
    result: list[dict[str, str | int]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.name in EXCLUDED_NAMES or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        result.append({
            "path": relative.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return result


def render(records: list[dict[str, str | int]]) -> str:
    lines = ["path\tsize_bytes\tsha256"]
    lines.extend(f"{row['path']}\t{row['size_bytes']}\t{row['sha256']}" for row in records)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    content = render(rows())
    if args.write:
        MANIFEST.write_text(content, encoding="utf-8")
        ANCHOR.write_text(f"{sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
        print(f"WROTE {len(content.splitlines()) - 1} manifest records")
        return 0
    errors: list[str] = []
    if not MANIFEST.is_file() or MANIFEST.read_text(encoding="utf-8") != content:
        errors.append("manifest content does not match the release tree")
    if not ANCHOR.is_file():
        errors.append("manifest checksum anchor is missing")
    elif MANIFEST.is_file():
        expected = f"{sha256(MANIFEST)}  {MANIFEST.name}\n"
        if ANCHOR.read_text(encoding="utf-8") != expected:
            errors.append("manifest checksum anchor does not match")
    if errors:
        print("RELEASE_MANIFEST_INVALID")
        for error in errors:
            print(f"- {error}")
        return 2
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        count = sum(1 for _ in csv.DictReader(handle, delimiter="\t"))
    print(f"RELEASE_MANIFEST_PASS records={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
