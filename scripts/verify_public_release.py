#!/usr/bin/env python3
"""Fail closed on public-release scope, structure and disclosure hazards."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    "", ".cff", ".csv", ".gitignore", ".json", ".md", ".py", ".toml",
    ".tsv", ".txt", ".yaml", ".yml",
}
SKIP_PARTS = {
    ".git", ".pytest_cache", ".tmp", ".venv", "__pycache__",
    "replay_outputs", "replay_figure",
}
FORBIDDEN = {
    "local absolute path": re.compile(r"/(?:Users|Volumes|private)/|/home/[^/\s]+/"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "cloud bucket": re.compile(r"(?:gs://|storage\.googleapis\.com)"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "OpenAI token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "CJK text": re.compile(
        r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]"
    ),
    "journal or review staging language": re.compile(
        r"\b(?:Nature Computational Science|Nature Methods|reviewer replay|confidential review)\b",
        re.IGNORECASE,
    ),
    "retired internal path": re.compile(r"(?:nature_methods|results/v0_3|configs/nature)", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_files() -> list[Path]:
    result = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts):
            continue
        if path.name == "uv.lock" or path.suffix.lower() in TEXT_SUFFIXES:
            result.append(path)
    return sorted(result)


def main() -> int:
    errors: list[str] = []
    required = [
        "README.md", "LICENSE", "DATA_LICENSE.md", "THIRD_PARTY_NOTICES.md",
        "CITATION.cff", "pyproject.toml", "uv.lock",
        "RELEASE_FILE_MANIFEST.tsv", "RELEASE_FILE_MANIFEST.sha256",
        "configs/benchmark_composition_policy_v1.json",
        "configs/public_run_registry_v1.yaml",
        "data/norman/condition_metrics.tsv", "data/norman/regime_manifest.csv",
        "expected/EXPECTED_OUTPUT_CHECKSUMS.tsv",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    metrics_path = ROOT / "data/norman/condition_metrics.tsv"
    regimes_path = ROOT / "data/norman/regime_manifest.csv"
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path, sep="\t")
        if metrics.shape != (920, 13):
            errors.append(f"unexpected condition metric shape: {metrics.shape}")
        if metrics["run_id"].nunique() != 10 or metrics["condition_id"].nunique() != 46:
            errors.append("condition metric identity counts differ from the public contract")
    if regimes_path.is_file():
        regimes = pd.read_csv(regimes_path)
        expected_columns = [
            "dataset_id", "split_id", "condition_id", "primary_arm",
            "high_additivity_violation", "high_effect",
            "additivity_violation_score", "effect_size_covariate",
        ]
        if regimes.shape != (46, 8) or list(regimes.columns) != expected_columns:
            errors.append("regime manifest differs from the eight-column public contract")

    for path in text_files():
        if path.name == "verify_public_release.py":
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 public text file: {path.relative_to(ROOT)}")
            continue
        for label, pattern in FORBIDDEN.items():
            if pattern.search(content):
                errors.append(f"{label}: {path.relative_to(ROOT)}")

    if errors:
        print("PUBLIC_RELEASE_INVALID")
        for error in errors:
            print(f"- {error}")
        return 2
    print("PUBLIC_RELEASE_PASS")
    print(f"condition_metrics_sha256={sha256(metrics_path)}")
    print(f"regime_manifest_sha256={sha256(regimes_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
