#!/usr/bin/env python3
"""Run the finite-benchmark composition analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crg.benchmark_composition import (  # noqa: E402
    BenchmarkCompositionError,
    build_from_policy,
    write_analysis,
)


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise BenchmarkCompositionError("cannot determine producer Git commit") from exc


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository-root", type=Path, default=ROOT)
    result.add_argument(
        "--policy", type=Path,
        default=ROOT / "configs/benchmark_composition_policy_v1.json",
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--producer-commit")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        artifacts = build_from_policy(
            args.repository_root, args.policy,
            producer_commit=args.producer_commit or _commit(),
        )
        output = write_analysis(args.output_dir, artifacts)
        report = {
            "status": "CALCULATED_POST_RESULT_SECONDARY_ANALYSIS",
            "output_dir": str(output),
            "n_conditions": artifacts.record["n_conditions"],
            "n_models": artifacts.record["n_models"],
            "composition_grid_points": artifacts.record["composition_grid_points"],
            "population_confidence_interval": False,
        }
        code = 0
    except (BenchmarkCompositionError, OSError, UnicodeError, ValueError) as exc:
        report = {"status": "INVALID", "error": str(exc)}
        code = 2
    print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
