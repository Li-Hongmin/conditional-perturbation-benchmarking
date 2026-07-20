#!/usr/bin/env python3
"""Verify a regenerated PNG preview by decoded visual content.

PNG files are containers: byte-level equality can fail when encoders write
different ancillary metadata, and Matplotlib can round the declared physical
size by one pixel on different platforms.  This verifier therefore decodes the
images, checks their dimensions against the figure contract, normalizes the
one-pixel size variation, and compares the rendered RGB content.  It does not
weaken the byte-exact checks applied separately to the scientific TSV outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_SIZE = (2161, 1429)
DIMENSION_TOLERANCE_PX = 1
NORMALIZED_WIDTH_PX = 720
MAX_NORMALIZED_MAE = 0.02
MIN_INK_CORRELATION = 0.95
MAX_FOREGROUND_FRACTION_DELTA = 0.02


class FigurePreviewError(ValueError):
    """Raised when a rendered preview violates the visual contract."""


def _load_rgb(path: Path) -> tuple[Image.Image, tuple[int, int]]:
    if not path.is_file():
        raise FigurePreviewError(f"PNG preview is missing: {path}")
    try:
        with Image.open(path) as source:
            source.load()
            size = source.size
            rgba = source.convert("RGBA")
    except Exception as exc:  # Pillow supplies format-specific exceptions.
        raise FigurePreviewError(f"PNG preview cannot be decoded: {path}") from exc

    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB"), size


def _within_expected_size(size: tuple[int, int], expected: tuple[int, int]) -> bool:
    return all(
        abs(observed - declared) <= DIMENSION_TOLERANCE_PX
        for observed, declared in zip(size, expected, strict=True)
    )


def _normalized_array(image: Image.Image, expected: tuple[int, int]) -> np.ndarray:
    height = round(NORMALIZED_WIDTH_PX * expected[1] / expected[0])
    resized = image.resize((NORMALIZED_WIDTH_PX, height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32) / 255.0


def compare_previews(
    candidate_path: Path,
    reference_path: Path,
    *,
    expected_size: tuple[int, int] = EXPECTED_SIZE,
) -> dict[str, object]:
    """Compare decoded previews and return machine-readable quality metrics."""

    candidate, candidate_size = _load_rgb(candidate_path)
    reference, reference_size = _load_rgb(reference_path)
    if not _within_expected_size(candidate_size, expected_size):
        raise FigurePreviewError(
            f"candidate dimensions {candidate_size} violate expected {expected_size} "
            f"±{DIMENSION_TOLERANCE_PX} px"
        )
    if not _within_expected_size(reference_size, expected_size):
        raise FigurePreviewError(
            f"reference dimensions {reference_size} violate expected {expected_size} "
            f"±{DIMENSION_TOLERANCE_PX} px"
        )

    candidate_rgb = _normalized_array(candidate, expected_size)
    reference_rgb = _normalized_array(reference, expected_size)
    absolute_difference = np.abs(candidate_rgb - reference_rgb)
    normalized_mae = float(absolute_difference.mean())

    luminance_weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    candidate_ink = 1.0 - candidate_rgb @ luminance_weights
    reference_ink = 1.0 - reference_rgb @ luminance_weights
    candidate_centered = candidate_ink.ravel() - float(candidate_ink.mean())
    reference_centered = reference_ink.ravel() - float(reference_ink.mean())
    denominator = float(
        np.linalg.norm(candidate_centered) * np.linalg.norm(reference_centered)
    )
    ink_correlation = (
        float(np.dot(candidate_centered, reference_centered) / denominator)
        if denominator > 0
        else -1.0
    )

    foreground_threshold = 8.0 / 255.0
    candidate_foreground = float((candidate_ink > foreground_threshold).mean())
    reference_foreground = float((reference_ink > foreground_threshold).mean())
    foreground_fraction_delta = abs(candidate_foreground - reference_foreground)

    failures: list[str] = []
    if normalized_mae > MAX_NORMALIZED_MAE:
        failures.append(
            f"normalized RGB MAE {normalized_mae:.6f} exceeds {MAX_NORMALIZED_MAE:.6f}"
        )
    if ink_correlation < MIN_INK_CORRELATION:
        failures.append(
            f"ink correlation {ink_correlation:.6f} is below {MIN_INK_CORRELATION:.6f}"
        )
    if foreground_fraction_delta > MAX_FOREGROUND_FRACTION_DELTA:
        failures.append(
            "foreground fraction delta "
            f"{foreground_fraction_delta:.6f} exceeds "
            f"{MAX_FOREGROUND_FRACTION_DELTA:.6f}"
        )
    if failures:
        raise FigurePreviewError("; ".join(failures))

    return {
        "status": "FIGURE_PREVIEW_PASS",
        "candidate_size": list(candidate_size),
        "reference_size": list(reference_size),
        "expected_size": list(expected_size),
        "dimension_tolerance_px": DIMENSION_TOLERANCE_PX,
        "normalized_rgb_mae": normalized_mae,
        "ink_correlation": ink_correlation,
        "candidate_foreground_fraction": candidate_foreground,
        "reference_foreground_fraction": reference_foreground,
        "foreground_fraction_delta": foreground_fraction_delta,
        "comparison_scope": "decoded pixels and dimensions; PNG metadata ignored",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_previews(args.candidate, args.reference)
    except FigurePreviewError as exc:
        print(f"FIGURE_PREVIEW_INVALID\n- {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
