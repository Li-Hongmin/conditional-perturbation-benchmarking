from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image, ImageDraw, PngImagePlugin
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_figure_preview import FigurePreviewError, compare_previews  # noqa: E402


def _preview(size: tuple[int, int] = (100, 60)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 46, 28), fill="#008C95")
    draw.line((10, 50, 90, 34), fill="#7A4EA3", width=3)
    return image


def _save(image: Image.Image, path: Path, metadata: str) -> None:
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Software", metadata)
    image.save(path, pnginfo=pnginfo)


def test_metadata_difference_does_not_fail_visual_comparison(tmp_path):
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    image = _preview()
    _save(image, reference, "encoder-a")
    _save(image, candidate, "encoder-b")
    assert reference.read_bytes() != candidate.read_bytes()

    result = compare_previews(candidate, reference, expected_size=(100, 60))

    assert result["status"] == "FIGURE_PREVIEW_PASS"
    assert result["normalized_rgb_mae"] == pytest.approx(0.0)
    assert result["comparison_scope"].startswith("decoded pixels")


def test_one_pixel_platform_rounding_is_normalized(tmp_path):
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    image = _preview()
    _save(image, reference, "reference")
    _save(image.resize((101, 61), Image.Resampling.LANCZOS), candidate, "candidate")

    result = compare_previews(candidate, reference, expected_size=(100, 60))

    assert result["candidate_size"] == [101, 61]
    assert result["reference_size"] == [100, 60]


def test_blank_or_wrong_preview_is_rejected(tmp_path):
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    _save(_preview(), reference, "reference")
    _save(Image.new("RGB", (100, 60), "white"), candidate, "candidate")

    with pytest.raises(FigurePreviewError, match="ink correlation"):
        compare_previews(candidate, reference, expected_size=(100, 60))


def test_dimension_drift_beyond_contract_is_rejected(tmp_path):
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    _save(_preview(), reference, "reference")
    _save(_preview((103, 60)), candidate, "candidate")

    with pytest.raises(FigurePreviewError, match="candidate dimensions"):
        compare_previews(candidate, reference, expected_size=(100, 60))
