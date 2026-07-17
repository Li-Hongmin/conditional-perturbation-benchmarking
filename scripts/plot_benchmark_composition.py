#!/usr/bin/env python3
"""Build the benchmark-composition figure from analysis tables.

The script renders a fixed-size, four-panel scientific figure.  It does not
recompute scientific results: every plotted quantity is read from the frozen
TSV outputs produced by ``run_benchmark_composition.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


WIDTH_MM = 183.0
HEIGHT_MM = 121.0
MM_PER_INCH = 25.4
PRIMARY_METRIC = "relative_l2_response_error"

MODEL_LABELS = {
    "pb_biolord_star": "biolord",
    "pb_cpa": "CPA",
    "pb_decoder_only": "DecoderOnly",
    "pb_latent_additive": "LatentAdditive",
    "pb_linear_additive": "LinearAdditive",
    "pb_sparse_additive_vae": "SAMS-VAE",
}

# LatentAdditive and CPA carry the finite-panel winner transition.  The other
# model colors retain the blue/orange/neutral vocabulary used by the project,
# but are intentionally less saturated and thinner in panel a.
MODEL_COLORS = {
    "pb_biolord_star": "#D47A22",
    "pb_cpa": "#7A4EA3",
    "pb_decoder_only": "#6F7F8F",
    "pb_latent_additive": "#008C95",
    "pb_linear_additive": "#2B6EA6",
    "pb_sparse_additive_vae": "#9B8EA6",
}
HIGHLIGHT_MODELS = {"pb_cpa", "pb_latent_additive"}

PAIR_LABELS = {
    frozenset(("pb_biolord_star", "pb_sparse_additive_vae")): "biolord – SAMS-VAE",
    frozenset(("pb_cpa", "pb_linear_additive")): "CPA – LinearAdditive",
    frozenset(("pb_cpa", "pb_decoder_only")): "CPA – DecoderOnly",
    frozenset(("pb_linear_additive", "pb_sparse_additive_vae")): "LinearAdditive – SAMS-VAE",
    frozenset(("pb_biolord_star", "pb_linear_additive")): "biolord – LinearAdditive",
    frozenset(("pb_cpa", "pb_latent_additive")): "CPA – LatentAdditive",
}

ABLATION_LABELS = {
    "continuous_effect_rank": "Effect magnitude",
    "continuous_additivity_violation_rank": "AV",
    "continuous_additive_axes": "Effect + AV",
    "continuous_axis_interaction": "Effect + AV + interaction",
}


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.2,
            "axes.labelsize": 6.4,
            "axes.titlesize": 7.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _read_table(input_dir: Path, filename: str, required: set[str]) -> pd.DataFrame:
    path = input_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen analysis table: {path}")
    table = pd.read_csv(path, sep="\t")
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")
    return table


def _load_inputs(input_dir: Path) -> dict[str, pd.DataFrame]:
    curves = _read_table(
        input_dir,
        "composition_risk_curves.tsv",
        {"metric_name", "lambda_high_av", "model_id", "metric_value", "is_winner"},
    )
    crossovers = _read_table(
        input_dir,
        "pairwise_crossovers.tsv",
        {
            "metric_name",
            "model_a",
            "model_b",
            "crossover_lambda",
            "point_status",
            "resampling_valid_fraction",
            "crossover_stability_low",
            "crossover_stability_high",
        },
    )
    regret = _read_table(
        input_dir,
        "selection_regret.tsv",
        {
            "metric_name",
            "lambda_high_av",
            "aggregate_winner_model_id",
            "conditional_winner_model_ids",
            "selection_regret",
            "aggregate_winner_is_optimal",
        },
    )
    ablation = _read_table(
        input_dir,
        "geometry_ablation_summary.tsv",
        {
            "analysis_type",
            "metric_name",
            "continuous_specification",
            "model_a",
            "delta_loo_mae_vs_intercept",
            "delta_loo_mae_vs_av_only",
        },
    )

    curves = curves.loc[curves["metric_name"].eq(PRIMARY_METRIC)].copy()
    crossovers = crossovers.loc[
        crossovers["metric_name"].eq(PRIMARY_METRIC)
        & crossovers["point_status"].eq("interior_or_endpoint_crossover")
    ].copy()
    regret = regret.loc[regret["metric_name"].eq(PRIMARY_METRIC)].copy()
    ablation = ablation.loc[
        ablation["metric_name"].eq(PRIMARY_METRIC)
        & ablation["analysis_type"].eq("continuous_pairwise")
        & ablation["continuous_specification"].isin(ABLATION_LABELS)
    ].copy()

    expected_models = set(MODEL_LABELS)
    observed_models = set(curves["model_id"].unique())
    if observed_models != expected_models:
        raise ValueError(
            f"Expected six frozen models {sorted(expected_models)}, got {sorted(observed_models)}"
        )
    counts = curves.groupby("model_id")["lambda_high_av"].nunique()
    if not counts.eq(1001).all():
        raise ValueError(f"Expected 1,001 composition points per model, got {counts.to_dict()}")
    if len(crossovers) != 6:
        raise ValueError(f"Expected 6 within-domain crossovers, got {len(crossovers)}")
    if regret["lambda_high_av"].nunique() != 1001:
        raise ValueError("Selection-regret table does not contain the frozen 1,001-point grid")
    if set(ablation["continuous_specification"].unique()) != set(ABLATION_LABELS):
        raise ValueError("Coordinate-ablation table is missing a requested specification")
    if ablation.groupby("continuous_specification").size().ne(15).any():
        raise ValueError("Expected 15 model-pair estimates for every ablation specification")

    return {
        "curves": curves,
        "crossovers": crossovers,
        "regret": regret,
        "ablation": ablation,
    }


def _style_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["left"].set_color("#30343B")
    ax.spines["bottom"].set_color("#30343B")
    ax.tick_params(colors="#30343B", pad=2)
    ax.xaxis.label.set_color("#20242A")
    ax.yaxis.label.set_color("#20242A")
    ax.title.set_color("#20242A")


def _panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.17, y: float = 1.07) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        va="top",
        ha="left",
        color="#161A1F",
        clip_on=False,
    )


def _plot_risk_trajectories(ax: mpl.axes.Axes, curves: pd.DataFrame) -> None:
    draw_order = [
        "pb_biolord_star",
        "pb_sparse_additive_vae",
        "pb_linear_additive",
        "pb_decoder_only",
        "pb_latent_additive",
        "pb_cpa",
    ]
    for model_id in draw_order:
        subset = curves.loc[curves["model_id"].eq(model_id)].sort_values("lambda_high_av")
        highlighted = model_id in HIGHLIGHT_MODELS
        ax.plot(
            subset["lambda_high_av"],
            subset["metric_value"],
            color=MODEL_COLORS[model_id],
            linewidth=1.8 if highlighted else 0.9,
            alpha=1.0 if highlighted else 0.72,
            zorder=3 if highlighted else 2,
        )

    winner_change = 0.8195
    ax.axvline(winner_change, color="#60666E", linestyle=(0, (3, 2)), linewidth=0.7, zorder=1)
    ax.text(
        winner_change + 0.012,
        0.985,
        r"point estimate $\lambda=0.82$",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=5.2,
        color="#555C66",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.34, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel(r"Target composition in two-regime slice, $\lambda_{\mathrm{high\ AV}}$")
    ax.set_ylabel(r"Conditional mean relative-$L_2$ error")
    ax.set_title("Six-model conditional risk trajectories", loc="left", pad=5)
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.45, alpha=0.8)

    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[m],
            lw=1.8 if m in HIGHLIGHT_MODELS else 1.0,
            alpha=1.0 if m in HIGHLIGHT_MODELS else 0.8,
            label=MODEL_LABELS[m],
        )
        for m in draw_order
    ]
    ax.legend(
        handles=handles,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0, 1.01),
        borderaxespad=0,
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.35,
        labelspacing=0.35,
    )
    _style_axis(ax)
    _panel_label(ax, "a", x=-0.14)


def _plot_crossovers(ax: mpl.axes.Axes, crossovers: pd.DataFrame) -> None:
    crossovers = crossovers.sort_values("crossover_lambda").reset_index(drop=True)
    y = np.arange(len(crossovers))
    labels = [PAIR_LABELS[frozenset((row.model_a, row.model_b))] for row in crossovers.itertuples()]
    low = crossovers["crossover_stability_low"].to_numpy(dtype=float)
    high = crossovers["crossover_stability_high"].to_numpy(dtype=float)
    point = crossovers["crossover_lambda"].to_numpy(dtype=float)
    valid = crossovers["resampling_valid_fraction"].to_numpy(dtype=float)

    for yi, x, lo, hi, vf in zip(y, point, low, high, valid):
        ax.hlines(yi, lo, hi, color="#A3A9B1", linewidth=1.1, zorder=1)
        ax.plot(
            x,
            yi,
            marker="o",
            markersize=3.5 + 2.0 * vf,
            color="#6B3FA0",
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=2,
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel(r"Crossover within two-regime slice, $\lambda_{\mathrm{high\ AV}}$")
    ax.set_title("Pairwise crossovers (6 of 15)", loc="left", pad=5)
    ax.grid(axis="x", color="#D8DDE3", linewidth=0.45, alpha=0.8)
    ax.text(
        0.98,
        0.02,
        "lines: conditional 95% stability intervals",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color="#69717B",
    )
    _style_axis(ax)
    _panel_label(ax, "b", x=-0.31)


def _plot_selection_regret(ax: mpl.axes.Axes, regret: pd.DataFrame) -> None:
    regret = regret.sort_values("lambda_high_av")
    x = regret["lambda_high_av"].to_numpy(dtype=float)
    y = regret["selection_regret"].to_numpy(dtype=float)
    aggregate_winner = str(regret["aggregate_winner_model_id"].iloc[0])
    conditional = regret["conditional_winner_model_ids"].astype(str)
    cpa_start = float(regret.loc[conditional.eq("pb_cpa"), "lambda_high_av"].min())

    ax.axvspan(0, cpa_start, color=MODEL_COLORS["pb_latent_additive"], alpha=0.08, lw=0)
    ax.axvspan(cpa_start, 1, color=MODEL_COLORS["pb_cpa"], alpha=0.09, lw=0)
    ax.plot(x, y, color="#2D596D", linewidth=1.7)
    ax.fill_between(x, 0, y, color="#2D596D", alpha=0.12, linewidth=0)
    ax.axvline(cpa_start, color="#60666E", linestyle=(0, (3, 2)), linewidth=0.7)

    max_idx = int(np.nanargmax(y))
    ax.plot(x[max_idx], y[max_idx], "o", color="#2D596D", markersize=3.5, zorder=3)
    ax.annotate(
        f"max = {y[max_idx]:.4f}",
        xy=(x[max_idx], y[max_idx]),
        xytext=(-5, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=5.4,
        color="#2D596D",
    )
    ax.text(
        0.37,
        0.88,
        "LatentAdditive optimal",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.5,
        color=MODEL_COLORS["pb_latent_additive"],
    )
    ax.text(
        0.91,
        0.88,
        "CPA optimal",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.5,
        color=MODEL_COLORS["pb_cpa"],
    )
    nonoptimal_pct = 100.0 * (~regret["aggregate_winner_is_optimal"].astype(bool)).mean()
    ax.text(
        0.02,
        0.96,
        f"full-test winner: {MODEL_LABELS[aggregate_winner]}; non-optimal on {nonoptimal_pct:.1f}% of slice",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.15,
        color="#555C66",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.019, y.max() * 1.16))
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel(r"Target composition in two-regime slice, $\lambda_{\mathrm{high\ AV}}$")
    ax.set_ylabel("Selection regret")
    ax.set_title("Regret of retaining the full-test winner", loc="left", pad=5)
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.45, alpha=0.8)
    _style_axis(ax)
    _panel_label(ax, "c", x=-0.14)


def _plot_ablation(ax: mpl.axes.Axes, ablation: pd.DataFrame) -> None:
    order = list(ABLATION_LABELS)
    rows = []
    for specification in order:
        subset = ablation.loc[ablation["continuous_specification"].eq(specification)]
        rows.append(
            {
                "specification": specification,
                "median_vs_intercept": float(subset["delta_loo_mae_vs_intercept"].median()),
                "median_vs_av": float(subset["delta_loo_mae_vs_av_only"].median()),
                "improves": int((subset["delta_loo_mae_vs_intercept"] < 0).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    y = np.arange(len(summary))[::-1]

    ax.axvline(0, color="#707780", linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    ax.scatter(
        summary["median_vs_intercept"],
        y,
        s=23,
        color="#2B6EA6",
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
        label="vs intercept",
    )
    ax.scatter(
        summary["median_vs_av"],
        y,
        s=21,
        facecolor="white",
        edgecolor="#B43C7A",
        linewidth=0.9,
        marker="D",
        zorder=3,
        label="vs AV-only",
    )
    for yi, row in zip(y, summary.itertuples()):
        ax.text(
            0.012,
            yi,
            f"{row.improves}/15",
            va="center",
            ha="left",
            fontsize=5.3,
            color="#2B6EA6",
        )
    ax.set_yticks(y, [ABLATION_LABELS[s] for s in summary["specification"]])
    ax.set_xlim(-0.05, 0.025)
    ax.set_xticks([-0.04, -0.02, 0, 0.02])
    ax.set_xlabel(r"Median $\Delta$LOO-MAE (negative is better)")
    ax.set_title("Coordinate ablation across 15 model pairs", loc="left", pad=17)
    ax.grid(axis="x", color="#D8DDE3", linewidth=0.45, alpha=0.8)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.002),
        ncol=2,
        borderaxespad=0,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    ax.text(
        0.90,
        1.003,
        "pairs improved\nvs intercept",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=4.8,
        color="#2B6EA6",
        linespacing=0.92,
        clip_on=False,
    )
    _style_axis(ax)
    _panel_label(ax, "d", x=-0.31)


def build_figure(tables: dict[str, pd.DataFrame]) -> plt.Figure:
    _configure_matplotlib()
    fig = plt.figure(figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.083,
        right=0.982,
        bottom=0.105,
        top=0.925,
        wspace=0.50,
        hspace=0.47,
        width_ratios=(1.22, 1.0),
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    _plot_risk_trajectories(ax_a, tables["curves"])
    _plot_crossovers(ax_b, tables["crossovers"])
    _plot_selection_regret(ax_c, tables["regret"])
    _plot_ablation(ax_d, tables["ablation"])
    return fig


def _export_and_check(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    qa_path = output_dir / f"{stem}.qa.json"

    # No tight bounding box: this preserves the declared 183 mm page width.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        bbox = ax.get_window_extent(renderer=renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            raise RuntimeError("A figure panel has an invalid rendering extent")

    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    # Matplotlib emits insignificant trailing spaces in SVG path elements.
    # Normalize them so generated vector artifacts pass repository whitespace checks.
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )

    from PIL import Image

    with Image.open(png_path) as image:
        png_size = image.size
    expected_png = (
        round(WIDTH_MM / MM_PER_INCH * 300),
        round(HEIGHT_MM / MM_PER_INCH * 300),
    )
    dimensions_within_tolerance = (
        abs(png_size[0] - expected_png[0]) <= 1
        and abs(png_size[1] - expected_png[1]) <= 1
    )
    if not dimensions_within_tolerance:
        raise RuntimeError(f"PNG dimensions {png_size} differ from expected {expected_png}")

    regret = tables["regret"]
    qa = {
        "figure_contract": {
            "backend": "Python/matplotlib",
            "width_mm": WIDTH_MM,
            "height_mm": HEIGHT_MM,
            "raster_dpi": 300,
            "editable_vector_outputs": [pdf_path.name, svg_path.name],
        },
        "source_checks": {
            "models": int(tables["curves"]["model_id"].nunique()),
            "composition_points_per_model": 1001,
            "within_domain_crossovers": int(len(tables["crossovers"])),
            "ablation_pairs_per_specification": 15,
        },
        "plotted_results": {
            "aggregate_winner": str(regret["aggregate_winner_model_id"].iloc[0]),
            "maximum_selection_regret": float(regret["selection_regret"].max()),
            "aggregate_winner_nonoptimal_grid_fraction": float(
                (~regret["aggregate_winner_is_optimal"].astype(bool)).mean()
            ),
        },
        "output_checks": {
            "png_pixels": list(png_size),
            "expected_png_pixels": list(expected_png),
            "pixel_tolerance": 1,
            "dimensions_within_tolerance": dimensions_within_tolerance,
            "all_outputs_nonempty": all(
                path.is_file() and path.stat().st_size > 0
                for path in (pdf_path, svg_path, png_path)
            ),
        },
    }
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    return {
        "pdf": pdf_path,
        "svg": svg_path,
        "png": png_path,
        "qa": qa_path,
        "qa_payload": qa,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/benchmark_composition_v1"),
        help="Directory containing the benchmark-composition TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory receiving PDF, SVG, PNG and QA JSON outputs.",
    )
    parser.add_argument(
        "--stem",
        default="benchmark_composition_landscape",
        help="Semantic output stem without an extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = _load_inputs(args.input_dir)
    fig = build_figure(tables)
    outputs = _export_and_check(fig, args.output_dir, args.stem, tables)
    plt.close(fig)
    print(json.dumps({key: str(value) for key, value in outputs.items() if key != "qa_payload"}, indent=2))


if __name__ == "__main__":
    main()
