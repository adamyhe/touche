"""Quantitative APA submatrix scores: named masks, tidy summaries, and clustered intervals.

Public API: `MaskSpec` and `default_masks` (the mask definitions),
`read_masks`/`write_masks` (JSON/YAML mask files), `build_masks` (resolve
specs to boolean pixel selections for a given window/resolution), and
`summarize_apa` (the tidy score table).

An APA heatmap is a picture; this turns it into comparable numbers with
declared numerators and denominators. Masks are defined in *base pairs of
offset from the anchor*, not in pixel indices, so the same mask file means
the same thing at 200 bp and at 2 kb resolution and a summary computed at
one resolution is comparable to one computed at another.

Matrix orientation follows `touche.apa.compute_apa`: columns are signed
offset from the bait (promoter) anchor and rows are signed offset from the
prey (enhancer) anchor, with rows in descending order so the plotted heatmap
puts `+window` at the top. `build_masks` reads the offsets off the matrix
labels rather than assuming an index layout, so it stays correct if the
binning changes.

Uncertainty is resampled over *chromosomes*, never over pixels. Pixels
inside one pileup are neither independent nor a sample of anything -- an
interval built from them is meaningless. Chromosomes are independent enough
to bootstrap over and are already the unit `compute_apa` accumulates by, so
`compute_apa(..., keep_chromosomes=True)` is all that is needed to get one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from touche.metadata import StatResult, method_info

Bound = float | None


@dataclass(frozen=True, slots=True)
class MaskSpec:
    """One named region of an APA matrix, in base pairs of offset from the anchors.

    `row_range`/`col_range` are inclusive `(low, high)` bounds on the prey
    (row) and bait (column) offset; `None` on either side means unbounded.
    `exclude` names other masks whose pixels are subtracted, which is how
    the stripes exclude the dot and how the ring becomes a donut rather than
    a filled square.
    """

    name: str
    row_range: tuple[Bound, Bound] = (None, None)
    col_range: tuple[Bound, Bound] = (None, None)
    exclude: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-serializable form, used by `write_masks` and result metadata."""
        return {
            "name": self.name,
            "row_range": list(self.row_range),
            "col_range": list(self.col_range),
            "exclude": list(self.exclude),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ApaScore:
    """One reported ratio, naming the two masks it divides."""

    name: str
    numerator: str
    denominator: str | None = None
    description: str = ""


DEFAULT_SCORES: tuple[ApaScore, ...] = (
    ApaScore("central_enrichment", "dot", "all", "Focal signal against the whole window's mean."),
    ApaScore("p2ll", "dot", "lower_left", "Peak to lower-left corner, the conventional APA ratio."),
    ApaScore("p2m", "dot", "global_background", "Peak to mean background outside the cross and ring."),
    ApaScore("promoter_stripe_enrichment", "promoter_stripe", "global_background", ""),
    ApaScore("enhancer_stripe_enrichment", "enhancer_stripe", "global_background", ""),
    ApaScore("ring_enrichment", "ring", "global_background", ""),
    ApaScore("dot_to_promoter_stripe", "dot", "promoter_stripe", "Focal versus promoter-anchored stripe signal."),
    ApaScore("dot_to_enhancer_stripe", "dot", "enhancer_stripe", "Focal versus enhancer-anchored stripe signal."),
)


def default_masks(
    window: int,
    *,
    pixels: int | None = None,
    dot_fraction: float = 0.1,
    ring_fraction: float = 0.2,
    corner_fraction: float = 0.2,
) -> list[MaskSpec]:
    """Standard mask set, scaled to `window` so it means the same at any resolution.

    `dot_fraction` sets the focal half-width as a fraction of `window`
    (0.1 -> the central +/-10% in each direction). `ring_fraction` sets the
    outer edge of the donut-shaped local neighbourhood, and `corner_fraction`
    the size of the lower-left corner block that `p2ll` divides by.

    Pass `pixels` (the pileup's per-side bin count) to floor the focal
    half-widths at one and two bin widths respectively. Without it, a coarse
    pileup can make `dot_fraction` fall inside the innermost bin and select
    no pixels at all, which turns every focal score into NaN.
    """

    for name, value in (("dot_fraction", dot_fraction), ("ring_fraction", ring_fraction), ("corner_fraction", corner_fraction)):
        if not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    if ring_fraction <= dot_fraction:
        raise ValueError("ring_fraction must exceed dot_fraction")

    step = (window / pixels) if pixels else 0.0
    dot = max(window * dot_fraction, step)
    ring = max(window * ring_fraction, 2 * step, dot + step)
    corner = window * (1.0 - corner_fraction)
    cross = ("dot", "promoter_stripe", "enhancer_stripe")
    return [
        MaskSpec("all", description="Every pixel in the window; the denominator for central enrichment."),
        MaskSpec("dot", (-dot, dot), (-dot, dot), description="Focal enrichment at both anchors."),
        MaskSpec(
            "promoter_stripe", (None, None), (-dot, dot), ("dot",),
            "Column band through the promoter anchor, excluding the dot.",
        ),
        MaskSpec(
            "enhancer_stripe", (-dot, dot), (None, None), ("dot",),
            "Row band through the enhancer anchor, excluding the dot.",
        ),
        MaskSpec(
            "ring", (-ring, ring), (-ring, ring), cross,
            "Local neighbourhood around the dot, with the stripe cross removed.",
        ),
        MaskSpec(
            "lower_left", (None, -corner), (None, -corner),
            description="Outer lower-left corner block, the conventional P2LL denominator.",
        ),
        MaskSpec(
            "global_background", (None, None), (None, None), (*cross, "ring"),
            "Everything outside the focal structures.",
        ),
    ]


def build_masks(
    row_offsets: np.ndarray, col_offsets: np.ndarray, specs: list[MaskSpec]
) -> dict[str, np.ndarray]:
    """Resolve mask specs to boolean pixel selections over a matrix with these offsets.

    Specs are resolved in order, so a spec may only exclude masks defined
    before it. An empty mask is returned as an all-false array rather than
    an error -- a window too small for a given mask is a real configuration
    to report, not a crash -- and `summarize_apa` reports its pixel count as
    zero and its statistics as NaN.
    """

    rows = np.asarray(row_offsets, dtype=np.float64)[:, None]
    cols = np.asarray(col_offsets, dtype=np.float64)[None, :]
    masks: dict[str, np.ndarray] = {}
    for spec in specs:
        selected = _in_range(rows, spec.row_range) & _in_range(cols, spec.col_range)
        selected = np.broadcast_to(selected, (rows.shape[0], cols.shape[1])).copy()
        for name in spec.exclude:
            if name not in masks:
                raise ValueError(f"Mask {spec.name!r} excludes {name!r}, which is not defined before it")
            selected &= ~masks[name]
        masks[spec.name] = selected
    return masks


def read_masks(path: str | Path) -> list[MaskSpec]:
    """Read mask definitions from a JSON or YAML list of mask objects."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if isinstance(payload, dict) and "masks" in payload:
        payload = payload["masks"]
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list of mask objects (or a 'masks' key holding one)")
    return [
        MaskSpec(
            name=entry["name"],
            row_range=tuple(entry.get("row_range", (None, None))),
            col_range=tuple(entry.get("col_range", (None, None))),
            exclude=tuple(entry.get("exclude", ())),
            description=entry.get("description", ""),
        )
        for entry in payload
    ]


def write_masks(specs: list[MaskSpec], path: str | Path) -> Path:
    """Write mask definitions as JSON (or YAML, by file extension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [spec.to_dict() for spec in specs]
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def summarize_apa(
    result: Any,
    *,
    masks: list[MaskSpec] | None = None,
    scores: tuple[ApaScore, ...] = DEFAULT_SCORES,
    bootstrap: int = 0,
    confidence: float = 0.95,
    seed: int = 0,
) -> StatResult:
    """Tidy quantitative summary of an APA pileup.

    `result` is an `touche.apa.ApaResult`. The returned table has one row per
    reported quantity, and every ratio carries its own numerator, denominator,
    and pixel counts, so nothing has to be reverse-engineered from the score.

    Set `bootstrap > 0` to add confidence intervals. That requires the result
    to have been computed with `compute_apa(..., keep_chromosomes=True)`,
    because the resampling unit is the chromosome. Without per-chromosome
    matrices the intervals are NaN and the reason is recorded as a warning --
    resampling pixels instead would produce an interval that looks
    informative and is not.
    """

    matrix, row_offsets, col_offsets = _matrix_and_offsets(result)
    specs = masks if masks is not None else default_masks(result.window, pixels=result.pixels)
    resolved = build_masks(row_offsets, col_offsets, specs)

    rows = [_mask_row(name, matrix, selected) for name, selected in resolved.items()]
    rows.extend(_score_row(score, matrix, resolved) for score in scores)
    table = pl.DataFrame(rows)

    warnings: list[str] = []
    chrom_matrices = getattr(result, "chrom_matrices", None)
    if bootstrap > 0 and chrom_matrices:
        table = table.with_columns(
            _bootstrap_bounds(chrom_matrices, resolved, scores, table, bootstrap, confidence, seed)
        )
        if len(chrom_matrices) < 10:
            warnings.append(
                f"Bootstrap resampled only {len(chrom_matrices)} chromosomes; intervals from this "
                "few independent units are unreliable."
            )
    else:
        table = table.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("ci_low"), pl.lit(None, dtype=pl.Float64).alias("ci_high")
        )
        if bootstrap > 0:
            warnings.append(
                "bootstrap was requested but the APA result has no per-chromosome matrices; "
                "re-run compute_apa(..., keep_chromosomes=True). Intervals are not estimated from "
                "pixels, which are not independent observations."
            )

    empty = [name for name, selected in resolved.items() if not selected.any()]
    if empty:
        warnings.append(f"These masks select no pixels at this window/resolution: {', '.join(empty)}")

    info = method_info(
        "apa_mask",
        unit_of_analysis="aggregated pileup over a pair set",
        cluster_unit="chromosome" if bootstrap > 0 and chrom_matrices else None,
        n_input=int(matrix.size),
        n_tested=int(matrix.size),
        seed=seed if bootstrap > 0 else None,
        parameters={
            "window": result.window,
            "pixels": result.pixels,
            "bootstrap": bootstrap,
            "confidence": confidence,
            "n_chromosomes": len(chrom_matrices) if chrom_matrices else 0,
            "masks": [spec.to_dict() for spec in specs],
        },
        warnings=warnings,
    )
    return StatResult(table=table, info=info)


def _in_range(offsets: np.ndarray, bounds: tuple[Bound, Bound]) -> np.ndarray:
    """Boolean selection of offsets inside an inclusive, possibly half-open bp range."""
    low, high = bounds
    selected = np.ones_like(offsets, dtype=bool)
    if low is not None:
        selected &= offsets >= low
    if high is not None:
        selected &= offsets <= high
    return selected


def _matrix_and_offsets(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull the numeric matrix and its signed row/column bp offsets off an `ApaResult`."""
    frame = result.matrix
    row_offsets = frame["bin_label"].to_numpy().astype(np.float64)
    col_offsets = np.array([float(name) for name in frame.columns if name != "bin_label"])
    matrix = frame.drop("bin_label").to_numpy().astype(np.float64)
    return matrix, row_offsets, col_offsets


def _mask_row(name: str, matrix: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    """One `<mask>_mean` summary row: the mask's own mean, sum, and pixel count."""
    values = matrix[selected]
    return {
        "metric": f"{name}_mean",
        "value": float(values.mean()) if values.size else float("nan"),
        "numerator_mask": name,
        "numerator": float(values.sum()) if values.size else float("nan"),
        "n_pixels_numerator": int(values.size),
        "denominator_mask": None,
        "denominator": None,
        "n_pixels_denominator": None,
    }


def _score_row(score: ApaScore, matrix: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    """One ratio row, carrying both means and both pixel counts alongside the ratio."""
    numerator, n_numerator = _mask_mean(matrix, masks, score.numerator)
    denominator, n_denominator = _mask_mean(matrix, masks, score.denominator)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = numerator / denominator if denominator is not None else numerator
    return {
        "metric": score.name,
        "value": float(value),
        "numerator_mask": score.numerator,
        "numerator": float(numerator),
        "n_pixels_numerator": n_numerator,
        "denominator_mask": score.denominator,
        "denominator": None if denominator is None else float(denominator),
        "n_pixels_denominator": n_denominator,
    }


def _mask_mean(matrix: np.ndarray, masks: dict[str, np.ndarray], name: str | None) -> tuple[float | None, int | None]:
    """Mean and pixel count of one named mask, or `(None, None)` for an absent denominator."""
    if name is None:
        return None, None
    if name not in masks:
        raise ValueError(f"Score references undefined mask {name!r}")
    values = matrix[masks[name]]
    return (float(values.mean()) if values.size else float("nan")), int(values.size)


def _bootstrap_bounds(
    chrom_matrices: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    scores: tuple[ApaScore, ...],
    table: pl.DataFrame,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> list[pl.Series]:
    """Percentile interval per metric from resampling whole chromosome pileups."""
    names = list(chrom_matrices)
    stacked = np.stack([chrom_matrices[name].astype(np.float64) for name in names])
    rng = np.random.default_rng(seed)

    replicates = np.empty((n_resamples, table.height), dtype=np.float64)
    for i in range(n_resamples):
        pooled = stacked[rng.integers(0, len(names), len(names))].sum(axis=0)
        rows = [_mask_row(name, pooled, selected)["value"] for name, selected in masks.items()]
        rows.extend(_score_row(score, pooled, masks)["value"] for score in scores)
        replicates[i] = rows

    alpha = 1.0 - confidence
    with np.errstate(invalid="ignore"):
        low = np.nanpercentile(replicates, 100 * alpha / 2, axis=0)
        high = np.nanpercentile(replicates, 100 * (1 - alpha / 2), axis=0)
    return [pl.Series("ci_low", low), pl.Series("ci_high", high)]
