"""Aggregate Peak Analysis (APA): pixel-binned contact matrix + 1D anchor signal.

Public API: `ApaResult`, `aggregate_apa` (file-driven wrapper), `compute_apa`
(in-memory compute), `write_apa_result`, `compare_apa_change`,
`plot_raw_apa_heatmap`, `plot_apa_change`. Everything prefixed `_` is an
internal helper for one of those entry points -- in particular
`_add_chrom_apa_numba`/`_apa_anchor_signal_numba`/`_apa_matrix_numba` wrap the
Numba counting kernels in `touche.numba.apa`, which `compute_apa` always
uses (there is no alternate counting path to choose between).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from touche.anchors import read_bed_anchors
from touche.contacts import build_contact_indexes, load_cached_contact_indexes
from touche.instrumentation import Instrumentation, make_instrumentation
from touche.models import ContactIndex
from touche.pairs import PairAnchors, read_bedpe, split_pair_anchors

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True, slots=True)
class ApaResult:
    """In-memory APA aggregate result for interactive use."""

    matrix: pl.DataFrame
    bait_signal: pl.DataFrame
    prey_signal: pl.DataFrame
    window: int
    pixels: int
    # Per-chromosome pileups, kept only when `compute_apa(keep_chromosomes=True)`
    # asked for them. They exist so `touche.apa_masks.summarize_apa` can build
    # confidence intervals by resampling whole chromosomes -- the smallest unit
    # of an APA pileup that is plausibly independent. Row order matches
    # `matrix`, so a resampled sum can be scored with the same masks.
    chrom_matrices: dict[str, "np.ndarray"] | None = None

    def plot(self, *, reference_style: bool = True) -> "Figure":
        """Render the pixel-binned matrix as a heatmap; see `plot_raw_apa_heatmap`."""
        return plot_raw_apa_heatmap(
            self.matrix,
            window=self.window,
            pixels=self.pixels,
            reference_style=reference_style,
        )

    def write(self, out_dir: str | Path, *, reference_style: bool = True) -> dict[str, Path]:
        """Write this result using the reference output filenames; see `write_apa_result`."""
        return write_apa_result(self, out_dir, reference_style=reference_style)


def aggregate_apa(
    pairs_path: str | Path,
    baits_path: str | Path,
    preys_path: str | Path,
    out_dir: str | Path,
    *,
    min_distance: int,
    max_distance: int,
    window: int,
    pixels: int,
    pairs_list: str | Path | None = None,
    source: str = "auto",
    shift: int = 75,
    reference_style: bool = True,
    index_strategy: str = "all",
    cache_dir: str | Path | None = None,
    cache_prefix: str = "contacts",
    require_cache: bool = False,
    summary_out: str | Path | None = None,
    masks: str | Path | None = None,
    bootstrap: int = 0,
    seed: int = 0,
    progress: bool | Instrumentation = False,
    profile: bool = False,
) -> dict[str, Path]:
    """Aggregate APA matrix and 1D anchor signal without per-bait temp files.

    `index_strategy="cache"` reads a persistent NPZ `ContactIndex` cache
    (building it first if missing) instead of re-parsing `pairs_path` --
    useful when `background count` is also run against the same sample,
    since both would otherwise each pay their own full pairs-file parse.

    `pairs_list` takes a BEDPE of explicit bait/prey pairs instead of
    piling up every bait crossed with every prey inside `[min_distance,
    max_distance]`. That list is then the pileup's pair universe exactly as
    given, which is what makes a pileup over a restricted hypothesis (a
    functional enhancer-promoter set, an imported loop call set) mean what
    it says. `baits_path`/`preys_path` are unused when it is supplied.

    `summary_out` additionally writes the quantitative mask summary (see
    `touche.apa_masks.summarize_apa`) and its metadata sidecar. `bootstrap`
    adds confidence intervals to that summary, which requires keeping
    per-chromosome pileups in memory for the duration of the run.
    """

    if index_strategy not in {"all", "cache"}:
        raise ValueError("index_strategy must be one of: all, cache")

    instrument = make_instrumentation(progress, profile=profile)
    with instrument.step("read inputs"):
        if index_strategy == "cache":
            cache_dir = _resolve_cache_dir(cache_dir, out_dir)
            indexes = load_cached_contact_indexes(
                pairs_path,
                cache_dir=cache_dir,
                cache_prefix=cache_prefix,
                source=source,
                include_metadata=True,
                require_cache=require_cache,
            )
        else:
            indexes = build_contact_indexes(pairs_path, source=source, cis_only=True)
        explicit_pairs = read_bedpe(pairs_list, cis_only=True) if pairs_list is not None else None
        baits = read_bed_anchors(baits_path) if explicit_pairs is None else pl.DataFrame()
        preys = read_bed_anchors(preys_path) if explicit_pairs is None else pl.DataFrame()
    result = compute_apa(
        indexes,
        baits,
        preys,
        min_distance=min_distance,
        max_distance=max_distance,
        window=window,
        pixels=pixels,
        pairs=explicit_pairs,
        shift=shift,
        keep_chromosomes=bootstrap > 0,
        progress=instrument,
    )
    with instrument.step("write apa outputs"):
        outputs = write_apa_result(result, out_dir, reference_style=reference_style)
    if summary_out is not None:
        with instrument.step("summarize apa"):
            from touche.apa_masks import read_masks, summarize_apa

            summary = summarize_apa(
                result,
                masks=read_masks(masks) if masks is not None else None,
                bootstrap=bootstrap,
                seed=seed,
            )
            outputs.update(
                {key: value for key, value in summary.write(summary_out).items()},
            )
            outputs["summary"] = outputs.pop("table")
            outputs["summary_metadata"] = outputs.pop("metadata")
    return outputs


def read_apa_matrix(path: str | Path) -> ApaResult:
    """Read a written `AggMat.csv` back into an `ApaResult`, inferring window and resolution.

    The matrix's own column labels are signed bp offsets, so `window` is the
    largest of them and `pixels` is half their count -- no need to re-supply
    the settings the pileup was built with. The result carries no
    per-chromosome matrices, so summaries computed from it cannot be
    bootstrapped; re-run `aggregate_apa` for that.
    """

    matrix = pl.read_csv(path)
    labels = [int(name) for name in matrix.columns if name != "bin_label"]
    if not labels:
        raise ValueError(f"{path} has no signed-offset matrix columns")
    signal = pl.DataFrame({"bin_label": labels, "contacts": [0] * len(labels)})
    return ApaResult(
        matrix=matrix,
        bait_signal=signal,
        prey_signal=signal,
        window=max(abs(label) for label in labels),
        pixels=len(labels) // 2,
    )


def compute_apa(
    indexes: dict[str, ContactIndex],
    baits: pl.DataFrame,
    preys: pl.DataFrame,
    *,
    min_distance: int,
    max_distance: int,
    window: int,
    pixels: int,
    pairs: pl.DataFrame | None = None,
    shift: int = 75,
    keep_chromosomes: bool = False,
    progress: bool | Instrumentation = False,
    profile: bool = False,
) -> ApaResult:
    """Compute APA matrix and 1D anchor signal from in-memory indexes and anchors.

    Pass `pairs` (a canonical pair table from `touche.pairs`) to pile up an
    explicit pair list instead of the distance-filtered product of `baits`
    and `preys`; `baits`/`preys` are then ignored.

    `keep_chromosomes=True` additionally retains each chromosome's own
    pileup on the result, which is what
    `touche.apa_masks.summarize_apa(..., bootstrap=N)` resamples to build
    confidence intervals. It costs one extra matrix per chromosome
    (a few megabytes at typical window/pixel settings), not one per pair.
    """

    if window % pixels != 0:
        raise ValueError("window must be divisible by pixels")
    instrument = make_instrumentation(progress, profile=profile)

    labels = _pixel_labels(window, pixels)
    n = len(labels)
    matrix_arr = np.zeros((n, n), dtype=np.int64)
    bait_signal_arr = np.zeros(n, dtype=np.int64)
    prey_signal_arr = np.zeros(n, dtype=np.int64)

    chrom_matrices: dict[str, np.ndarray] = {}
    explicit = split_pair_anchors(pairs) if pairs is not None else None
    chrom_list = (
        list(explicit) if explicit is not None else baits["chr"].unique(maintain_order=True).to_list()
    )
    chrom_iter = instrument.iter(
        chrom_list,
        total=len(chrom_list),
        desc="apa chromosomes",
        unit="chrom",
    )
    for chrom in chrom_iter:
        index = indexes.get(chrom)
        if index is None:
            continue
        anchors = explicit.get(chrom) if explicit is not None else None
        if explicit is not None and anchors is None:
            continue
        chrom_baits = anchors.baits if anchors else baits.filter(pl.col("chr") == chrom)
        chrom_preys = anchors.preys if anchors else preys.filter(pl.col("chr") == chrom)
        if chrom_preys.is_empty():
            continue

        pos_a, pos_b = _shifted_positions(index, shift=shift)
        long_range = np.abs(pos_b - pos_a) > (min_distance - window)

        # When per-chromosome pileups are wanted, accumulate into a scratch
        # matrix and fold it into the total afterwards, so the genome-wide
        # matrix stays bit-identical to the single-accumulator path.
        target = np.zeros((n, n), dtype=np.int64) if keep_chromosomes else matrix_arr
        _add_chrom_apa_numba(
            target,
            bait_signal_arr,
            prey_signal_arr,
            pos_a,
            pos_b,
            long_range,
            chrom_baits,
            chrom_preys,
            anchors,
            min_distance=min_distance,
            max_distance=max_distance,
            window=window,
            pixels=pixels,
        )
        if keep_chromosomes:
            matrix_arr += target
            chrom_matrices[str(chrom)] = target[::-1]

    matrix_df = _matrix_to_frame(list(reversed(labels)), labels, matrix_arr[::-1])
    bait_signal_df = pl.DataFrame({"bin_label": labels, "contacts": bait_signal_arr})
    prey_signal_df = pl.DataFrame({"bin_label": labels, "contacts": prey_signal_arr})
    return ApaResult(
        matrix=matrix_df,
        bait_signal=bait_signal_df,
        prey_signal=prey_signal_df,
        window=window,
        pixels=pixels,
        chrom_matrices=chrom_matrices if keep_chromosomes else None,
    )


def _add_chrom_apa_numba(
    matrix: np.ndarray,
    bait_signal: np.ndarray,
    prey_signal: np.ndarray,
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    long_range: np.ndarray,
    chrom_baits: pl.DataFrame,
    chrom_preys: pl.DataFrame,
    anchors: PairAnchors | None = None,
    *,
    min_distance: int,
    max_distance: int,
    window: int,
    pixels: int,
) -> None:
    """Accumulate one chromosome's contribution to `matrix`/`bait_signal`/`prey_signal` in place.

    Without `anchors`, baits are filtered to those with at least one prey
    candidate in `[min_distance, max_distance]` before the (expensive) numba
    matrix kernel runs, since most baits on a chromosome have none. With
    `anchors`, the pair grouping comes from the explicit pair list instead
    and no distance filter is applied.
    """
    bait_centers = chrom_baits["center"].to_numpy().astype(np.int64)
    bait_strands = _strand_codes(chrom_baits["strand"])
    prey_centers = chrom_preys["center"].to_numpy().astype(np.int64)
    prey_strands = _strand_codes(chrom_preys["strand"])

    order_a = np.argsort(pos_a, kind="stable")
    order_b = np.argsort(pos_b, kind="stable")
    sorted_pos_a = pos_a[order_a]
    sorted_pos_b = pos_b[order_b]

    if anchors is None:
        group_bait_indexes, group_starts, pair_prey_indexes = _distance_pair_groups(
            bait_centers, prey_centers, min_distance=min_distance, max_distance=max_distance
        )
    else:
        group_bait_indexes, group_starts, pair_prey_indexes = _explicit_pair_groups(
            anchors.bait_index, anchors.prey_index
        )
    active_bait = np.zeros(bait_centers.shape[0], dtype=bool)
    active_bait[group_bait_indexes] = True

    if active_bait.any():
        bait_values = _apa_anchor_signal_numba(
            pos_a,
            pos_b,
            sorted_pos_a,
            order_a,
            sorted_pos_b,
            order_b,
            bait_centers[active_bait],
            bait_strands[active_bait],
            long_range,
            window=window,
            pixels=pixels,
        )
        bait_signal += bait_values.sum(axis=0)

    prey_values = _apa_anchor_signal_numba(
        pos_a,
        pos_b,
        sorted_pos_a,
        order_a,
        sorted_pos_b,
        order_b,
        prey_centers,
        prey_strands,
        long_range,
        window=window,
        pixels=pixels,
    )
    prey_signal += prey_values.sum(axis=0)

    if group_bait_indexes:
        matrix_values = _apa_matrix_numba(
            pos_a,
            pos_b,
            sorted_pos_a,
            order_a,
            sorted_pos_b,
            order_b,
            bait_centers,
            bait_strands,
            prey_centers,
            prey_strands,
            np.asarray(group_bait_indexes, dtype=np.int64),
            np.asarray(group_starts, dtype=np.int64),
            np.asarray(pair_prey_indexes, dtype=np.int64),
            long_range,
            window=window,
            pixels=pixels,
        )
        matrix += matrix_values


def _distance_pair_groups(
    bait_centers: np.ndarray, prey_centers: np.ndarray, *, min_distance: int, max_distance: int
) -> tuple[list[int], list[int], list[int]]:
    """Bait/prey grouping from the implicit distance-filtered pair universe."""
    group_bait_indexes: list[int] = []
    group_starts: list[int] = [0]
    pair_prey_indexes: list[int] = []
    for bait_index, bait_center in enumerate(bait_centers):
        distances = np.abs(prey_centers - bait_center)
        candidate_indexes = np.flatnonzero((distances >= min_distance) & (distances <= max_distance))
        if len(candidate_indexes):
            group_bait_indexes.append(bait_index)
            pair_prey_indexes.extend(candidate_indexes.tolist())
            group_starts.append(len(pair_prey_indexes))
    return group_bait_indexes, group_starts, pair_prey_indexes


def _explicit_pair_groups(
    bait_index: np.ndarray, prey_index: np.ndarray
) -> tuple[list[int], list[int], list[int]]:
    """Bait/prey grouping from an explicit pair list, in the kernel's grouped-CSR layout."""
    order = np.argsort(bait_index, kind="mergesort")
    sorted_baits = bait_index[order]
    boundaries = np.flatnonzero(np.diff(sorted_baits)) + 1
    group_bait_indexes = sorted_baits[np.concatenate([[0], boundaries])].tolist() if order.size else []
    starts = np.concatenate([[0], boundaries, [order.size]]).tolist() if order.size else [0]
    return group_bait_indexes, [int(value) for value in starts], prey_index[order].tolist()


def _apa_anchor_signal_numba(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    sorted_pos_a: np.ndarray,
    order_a: np.ndarray,
    sorted_pos_b: np.ndarray,
    order_b: np.ndarray,
    centers: np.ndarray,
    strand_codes: np.ndarray,
    contact_mask: np.ndarray,
    *,
    window: int,
    pixels: int,
) -> np.ndarray:
    """Cast inputs to the dtypes `touche.numba.apa.apa_anchor_signal_numba` expects, then call it."""
    from touche.numba.apa import apa_anchor_signal_numba

    return apa_anchor_signal_numba(
        pos_a.astype(np.int64, copy=False),
        pos_b.astype(np.int64, copy=False),
        sorted_pos_a.astype(np.int64, copy=False),
        order_a.astype(np.int64, copy=False),
        sorted_pos_b.astype(np.int64, copy=False),
        order_b.astype(np.int64, copy=False),
        centers.astype(np.int64, copy=False),
        strand_codes.astype(np.int64, copy=False),
        contact_mask.astype(np.bool_, copy=False),
        int(window),
        int(pixels),
    )


def _apa_matrix_numba(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    sorted_pos_a: np.ndarray,
    order_a: np.ndarray,
    sorted_pos_b: np.ndarray,
    order_b: np.ndarray,
    bait_centers: np.ndarray,
    bait_strands: np.ndarray,
    prey_centers: np.ndarray,
    prey_strands: np.ndarray,
    group_bait_index: np.ndarray,
    group_start: np.ndarray,
    pair_prey_index: np.ndarray,
    long_range: np.ndarray,
    *,
    window: int,
    pixels: int,
) -> np.ndarray:
    """Cast inputs to the dtypes `touche.numba.apa.apa_matrix_numba` expects, then call it."""
    from numba import get_num_threads

    from touche.numba.apa import apa_matrix_numba

    return apa_matrix_numba(
        pos_a.astype(np.int64, copy=False),
        pos_b.astype(np.int64, copy=False),
        sorted_pos_a.astype(np.int64, copy=False),
        order_a.astype(np.int64, copy=False),
        sorted_pos_b.astype(np.int64, copy=False),
        order_b.astype(np.int64, copy=False),
        bait_centers.astype(np.int64, copy=False),
        bait_strands.astype(np.int64, copy=False),
        prey_centers.astype(np.int64, copy=False),
        prey_strands.astype(np.int64, copy=False),
        group_bait_index.astype(np.int64, copy=False),
        group_start.astype(np.int64, copy=False),
        pair_prey_index.astype(np.int64, copy=False),
        long_range.astype(np.bool_, copy=False),
        int(window),
        int(pixels),
        get_num_threads(),
    )


def _strand_codes(strands: pl.Series) -> np.ndarray:
    """Map a `"+"`/`"-"` strand series to `+1`/`-1`."""
    return np.where(strands.to_numpy() == "-", -1, 1).astype(np.int64)


def _resolve_cache_dir(cache_dir: str | Path | None, out_dir: str | Path) -> Path:
    """Default to a `contact_index_cache/` directory next to `out_dir`."""
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(out_dir) / "contact_index_cache"


def write_apa_result(
    result: ApaResult,
    out_dir: str | Path,
    *,
    reference_style: bool = True,
) -> dict[str, Path]:
    """Write an in-memory APA result using the reference output filenames."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "AggMat.csv"
    heatmap_path = out_dir / "AggHeatmap.svg"
    bait_signal_path = out_dir / "baits_genome_wide_contacts.csv"
    prey_signal_path = out_dir / "preys_genome_wide_contacts.csv"

    result.matrix.write_csv(matrix_path)
    result.bait_signal.write_csv(bait_signal_path)
    result.prey_signal.write_csv(prey_signal_path)
    fig = plot_raw_apa_heatmap(
        result.matrix,
        heatmap_path,
        window=result.window,
        pixels=result.pixels,
        reference_style=reference_style,
    )
    _close_figure(fig)

    return {
        "matrix": matrix_path,
        "heatmap": heatmap_path,
        "baits_signal": bait_signal_path,
        "preys_signal": prey_signal_path,
    }


def compare_apa_change(
    control_apa: str | Path,
    treatment_apa: str | Path,
    control_baits: str | Path,
    control_preys: str | Path,
    treatment_baits: str | Path,
    treatment_preys: str | Path,
    *,
    bait_count: int,
    prey_count: int,
    out: str | Path | None = None,
    matrix_out: str | Path | None = None,
    window: int = 10_000,
    pixels: int = 50,
    reference_style: bool = True,
) -> pl.DataFrame:
    """Calculate and optionally plot 1D-normalized inter-sample APA change."""

    control = pl.read_csv(control_apa)
    treatment = pl.read_csv(treatment_apa)

    control_rows, control_cols, control_values = _matrix_labels_and_values(control)
    treatment_rows, treatment_cols, treatment_values = _matrix_labels_and_values(treatment)
    treatment_values = _reindex_matrix(
        treatment_rows, treatment_cols, treatment_values, control_rows, control_cols
    )

    control_bait_signal = _read_signal(control_baits)
    treatment_bait_signal = _read_signal(treatment_baits)
    control_prey_signal = _read_signal(control_preys)
    treatment_prey_signal = _read_signal(treatment_preys)

    control_bait_values = _signal_values_by_label(control_bait_signal, control_cols) / bait_count
    treatment_bait_values = (
        _signal_values_by_label(treatment_bait_signal, control_cols) / bait_count
    )
    control_prey_values = _signal_values_by_label(control_prey_signal, control_rows) / prey_count
    treatment_prey_values = (
        _signal_values_by_label(treatment_prey_signal, control_rows) / prey_count
    )

    expected_change = (treatment_prey_values[:, None] + treatment_bait_values[None, :]) / (
        control_prey_values[:, None] + control_bait_values[None, :]
    )
    observed_change = treatment_values / control_values
    ratio = observed_change / expected_change
    ratio = np.where(np.isinf(ratio), np.nan, ratio)
    obs_over_exp = _matrix_to_frame(control_rows, control_cols, ratio)

    if matrix_out is not None:
        Path(matrix_out).parent.mkdir(parents=True, exist_ok=True)
        obs_over_exp.write_csv(matrix_out)

    if out is not None:
        fig = plot_apa_change(
            obs_over_exp, out, window=window, pixels=pixels, reference_style=reference_style
        )
        _close_figure(fig)

    return obs_over_exp


def plot_raw_apa_heatmap(
    matrix: pl.DataFrame,
    out: str | Path | None = None,
    *,
    window: int,
    pixels: int,
    reference_style: bool = True,
) -> "Figure":
    """Render an APA matrix (from `compute_apa`/`aggregate_apa`) as a heatmap; save to `out` if given."""
    import matplotlib

    matplotlib.use("Agg")
    import seaborn as sns

    values = matrix.drop("bin_label").to_numpy()
    cmap = sns.color_palette("YlOrRd") if reference_style else "viridis"
    ax = sns.heatmap(values, cmap=cmap, square=True)
    labels = [f"-{int(window / 1000)}kb", "0", f"{int(window / 1000)}kb"]
    ax.set_xticks([0, pixels, pixels * 2], labels)
    ax.set_yticks([0, pixels, pixels * 2], [labels[2], "0", labels[0]])
    ax.figure.tight_layout()
    if out is not None:
        ax.figure.savefig(out)
    return ax.figure


def plot_apa_change(
    matrix: pl.DataFrame,
    out: str | Path | None = None,
    *,
    window: int = 10_000,
    pixels: int = 50,
    reference_style: bool = True,
) -> "Figure":
    """Render an observed/expected APA-change matrix (from `compare_apa_change`) as a log2 heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import seaborn as sns

    values = matrix.drop("bin_label").to_numpy().astype(float)
    vmax = 1 if reference_style else None
    vmin = -1 if reference_style else None
    ax = sns.heatmap(np.log2(values), cmap="RdYlBu_r", square=True, vmax=vmax, vmin=vmin)
    labels = [f"-{int(window / 1000)}kb", "0", f"{int(window / 1000)}kb"]
    ax.set_xticks([0, pixels, pixels * 2], labels)
    ax.set_xlabel("Distane to Promoter TSS", size=16)
    ax.set_yticks([0, pixels, pixels * 2], [labels[2], "0", labels[0]])
    ax.set_ylabel("Distane to Enhancer TSS", size=16)
    ax.figure.tight_layout()
    if out is not None:
        ax.figure.savefig(out)
    return ax.figure


def _read_signal(path: str | Path) -> pl.DataFrame:
    """Read a `baits_genome_wide_contacts.csv`/`preys_genome_wide_contacts.csv` file."""
    frame = pl.read_csv(path)
    if "contacts" not in frame.columns:
        raise ValueError(f"Expected a 'contacts' column in {path}")
    return frame


def _matrix_labels_and_values(frame: pl.DataFrame) -> tuple[list[int], list[int], np.ndarray]:
    """Split an `AggMat.csv`-shaped frame into `(row labels, column labels, value matrix)`."""
    row_labels = [int(v) for v in frame["bin_label"].to_list()]
    col_names = [c for c in frame.columns if c != "bin_label"]
    values = frame.select(col_names).to_numpy().astype(float)
    col_labels = [int(c) for c in col_names]
    return row_labels, col_labels, values


def _reindex_matrix(
    row_labels: list[int],
    col_labels: list[int],
    values: np.ndarray,
    target_rows: list[int],
    target_cols: list[int],
) -> np.ndarray:
    """Reindex `values` onto `target_rows`/`target_cols`, filling missing labels with NaN.

    Used by `compare_apa_change` when control and treatment matrices were
    aggregated with slightly different bin labels (e.g. different `window`).
    """
    if row_labels == target_rows and col_labels == target_cols:
        return values
    row_index = {label: i for i, label in enumerate(row_labels)}
    col_index = {label: i for i, label in enumerate(col_labels)}
    out = np.full((len(target_rows), len(target_cols)), np.nan)
    for i, row_label in enumerate(target_rows):
        ri = row_index.get(row_label)
        if ri is None:
            continue
        for j, col_label in enumerate(target_cols):
            ci = col_index.get(col_label)
            if ci is not None:
                out[i, j] = values[ri, ci]
    return out


def _signal_values_by_label(frame: pl.DataFrame, labels: list[int]) -> np.ndarray:
    """Look up `frame`'s `contacts` values by `bin_label`, in the order given by `labels`."""
    lookup = dict(zip(frame["bin_label"].to_list(), frame["contacts"].to_list()))
    return np.array([lookup.get(label, np.nan) for label in labels], dtype=float)


def _matrix_to_frame(
    row_labels: list[int], col_labels: list[int], values: np.ndarray
) -> pl.DataFrame:
    """Build the `bin_label` + one-column-per-bin `pl.DataFrame` layout APA matrices are stored in."""
    data: dict[str, object] = {"bin_label": row_labels}
    for j, label in enumerate(col_labels):
        data[str(label)] = values[:, j]
    return pl.DataFrame(data)


def _shifted_positions(index: ContactIndex, *, shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Shift each read's position `shift` bp in its 5'->3' direction (matching the reference workflow)."""
    pos_a = np.where(_is_plus_strand(index.strand_a), index.pos_a + shift, index.pos_a - shift)
    pos_b = np.where(_is_plus_strand(index.strand_b), index.pos_b + shift, index.pos_b - shift)
    return pos_a.astype(np.int64), pos_b.astype(np.int64)


def _is_plus_strand(strands: np.ndarray) -> np.ndarray:
    """Accept either `ContactIndex`'s `+1`/`-1` codes or raw `"+"`/`"-"` strings."""
    if np.issubdtype(strands.dtype, np.integer):
        return strands > 0
    return strands == "+"


def _pixel_labels(window: int, pixels: int) -> list[int]:
    """Signed bin-center offsets from `-window` to `+window`, e.g. `[-10000, ..., -200, 200, ..., 10000]`."""
    step = window // pixels
    return list(range(-window, 0, step)) + list(range(step, window + 1, step))


def _close_figure(fig: "Figure") -> None:
    """Release a matplotlib figure's memory once it's been saved/embedded."""
    import matplotlib.pyplot as plt

    plt.close(fig)
