from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from touche.anchors import read_bed_anchors
from touche.backends import validate_backend
from touche.contacts import build_contact_indexes
from touche.instrumentation import Instrumentation, make_instrumentation
from touche.models import ContactIndex

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

    def plot(self, *, reference_style: bool = True) -> "Figure":
        return plot_raw_apa_heatmap(
            self.matrix,
            window=self.window,
            pixels=self.pixels,
            reference_style=reference_style,
        )

    def write(self, out_dir: str | Path, *, reference_style: bool = True) -> dict[str, Path]:
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
    source: str = "auto",
    shift: int = 75,
    reference_style: bool = True,
    backend: str = "numpy",
    progress: bool | Instrumentation = False,
    profile: bool = False,
) -> dict[str, Path]:
    """Aggregate APA matrix and 1D anchor signal without per-bait temp files."""

    instrument = make_instrumentation(progress, profile=profile)
    with instrument.step("read inputs"):
        indexes = build_contact_indexes(pairs_path, source=source, cis_only=True)
        baits = read_bed_anchors(baits_path)
        preys = read_bed_anchors(preys_path)
    result = compute_apa(
        indexes,
        baits,
        preys,
        min_distance=min_distance,
        max_distance=max_distance,
        window=window,
        pixels=pixels,
        shift=shift,
        backend=backend,
        progress=instrument,
    )
    with instrument.step("write apa outputs"):
        return write_apa_result(result, out_dir, reference_style=reference_style)


def compute_apa(
    indexes: dict[str, ContactIndex],
    baits: pl.DataFrame,
    preys: pl.DataFrame,
    *,
    min_distance: int,
    max_distance: int,
    window: int,
    pixels: int,
    shift: int = 75,
    backend: str = "numpy",
    progress: bool | Instrumentation = False,
    profile: bool = False,
) -> ApaResult:
    """Compute APA matrix and 1D anchor signal from in-memory indexes and anchors."""

    if window % pixels != 0:
        raise ValueError("window must be divisible by pixels")
    backend = validate_backend(backend)
    instrument = make_instrumentation(progress, profile=profile)

    labels = _pixel_labels(window, pixels)
    n = len(labels)
    matrix_arr = np.zeros((n, n), dtype=np.int64)
    bait_signal_arr = np.zeros(n, dtype=np.int64)
    prey_signal_arr = np.zeros(n, dtype=np.int64)

    chrom_list = baits["chr"].unique(maintain_order=True).to_list()
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
        chrom_baits = baits.filter(pl.col("chr") == chrom)
        chrom_preys = preys.filter(pl.col("chr") == chrom)
        if chrom_preys.is_empty():
            continue

        pos_a, pos_b = _shifted_positions(index, shift=shift)
        long_range = np.abs(pos_b - pos_a) > (min_distance - window)
        prey_centers = chrom_preys["center"].to_numpy().astype(np.int64)
        prey_strands = chrom_preys["strand"].to_list()

        if backend == "numba":
            _add_chrom_apa_numba(
                matrix_arr,
                bait_signal_arr,
                prey_signal_arr,
                pos_a,
                pos_b,
                long_range,
                chrom_baits,
                chrom_preys,
                min_distance=min_distance,
                max_distance=max_distance,
                window=window,
                pixels=pixels,
            )
            continue

        bait_centers = chrom_baits["center"].to_numpy().astype(np.int64)
        bait_strands = chrom_baits["strand"].to_list()
        for bait_idx in range(len(bait_centers)):
            bait_center = int(bait_centers[bait_idx])
            bait_strand = bait_strands[bait_idx]
            distances = np.abs(prey_centers - bait_center)
            candidate_mask = (distances >= min_distance) & (distances <= max_distance)
            if not np.any(candidate_mask):
                continue
            _add_anchor_signal(
                bait_signal_arr,
                pos_a,
                pos_b,
                center=bait_center,
                strand=bait_strand,
                window=window,
                pixels=pixels,
                mask=long_range,
            )
            bait_mask = (
                _in_window(pos_a, bait_center - window, bait_center + window)
                | _in_window(pos_b, bait_center - window, bait_center + window)
            ) & long_range
            for prey_idx in np.flatnonzero(candidate_mask):
                _add_pair_matrix(
                    matrix_arr,
                    pos_a,
                    pos_b,
                    bait_center=bait_center,
                    bait_strand=bait_strand,
                    prey_center=int(prey_centers[prey_idx]),
                    prey_strand=prey_strands[prey_idx],
                    window=window,
                    pixels=pixels,
                    mask=bait_mask,
                )

        for prey_idx in range(len(prey_centers)):
            _add_anchor_signal(
                prey_signal_arr,
                pos_a,
                pos_b,
                center=int(prey_centers[prey_idx]),
                strand=prey_strands[prey_idx],
                window=window,
                pixels=pixels,
                mask=long_range,
            )

    matrix_df = _matrix_to_frame(list(reversed(labels)), labels, matrix_arr[::-1])
    bait_signal_df = pl.DataFrame({"bin_label": labels, "contacts": bait_signal_arr})
    prey_signal_df = pl.DataFrame({"bin_label": labels, "contacts": prey_signal_arr})
    return ApaResult(
        matrix=matrix_df,
        bait_signal=bait_signal_df,
        prey_signal=prey_signal_df,
        window=window,
        pixels=pixels,
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
    *,
    min_distance: int,
    max_distance: int,
    window: int,
    pixels: int,
) -> None:
    bait_centers = chrom_baits["center"].to_numpy().astype(np.int64)
    bait_strands = _strand_codes(chrom_baits["strand"])
    prey_centers = chrom_preys["center"].to_numpy().astype(np.int64)
    prey_strands = _strand_codes(chrom_preys["strand"])

    pair_bait_indexes: list[int] = []
    pair_prey_indexes: list[int] = []
    active_bait = np.zeros(bait_centers.shape[0], dtype=bool)
    for bait_index, bait_center in enumerate(bait_centers):
        distances = np.abs(prey_centers - bait_center)
        candidate_indexes = np.flatnonzero(
            (distances >= min_distance) & (distances <= max_distance)
        )
        if len(candidate_indexes):
            active_bait[bait_index] = True
            pair_bait_indexes.extend([bait_index] * len(candidate_indexes))
            pair_prey_indexes.extend(candidate_indexes.tolist())

    if active_bait.any():
        bait_values = _apa_anchor_signal_numba(
            pos_a,
            pos_b,
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
        prey_centers,
        prey_strands,
        long_range,
        window=window,
        pixels=pixels,
    )
    prey_signal += prey_values.sum(axis=0)

    if pair_bait_indexes:
        matrix_values = _apa_matrix_numba(
            pos_a,
            pos_b,
            bait_centers,
            bait_strands,
            prey_centers,
            prey_strands,
            np.asarray(pair_bait_indexes, dtype=np.int64),
            np.asarray(pair_prey_indexes, dtype=np.int64),
            long_range,
            window=window,
            pixels=pixels,
        )
        matrix += matrix_values


def _apa_anchor_signal_numba(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    centers: np.ndarray,
    strand_codes: np.ndarray,
    contact_mask: np.ndarray,
    *,
    window: int,
    pixels: int,
) -> np.ndarray:
    from touche.numba_kernels import apa_anchor_signal_numba

    return apa_anchor_signal_numba(
        pos_a.astype(np.int64, copy=False),
        pos_b.astype(np.int64, copy=False),
        centers.astype(np.int64, copy=False),
        strand_codes.astype(np.int64, copy=False),
        contact_mask.astype(np.bool_, copy=False),
        int(window),
        int(pixels),
    )


def _apa_matrix_numba(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    bait_centers: np.ndarray,
    bait_strands: np.ndarray,
    prey_centers: np.ndarray,
    prey_strands: np.ndarray,
    pair_bait_index: np.ndarray,
    pair_prey_index: np.ndarray,
    long_range: np.ndarray,
    *,
    window: int,
    pixels: int,
) -> np.ndarray:
    from touche.numba_kernels import apa_matrix_numba

    return apa_matrix_numba(
        pos_a.astype(np.int64, copy=False),
        pos_b.astype(np.int64, copy=False),
        bait_centers.astype(np.int64, copy=False),
        bait_strands.astype(np.int64, copy=False),
        prey_centers.astype(np.int64, copy=False),
        prey_strands.astype(np.int64, copy=False),
        pair_bait_index.astype(np.int64, copy=False),
        pair_prey_index.astype(np.int64, copy=False),
        long_range.astype(np.bool_, copy=False),
        int(window),
        int(pixels),
    )


def _strand_codes(strands: pl.Series) -> np.ndarray:
    return np.where(strands.to_numpy() == "-", -1, 1).astype(np.int64)


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
    frame = pl.read_csv(path)
    if "contacts" not in frame.columns:
        raise ValueError(f"Expected a 'contacts' column in {path}")
    return frame


def _matrix_labels_and_values(frame: pl.DataFrame) -> tuple[list[int], list[int], np.ndarray]:
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
    lookup = dict(zip(frame["bin_label"].to_list(), frame["contacts"].to_list()))
    return np.array([lookup.get(label, np.nan) for label in labels], dtype=float)


def _matrix_to_frame(
    row_labels: list[int], col_labels: list[int], values: np.ndarray
) -> pl.DataFrame:
    data: dict[str, object] = {"bin_label": row_labels}
    for j, label in enumerate(col_labels):
        data[str(label)] = values[:, j]
    return pl.DataFrame(data)


def _shifted_positions(index: ContactIndex, *, shift: int) -> tuple[np.ndarray, np.ndarray]:
    pos_a = np.where(_is_plus_strand(index.strand_a), index.pos_a + shift, index.pos_a - shift)
    pos_b = np.where(_is_plus_strand(index.strand_b), index.pos_b + shift, index.pos_b - shift)
    return pos_a.astype(np.int64), pos_b.astype(np.int64)


def _is_plus_strand(strands: np.ndarray) -> np.ndarray:
    if np.issubdtype(strands.dtype, np.integer):
        return strands > 0
    return strands == "+"


def _pixel_labels(window: int, pixels: int) -> list[int]:
    step = window // pixels
    return list(range(-window, 0, step)) + list(range(step, window + 1, step))


def _oriented_bins(center: int, strand: str, *, window: int, pixels: int) -> list[tuple[int, int]]:
    step = window // pixels
    if strand == "-":
        return [(start - step, start) for start in range(center + window, center - window, -step)]
    return [(start, start + step) for start in range(center - window, center + window, step)]


def _in_window(values: np.ndarray, start: int, end: int) -> np.ndarray:
    return (values >= start) & (values <= end)


def _add_anchor_signal(
    signal: np.ndarray,
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    *,
    center: int,
    strand: str,
    window: int,
    pixels: int,
    mask: np.ndarray,
) -> None:
    bins = _oriented_bins(center, strand, window=window, pixels=pixels)
    for i, (start, end) in enumerate(bins):
        count = np.count_nonzero(
            mask & (_in_window(pos_a, start, end) | _in_window(pos_b, start, end))
        )
        signal[i] += count


def _add_pair_matrix(
    matrix: np.ndarray,
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    *,
    bait_center: int,
    bait_strand: str,
    prey_center: int,
    prey_strand: str,
    window: int,
    pixels: int,
    mask: np.ndarray,
) -> None:
    bait_bins = _oriented_bins(bait_center, bait_strand, window=window, pixels=pixels)
    prey_bins = _oriented_bins(prey_center, prey_strand, window=window, pixels=pixels)
    masked_a = pos_a[mask]
    masked_b = pos_b[mask]
    for col_i, (bait_start, bait_end) in enumerate(bait_bins):
        side_a_in_bait = _in_window(masked_a, bait_start, bait_end)
        side_b_in_bait = _in_window(masked_b, bait_start, bait_end)
        if not np.any(side_a_in_bait | side_b_in_bait):
            continue
        for row_i, (prey_start, prey_end) in enumerate(prey_bins):
            side_a_in_prey = _in_window(masked_a, prey_start, prey_end)
            side_b_in_prey = _in_window(masked_b, prey_start, prey_end)
            count = np.count_nonzero(
                (side_a_in_bait & side_b_in_prey) | (side_b_in_bait & side_a_in_prey)
            )
            if count:
                matrix[row_i, col_i] += count


def _close_figure(fig: "Figure") -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)
