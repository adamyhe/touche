from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from touche.anchors import read_bed_anchors
from touche.contacts import build_contact_indexes


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
) -> dict[str, Path]:
    """Aggregate APA matrix and 1D anchor signal without per-bait temp files."""

    if window % pixels != 0:
        raise ValueError("window must be divisible by pixels")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = _pixel_labels(window, pixels)
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=np.int64)
    bait_signal = pd.DataFrame(0, index=labels, columns=["contacts"], dtype=np.int64)
    prey_signal = pd.DataFrame(0, index=labels, columns=["contacts"], dtype=np.int64)

    indexes = build_contact_indexes(pairs_path, source=source, cis_only=True)
    baits = read_bed_anchors(baits_path)
    preys = read_bed_anchors(preys_path)

    for chrom, chrom_baits in baits.groupby("chr", sort=False):
        index = indexes.get(chrom)
        if index is None:
            continue
        chrom_preys = preys.loc[preys["chr"] == chrom]
        if chrom_preys.empty:
            continue

        pos_a, pos_b = _shifted_positions(index, shift=shift)
        long_range = np.abs(pos_b - pos_a) > (min_distance - window)
        prey_centers = chrom_preys["center"].to_numpy(dtype=np.int64)

        for bait in chrom_baits.itertuples(index=False):
            distances = np.abs(prey_centers - bait.center)
            candidate_preys = chrom_preys.loc[
                (distances >= min_distance) & (distances <= max_distance)
            ]
            if candidate_preys.empty:
                continue
            _add_anchor_signal(
                bait_signal,
                pos_a,
                pos_b,
                center=int(bait.center),
                strand=str(bait.strand),
                window=window,
                pixels=pixels,
                mask=long_range,
            )
            bait_mask = (
                (_in_window(pos_a, int(bait.center) - window, int(bait.center) + window))
                | (_in_window(pos_b, int(bait.center) - window, int(bait.center) + window))
            ) & long_range
            for prey in candidate_preys.itertuples(index=False):
                _add_pair_matrix(
                    matrix,
                    pos_a,
                    pos_b,
                    bait_center=int(bait.center),
                    bait_strand=str(bait.strand),
                    prey_center=int(prey.center),
                    prey_strand=str(prey.strand),
                    window=window,
                    pixels=pixels,
                    mask=bait_mask,
                )

        for prey in chrom_preys.itertuples(index=False):
            _add_anchor_signal(
                prey_signal,
                pos_a,
                pos_b,
                center=int(prey.center),
                strand=str(prey.strand),
                window=window,
                pixels=pixels,
                mask=long_range,
            )

    agg_mat = matrix.iloc[::-1]
    matrix_path = out_dir / "AggMat.csv"
    heatmap_path = out_dir / "AggHeatmap.svg"
    bait_signal_path = out_dir / "baits_genome_wide_contacts.csv"
    prey_signal_path = out_dir / "preys_genome_wide_contacts.csv"

    agg_mat.to_csv(matrix_path)
    bait_signal.to_csv(bait_signal_path)
    prey_signal.to_csv(prey_signal_path)
    plot_raw_apa_heatmap(
        agg_mat, heatmap_path, window=window, pixels=pixels, reference_style=reference_style
    )

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
) -> pd.DataFrame:
    """Calculate and optionally plot 1D-normalized inter-sample APA change."""

    control = _read_matrix(control_apa)
    treatment = _read_matrix(treatment_apa)
    treatment = treatment.reindex(index=control.index, columns=control.columns)

    control_bait_signal = _read_signal(control_baits) / bait_count
    treatment_bait_signal = _read_signal(treatment_baits) / bait_count
    control_prey_signal = _read_signal(control_preys) / prey_count
    treatment_prey_signal = _read_signal(treatment_preys) / prey_count

    control_bait_values = control_bait_signal.reindex(control.columns).to_numpy(dtype=float)
    treatment_bait_values = treatment_bait_signal.reindex(control.columns).to_numpy(dtype=float)
    control_prey_values = control_prey_signal.reindex(control.index).to_numpy(dtype=float)
    treatment_prey_values = treatment_prey_signal.reindex(control.index).to_numpy(dtype=float)

    expected_change = (treatment_prey_values[:, None] + treatment_bait_values[None, :]) / (
        control_prey_values[:, None] + control_bait_values[None, :]
    )
    observed_change = treatment.to_numpy(dtype=float) / control.to_numpy(dtype=float)
    obs_over_exp = pd.DataFrame(
        observed_change / expected_change,
        index=control.index,
        columns=control.columns,
    ).replace([np.inf, -np.inf], np.nan)

    if matrix_out is not None:
        Path(matrix_out).parent.mkdir(parents=True, exist_ok=True)
        obs_over_exp.to_csv(matrix_out)

    if out is not None:
        plot_apa_change(
            obs_over_exp, out, window=window, pixels=pixels, reference_style=reference_style
        )

    return obs_over_exp


def plot_raw_apa_heatmap(
    matrix: pd.DataFrame,
    out: str | Path,
    *,
    window: int,
    pixels: int,
    reference_style: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cmap = sns.color_palette("YlOrRd") if reference_style else "viridis"
    ax = sns.heatmap(matrix, cmap=cmap, square=True)
    labels = [f"-{int(window / 1000)}kb", "0", f"{int(window / 1000)}kb"]
    ax.set_xticks([0, pixels, pixels * 2], labels)
    ax.set_yticks([0, pixels, pixels * 2], [labels[2], "0", labels[0]])
    ax.figure.tight_layout()
    ax.figure.savefig(out)
    plt.close(ax.figure)


def plot_apa_change(
    matrix: pd.DataFrame,
    out: str | Path,
    *,
    window: int = 10_000,
    pixels: int = 50,
    reference_style: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    vmax = 1 if reference_style else None
    vmin = -1 if reference_style else None
    ax = sns.heatmap(
        np.log2(matrix.astype(float)), cmap="RdYlBu_r", square=True, vmax=vmax, vmin=vmin
    )
    labels = [f"-{int(window / 1000)}kb", "0", f"{int(window / 1000)}kb"]
    ax.set_xticks([0, pixels, pixels * 2], labels)
    ax.set_xlabel("Distane to Promoter TSS", size=16)
    ax.set_yticks([0, pixels, pixels * 2], [labels[2], "0", labels[0]])
    ax.set_ylabel("Distane to Enhancer TSS", size=16)
    ax.figure.tight_layout()
    ax.figure.savefig(out)
    plt.close(ax.figure)


def _read_matrix(path: str | Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)
    return matrix


def _read_signal(path: str | Path) -> pd.Series:
    data = pd.read_csv(path, index_col=0)
    if "contacts" not in data.columns:
        raise ValueError(f"Expected a 'contacts' column in {path}")
    data.index = data.index.astype(int)
    return data["contacts"].astype(float)


def _shifted_positions(index, *, shift: int) -> tuple[np.ndarray, np.ndarray]:
    pos_a = np.where(index.strand_a == "+", index.pos_a + shift, index.pos_a - shift)
    pos_b = np.where(index.strand_b == "+", index.pos_b + shift, index.pos_b - shift)
    return pos_a.astype(np.int64), pos_b.astype(np.int64)


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
    signal: pd.DataFrame,
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    *,
    center: int,
    strand: str,
    window: int,
    pixels: int,
    mask: np.ndarray,
) -> None:
    labels = signal.index.to_list()
    for label, (start, end) in zip(
        labels, _oriented_bins(center, strand, window=window, pixels=pixels)
    ):
        count = np.count_nonzero(
            mask & (_in_window(pos_a, start, end) | _in_window(pos_b, start, end))
        )
        signal.loc[label, "contacts"] += int(count)


def _add_pair_matrix(
    matrix: pd.DataFrame,
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
    labels = matrix.index.to_list()
    bait_bins = _oriented_bins(bait_center, bait_strand, window=window, pixels=pixels)
    prey_bins = _oriented_bins(prey_center, prey_strand, window=window, pixels=pixels)
    masked_a = pos_a[mask]
    masked_b = pos_b[mask]
    for col_label, (bait_start, bait_end) in zip(labels, bait_bins):
        side_a_in_bait = _in_window(masked_a, bait_start, bait_end)
        side_b_in_bait = _in_window(masked_b, bait_start, bait_end)
        if not np.any(side_a_in_bait | side_b_in_bait):
            continue
        for row_label, (prey_start, prey_end) in zip(labels, prey_bins):
            side_a_in_prey = _in_window(masked_a, prey_start, prey_end)
            side_b_in_prey = _in_window(masked_b, prey_start, prey_end)
            count = np.count_nonzero(
                (side_a_in_bait & side_b_in_prey) | (side_b_in_bait & side_a_in_prey)
            )
            if count:
                matrix.loc[row_label, col_label] += int(count)
