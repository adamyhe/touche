from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from touche.anchors import read_bed_anchors
from touche.backends import validate_backend
from touche.contacts import build_contact_indexes
from touche.instrumentation import Instrumentation, make_instrumentation
from touche.models import ContactIndex, NamedDepth, NamedPath

if TYPE_CHECKING:
    from matplotlib.figure import Figure

BACKGROUND_COLUMNS = ["chr", "promoter", "enhancer", "EP_contacts", "BG_contacts"]
PAIR_COLUMNS = ["chr", "promoter", "enhancer"]
_BACKGROUND_SCHEMA = {
    "chr": pl.Utf8,
    "promoter": pl.Int64,
    "enhancer": pl.Int64,
    "EP_contacts": pl.Int64,
    "BG_contacts": pl.Int64,
}


def count_ep_and_background(
    pairs_path: str | Path,
    baits_path: str | Path,
    preys_path: str | Path,
    out_path: str | Path,
    *,
    min_distance: int,
    max_distance: int,
    window: int,
    min_bg_distance: int,
    max_bg_distance: int,
    source: str = "auto",
    backend: str = "numpy",
    progress: bool | Instrumentation = False,
    profile: bool = False,
) -> pl.DataFrame:
    """Count anchor-to-anchor and local-background contacts for bait/prey pairs."""

    instrument = make_instrumentation(progress, profile=profile)
    with instrument.step("read inputs"):
        indexes = build_contact_indexes(
            pairs_path,
            source=source,
            cis_only=True,
            include_metadata=False,
        )
        baits = read_bed_anchors(baits_path)
        preys = read_bed_anchors(preys_path)
    result = compute_ep_and_background(
        indexes,
        baits,
        preys,
        min_distance=min_distance,
        max_distance=max_distance,
        window=window,
        min_bg_distance=min_bg_distance,
        max_bg_distance=max_bg_distance,
        backend=backend,
        progress=instrument,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with instrument.step("write counts"):
        result.write_csv(out_path, include_header=False, separator="\t")
    return result


def compute_ep_and_background(
    indexes: dict[str, ContactIndex],
    baits: pl.DataFrame,
    preys: pl.DataFrame,
    *,
    min_distance: int,
    max_distance: int,
    window: int,
    min_bg_distance: int,
    max_bg_distance: int,
    backend: str = "numpy",
    progress: bool | Instrumentation = False,
    profile: bool = False,
) -> pl.DataFrame:
    """Count EP and local-background contacts from in-memory indexes and anchors."""

    backend = validate_backend(backend)
    instrument = make_instrumentation(progress, profile=profile)
    rows: list[dict[str, object]] = []
    chrom_list = baits["chr"].unique(maintain_order=True).to_list()

    chrom_iter = instrument.iter(
        chrom_list,
        total=len(chrom_list),
        desc="background chromosomes",
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
        prey_centers = chrom_preys["center"].to_numpy().astype(np.int64)
        bait_centers = chrom_baits["center"].to_numpy().astype(np.int64)
        pair_bait_indexes: list[int] = []
        pair_prey_indexes: list[int] = []
        for bait_index, bait_center in enumerate(bait_centers):
            distances = np.abs(prey_centers - bait_center)
            candidate_indexes = np.flatnonzero(
                (distances >= min_distance) & (distances <= max_distance)
            )
            pair_bait_indexes.extend([bait_index] * len(candidate_indexes))
            pair_prey_indexes.extend(candidate_indexes.tolist())
        if not pair_bait_indexes:
            continue

        if backend == "numba":
            ep_counts, bg_counts = _count_ep_background_pairs_numba(
                index.pos_a,
                index.pos_b,
                bait_centers,
                prey_centers,
                np.asarray(pair_bait_indexes, dtype=np.int64),
                np.asarray(pair_prey_indexes, dtype=np.int64),
                window=window,
                min_bg_distance=min_bg_distance,
                max_bg_distance=max_bg_distance,
            )
            for pair_index, (bait_index, prey_index) in enumerate(
                zip(pair_bait_indexes, pair_prey_indexes, strict=True)
            ):
                rows.append(
                    {
                        "chr": chrom,
                        "promoter": int(bait_centers[bait_index]),
                        "enhancer": int(prey_centers[prey_index]),
                        "EP_contacts": int(ep_counts[pair_index]),
                        "BG_contacts": int(bg_counts[pair_index]),
                    }
                )
            continue

        for bait_index, prey_index in zip(pair_bait_indexes, pair_prey_indexes, strict=True):
            bait_center = int(bait_centers[bait_index])
            prey_center = int(prey_centers[prey_index])
            ep_contacts = _count_between_windows(
                index.pos_a,
                index.pos_b,
                bait_center - window,
                bait_center + window,
                prey_center - window,
                prey_center + window,
            )
            bait_to_prey_bg = _count_anchor_to_background(
                index.pos_a,
                index.pos_b,
                bait_center,
                prey_center,
                window=window,
                min_bg_distance=min_bg_distance,
                max_bg_distance=max_bg_distance,
            )
            prey_to_bait_bg = _count_anchor_to_background(
                index.pos_a,
                index.pos_b,
                prey_center,
                bait_center,
                window=window,
                min_bg_distance=min_bg_distance,
                max_bg_distance=max_bg_distance,
            )
            rows.append(
                {
                    "chr": chrom,
                    "promoter": bait_center,
                    "enhancer": prey_center,
                    "EP_contacts": ep_contacts,
                    "BG_contacts": bait_to_prey_bg + prey_to_bait_bg,
                }
            )

    return pl.DataFrame(rows, schema=_BACKGROUND_SCHEMA)


def _count_ep_background_pairs_numba(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    bait_centers: np.ndarray,
    prey_centers: np.ndarray,
    pair_bait_index: np.ndarray,
    pair_prey_index: np.ndarray,
    *,
    window: int,
    min_bg_distance: int,
    max_bg_distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    from touche.numba_kernels import count_ep_background_pairs_numba

    return count_ep_background_pairs_numba(
        pos_a.astype(np.int64, copy=False),
        pos_b.astype(np.int64, copy=False),
        bait_centers.astype(np.int64, copy=False),
        prey_centers.astype(np.int64, copy=False),
        pair_bait_index.astype(np.int64, copy=False),
        pair_prey_index.astype(np.int64, copy=False),
        int(window),
        int(min_bg_distance),
        int(max_bg_distance),
    )


def compare_background_ratios(
    control: NamedPath,
    treatments: list[NamedPath],
    depths: dict[str, int],
    *,
    min_ep_cpb: float = 8.0,
    out_dir: str | Path | None = None,
    table_out: str | Path | None = None,
    reference_style: bool = True,
) -> tuple[pl.DataFrame, dict[str, Path]]:
    """Compare EP/background ratios across control and treatment samples."""

    if not treatments:
        raise ValueError("At least one treatment sample is required")
    sample_order = [control.name] + [sample.name for sample in treatments]
    missing_depths = [sample for sample in sample_order if sample not in depths]
    if missing_depths:
        raise ValueError(f"Missing sequencing depths for samples: {', '.join(missing_depths)}")

    merged = _merge_samples([control, *treatments])
    for sample in sample_order:
        depth_scale = depths[sample] / 10_000_000_000
        merged = merged.with_columns(
            (pl.col(f"EP_contacts_{sample}") / depth_scale).alias(f"EP_CPB_{sample}")
        )

    passes_threshold = pl.any_horizontal(
        [pl.col(f"EP_CPB_{sample}") > min_ep_cpb for sample in sample_order]
    )
    positive_all = pl.all_horizontal([pl.col(f"EP_CPB_{sample}") > 0 for sample in sample_order])
    merged = merged.filter(passes_threshold & positive_all)

    numeric_columns = [c for c, dtype in zip(merged.columns, merged.dtypes) if dtype.is_numeric()]
    merged = merged.with_columns(
        [
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c in numeric_columns
        ]
    ).drop_nulls()

    if table_out is not None:
        Path(table_out).parent.mkdir(parents=True, exist_ok=True)
        merged.write_csv(table_out, separator="\t")

    plot_paths: dict[str, Path] = {}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for left, right in _reference_pair_order(
            control.name, [sample.name for sample in treatments]
        ):
            path = out_dir / f"{right}_vs_{left}.svg"
            fig = plot_background_scatter(
                merged,
                x_sample=left,
                y_sample=right,
                out_path=path,
                reference_style=reference_style,
            )
            _close_figure(fig)
            plot_paths[f"{right}_vs_{left}"] = path
    return merged, plot_paths


def _in_window(values: np.ndarray, start: int, end: int) -> np.ndarray:
    return (values >= start) & (values <= end)


def _in_background(
    values: np.ndarray, center: int, min_distance: int, max_distance: int
) -> np.ndarray:
    left = _in_window(values, center - max_distance, center - min_distance)
    right = _in_window(values, center + min_distance, center + max_distance)
    return left | right


def _count_between_windows(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    start_1: int,
    end_1: int,
    start_2: int,
    end_2: int,
) -> int:
    side_a_in_1 = _in_window(pos_a, start_1, end_1)
    side_b_in_1 = _in_window(pos_b, start_1, end_1)
    side_a_in_2 = _in_window(pos_a, start_2, end_2)
    side_b_in_2 = _in_window(pos_b, start_2, end_2)
    return int(np.count_nonzero((side_a_in_1 & side_b_in_2) | (side_b_in_1 & side_a_in_2)))


def _count_anchor_to_background(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    anchor_center: int,
    background_center: int,
    *,
    window: int,
    min_bg_distance: int,
    max_bg_distance: int,
) -> int:
    side_a_in_anchor = _in_window(pos_a, anchor_center - window, anchor_center + window)
    side_b_in_anchor = _in_window(pos_b, anchor_center - window, anchor_center + window)
    side_a_in_bg = _in_background(pos_a, background_center, min_bg_distance, max_bg_distance)
    side_b_in_bg = _in_background(pos_b, background_center, min_bg_distance, max_bg_distance)
    return int(
        np.count_nonzero((side_a_in_anchor & side_b_in_bg) | (side_b_in_anchor & side_a_in_bg))
    )


def plot_background_scatter(
    data: pl.DataFrame,
    *,
    x_sample: str,
    y_sample: str,
    out_path: str | Path | None = None,
    reference_style: bool = True,
) -> "Figure":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    x_col = f"ratio_{x_sample}"
    y_col = f"ratio_{y_sample}"
    plot_data = data.filter((pl.col(x_col) > 0) & (pl.col(y_col) > 0))
    x_values = plot_data[x_col].to_numpy()
    y_values = plot_data[y_col].to_numpy()
    values = np.vstack([x_values, y_values])
    colors = _safe_kde(values)

    figsize = (8, 8)
    label_size = 30
    tick_size = 20
    if not reference_style:
        label_size = 14
        tick_size = 10

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        x=x_values,
        y=y_values,
        c=colors,
        cmap="jet",
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ax.get_ylim())
    xpoints = ypoints = ax.get_xlim()
    ax.plot(xpoints, ypoints, linestyle="--", color="k", lw=3, scalex=False, scaley=False)
    ax.set_ylabel(y_sample, size=label_size)
    ax.set_xlabel(x_sample, size=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path)
    return fig


def parse_named_path(value: str) -> NamedPath:
    name, raw_path = _split_name_value(value)
    return NamedPath(name=name, path=Path(raw_path))


def parse_named_depth(value: str) -> NamedDepth:
    name, raw_depth = _split_name_value(value)
    return NamedDepth(name=name, depth=int(raw_depth))


def _read_background(path: str | Path, sample: str) -> pl.DataFrame:
    data = pl.read_csv(path, separator="\t", has_header=False, new_columns=BACKGROUND_COLUMNS)
    data = data.with_columns(
        (pl.col("EP_contacts") / pl.col("BG_contacts")).alias(f"ratio_{sample}")
    )
    return data.rename(
        {
            "EP_contacts": f"EP_contacts_{sample}",
            "BG_contacts": f"BG_contacts_{sample}",
        }
    )


def _merge_samples(samples: list[NamedPath]) -> pl.DataFrame:
    merged = _read_background(samples[0].path, samples[0].name)
    for sample in samples[1:]:
        merged = merged.join(_read_background(sample.path, sample.name), how="inner", on=PAIR_COLUMNS)
    return merged


def _reference_pair_order(control: str, treatments: list[str]) -> list[tuple[str, str]]:
    pairs = [(control, treatment) for treatment in treatments]
    pairs.extend(combinations(treatments, 2))
    return pairs


def _safe_kde(values: np.ndarray) -> np.ndarray:
    if values.shape[1] < 2:
        return np.ones(values.shape[1])
    try:
        return gaussian_kde(values)(values)
    except np.linalg.LinAlgError:
        return np.ones(values.shape[1])


def _split_name_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=VALUE, got {value!r}")
    name, raw_value = value.split("=", 1)
    if not name or not raw_value:
        raise ValueError(f"Expected NAME=VALUE, got {value!r}")
    return name, raw_value


def _close_figure(fig: "Figure") -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)
