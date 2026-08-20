"""Enhancer-promoter (EP) vs. local-background contact counting and cross-sample comparison.

Public API: `count_ep_and_background` (file-driven wrapper), `compute_ep_and_background`
(in-memory compute), `compare_background_ratios`, `plot_background_scatter`,
`parse_named_path`, `parse_named_depth`. Everything prefixed `_` is internal,
including `_count_ep_background_pairs_numba`, which wraps the Numba counting
kernel in `touche.numba.background` that `compute_ep_and_background` always
uses. `_safe_kde` is the one place `scipy.stats.gaussian_kde` is called --
only for the scatter plot's point-density coloring, not the numeric output --
and fits on a capped random subsample at large point counts to avoid
`gaussian_kde`'s O(n^2) evaluation cost, still coloring every point.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from touche.anchors import read_bed_anchors
from touche.contacts import build_contact_indexes, load_cached_contact_indexes
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
    index_strategy: str = "all",
    cache_dir: str | Path | None = None,
    cache_prefix: str = "contacts",
    require_cache: bool = False,
    progress: bool | Instrumentation = False,
    profile: bool = False,
) -> pl.DataFrame:
    """Count anchor-to-anchor and local-background contacts for bait/prey pairs.

    `index_strategy="cache"` reads a persistent NPZ `ContactIndex` cache
    (building it first if missing) instead of re-parsing `pairs_path` --
    useful when `apa aggregate` is also run against the same sample, since
    both would otherwise each pay their own full pairs-file parse. The
    cache only needs positions (no strand/mapq), but a cache shared with
    `apa aggregate` may include metadata anyway -- harmless, just ignored.
    """

    if index_strategy not in {"all", "cache"}:
        raise ValueError("index_strategy must be one of: all, cache")

    instrument = make_instrumentation(progress, profile=profile)
    with instrument.step("read inputs"):
        if index_strategy == "cache":
            cache_dir = _resolve_cache_dir(cache_dir, out_path)
            indexes = load_cached_contact_indexes(
                pairs_path,
                cache_dir=cache_dir,
                cache_prefix=cache_prefix,
                source=source,
                include_metadata=False,
                require_cache=require_cache,
            )
        else:
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
    progress: bool | Instrumentation = False,
    profile: bool = False,
) -> pl.DataFrame:
    """Count EP and local-background contacts from in-memory indexes and anchors."""

    instrument = make_instrumentation(progress, profile=profile)
    frames: list[pl.DataFrame] = []
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
        pair_bait_indexes, pair_prey_indexes = _candidate_pair_indexes(
            bait_centers,
            prey_centers,
            min_distance=min_distance,
            max_distance=max_distance,
        )
        if not len(pair_bait_indexes):
            continue

        ep_counts, bg_counts = _count_ep_background_pairs_numba(
            index.pos_a,
            index.pos_b,
            bait_centers,
            prey_centers,
            pair_bait_indexes,
            pair_prey_indexes,
            window=window,
            min_bg_distance=min_bg_distance,
            max_bg_distance=max_bg_distance,
        )
        frames.append(
            pl.DataFrame(
                {
                    "chr": [chrom] * len(pair_bait_indexes),
                    "promoter": bait_centers[pair_bait_indexes],
                    "enhancer": prey_centers[pair_prey_indexes],
                    "EP_contacts": ep_counts,
                    "BG_contacts": bg_counts,
                },
                schema=_BACKGROUND_SCHEMA,
            )
        )

    if not frames:
        return pl.DataFrame(schema=_BACKGROUND_SCHEMA)
    return pl.concat(frames)


def _candidate_pair_indexes(
    bait_centers: np.ndarray,
    prey_centers: np.ndarray,
    *,
    min_distance: int,
    max_distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    pair_bait_indexes: list[int] = []
    pair_prey_indexes: list[int] = []
    if prey_centers.shape[0] > 1 and np.any(prey_centers[:-1] > prey_centers[1:]):
        for bait_index, bait_center in enumerate(bait_centers):
            distances = np.abs(prey_centers - bait_center)
            candidate_indexes = np.flatnonzero(
                (distances >= min_distance) & (distances <= max_distance)
            )
            pair_bait_indexes.extend([bait_index] * len(candidate_indexes))
            pair_prey_indexes.extend(candidate_indexes.tolist())
    else:
        for bait_index, bait_center in enumerate(bait_centers):
            left_start = int(bait_center - max_distance)
            left_stop = int(bait_center - min_distance)
            right_start = int(bait_center + min_distance)
            right_stop = int(bait_center + max_distance)

            left = int(np.searchsorted(prey_centers, left_start, side="left"))
            right = int(np.searchsorted(prey_centers, left_stop, side="right"))
            pair_bait_indexes.extend([bait_index] * (right - left))
            pair_prey_indexes.extend(range(left, right))

            left = int(np.searchsorted(prey_centers, right_start, side="left"))
            right = int(np.searchsorted(prey_centers, right_stop, side="right"))
            pair_bait_indexes.extend([bait_index] * (right - left))
            pair_prey_indexes.extend(range(left, right))

    return (
        np.asarray(pair_bait_indexes, dtype=np.int64),
        np.asarray(pair_prey_indexes, dtype=np.int64),
    )


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
    """Cast inputs to the dtypes `touche.numba.background.count_ep_background_pairs_numba` expects, sorting by `pos_a` first if not already ascending."""
    from touche.numba.background import count_ep_background_pairs_numba

    pos_a = pos_a.astype(np.int64, copy=False)
    pos_b = pos_b.astype(np.int64, copy=False)
    if pos_a.shape[0] > 1 and np.any(pos_a[:-1] > pos_a[1:]):
        order = np.argsort(pos_a, kind="mergesort")
        pos_a = pos_a[order]
        pos_b = pos_b[order]

    return count_ep_background_pairs_numba(
        pos_a,
        pos_b,
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



def plot_background_scatter(
    data: pl.DataFrame,
    *,
    x_sample: str,
    y_sample: str,
    out_path: str | Path | None = None,
    reference_style: bool = True,
) -> "Figure":
    """Log-log scatter of EP/background ratios for two samples, colored by local point density."""
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
    """Parse a `NAME=PATH` CLI argument, e.g. `--treatment flv=flv_background.tsv`."""
    name, raw_path = _split_name_value(value)
    return NamedPath(name=name, path=Path(raw_path))


def parse_named_depth(value: str) -> NamedDepth:
    """Parse a `NAME=INTEGER` CLI argument, e.g. `--depth flv=200000000`."""
    name, raw_depth = _split_name_value(value)
    return NamedDepth(name=name, depth=int(raw_depth))


def _read_background(path: str | Path, sample: str) -> pl.DataFrame:
    """Read one sample's `count_ep_and_background` TSV output and suffix its columns by `sample`."""
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
    """Inner-join every sample's background counts on shared bait/prey pairs."""
    merged = _read_background(samples[0].path, samples[0].name)
    for sample in samples[1:]:
        merged = merged.join(_read_background(sample.path, sample.name), how="inner", on=PAIR_COLUMNS)
    return merged


def _reference_pair_order(control: str, treatments: list[str]) -> list[tuple[str, str]]:
    """Every `treatment vs. control` pair, plus every `treatment vs. treatment` pair, to plot."""
    pairs = [(control, treatment) for treatment in treatments]
    pairs.extend(combinations(treatments, 2))
    return pairs


def _resolve_cache_dir(cache_dir: str | Path | None, out_path: str | Path) -> Path:
    """Default to a `contact_index_cache/` directory next to `out_path`."""
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(out_path).parent / "contact_index_cache"


def _safe_kde(values: np.ndarray, *, max_fit_points: int = 5000, seed: int = 0) -> np.ndarray:
    """Gaussian KDE point-density estimate, falling back to uniform color on a degenerate input.

    `gaussian_kde(values)(values)` is O(n_fit * n_eval) with no faster path in
    scipy -- fitting and evaluating on every point is O(n^2) and dominates
    `background compare`'s wall time at real scale (tens of seconds per plot
    at ~68k rows). Every point is still colored (`n_eval` stays the full
    `values`); only the fit is capped to a random subsample of at most
    `max_fit_points`, since KDE bandwidth already smooths over local
    neighborhoods -- a several-thousand-point subsample gives a visually
    indistinguishable density estimate. `seed` is fixed by default so the
    same input always produces the same plot.
    """
    n = values.shape[1]
    if n < 2:
        return np.ones(n)
    fit_values = values
    if n > max_fit_points:
        indices = np.random.default_rng(seed).choice(n, size=max_fit_points, replace=False)
        fit_values = values[:, indices]
    try:
        return gaussian_kde(fit_values)(values)
    except np.linalg.LinAlgError:
        return np.ones(n)


def _split_name_value(value: str) -> tuple[str, str]:
    """Split a `NAME=VALUE` CLI argument, shared by `parse_named_path`/`parse_named_depth`."""
    if "=" not in value:
        raise ValueError(f"Expected NAME=VALUE, got {value!r}")
    name, raw_value = value.split("=", 1)
    if not name or not raw_value:
        raise ValueError(f"Expected NAME=VALUE, got {value!r}")
    return name, raw_value


def _close_figure(fig: "Figure") -> None:
    """Release a matplotlib figure's memory once it's been saved/embedded."""
    import matplotlib.pyplot as plt

    plt.close(fig)
