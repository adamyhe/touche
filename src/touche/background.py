from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from touche.anchors import read_bed_anchors
from touche.contacts import build_contact_indexes
from touche.models import NamedDepth, NamedPath

BACKGROUND_COLUMNS = ["chr", "promoter", "enhancer", "EP_contacts", "BG_contacts"]
PAIR_COLUMNS = ["chr", "promoter", "enhancer"]


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
) -> pd.DataFrame:
    """Count anchor-to-anchor and local-background contacts for bait/prey pairs."""

    indexes = build_contact_indexes(pairs_path, source=source, cis_only=True)
    baits = read_bed_anchors(baits_path)
    preys = read_bed_anchors(preys_path)
    rows: list[dict[str, object]] = []

    for chrom, chrom_baits in baits.groupby("chr", sort=False):
        index = indexes.get(chrom)
        if index is None:
            continue
        chrom_preys = preys.loc[preys["chr"] == chrom]
        if chrom_preys.empty:
            continue
        prey_centers = chrom_preys["center"].to_numpy(dtype=np.int64)
        for bait in chrom_baits.itertuples(index=False):
            distances = np.abs(prey_centers - bait.center)
            candidate_preys = chrom_preys.loc[
                (distances >= min_distance) & (distances <= max_distance)
            ]
            for prey in candidate_preys.itertuples(index=False):
                ep_contacts = _count_between_windows(
                    index.pos_a,
                    index.pos_b,
                    bait.center - window,
                    bait.center + window,
                    prey.center - window,
                    prey.center + window,
                )
                bait_to_prey_bg = _count_anchor_to_background(
                    index.pos_a,
                    index.pos_b,
                    bait.center,
                    prey.center,
                    window=window,
                    min_bg_distance=min_bg_distance,
                    max_bg_distance=max_bg_distance,
                )
                prey_to_bait_bg = _count_anchor_to_background(
                    index.pos_a,
                    index.pos_b,
                    prey.center,
                    bait.center,
                    window=window,
                    min_bg_distance=min_bg_distance,
                    max_bg_distance=max_bg_distance,
                )
                rows.append(
                    {
                        "chr": chrom,
                        "promoter": int(bait.center),
                        "enhancer": int(prey.center),
                        "EP_contacts": int(ep_contacts),
                        "BG_contacts": int(bait_to_prey_bg + prey_to_bait_bg),
                    }
                )

    result = pd.DataFrame(rows, columns=BACKGROUND_COLUMNS)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", header=False, index=False)
    return result


def compare_background_ratios(
    control: NamedPath,
    treatments: list[NamedPath],
    depths: dict[str, int],
    *,
    min_ep_cpb: float = 8.0,
    out_dir: str | Path | None = None,
    table_out: str | Path | None = None,
    reference_style: bool = True,
) -> tuple[pd.DataFrame, dict[str, Path]]:
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
        merged[f"EP_CPB_{sample}"] = merged[f"EP_contacts_{sample}"] / depth_scale

    passes_threshold = np.logical_or.reduce(
        [merged[f"EP_CPB_{sample}"] > min_ep_cpb for sample in sample_order]
    )
    positive_all = np.logical_and.reduce(
        [merged[f"EP_CPB_{sample}"] > 0 for sample in sample_order]
    )
    merged = merged.loc[passes_threshold & positive_all].replace([np.inf, -np.inf], np.nan).dropna()

    if table_out is not None:
        Path(table_out).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(table_out, sep="\t", index=False)

    plot_paths: dict[str, Path] = {}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for left, right in _reference_pair_order(
            control.name, [sample.name for sample in treatments]
        ):
            path = out_dir / f"{right}_vs_{left}.svg"
            plot_background_scatter(
                merged,
                x_sample=left,
                y_sample=right,
                out_path=path,
                reference_style=reference_style,
            )
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
    data: pd.DataFrame,
    *,
    x_sample: str,
    y_sample: str,
    out_path: str | Path,
    reference_style: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    x_col = f"ratio_{x_sample}"
    y_col = f"ratio_{y_sample}"
    plot_data = data.loc[(data[x_col] > 0) & (data[y_col] > 0)].copy()
    values = np.vstack([plot_data[x_col], plot_data[y_col]])
    colors = _safe_kde(values)

    figsize = (8, 8)
    label_size = 30
    tick_size = 20
    if not reference_style:
        label_size = 14
        tick_size = 10

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=plot_data,
        x=x_col,
        y=y_col,
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
    fig.savefig(out_path)
    plt.close(fig)


def parse_named_path(value: str) -> NamedPath:
    name, raw_path = _split_name_value(value)
    return NamedPath(name=name, path=Path(raw_path))


def parse_named_depth(value: str) -> NamedDepth:
    name, raw_depth = _split_name_value(value)
    return NamedDepth(name=name, depth=int(raw_depth))


def _read_background(path: str | Path, sample: str) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", names=BACKGROUND_COLUMNS)
    data[f"ratio_{sample}"] = data["EP_contacts"] / data["BG_contacts"]
    return data.rename(
        columns={
            "EP_contacts": f"EP_contacts_{sample}",
            "BG_contacts": f"BG_contacts_{sample}",
        }
    )


def _merge_samples(samples: list[NamedPath]) -> pd.DataFrame:
    merged = _read_background(samples[0].path, samples[0].name)
    for sample in samples[1:]:
        merged = merged.merge(
            _read_background(sample.path, sample.name), how="inner", on=PAIR_COLUMNS
        )
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
