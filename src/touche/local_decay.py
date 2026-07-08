from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from touche.contacts import build_contact_indexes
from touche.io import open_text
from touche.models import ContactIndex
from touche.stats import fisher_greater

CONTACT_COLUMNS = [
    "target_site.chr",
    "target_site.center",
    "target_promoter.center",
    "directional_distance",
    "observed",
    "expected",
]

PAIR_KEY_COLUMNS = [
    "target_site.chr",
    "target_promoter.center",
    "target_site.center",
]

LOCAL_DECAY_OUTPUT_COLUMNS = [
    "chr",
    "bait_center",
    "prey_center",
    "directional_distance",
    "p_value",
    "observed",
    "expected",
    "observed_background",
    "expected_background",
]


def call_local_decay(
    baits_path: str | Path,
    preys_path: str | Path,
    pairs_path: str | Path,
    out_path: str | Path,
    *,
    dist: int = 1_000_000,
    cap: int = 2_000,
    min_distance: int = 5_000,
    source: str = "auto",
    lowess_window: int = 5_000,
    lowess_delta: float = 16.0,
) -> pd.DataFrame:
    """Call bait-prey contacts normalized by local distance decay.

    This ports the reference ``ContactCaller_microC.py`` workflow without
    materializing one contact file per bait. Output intentionally keeps the
    reference nine-column, headerless layout.
    """

    if dist <= 0:
        raise ValueError("dist must be positive")
    if cap < 0:
        raise ValueError("cap must be non-negative")

    baits = _read_center_anchors(baits_path)
    preys = _read_center_anchors(preys_path)
    indexes = build_contact_indexes(pairs_path, source=source, cis_only=True)

    records: list[dict[str, float | int | str]] = []
    for chrom, chrom_baits in baits.groupby("chr", sort=False):
        chrom_preys = preys.loc[preys["chr"] == chrom].sort_values("center")
        index = indexes.get(chrom)
        if index is None or chrom_preys.empty:
            continue
        normalized = _ordered_cis_index(index)
        prey_centers = chrom_preys["center"].to_numpy(dtype=np.int64)
        for bait_center in chrom_baits["center"].to_numpy(dtype=np.int64):
            start = int(max(0, bait_center - dist))
            stop = int(bait_center + dist)
            left = np.searchsorted(prey_centers, start, side="left")
            right = np.searchsorted(prey_centers, stop, side="right")
            if left == right:
                continue
            candidate_preys = prey_centers[left:right]
            records.extend(
                _call_bait_contacts(
                    normalized,
                    int(bait_center),
                    candidate_preys,
                    dist=dist,
                    cap=cap,
                    min_distance=min_distance,
                    lowess_window=lowess_window,
                    lowess_delta=lowess_delta,
                )
            )

    calls = pd.DataFrame.from_records(records, columns=LOCAL_DECAY_OUTPUT_COLUMNS)
    calls.to_csv(out_path, sep="\t", header=False, index=False)
    return calls


def assign_pair_types(
    contacts_path: str | Path,
    functional_path: str | Path,
    nonfunctional_path: str | Path,
    out_path: str | Path,
) -> pd.DataFrame:
    """Assign local-decay contacts to positive, negative, or other pair classes."""

    contacts = pd.read_csv(
        contacts_path,
        sep="\t",
        usecols=[0, 1, 2, 3, 5, 6],
        names=CONTACT_COLUMNS,
    )
    contacts["distance"] = contacts["directional_distance"].abs()

    functional_keys = _read_pair_keys(functional_path)
    nonfunctional_keys = _read_pair_keys(nonfunctional_path)
    contact_keys = pd.MultiIndex.from_frame(contacts[PAIR_KEY_COLUMNS])

    contacts["PosNeg"] = np.select(
        [
            contact_keys.isin(functional_keys),
            contact_keys.isin(nonfunctional_keys),
        ],
        [
            "positive",
            "negative",
        ],
        default="other",
    )
    contacts.to_csv(out_path, sep="\t")
    return contacts


def plot_pair_type_distribution(
    assignments_path: str | Path,
    out_path: str | Path,
    *,
    min_contacts: int = 1,
    min_distance: int = 15_000,
    plot_table_out: str | Path | None = None,
    reference_style: bool = True,
) -> pd.DataFrame:
    """Plot observed/expected contact distributions by assigned pair type."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    contacts = pd.read_csv(assignments_path, sep="\t", index_col=0)
    contacts.index = range(len(contacts.index))
    filtered = contacts.loc[
        (contacts["observed"] >= min_contacts)
        & (contacts["expected"] >= min_contacts)
        & (contacts["distance"] >= min_distance)
    ].copy()
    filtered["Obs/Exp"] = np.log2(filtered["observed"] / filtered["expected"])

    if plot_table_out is not None:
        filtered.to_csv(plot_table_out, sep="\t", index=False)

    if reference_style:
        figsize = (6, 8)
        palette = ["#ffa600", "#bc5090", "gray"]
        order = ["positive", "negative", "other"]
        xtick_labels = ["Functional", "Nonfunctional", "Other"]
    else:
        figsize = (6, 6)
        palette = "deep"
        order = sorted(filtered["PosNeg"].dropna().unique())
        xtick_labels = order

    plt.figure(figsize=figsize)
    sns.violinplot(
        x="PosNeg",
        y="Obs/Exp",
        hue="PosNeg",
        data=filtered,
        showfliers=False,
        palette=palette,
        order=order,
        hue_order=order,
        inner="quartile",
        legend=False,
    )
    plt.yticks(size=16)
    plt.ylabel("Normalized contacts (log2)", size=24)
    plt.xlabel("Pair Type", size=24)
    plt.xticks(range(len(xtick_labels)), xtick_labels, size=16)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return filtered


def _read_pair_keys(path: str | Path) -> pd.MultiIndex:
    data = pd.read_csv(path)
    missing = [column for column in PAIR_KEY_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required pair key columns in {path}: {', '.join(missing)}")
    return pd.MultiIndex.from_frame(data[PAIR_KEY_COLUMNS])


def _read_center_anchors(path: str | Path) -> pd.DataFrame:
    rows: list[tuple[str, int]] = []
    with open_text(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split("\t")
            if len(fields) < 2:
                raise ValueError(f"Expected at least two columns in {path} line {line_number}")
            chrom = fields[0]
            if len(fields) >= 3:
                center = (int(fields[1]) + int(fields[2])) // 2
            else:
                center = int(fields[1])
            rows.append((chrom, center))
    return pd.DataFrame(rows, columns=["chr", "center"])


def _ordered_cis_index(index: ContactIndex) -> ContactIndex:
    pos_a = np.minimum(index.pos_a, index.pos_b).astype(np.int64, copy=False)
    pos_b = np.maximum(index.pos_a, index.pos_b).astype(np.int64, copy=False)
    order = np.argsort(pos_a, kind="mergesort")
    return ContactIndex(
        chrom=index.chrom,
        pos_a=pos_a[order],
        pos_b=pos_b[order],
        strand_a=index.strand_a[order],
        strand_b=index.strand_b[order],
        mapq_a=index.mapq_a[order],
        mapq_b=index.mapq_b[order],
    )


def _call_bait_contacts(
    index: ContactIndex,
    bait_center: int,
    prey_centers: np.ndarray,
    *,
    dist: int,
    cap: int,
    min_distance: int,
    lowess_window: int,
    lowess_delta: float,
) -> list[dict[str, float | int | str]]:
    in_region = ((bait_center - dist) <= index.pos_a) & (index.pos_a <= (bait_center + dist)) | (
        ((bait_center - dist) <= index.pos_b) & (index.pos_b <= (bait_center + dist))
    )
    pos_a = index.pos_a[in_region]
    pos_b = index.pos_b[in_region]
    if pos_a.size == 0:
        return []

    distances = pos_b - pos_a
    max_distance = int(min(max(int(distances.max()), 1), dist * 2))
    counts = np.histogram(distances, bins=max_distance, range=(0, max_distance))[0].astype(float)
    counts_zero = np.where(counts != 0, 0.0, 1.0)
    zero_model = fit_zero_inflation_model(
        counts_zero,
        dist=dist,
        winsize=lowess_window,
        delta=lowess_delta,
    )
    bg_pdf = fit_distance_decay_model(
        counts,
        zero_model,
        distances,
        dist=dist,
        winsize=lowess_window,
        delta=lowess_delta,
    )

    bait_start = bait_center - cap
    bait_stop = bait_center + cap
    plus = pos_b[(bait_start <= pos_a) & (pos_a <= bait_stop)]
    minus = pos_a[(bait_start <= pos_b) & (pos_b <= bait_stop)]
    histogram_bins = len(counts)

    records: list[dict[str, float | int | str]] = []
    for prey_center in prey_centers:
        directional_distance = int(prey_center - bait_center)
        if abs(directional_distance) <= min_distance:
            continue
        contact_positions = plus if directional_distance > 0 else minus
        strand_distance = abs(directional_distance)
        prey_start = int(prey_center - cap)
        prey_stop = int(prey_center + cap)
        exp_start = max(0, int(strand_distance - cap))
        exp_stop = min(len(bg_pdf) - 1, int(strand_distance + cap))
        exp_prob = float(bg_pdf[exp_start : exp_stop + 1].sum()) if exp_stop >= exp_start else 0.0
        expected = float(len(contact_positions) * exp_prob)
        observed = int(((prey_start <= contact_positions) & (contact_positions <= prey_stop)).sum())
        p_value = fisher_greater(
            observed,
            expected,
            histogram_bins - observed,
            histogram_bins - expected,
        )
        records.append(
            {
                "chr": _with_chr_prefix(index.chrom),
                "bait_center": int(bait_center),
                "prey_center": int(prey_center),
                "directional_distance": directional_distance,
                "p_value": p_value,
                "observed": observed,
                "expected": expected,
                "observed_background": histogram_bins - observed,
                "expected_background": histogram_bins - expected,
            }
        )
    return records


def fit_zero_inflation_model(
    contact_counts_zero: np.ndarray,
    *,
    dist: int = 1_000_000,
    winsize: int = 5_000,
    delta: float = 16.0,
) -> np.ndarray:
    """Fit the reference zero-inflation LOWESS model."""

    model: list[np.ndarray] = []
    counts = np.asarray(contact_counts_zero, dtype=float)
    target_len = min(dist, len(counts))
    if target_len <= 0:
        return np.asarray([], dtype=float)
    winsize = max(1, min(winsize, target_len))
    for start in range(0, target_len, winsize):
        stop = min(start + winsize, target_len)
        zero_pdf = np.zeros(stop - start, dtype=float)
        for offset, k in enumerate(range(start, stop)):
            if k >= 50 and (k + 50) <= (target_len - 1):
                zero_pdf[offset] = counts[k - 50 : k + 50].sum() / 100.0
        pos = np.arange(1, len(zero_pdf) + 1, dtype=float)
        smoothed = _safe_lowess(zero_pdf, pos, frac=0.01, it=3, delta=delta)
        model.append(0.5 * (1 - smoothed))
    return np.concatenate(model)


def fit_distance_decay_model(
    contact_counts: np.ndarray,
    zero_model: np.ndarray,
    distances: np.ndarray,
    *,
    dist: int = 1_000_000,
    winsize: int = 5_000,
    delta: float = 16.0,
) -> np.ndarray:
    """Fit the reference distance-decay LOWESS model."""

    target_len = min(dist, len(contact_counts), len(zero_model))
    if target_len <= 0:
        return np.asarray([], dtype=float)
    counts = np.asarray(contact_counts[:target_len], dtype=float)
    zero = np.asarray(zero_model[:target_len], dtype=float)
    pos = np.arange(1, target_len + 1, dtype=float)
    winsize = max(1, min(winsize, target_len))

    seed_len = min(1_000, target_len)
    bg_model = _safe_lowess(counts[:seed_len], pos[:seed_len], frac=0.05, it=3, delta=0.0)
    for chunk_index, start in enumerate(range(0, target_len, winsize)):
        stop = min(start + winsize, target_len)
        extension_stop = min(stop + 300, target_len)
        chunk_pos = pos[start:extension_stop]
        pseudo_counts = counts[start:extension_stop] + zero[start:extension_stop]
        smoothed = _safe_lowess(pseudo_counts, chunk_pos, frac=0.01, it=3, delta=delta)
        if len(smoothed) <= 300 or len(bg_model) <= 300:
            bg_model = smoothed
            continue
        tail = bg_model[-300:]
        if chunk_index < 1 and len(smoothed) > 1_001:
            head = smoothed[700:1001]
            counts_tail = smoothed[1001:]
        else:
            head = smoothed[:300]
            counts_tail = smoothed[300:]
        overlap = min(len(tail), len(head))
        merge = (tail[-overlap:] + head[:overlap]) / 2.0
        bg_model = np.concatenate([bg_model[:-overlap], merge, counts_tail])

    bg_model = np.asarray(bg_model[:target_len], dtype=float)
    reads = int((np.asarray(distances) < dist).sum())
    if reads == 0:
        return np.zeros_like(bg_model)
    return bg_model / reads


def _safe_lowess(
    endog: np.ndarray,
    exog: np.ndarray,
    *,
    frac: float,
    it: int,
    delta: float,
) -> np.ndarray:
    if len(endog) <= 2:
        return np.asarray(endog, dtype=float)
    return np.asarray(
        lowess(endog, exog, frac=frac, it=it, delta=delta, return_sorted=False), dtype=float
    )


def _with_chr_prefix(chrom: str) -> str:
    return chrom if chrom.startswith("chr") else f"chr{chrom}"
