from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from statsmodels.nonparametric.smoothers_lowess import lowess

from touche.backends import (
    DEFAULT_BACKEND,
    DEFAULT_FISHER_BACKEND,
    DEFAULT_LOWESS_BACKEND,
    validate_backend,
    validate_fisher_backend,
    validate_lowess_backend,
)
from touche.contacts import (
    build_contact_indexes,
    build_npz_cache,
    load_npz_cache,
    load_npz_cache_manifest,
)
from touche.instrumentation import Instrumentation, make_instrumentation
from touche.io import open_text
from touche.models import ContactIndex
from touche.stats import fisher_greater_batch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

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

_LOCAL_DECAY_SCHEMA = {
    "chr": pl.Utf8,
    "bait_center": pl.Int64,
    "prey_center": pl.Int64,
    "directional_distance": pl.Int64,
    "p_value": pl.Float64,
    "observed": pl.Int64,
    "expected": pl.Float64,
    "observed_background": pl.Int64,
    "expected_background": pl.Float64,
}


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
    backend: str = DEFAULT_BACKEND,
    lowess_backend: str = DEFAULT_LOWESS_BACKEND,
    fisher_backend: str = DEFAULT_FISHER_BACKEND,
    lowess_iterations: int = 3,
    index_strategy: str = "cache",
    cache_dir: str | Path | None = None,
    cache_prefix: str = "contacts",
    require_cache: bool = False,
    progress: bool | Instrumentation = False,
    profile: bool = False,
) -> pl.DataFrame:
    """Call bait-prey contacts normalized by local distance decay.

    This ports the reference ``ContactCaller_microC.py`` workflow without
    materializing one contact file per bait. Output intentionally keeps the
    reference nine-column, headerless layout.
    """

    if dist <= 0:
        raise ValueError("dist must be positive")
    if cap < 0:
        raise ValueError("cap must be non-negative")
    if index_strategy not in {"all", "chromosome", "cache"}:
        raise ValueError("index_strategy must be one of: all, chromosome, cache")

    instrument = make_instrumentation(progress, profile=profile)
    with instrument.step("read inputs"):
        baits = read_center_anchors(baits_path)
        preys = read_center_anchors(preys_path)
    if index_strategy == "cache":
        cache_dir = _resolve_cache_dir(cache_dir, out_path)
        with instrument.step("prepare contact cache"):
            _ensure_local_decay_cache(
                pairs_path,
                cache_dir=cache_dir,
                cache_prefix=cache_prefix,
                source=source,
                require_cache=require_cache,
            )
        calls = _call_local_decay_from_cache(
            baits,
            preys,
            cache_dir=cache_dir,
            cache_prefix=cache_prefix,
            dist=dist,
            cap=cap,
            min_distance=min_distance,
            lowess_window=lowess_window,
            lowess_delta=lowess_delta,
            backend=backend,
            lowess_backend=lowess_backend,
            fisher_backend=fisher_backend,
            lowess_iterations=lowess_iterations,
            progress=instrument,
        )
    elif index_strategy == "chromosome":
        calls = _call_local_decay_by_chromosome(
            baits,
            preys,
            pairs_path,
            dist=dist,
            cap=cap,
            min_distance=min_distance,
            source=source,
            lowess_window=lowess_window,
            lowess_delta=lowess_delta,
            backend=backend,
            lowess_backend=lowess_backend,
            fisher_backend=fisher_backend,
            lowess_iterations=lowess_iterations,
            progress=instrument,
        )
    else:
        with instrument.step("build contact indexes"):
            indexes = build_contact_indexes(
                pairs_path,
                source=source,
                cis_only=True,
                include_metadata=False,
            )
        calls = compute_local_decay(
            indexes,
            baits,
            preys,
            dist=dist,
            cap=cap,
            min_distance=min_distance,
            lowess_window=lowess_window,
            lowess_delta=lowess_delta,
            backend=backend,
            lowess_backend=lowess_backend,
            fisher_backend=fisher_backend,
            lowess_iterations=lowess_iterations,
            progress=instrument,
        )
    with instrument.step("write calls"):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        calls.write_csv(out_path, include_header=False, separator="\t")
    return calls


def _resolve_cache_dir(cache_dir: str | Path | None, out_path: str | Path) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(out_path).parent / "contact_index_cache"


def _ensure_local_decay_cache(
    pairs_path: str | Path,
    *,
    cache_dir: str | Path,
    cache_prefix: str,
    source: str,
    require_cache: bool = False,
) -> None:
    manifest_path = Path(cache_dir) / f"{cache_prefix}.manifest.json"
    if manifest_path.exists():
        return
    if require_cache:
        raise FileNotFoundError(
            f"Required local-decay cache manifest is missing: {manifest_path}. "
            "Run `touche preprocess build-cache` first or disable require_cache."
        )
    build_npz_cache(
        pairs_path,
        cache_dir,
        source=source,
        prefix=cache_prefix,
        cis_only=True,
        include_metadata=False,
        index_strategy="chromosome",
    )


def _call_local_decay_by_chromosome(
    baits: pl.DataFrame,
    preys: pl.DataFrame,
    pairs_path: str | Path,
    *,
    dist: int,
    cap: int,
    min_distance: int,
    source: str,
    lowess_window: int,
    lowess_delta: float,
    backend: str,
    lowess_backend: str,
    fisher_backend: str,
    lowess_iterations: int,
    progress: Instrumentation,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    chrom_list = baits["chr"].unique(maintain_order=True).to_list()
    chrom_iter = progress.iter(
        chrom_list,
        total=len(chrom_list),
        desc="local-decay index chromosomes",
        unit="chrom",
    )
    for chrom in chrom_iter:
        chrom_baits = baits.filter(pl.col("chr") == chrom)
        chrom_preys = preys.filter(pl.col("chr") == chrom)
        if chrom_preys.is_empty():
            continue
        with progress.step(f"build contact index {chrom}"):
            indexes = build_contact_indexes(
                pairs_path,
                source=source,
                cis_only=True,
                include_metadata=False,
                chromosomes={str(chrom)},
            )
        if chrom not in indexes:
            continue
        frames.append(
            compute_local_decay(
                indexes,
                chrom_baits,
                chrom_preys,
                dist=dist,
                cap=cap,
                min_distance=min_distance,
                lowess_window=lowess_window,
                lowess_delta=lowess_delta,
                backend=backend,
                lowess_backend=lowess_backend,
                fisher_backend=fisher_backend,
                lowess_iterations=lowess_iterations,
                progress=progress,
            )
        )
    if not frames:
        return pl.DataFrame(schema=_LOCAL_DECAY_SCHEMA)
    return pl.concat(frames)


def _call_local_decay_from_cache(
    baits: pl.DataFrame,
    preys: pl.DataFrame,
    *,
    cache_dir: str | Path | None,
    cache_prefix: str,
    dist: int,
    cap: int,
    min_distance: int,
    lowess_window: int,
    lowess_delta: float,
    backend: str,
    lowess_backend: str,
    fisher_backend: str,
    lowess_iterations: int,
    progress: Instrumentation,
) -> pl.DataFrame:
    if cache_dir is None:
        raise ValueError("cache_dir is required when loading local-decay indexes from cache")
    cache_paths = load_npz_cache_manifest(cache_dir, prefix=cache_prefix)
    frames: list[pl.DataFrame] = []
    chrom_list = baits["chr"].unique(maintain_order=True).to_list()
    chrom_iter = progress.iter(
        chrom_list,
        total=len(chrom_list),
        desc="local-decay cache chromosomes",
        unit="chrom",
    )
    for chrom in chrom_iter:
        chrom_baits = baits.filter(pl.col("chr") == chrom)
        chrom_preys = preys.filter(pl.col("chr") == chrom)
        cache_path = cache_paths.get(str(chrom))
        if chrom_preys.is_empty() or cache_path is None:
            continue
        with progress.step(f"load contact cache {chrom}"):
            index = load_npz_cache(cache_path, include_metadata=False)
        frames.append(
            compute_local_decay(
                {str(chrom): index},
                chrom_baits,
                chrom_preys,
                dist=dist,
                cap=cap,
                min_distance=min_distance,
                lowess_window=lowess_window,
                lowess_delta=lowess_delta,
                backend=backend,
                lowess_backend=lowess_backend,
                fisher_backend=fisher_backend,
                lowess_iterations=lowess_iterations,
                progress=progress,
            )
        )
    if not frames:
        return pl.DataFrame(schema=_LOCAL_DECAY_SCHEMA)
    return pl.concat(frames)


def compute_local_decay(
    indexes: dict[str, ContactIndex],
    baits: pl.DataFrame,
    preys: pl.DataFrame,
    *,
    dist: int = 1_000_000,
    cap: int = 2_000,
    min_distance: int = 5_000,
    lowess_window: int = 5_000,
    lowess_delta: float = 16.0,
    backend: str = DEFAULT_BACKEND,
    lowess_backend: str = DEFAULT_LOWESS_BACKEND,
    fisher_backend: str = DEFAULT_FISHER_BACKEND,
    lowess_iterations: int = 3,
    progress: bool | Instrumentation = False,
    profile: bool = False,
) -> pl.DataFrame:
    """Call local-decay contacts from in-memory contact indexes and center anchors."""

    if dist <= 0:
        raise ValueError("dist must be positive")
    if cap < 0:
        raise ValueError("cap must be non-negative")
    backend = validate_backend(backend)
    lowess_backend = validate_lowess_backend(lowess_backend)
    fisher_backend = validate_fisher_backend(fisher_backend)
    if lowess_iterations < 0:
        raise ValueError("lowess_iterations must be non-negative")

    instrument = make_instrumentation(progress, profile=profile)
    records: list[dict[str, float | int | str]] = []
    chrom_list = baits["chr"].unique(maintain_order=True).to_list() if baits.height else []

    # Only show a chromosome-level bar when there's more than one chromosome to
    # show progress across -- callers that already iterate chromosomes
    # themselves (e.g. _call_local_decay_from_cache) invoke this once per
    # chromosome, where a single-item bar would be redundant. The per-chromosome
    # bait bar below is created fresh (and closed) each iteration so it nests
    # cleanly under this one instead of fighting it for the same terminal line.
    chrom_source = chrom_list
    if len(chrom_list) > 1:
        chrom_source = instrument.iter(
            chrom_list,
            total=len(chrom_list),
            desc="local-decay chromosomes",
            unit="chrom",
        )

    for chrom in chrom_source:
        chrom_baits = baits.filter(pl.col("chr") == chrom)
        chrom_preys = preys.filter(pl.col("chr") == chrom).sort("center")
        index = indexes.get(chrom)
        if index is None or chrom_preys.is_empty():
            continue
        normalized = _ordered_cis_index(index)
        prey_centers = chrom_preys["center"].to_numpy().astype(np.int64)
        bait_centers = chrom_baits["center"].to_numpy().astype(np.int64)
        bait_iter = instrument.iter(
            bait_centers,
            total=len(bait_centers),
            desc=f"local-decay baits ({chrom})",
            unit="bait",
            leave=False,
        )
        for bait_center in bait_iter:
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
                    backend=backend,
                    lowess_backend=lowess_backend,
                    fisher_backend=fisher_backend,
                    lowess_iterations=lowess_iterations,
                )
            )

    return pl.DataFrame(records, schema=_LOCAL_DECAY_SCHEMA)


def assign_pair_types(
    contacts_path: str | Path,
    functional_path: str | Path,
    nonfunctional_path: str | Path,
    out_path: str | Path,
) -> pl.DataFrame:
    """Assign local-decay contacts to positive, negative, or other pair classes."""

    raw = pl.read_csv(contacts_path, separator="\t", has_header=False)
    source_columns = ["column_1", "column_2", "column_3", "column_4", "column_6", "column_7"]
    contacts = raw.select(source_columns)
    contacts.columns = CONTACT_COLUMNS
    contacts = contacts.with_columns(pl.col("directional_distance").abs().alias("distance"))

    functional_keys = _read_pair_keys(functional_path)
    nonfunctional_keys = _read_pair_keys(nonfunctional_path)
    key_struct = pl.struct(PAIR_KEY_COLUMNS)

    contacts = contacts.with_columns(
        pl.when(key_struct.is_in(functional_keys.implode()))
        .then(pl.lit("positive"))
        .when(key_struct.is_in(nonfunctional_keys.implode()))
        .then(pl.lit("negative"))
        .otherwise(pl.lit("other"))
        .alias("PosNeg")
    )
    contacts.write_csv(out_path, separator="\t")
    return contacts


def plot_pair_type_distribution(
    assignments: str | Path | pl.DataFrame,
    out_path: str | Path | None = None,
    *,
    min_contacts: int = 1,
    min_distance: int = 15_000,
    plot_table_out: str | Path | None = None,
    reference_style: bool = True,
) -> tuple[pl.DataFrame, "Figure"]:
    """Plot observed/expected contact distributions by assigned pair type."""

    import matplotlib

    matplotlib.use("Agg")
    import seaborn as sns

    if isinstance(assignments, pl.DataFrame):
        contacts = assignments
    else:
        contacts = pl.read_csv(assignments, separator="\t")

    filtered = contacts.filter(
        (pl.col("observed") >= min_contacts)
        & (pl.col("expected") >= min_contacts)
        & (pl.col("distance") >= min_distance)
    ).with_columns((pl.col("observed") / pl.col("expected")).log(2).alias("Obs/Exp"))

    if plot_table_out is not None:
        filtered.write_csv(plot_table_out, separator="\t")

    if reference_style:
        figsize = (6, 8)
        palette = ["#ffa600", "#bc5090", "gray"]
        order = ["positive", "negative", "other"]
        xtick_labels = ["Functional", "Nonfunctional", "Other"]
    else:
        figsize = (6, 6)
        palette = "deep"
        order = sorted(filtered["PosNeg"].drop_nulls().unique().to_list())
        xtick_labels = order

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        x=filtered["PosNeg"].to_numpy(),
        y=filtered["Obs/Exp"].to_numpy(),
        hue=filtered["PosNeg"].to_numpy(),
        showfliers=False,
        palette=palette,
        order=order,
        hue_order=order,
        inner="quartile",
        legend=False,
        ax=ax,
    )
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel("Normalized contacts (log2)", size=24)
    ax.set_xlabel("Pair Type", size=24)
    ax.set_xticks(range(len(xtick_labels)), xtick_labels, size=16)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path)
    return filtered, fig


def _read_pair_keys(path: str | Path) -> pl.Series:
    data = pl.read_csv(path)
    missing = [column for column in PAIR_KEY_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required pair key columns in {path}: {', '.join(missing)}")
    return data.select(pl.struct(PAIR_KEY_COLUMNS).alias("key"))["key"]


def read_center_anchors(path: str | Path) -> pl.DataFrame:
    """Read local-decay two-column or BED-like anchors with integer centers."""

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
    return pl.DataFrame(rows, schema=["chr", "center"], orient="row")


_read_center_anchors = read_center_anchors


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
    backend: str,
    lowess_backend: str,
    fisher_backend: str,
    lowess_iterations: int,
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
        backend=lowess_backend,
        iterations=lowess_iterations,
    )
    bg_pdf = fit_distance_decay_model(
        counts,
        zero_model,
        distances,
        dist=dist,
        winsize=lowess_window,
        delta=lowess_delta,
        backend=lowess_backend,
        iterations=lowess_iterations,
    )

    bait_start = bait_center - cap
    bait_stop = bait_center + cap
    plus = pos_b[(bait_start <= pos_a) & (pos_a <= bait_stop)]
    minus = pos_a[(bait_start <= pos_b) & (pos_b <= bait_stop)]
    histogram_bins = len(counts)

    if backend == "numba":
        observed_values, directional_distances, contact_counts = _local_decay_observed_numba(
            plus,
            minus,
            bait_center,
            prey_centers,
            cap=cap,
            min_distance=min_distance,
        )
    else:
        observed_values = []
        directional_distances = []
        contact_counts = []
        for prey_center in prey_centers:
            directional_distance = int(prey_center - bait_center)
            directional_distances.append(directional_distance)
            if abs(directional_distance) <= min_distance:
                observed_values.append(0)
                contact_counts.append(-1)
                continue
            contact_positions = plus if directional_distance > 0 else minus
            prey_start = int(prey_center - cap)
            prey_stop = int(prey_center + cap)
            observed_values.append(
                int(((prey_start <= contact_positions) & (contact_positions <= prey_stop)).sum())
            )
            contact_counts.append(len(contact_positions))

    prey_centers = np.asarray(prey_centers, dtype=np.int64)
    directional_distances = np.asarray(directional_distances, dtype=np.int64)
    observed_values = np.asarray(observed_values, dtype=np.int64)
    contact_counts = np.asarray(contact_counts, dtype=np.int64)

    keep = np.abs(directional_distances) > min_distance
    prey_centers = prey_centers[keep]
    directional_distances = directional_distances[keep]
    observed_values = observed_values[keep]
    contact_counts = contact_counts[keep]
    if prey_centers.size == 0:
        return []

    # Windowed sum of bg_pdf via a cumulative-sum lookup, vectorized across all
    # preys at once instead of a fresh slice-sum + scipy call per prey.
    strand_distances = np.abs(directional_distances)
    exp_start = np.maximum(0, strand_distances - cap)
    exp_stop = np.minimum(len(bg_pdf) - 1, strand_distances + cap)
    valid_window = exp_stop >= exp_start
    cumsum = np.concatenate([[0.0], np.cumsum(bg_pdf)])
    safe_start = np.clip(exp_start, 0, len(bg_pdf))
    safe_stop = np.clip(exp_stop + 1, 0, len(bg_pdf))
    exp_prob = np.where(valid_window, cumsum[safe_stop] - cumsum[safe_start], 0.0)

    expected = contact_counts.astype(np.float64) * exp_prob
    p_values = fisher_greater_batch(
        observed_values.astype(np.float64),
        expected,
        (histogram_bins - observed_values).astype(np.float64),
        histogram_bins - expected,
        backend=fisher_backend,
    )

    chrom = index.chrom if index.chrom.startswith("chr") else f"chr{index.chrom}"
    return [
        {
            "chr": chrom,
            "bait_center": int(bait_center),
            "prey_center": int(prey_center),
            "directional_distance": int(directional_distance),
            "p_value": float(p_value),
            "observed": int(observed),
            "expected": float(expected_value),
            "observed_background": int(histogram_bins - observed),
            "expected_background": float(histogram_bins - expected_value),
        }
        for prey_center, directional_distance, p_value, observed, expected_value in zip(
            prey_centers, directional_distances, p_values, observed_values, expected, strict=True
        )
    ]


def _local_decay_observed_numba(
    plus: np.ndarray,
    minus: np.ndarray,
    bait_center: int,
    prey_centers: np.ndarray,
    *,
    cap: int,
    min_distance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from touche.numba_kernels import local_decay_observed_counts_numba

    return local_decay_observed_counts_numba(
        plus.astype(np.int64, copy=False),
        minus.astype(np.int64, copy=False),
        int(bait_center),
        prey_centers.astype(np.int64, copy=False),
        int(cap),
        int(min_distance),
    )


def fit_zero_inflation_model(
    contact_counts_zero: np.ndarray,
    *,
    dist: int = 1_000_000,
    winsize: int = 5_000,
    delta: float = 16.0,
    backend: str = DEFAULT_LOWESS_BACKEND,
    iterations: int = 3,
) -> np.ndarray:
    """Fit the reference zero-inflation LOWESS model."""

    counts = np.asarray(contact_counts_zero, dtype=float)
    target_len = min(dist, len(counts))
    if target_len <= 0:
        return np.asarray([], dtype=float)
    winsize = max(1, min(winsize, target_len))

    # Vectorized windowed-average via a cumulative-sum lookup, replacing an
    # O(target_len) pure-Python loop (a single-threaded bottleneck regardless
    # of backend/lowess_backend/fisher_backend, since target_len is `dist` --
    # up to 1,000,000 -- evaluated once per bait).
    cumsum = np.concatenate([[0.0], np.cumsum(counts[:target_len])])
    k = np.arange(target_len)
    valid = (k >= 50) & (k + 50 <= target_len - 1)
    lo = np.clip(k - 50, 0, target_len)
    hi = np.clip(k + 50, 0, target_len)
    zero_pdf_full = np.where(valid, (cumsum[hi] - cumsum[lo]) / 100.0, 0.0)

    model: list[np.ndarray] = []
    for start in range(0, target_len, winsize):
        stop = min(start + winsize, target_len)
        zero_pdf = zero_pdf_full[start:stop]
        pos = np.arange(1, len(zero_pdf) + 1, dtype=float)
        smoothed = _safe_lowess(
            zero_pdf,
            pos,
            frac=0.01,
            it=iterations,
            delta=delta,
            backend=backend,
        )
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
    backend: str = DEFAULT_LOWESS_BACKEND,
    iterations: int = 3,
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
    bg_model = _safe_lowess(
        counts[:seed_len],
        pos[:seed_len],
        frac=0.05,
        it=iterations,
        delta=0.0,
        backend=backend,
    )
    for chunk_index, start in enumerate(range(0, target_len, winsize)):
        stop = min(start + winsize, target_len)
        extension_stop = min(stop + 300, target_len)
        chunk_pos = pos[start:extension_stop]
        pseudo_counts = counts[start:extension_stop] + zero[start:extension_stop]
        smoothed = _safe_lowess(
            pseudo_counts,
            chunk_pos,
            frac=0.01,
            it=iterations,
            delta=delta,
            backend=backend,
        )
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
    backend: str = DEFAULT_LOWESS_BACKEND,
) -> np.ndarray:
    if len(endog) <= 2:
        return np.asarray(endog, dtype=float)
    if backend == "numba":
        return _lowess_evenly_spaced_numba(endog, frac=frac, it=it, delta=delta)
    if backend != "statsmodels":
        raise ValueError("lowess backend must be one of: statsmodels, numba")
    return np.asarray(
        lowess(
            endog,
            exog,
            frac=frac,
            it=it,
            delta=delta,
            is_sorted=True,
            missing="none",
            return_sorted=False,
        ),
        dtype=float,
    )


def _lowess_evenly_spaced_numba(
    endog: np.ndarray, *, frac: float, it: int, delta: float
) -> np.ndarray:
    from touche.numba_kernels import lowess_evenly_spaced_numba

    return lowess_evenly_spaced_numba(
        np.asarray(endog, dtype=np.float64), float(frac), int(it), float(delta)
    )
