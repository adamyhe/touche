"""Import and export of external contact-significance and loop calls.

Public API: `LOOP_FORMATS` (the supported formats), `read_loop_calls`
(import any of them onto the canonical pair schema), `write_loop_calls`
(export a `touche` table as BEDPE or a FitHiC2-style significance table),
and `annotate_pairs` (attach imported calls to a `touche` pair table by
anchor overlap).

`touche` does not reimplement FitHiC2, MaxHiC, HiC-DC+, Mustache, Peakachu,
Chromosight, or HiCCUPS. Those are mature, well-validated, and make
different modelling assumptions from each other; the useful thing is to
interoperate with them, not to add an eighth caller. This module is the
narrow contract for doing that: read their output into the same columns and
the same `pair_id`s `touche` uses everywhere else, so imported calls can be
filtered, compared, and joined alongside native ones.

Method-specific scores are *not* collapsed into one column named
`probability`. Each import keeps `method` and `source_format`, and a
p-value from a binomial fragment-level model, a posterior probability from
a supervised classifier, and a pattern-matching correlation score are left
distinguishable, because they are not the same quantity.

Column resolution is by header name, with aliases, and a mismatch raises an
error listing the header actually found rather than silently mapping the
wrong column. Every format's mapping can be overridden per call with
`columns=`, which is also the escape hatch when an upstream tool changes
its output between versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from touche.pairs import finalize_pair_table

CALL_COLUMNS = [
    "pair_id",
    "bait_id",
    "prey_id",
    "chrom",
    "bait_chrom",
    "bait_start",
    "bait_end",
    "bait_center",
    "prey_chrom",
    "prey_start",
    "prey_end",
    "prey_center",
    "is_cis",
    "distance",
    "observed",
    "expected",
    "p_value",
    "q_value",
    "score",
    "method",
    "source_format",
]

_OPTIONAL_FIELDS = ("observed", "expected", "p_value", "q_value", "score")


@dataclass(frozen=True, slots=True)
class LoopFormat:
    """How to read one external tool's call output onto the canonical schema.

    `columns` maps a canonical field to the candidate source column names to
    look for (matched case- and punctuation-insensitively), or to a 0-based
    column index for headerless formats. `point_anchors` marks formats whose
    anchors are single coordinates (fragment or bin midpoints) rather than
    intervals; those become the one-base interval `[pos, pos + 1)`, which
    puts their center exactly on the reported coordinate.

    `p_transform`/`q_transform` name the scale the file reports significance
    on, so `-log10` and `-ln` outputs are converted back to probabilities on
    import instead of being compared against raw p-values.
    """

    name: str
    columns: dict[str, tuple[str, ...] | int]
    has_header: bool = True
    separator: str = "\t"
    comment_prefix: str | None = "#"
    point_anchors: bool = False
    p_transform: str | None = None
    q_transform: str | None = None
    reference: str = ""
    notes: str = ""
    required: tuple[str, ...] = field(
        default=("bait_chrom", "bait_start", "prey_chrom", "prey_start")
    )


LOOP_FORMATS: dict[str, LoopFormat] = {
    "fithic2": LoopFormat(
        name="fithic2",
        columns={
            "bait_chrom": ("chr1",),
            "bait_start": ("fragmentMid1", "fragment_mid1", "mid1"),
            "prey_chrom": ("chr2",),
            "prey_start": ("fragmentMid2", "fragment_mid2", "mid2"),
            "observed": ("contactCount", "contact_count"),
            "p_value": ("p-value", "p_value", "pvalue"),
            "q_value": ("q-value", "q_value", "qvalue"),
        },
        point_anchors=True,
        reference="FitHiC2 .significances.txt",
    ),
    "hicdcplus": LoopFormat(
        name="hicdcplus",
        columns={
            "bait_chrom": ("chrI", "chr_i", "chr1"),
            "bait_start": ("startI", "start_i", "start1"),
            "bait_end": ("endI", "end_i", "end1"),
            "prey_chrom": ("chrJ", "chr_j", "chr2"),
            "prey_start": ("startJ", "start_j", "start2"),
            "prey_end": ("endJ", "end_j", "end2"),
            "observed": ("counts", "count"),
            "expected": ("mu",),
            "p_value": ("pvalue", "p_value", "p-value"),
            "q_value": ("qvalue", "q_value", "q-value", "fdr"),
        },
        reference="HiC-DC+ exported gi_list significance table",
    ),
    "maxhic": LoopFormat(
        name="maxhic",
        columns={
            "bait_chrom": ("chr1", "chromosome1"),
            "bait_start": ("start1", "x_start"),
            "bait_end": ("end1", "x_end"),
            "prey_chrom": ("chr2", "chromosome2"),
            "prey_start": ("start2", "y_start"),
            "prey_end": ("end2", "y_end"),
            "observed": ("observed_interactions", "read_count", "observed"),
            "expected": ("exp_interactions", "expected_interactions", "expected"),
            "p_value": ("neg_ln_p_val", "neg_log_p_val", "p_val", "p_value"),
            "q_value": ("neg_ln_q_val", "neg_log_q_val", "q_val", "q_value"),
        },
        p_transform="neg_ln",
        q_transform="neg_ln",
        reference="MaxHiC cis_interactions.txt / trans_interactions.txt",
        notes=(
            "MaxHiC reports significance on a negative-log scale and has changed its "
            "column names between releases. If import raises on a missing column, pass "
            "columns= with the names in your file's header, and p_transform=None if "
            "your build reports raw probabilities."
        ),
    ),
    "bedpe": LoopFormat(
        name="bedpe",
        columns={
            "bait_chrom": 0, "bait_start": 1, "bait_end": 2,
            "prey_chrom": 3, "prey_start": 4, "prey_end": 5, "score": 7,
        },
        has_header=False,
        reference="Generic BEDPE loop list",
    ),
    "mustache": LoopFormat(
        name="mustache",
        columns={
            "bait_chrom": ("BIN1_CHR", "BIN1_CHROMOSOME", "chr1"),
            "bait_start": ("BIN1_START", "x1"),
            "bait_end": ("BIN1_END", "x2"),
            "prey_chrom": ("BIN2_CHROMOSOME", "BIN2_CHR", "chr2"),
            "prey_start": ("BIN2_START", "y1"),
            "prey_end": ("BIN2_END", "y2"),
            "q_value": ("FDR", "fdr"),
            "score": ("DETECTION_SCALE", "detection_scale"),
        },
        reference="Mustache .tsv loop calls",
    ),
    "chromosight": LoopFormat(
        name="chromosight",
        columns={
            "bait_chrom": ("chrom1",),
            "bait_start": ("start1",),
            "bait_end": ("end1",),
            "prey_chrom": ("chrom2",),
            "prey_start": ("start2",),
            "prey_end": ("end2",),
            "score": ("score",),
            "p_value": ("pvalue", "p_value"),
            "q_value": ("qvalue", "q_value"),
        },
        separator=",",
        reference="Chromosight detect .tsv/.csv output",
    ),
    "peakachu": LoopFormat(
        name="peakachu",
        columns={
            "bait_chrom": 0, "bait_start": 1, "bait_end": 2,
            "prey_chrom": 3, "prey_start": 4, "prey_end": 5, "score": 6,
        },
        has_header=False,
        reference="Peakachu BEDPE loops; score is the classifier probability, not a p-value",
    ),
    "hiccups": LoopFormat(
        name="hiccups",
        columns={
            "bait_chrom": ("chr1", "chrom1", "#chr1"),
            "bait_start": ("x1", "start1"),
            "bait_end": ("x2", "end1"),
            "prey_chrom": ("chr2", "chrom2"),
            "prey_start": ("y1", "start2"),
            "prey_end": ("y2", "end2"),
            "observed": ("o", "observed"),
            "expected": ("expectedDonut", "e_donut", "expected_donut"),
            "q_value": ("fdrDonut", "fdr_donut", "fdr_bl", "fdrBL"),
        },
        reference="Juicer HiCCUPS merged_loops.bedpe",
    ),
}


def read_loop_calls(
    path: str | Path,
    *,
    format: str = "bedpe",
    columns: dict[str, str | int] | None = None,
    method: str | None = None,
    id_style: str = "coord",
    cis_only: bool = False,
) -> pl.DataFrame:
    """Read an external tool's calls onto the canonical pair schema.

    Returns `CALL_COLUMNS`: the same anchor and `pair_id` columns
    `touche.pairs` produces, plus whichever of `observed`, `expected`,
    `p_value`, `q_value`, and `score` the source provides (null when it does
    not), plus `method` and `source_format` so the origin of a score is
    never ambiguous after a concatenation.

    `columns` overrides the format's mapping field by field; pass a header
    name or a 0-based index. `method` overrides the recorded method label,
    which defaults to the format name.

    Coordinates are imported exactly as written, as 0-based half-open
    intervals. `touche` does not know an external tool's genome build or
    chromosome naming, so it does not translate either -- check that `chr1`
    versus `1` matches your anchors before joining, or nothing will overlap.
    """

    if format not in LOOP_FORMATS:
        raise ValueError(f"format must be one of: {', '.join(sorted(LOOP_FORMATS))}")
    spec = LOOP_FORMATS[format]
    mapping: dict[str, tuple[str, ...] | str | int] = {**spec.columns, **(columns or {})}

    raw = pl.read_csv(
        path,
        separator=spec.separator,
        has_header=spec.has_header,
        comment_prefix=spec.comment_prefix if not spec.has_header else None,
        infer_schema_length=10_000,
    )
    resolved = {field: _resolve_column(raw, field, source, spec) for field, source in mapping.items()}
    missing = [field for field in spec.required if resolved.get(field) is None]
    if missing:
        raise ValueError(
            f"{path} is missing the {format} column(s) for {', '.join(missing)}. "
            f"Header found: {', '.join(raw.columns)}. Pass columns={{...}} to map them explicitly."
        )
    # Every `spec.required` field resolved, so these lookups are total.
    required = {field: name for field, name in resolved.items() if name is not None}

    frame = pl.DataFrame(
        {
            "bait_chrom": raw[required["bait_chrom"]].cast(pl.Utf8),
            "prey_chrom": raw[required["prey_chrom"]].cast(pl.Utf8),
            "bait_start": raw[required["bait_start"]].cast(pl.Int64),
            "prey_start": raw[required["prey_start"]].cast(pl.Int64),
        }
    )
    frame = frame.with_columns(
        _interval_end(raw, resolved.get("bait_end"), "bait_start", spec),
        _interval_end(raw, resolved.get("prey_end"), "prey_start", spec),
    )
    frame = frame.with_columns(
        ((pl.col("bait_start") + pl.col("bait_end")) // 2).alias("bait_center"),
        ((pl.col("prey_start") + pl.col("prey_end")) // 2).alias("prey_center"),
        pl.lit(".").alias("bait_strand"),
        pl.lit(".").alias("prey_strand"),
    )
    frame = frame.with_columns(
        pl.when(pl.col("bait_chrom") == pl.col("prey_chrom"))
        .then(pl.col("prey_center") - pl.col("bait_center"))
        .otherwise(None)
        .alias("directional_distance")
    )
    frame = frame.with_columns(pl.col("directional_distance").abs().alias("distance"))
    for name in _OPTIONAL_FIELDS:
        frame = frame.with_columns(_optional_column(raw, resolved.get(name), name, spec))
    frame = frame.with_columns(
        pl.lit(method or spec.name).alias("method"), pl.lit(spec.name).alias("source_format")
    )
    if cis_only:
        frame = frame.filter(pl.col("bait_chrom") == pl.col("prey_chrom"))

    extras = [*_OPTIONAL_FIELDS, "method", "source_format"]
    return finalize_pair_table(frame, id_style=id_style, extra_columns=extras).select(CALL_COLUMNS)


def write_loop_calls(
    calls: pl.DataFrame, path: str | Path, *, format: str = "bedpe", score_col: str = "score"
) -> Path:
    """Export a `touche` pair/result table for a downstream tool.

    `format="bedpe"` writes the six anchor columns plus `pair_id` as the
    name and `score_col` as the score -- the interchange every loop-calling
    and genome-browser tool reads. `format="fithic2"` writes the FitHiC2
    significance layout (`chr1 fragmentMid1 chr2 fragmentMid2 contactCount
    p-value q-value`), using anchor centers as the fragment midpoints, so
    `touche` calls can be fed to tooling built around that table.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "bedpe":
        score = pl.col(score_col).cast(pl.Utf8) if score_col in calls.columns else pl.lit(".")
        out = calls.select(
            pl.col("bait_chrom"), pl.col("bait_start"), pl.col("bait_end"),
            pl.col("prey_chrom"), pl.col("prey_start"), pl.col("prey_end"),
            pl.col("pair_id"), score.alias("score"),
            _or_null(calls, "bait_strand", ".").alias("strand1"),
            _or_null(calls, "prey_strand", ".").alias("strand2"),
        )
        out.write_csv(path, separator="\t", include_header=False)
        return path
    if format == "fithic2":
        out = calls.select(
            pl.col("bait_chrom").alias("chr1"),
            _center(calls, "bait").alias("fragmentMid1"),
            pl.col("prey_chrom").alias("chr2"),
            _center(calls, "prey").alias("fragmentMid2"),
            _or_null(calls, "observed", 0).alias("contactCount"),
            _or_null(calls, "p_value", None).alias("p-value"),
            _or_null(calls, "q_value", None).alias("q-value"),
        )
        out.write_csv(path, separator="\t")
        return path
    raise ValueError("format must be one of: bedpe, fithic2")


def annotate_pairs(
    pairs: pl.DataFrame,
    calls: pl.DataFrame,
    *,
    prefix: str = "loop_",
    slop: int = 0,
    keep: tuple[str, ...] = ("p_value", "q_value", "score", "observed", "expected", "method"),
) -> pl.DataFrame:
    """Attach external calls to a `touche` pair table by anchor overlap.

    Exact `pair_id` equality almost never works across tools: `touche`
    anchors are points and external callers report bins, so a match has to
    be "this pair's bait center falls inside that call's first anchor and
    its prey center inside the second". That is what this does, in both
    anchor orders, optionally widened by `slop` bases on each side.

    Adds `{prefix}matched` (boolean) plus the requested `keep` columns from
    the best-matching call, prefixed. When several calls match one pair, the
    one with the smallest `q_value` wins (then `p_value`, then the largest
    `score`), so the annotation is deterministic rather than
    whichever-came-first.
    """

    if slop < 0:
        raise ValueError("slop must be non-negative")
    available = [name for name in keep if name in calls.columns]
    matched_index = _match_calls(pairs, calls, slop=slop)

    hit = matched_index >= 0
    annotated = pairs.with_columns(pl.Series(f"{prefix}matched", hit, dtype=pl.Boolean))
    if calls.is_empty():
        return annotated.with_columns(
            [pl.lit(None, dtype=calls.schema[name]).alias(f"{prefix}{name}") for name in available]
        )
    for name in available:
        # Gather first, then null out the misses, so the annotation keeps the
        # source column's dtype instead of degrading to an object column.
        gathered = calls[name].gather(np.maximum(matched_index, 0))
        annotated = annotated.with_columns(
            pl.when(pl.Series(hit)).then(gathered).otherwise(None).alias(f"{prefix}{name}")
        )
    return annotated


def _resolve_column(
    raw: pl.DataFrame, field_name: str, source: tuple[str, ...] | str | int, spec: LoopFormat
) -> str | None:
    """Find the actual header name (or positional column) backing one canonical field."""
    if isinstance(source, int):
        return raw.columns[source] if source < raw.width else None
    if isinstance(source, str):
        source = (source,)
    normalized = {_normalize(name): name for name in raw.columns}
    for candidate in source:
        if _normalize(candidate) in normalized:
            return normalized[_normalize(candidate)]
    if field_name in spec.required and spec.notes:
        # Surfaced by the caller's error; the note explains the likely cause.
        return None
    return None


def _normalize(name: str) -> str:
    """Header names differ across tool versions only by case and punctuation; compare without either."""
    return "".join(character for character in name.lower() if character.isalnum())


def _interval_end(
    raw: pl.DataFrame, column: str | None, start_field: str, spec: LoopFormat
) -> pl.Expr | pl.Series:
    """Anchor end, taken from the file or synthesized as a one-base point anchor."""
    end_field = start_field.replace("_start", "_end")
    if column is not None and not spec.point_anchors:
        return raw[column].cast(pl.Int64).alias(end_field)
    return (pl.col(start_field) + 1).alias(end_field)


def _optional_column(
    raw: pl.DataFrame, column: str | None, name: str, spec: LoopFormat
) -> pl.Expr | pl.Series:
    """One optional score column, back-transformed from a negative-log scale if needed."""
    if column is None:
        return pl.lit(None, dtype=pl.Float64).alias(name)
    source = raw[column]
    if source.dtype == pl.Utf8:
        # BED-family formats write "." for an absent value; that is a null,
        # not a parse failure, and `write_loop_calls` emits it too.
        source = source.replace({".": None})
    try:
        values = source.cast(pl.Float64)
    except pl.exceptions.InvalidOperationError as error:
        raise ValueError(
            f"Column {column!r} was mapped to {name!r} for format {spec.name!r} but is not numeric. "
            "Pass columns={...} to point it at the right column."
        ) from error
    transform = spec.p_transform if name == "p_value" else spec.q_transform if name == "q_value" else None
    if transform == "neg_log10":
        values = pl.Series(name, np.power(10.0, -values.to_numpy()))
    elif transform == "neg_ln":
        values = pl.Series(name, np.exp(-values.to_numpy()))
    return values.alias(name)


def _or_null(frame: pl.DataFrame, column: str, default: object) -> pl.Expr:
    """`pl.col(column)` when present, otherwise a literal default -- for export layouts."""
    return pl.col(column) if column in frame.columns else pl.lit(default)


def _center(frame: pl.DataFrame, role: str) -> pl.Expr:
    """Anchor center, from the explicit column if present or derived from the interval."""
    if f"{role}_center" in frame.columns:
        return pl.col(f"{role}_center")
    return (pl.col(f"{role}_start") + pl.col(f"{role}_end")) // 2


def _match_calls(pairs: pl.DataFrame, calls: pl.DataFrame, *, slop: int) -> np.ndarray:
    """Index of the best matching call per pair, or -1.

    Per chromosome, calls are sorted by widened anchor start and candidates
    are found with `np.searchsorted`, bounded on the left by the widest
    anchor on that chromosome -- the same bounded-slice trick local-decay
    uses for contact windows. That keeps this linearithmic instead of
    comparing every pair against every call. Both anchor orders are tried,
    since a caller's anchor 1 need not be the bait.
    """

    matched = np.full(pairs.height, -1, dtype=np.int64)
    if pairs.is_empty() or calls.is_empty():
        return matched
    best = np.full(pairs.height, np.inf, dtype=np.float64)

    priority = _call_priority(calls)
    call_chroms = calls["bait_chrom"].to_numpy()
    call_partners = calls["prey_chrom"].to_numpy()
    bounds = {
        "bait": (calls["bait_start"].to_numpy() - slop, calls["bait_end"].to_numpy() + slop),
        "prey": (calls["prey_start"].to_numpy() - slop, calls["prey_end"].to_numpy() + slop),
    }

    pair_chroms = pairs["bait_chrom"].to_numpy()
    bait_centers = pairs["bait_center"].to_numpy()
    prey_centers = pairs["prey_center"].to_numpy()
    for chrom in np.unique(pair_chroms):
        pair_rows = np.flatnonzero(pair_chroms == chrom)
        call_rows = np.flatnonzero((call_chroms == chrom) & (call_partners == chrom))
        if call_rows.size == 0:
            continue
        for left, right in (("bait", "prey"), ("prey", "bait")):
            _fill_matches(
                matched,
                best,
                pair_rows,
                bait_centers[pair_rows],
                prey_centers[pair_rows],
                call_rows,
                priority[call_rows],
                *(array[call_rows] for array in bounds[left]),
                *(array[call_rows] for array in bounds[right]),
            )
    return matched


def _call_priority(calls: pl.DataFrame) -> np.ndarray:
    """Rank each call so a pair overlapping several picks the same one every time."""
    keys = []
    for name, ascending in (("q_value", True), ("p_value", True), ("score", False)):
        if name in calls.columns:
            values = calls[name].cast(pl.Float64).to_numpy()
            keys.append(np.where(np.isnan(values), np.inf, values if ascending else -values))
    if not keys:
        return np.arange(calls.height, dtype=np.float64)
    return np.lexsort(keys[::-1]).argsort().astype(np.float64)


def _fill_matches(
    matched: np.ndarray,
    best: np.ndarray,
    pair_rows: np.ndarray,
    bait_centers: np.ndarray,
    prey_centers: np.ndarray,
    call_rows: np.ndarray,
    priority: np.ndarray,
    left_low: np.ndarray,
    left_high: np.ndarray,
    right_low: np.ndarray,
    right_high: np.ndarray,
) -> None:
    """Record the highest-priority call whose two anchors contain each pair's two centers."""
    start_order = np.argsort(left_low, kind="mergesort")
    sorted_low = left_low[start_order]
    widest = int(np.max(left_high - left_low))
    for position, (bait, prey) in enumerate(zip(bait_centers, prey_centers)):
        stop = int(np.searchsorted(sorted_low, bait, side="right"))
        start = int(np.searchsorted(sorted_low, bait - widest, side="left"))
        if start >= stop:
            continue
        candidates = start_order[start:stop]
        hit = candidates[
            (left_high[candidates] >= bait)
            & (right_low[candidates] <= prey)
            & (right_high[candidates] >= prey)
        ]
        if hit.size == 0:
            continue
        winner = hit[np.argmin(priority[hit])]
        row = pair_rows[position]
        if priority[winner] < best[row]:
            best[row] = priority[winner]
            matched[row] = call_rows[winner]
