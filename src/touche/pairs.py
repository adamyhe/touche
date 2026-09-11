"""Canonical bait/prey pair table: stable `pair_id`s, BEDPE I/O, anchor expansion.

Public API: `PAIR_COLUMNS`/`PAIR_SCHEMA` (the canonical anchor-table layout
every statistical module keys on), `make_pair_ids`, `build_pair_table`
(Cartesian expansion of bait/prey anchor BEDs with a distance filter --
the implicit pair universe `background`/`apa` have always used),
`read_bedpe`/`write_bedpe` (explicit user-supplied pair lists), and
`attach_pair_ids` (re-derive `pair_id` for a frame that already carries
anchor coordinates).

Coordinate convention: every start/end in this module is 0-based half-open,
matching BED/BEDPE. `center` is `(start + end) // 2`, matching
`touche.anchors.read_bed_anchors`, so an explicit pair list and an
anchor-BED-derived one place the same anchor at the same base.

`pair_id` identity is *center-based and unordered*. Every `touche` analysis
reduces an anchor to its center before counting anything, so the center pair
-- not the interval pair -- is the unit actually being measured, and it is
what the id is built from. Two anchors that share a center are one anchor as
far as `touche` is concerned. The two anchors are then sorted before the id
is formed, so a BEDPE row and its anchor-swapped twin share one `pair_id`
while keeping whichever bait/prey roles the caller assigned.

Together those two properties are what let a `pair_id` join across
`local-decay` (which takes point center anchors), `background`, and `apa`
(which take BED intervals) for the same biological pair. Full interval
coordinates are still carried in the table as provenance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from touche.anchors import read_bed_anchors

PAIR_COLUMNS = [
    "pair_id",
    "bait_id",
    "prey_id",
    "chrom",
    "bait_chrom",
    "bait_start",
    "bait_end",
    "bait_strand",
    "bait_center",
    "prey_chrom",
    "prey_start",
    "prey_end",
    "prey_strand",
    "prey_center",
    "is_cis",
    "distance",
    "directional_distance",
]

PAIR_SCHEMA: dict[str, pl.DataType] = {
    "pair_id": pl.Utf8,
    "bait_id": pl.Utf8,
    "prey_id": pl.Utf8,
    "chrom": pl.Utf8,
    "bait_chrom": pl.Utf8,
    "bait_start": pl.Int64,
    "bait_end": pl.Int64,
    "bait_strand": pl.Utf8,
    "bait_center": pl.Int64,
    "prey_chrom": pl.Utf8,
    "prey_start": pl.Int64,
    "prey_end": pl.Int64,
    "prey_strand": pl.Utf8,
    "prey_center": pl.Int64,
    "is_cis": pl.Boolean,
    "distance": pl.Int64,
    "directional_distance": pl.Int64,
}

BEDPE_CORE_COLUMNS = [
    "chrom1",
    "start1",
    "end1",
    "chrom2",
    "start2",
    "end2",
    "name",
    "score",
    "strand1",
    "strand2",
]

_ID_STYLES = {"coord", "digest"}
_DUPLICATE_POLICIES = {"error", "first", "keep"}


def make_pair_ids(
    bait_chrom: pl.Expr | str,
    bait_center: pl.Expr | str,
    prey_chrom: pl.Expr | str,
    prey_center: pl.Expr | str,
    *,
    style: str = "coord",
) -> pl.Expr:
    """Deterministic, anchor-order-invariant `pair_id` expression.

    The id is built from the two anchor centers sorted as `chrom:center`
    strings, so it depends only on the unordered center pair -- never on row
    order, chunking, or which anchor the caller called the bait.
    `style="coord"` yields the readable `chrA:c|chrB:c` key; `style="digest"`
    yields the first 16 hex digits of that same string's BLAKE2b digest,
    which is shorter but no longer human-readable. Both are stable across
    `touche`, `polars`, and Python versions because neither relies on a
    built-in hash.
    """

    if style not in _ID_STYLES:
        raise ValueError(f"style must be one of: {', '.join(sorted(_ID_STYLES))}")
    left = _anchor_key(bait_chrom, bait_center)
    right = _anchor_key(prey_chrom, prey_center)
    ordered = pl.when(left <= right).then(left + pl.lit("|") + right).otherwise(right + pl.lit("|") + left)
    if style == "coord":
        return ordered.alias("pair_id")
    return ordered.map_elements(_digest, return_dtype=pl.Utf8).alias("pair_id")


def attach_pair_ids(pairs: pl.DataFrame, *, style: str = "coord") -> pl.DataFrame:
    """Add (or overwrite) `pair_id`, `bait_id`, and `prey_id` on a frame carrying anchor centers."""

    missing = [c for c in ("bait_chrom", "bait_center", "prey_chrom", "prey_center") if c not in pairs.columns]
    if missing:
        raise ValueError(f"Cannot derive pair_id, missing anchor columns: {', '.join(missing)}")
    return pairs.with_columns(
        make_pair_ids("bait_chrom", "bait_center", "prey_chrom", "prey_center", style=style),
        _anchor_key("bait_chrom", "bait_center").alias("bait_id"),
        _anchor_key("prey_chrom", "prey_center").alias("prey_id"),
    )


def build_pair_table(
    baits: pl.DataFrame,
    preys: pl.DataFrame,
    *,
    min_distance: int = 0,
    max_distance: int | None = None,
    id_style: str = "coord",
) -> pl.DataFrame:
    """Expand bait and prey anchor frames into the canonical cis pair table.

    This is the implicit pair universe `background count` and `apa aggregate`
    have always analyzed -- every bait crossed with every same-chromosome prey
    whose center distance falls in `[min_distance, max_distance]` -- made
    explicit so it can be exported, filtered, annotated, or replaced with a
    user-supplied BEDPE via `read_bedpe`.

    `baits`/`preys` are `touche.anchors.read_bed_anchors` frames (`chr`,
    `start`, `end`, `strand`, `center`).
    """

    if min_distance < 0:
        raise ValueError("min_distance must be non-negative")
    if max_distance is not None and max_distance < min_distance:
        raise ValueError("max_distance must be >= min_distance")

    left = _prefix_anchors(baits, "bait")
    right = _prefix_anchors(preys, "prey")
    if left.is_empty() or right.is_empty():
        return pl.DataFrame(schema=PAIR_SCHEMA)

    pairs = left.join(right, left_on="bait_chrom", right_on="prey_chrom", how="inner", suffix="_prey")
    # The join key is consumed by polars, so put the (identical) prey chrom back.
    pairs = pairs.with_columns(pl.col("bait_chrom").alias("prey_chrom"))
    pairs = pairs.with_columns((pl.col("prey_center") - pl.col("bait_center")).alias("directional_distance"))
    pairs = pairs.with_columns(pl.col("directional_distance").abs().alias("distance"))
    pairs = pairs.filter(pl.col("distance") >= min_distance)
    if max_distance is not None:
        pairs = pairs.filter(pl.col("distance") <= max_distance)
    return _finalize_pairs(pairs, id_style=id_style)


def pair_table_from_centers(
    chrom: pl.Series | list[str],
    bait_center: pl.Series | list[int],
    prey_center: pl.Series | list[int],
    *,
    id_style: str = "coord",
) -> pl.DataFrame:
    """Canonical cis pair table for point (center-only) anchors.

    `local-decay` takes two-column `chr`/`center` anchor files and never sees
    an interval, so its rows are materialized here as the one-base intervals
    `[center, center + 1)`. The resulting `pair_id`s are center-based and
    therefore join directly against interval-anchored tables from
    `build_pair_table`/`read_bedpe` for the same biological pair.
    """

    frame = pl.DataFrame(
        {
            "bait_chrom": pl.Series(chrom, dtype=pl.Utf8),
            "bait_center": pl.Series(bait_center, dtype=pl.Int64),
            "prey_center": pl.Series(prey_center, dtype=pl.Int64),
        }
    )
    frame = frame.with_columns(
        pl.col("bait_chrom").alias("prey_chrom"),
        pl.col("bait_center").alias("bait_start"),
        (pl.col("bait_center") + 1).alias("bait_end"),
        pl.col("prey_center").alias("prey_start"),
        (pl.col("prey_center") + 1).alias("prey_end"),
        pl.lit(".").alias("bait_strand"),
        pl.lit(".").alias("prey_strand"),
        (pl.col("prey_center") - pl.col("bait_center")).alias("directional_distance"),
    )
    frame = frame.with_columns(pl.col("directional_distance").abs().alias("distance"))
    return _finalize_pairs(frame, id_style=id_style)


def read_bedpe(
    path: str | Path,
    *,
    bait_anchor: str = "first",
    id_style: str = "coord",
    on_duplicate: str = "first",
    cis_only: bool = False,
) -> pl.DataFrame:
    """Read an explicit BEDPE pair list into the canonical pair table.

    Accepts the standard BEDPE column order (`chrom1 start1 end1 chrom2
    start2 end2 [name score strand1 strand2 ...]`); columns past `strand2`
    are carried through unchanged as extra annotation/covariate columns, so a
    caller can attach per-pair labels (functional class, activity, coverage)
    in the same file. Coordinates are read as 0-based half-open and are not
    re-interpreted.

    `bait_anchor` assigns roles: `"first"`/`"second"` take the named anchor as
    the bait; `"left"`/`"right"` assign by genomic position, which makes the
    role assignment itself anchor-order-invariant. Trans pairs are kept unless
    `cis_only=True`; their `chrom` and `distance` are null, since neither is
    defined across chromosomes. `on_duplicate` controls repeated `pair_id`s:
    `"first"` keeps the first occurrence, `"keep"` keeps every row, `"error"`
    raises.
    """

    if bait_anchor not in {"first", "second", "left", "right"}:
        raise ValueError("bait_anchor must be one of: first, second, left, right")
    if on_duplicate not in _DUPLICATE_POLICIES:
        raise ValueError(f"on_duplicate must be one of: {', '.join(sorted(_DUPLICATE_POLICIES))}")

    raw = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        comment_prefix="#",
        infer_schema_length=10_000,
        truncate_ragged_lines=False,
    )
    if raw.width < 6:
        raise ValueError(f"Expected at least six BEDPE columns in {path}, found {raw.width}")
    names = BEDPE_CORE_COLUMNS[: raw.width] + [f"extra_{i}" for i in range(raw.width - len(BEDPE_CORE_COLUMNS))]
    raw.columns = names
    raw = raw.with_columns(
        pl.col("chrom1").cast(pl.Utf8),
        pl.col("chrom2").cast(pl.Utf8),
        pl.col("start1").cast(pl.Int64),
        pl.col("end1").cast(pl.Int64),
        pl.col("start2").cast(pl.Int64),
        pl.col("end2").cast(pl.Int64),
    )
    _validate_bedpe_intervals(raw, path)
    for column, default in (("name", None), ("score", None), ("strand1", "."), ("strand2", ".")):
        if column not in raw.columns:
            raw = raw.with_columns(pl.lit(default).alias(column))
    raw = raw.with_columns(
        pl.col("strand1").cast(pl.Utf8).fill_null("."),
        pl.col("strand2").cast(pl.Utf8).fill_null("."),
    )

    swap = _bait_anchor_swap(bait_anchor)
    pairs = raw.with_columns(
        [
            pl.when(swap).then(pl.col(f"{field}2")).otherwise(pl.col(f"{field}1")).alias(f"bait_{name}")
            for field, name in (("chrom", "chrom"), ("start", "start"), ("end", "end"), ("strand", "strand"))
        ]
        + [
            pl.when(swap).then(pl.col(f"{field}1")).otherwise(pl.col(f"{field}2")).alias(f"prey_{name}")
            for field, name in (("chrom", "chrom"), ("start", "start"), ("end", "end"), ("strand", "strand"))
        ]
    )
    pairs = pairs.with_columns(
        ((pl.col("bait_start") + pl.col("bait_end")) // 2).alias("bait_center"),
        ((pl.col("prey_start") + pl.col("prey_end")) // 2).alias("prey_center"),
    )
    pairs = pairs.with_columns(
        pl.when(pl.col("bait_chrom") == pl.col("prey_chrom"))
        .then(pl.col("prey_center") - pl.col("bait_center"))
        .otherwise(None)
        .alias("directional_distance")
    )
    pairs = pairs.with_columns(pl.col("directional_distance").abs().alias("distance"))
    if cis_only:
        pairs = pairs.filter(pl.col("bait_chrom") == pl.col("prey_chrom"))

    extra = [c for c in pairs.columns if c.startswith("extra_")]
    named = pairs["name"] if pairs["name"].null_count() < pairs.height else None
    result = _finalize_pairs(pairs, id_style=id_style, extra_columns=extra)
    if named is not None:
        result = result.with_columns(named.alias("name"))
    if "score" in pairs.columns and pairs["score"].null_count() < pairs.height:
        result = result.with_columns(pairs["score"].alias("score"))

    if on_duplicate == "error":
        duplicated = result.filter(pl.col("pair_id").is_duplicated())
        if duplicated.height:
            example = duplicated["pair_id"][0]
            raise ValueError(f"{path} contains {duplicated.height} duplicate pair rows (e.g. {example!r})")
    elif on_duplicate == "first":
        result = result.unique(subset="pair_id", keep="first", maintain_order=True)
    return result


def write_bedpe(pairs: pl.DataFrame, path: str | Path, *, score_col: str | None = None) -> Path:
    """Write a canonical pair table back out as BEDPE, round-tripping coordinates.

    Anchor 1 is always the bait and anchor 2 the prey, `name` is the
    `pair_id`, and `score` comes from `score_col` when given (otherwise `.`).
    Reading the result back with `read_bedpe(bait_anchor="first")` reproduces
    the same `pair_id`s and coordinates.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    score = pl.col(score_col).cast(pl.Utf8) if score_col else pl.lit(".")
    out = pairs.select(
        pl.col("bait_chrom").alias("chrom1"),
        pl.col("bait_start").alias("start1"),
        pl.col("bait_end").alias("end1"),
        pl.col("prey_chrom").alias("chrom2"),
        pl.col("prey_start").alias("start2"),
        pl.col("prey_end").alias("end2"),
        pl.col("pair_id").alias("name"),
        score.alias("score"),
        pl.col("bait_strand").alias("strand1"),
        pl.col("prey_strand").alias("strand2"),
    )
    out.write_csv(path, separator="\t", include_header=False)
    return path


def read_pair_anchors(path: str | Path) -> pl.DataFrame:
    """Read a BED anchor file the way the pair table expects it; see `touche.anchors.read_bed_anchors`."""
    return read_bed_anchors(path)


def _anchor_key(chrom: pl.Expr | str, center: pl.Expr | str) -> pl.Expr:
    """`chrom:center` string expression, the atom both `pair_id` styles are built from."""
    chrom_expr = pl.col(chrom) if isinstance(chrom, str) else chrom
    center_expr = pl.col(center) if isinstance(center, str) else center
    return chrom_expr.cast(pl.Utf8) + pl.lit(":") + center_expr.cast(pl.Int64).cast(pl.Utf8)


def _digest(key: str) -> str:
    """16-hex-digit BLAKE2b digest of a canonical anchor key; backs `id_style="digest"`."""
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()


def _bait_anchor_swap(bait_anchor: str) -> pl.Expr:
    """Boolean expression selecting rows whose BEDPE anchor 2 should become the bait."""
    if bait_anchor == "first":
        return pl.lit(False)
    if bait_anchor == "second":
        return pl.lit(True)
    anchor_1_is_left = (pl.col("chrom1") < pl.col("chrom2")) | (
        (pl.col("chrom1") == pl.col("chrom2")) & (pl.col("start1") <= pl.col("start2"))
    )
    return anchor_1_is_left if bait_anchor == "right" else ~anchor_1_is_left


def _prefix_anchors(anchors: pl.DataFrame, role: str) -> pl.DataFrame:
    """Rename a `read_bed_anchors` frame's columns into `{role}_*` pair-table columns."""
    return anchors.select(
        pl.col("chr").cast(pl.Utf8).alias(f"{role}_chrom"),
        pl.col("start").cast(pl.Int64).alias(f"{role}_start"),
        pl.col("end").cast(pl.Int64).alias(f"{role}_end"),
        pl.col("strand").cast(pl.Utf8).alias(f"{role}_strand"),
        pl.col("center").cast(pl.Int64).alias(f"{role}_center"),
    )


def _validate_bedpe_intervals(raw: pl.DataFrame, path: str | Path) -> None:
    """Reject negative or empty BEDPE intervals rather than silently producing bad centers."""
    for suffix in ("1", "2"):
        bad = raw.filter((pl.col(f"start{suffix}") < 0) | (pl.col(f"end{suffix}") <= pl.col(f"start{suffix}")))
        if bad.height:
            raise ValueError(
                f"{path} has {bad.height} rows with an invalid anchor {suffix} interval "
                "(expected 0-based half-open with start >= 0 and end > start)"
            )


def _finalize_pairs(
    pairs: pl.DataFrame,
    *,
    id_style: str,
    extra_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Derive `pair_id`/`bait_id`/`prey_id`/`chrom`/`is_cis` and order columns as `PAIR_COLUMNS`."""
    pairs = pairs.with_columns(
        make_pair_ids("bait_chrom", "bait_center", "prey_chrom", "prey_center", style=id_style),
        _anchor_key("bait_chrom", "bait_center").alias("bait_id"),
        _anchor_key("prey_chrom", "prey_center").alias("prey_id"),
        (pl.col("bait_chrom") == pl.col("prey_chrom")).alias("is_cis"),
    )
    pairs = pairs.with_columns(
        pl.when(pl.col("is_cis")).then(pl.col("bait_chrom")).otherwise(None).alias("chrom"),
        pl.when(pl.col("is_cis")).then(pl.col("distance")).otherwise(None).alias("distance"),
        pl.when(pl.col("is_cis")).then(pl.col("directional_distance")).otherwise(None).alias("directional_distance"),
    )
    ordered = pairs.select([pl.col(name).cast(dtype) for name, dtype in PAIR_SCHEMA.items()])
    if extra_columns:
        ordered = ordered.with_columns([pairs[name] for name in extra_columns])
    return ordered
