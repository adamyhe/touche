"""Build `ContactIndex` objects from pairs files and cache them as chromosome-sharded NPZ files.

Public API: `build_contact_indexes`, `build_npz_cache`, `load_npz_cache`,
`load_npz_cache_manifest`, `write_npz_cache`, `ensure_npz_cache`,
`load_cached_contact_indexes`. Everything else (the
`_build_*`/`_write_*`/`_contact_index_from_frame` helpers) is internal
plumbing specific to one of the two `index_strategy` code paths in
`build_npz_cache` -- see CLAUDE.md's "Contact indexing strategies" section.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Collection
from pathlib import Path

import numpy as np
import polars as pl

from touche import __version__
from touche.io import scan_pairs
from touche.models import ContactIndex
from touche.pair_stats import compute_pair_stats, write_qc_payload

_METADATA_COLUMNS = ["strand_a", "strand_b", "mapq_a", "mapq_b"]


def _strand_code_expr(column: str) -> pl.Expr:
    """Map a `"+"`/`"-"` strand column to `+1`/`-1` (the numeric convention `ContactIndex` uses)."""
    return pl.when(pl.col(column) == "+").then(1).otherwise(-1).cast(pl.Int8).alias(column)


def _prepared_pairs_lazyframe(
    pairs_path: str | Path,
    *,
    source: str,
    cis_only: bool,
    include_metadata: bool,
) -> pl.LazyFrame:
    """Scan `pairs_path` and select/encode only the columns `build_contact_indexes` needs."""
    lf = scan_pairs(pairs_path, source=source)
    if cis_only:
        lf = lf.filter(pl.col("chrom_a") == pl.col("chrom_b"))
    columns = ["chrom_a", "pos_a", "pos_b"]
    if include_metadata:
        lf = lf.with_columns(_strand_code_expr("strand_a"), _strand_code_expr("strand_b"))
        columns += _METADATA_COLUMNS
    return lf.select(columns)


def build_contact_indexes(
    pairs_path: str | Path,
    *,
    source: str = "auto",
    cis_only: bool = True,
    include_metadata: bool = True,
    chromosomes: Collection[str] | None = None,
) -> dict[str, ContactIndex]:
    """Build chromosome-sharded numeric indexes from canonical/distiller pairs."""

    lf = _prepared_pairs_lazyframe(
        pairs_path, source=source, cis_only=cis_only, include_metadata=include_metadata
    )
    if chromosomes is not None:
        lf = lf.filter(pl.col("chrom_a").is_in(list(chromosomes)))

    df = lf.collect(engine="streaming")
    indexes: dict[str, ContactIndex] = {}
    for chrom in df["chrom_a"].unique(maintain_order=True).to_list():
        chrom_df = df.filter(pl.col("chrom_a") == chrom)
        indexes[chrom] = _contact_index_from_frame(
            chrom, chrom_df, include_metadata=include_metadata
        )
    return indexes


def write_npz_cache(
    indexes: dict[str, ContactIndex],
    cache_dir: str | Path,
    *,
    prefix: str = "contacts",
    compressed: bool = False,
    source: str | Path | None = None,
) -> list[Path]:
    """Write one versioned NPZ cache file per chromosome."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "touche_version": __version__,
        "source": str(source) if source is not None else None,
        "prefix": prefix,
        "compressed": compressed,
        "chromosomes": {},
    }
    saver = np.savez_compressed if compressed else np.savez
    written: list[Path] = []
    for chrom, index in sorted(indexes.items()):
        path = _write_npz_cache_shard(
            cache_dir,
            prefix=prefix,
            index=index,
            saver=saver,
            include_metadata=True,
        )
        manifest["chromosomes"][chrom] = {"path": path.name, "rows": index.size}
        written.append(path)

    manifest_path = cache_dir / f"{prefix}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written.append(manifest_path)
    return written


def load_npz_cache(path: str | Path, *, include_metadata: bool = True) -> ContactIndex:
    """Load one chromosome shard written by `write_npz_cache`/`build_npz_cache`.

    If `include_metadata` is False, or the shard was written without strand/
    mapq arrays, those fields are filled with zeros rather than read.
    """
    with np.load(path, allow_pickle=False) as data:
        pos_a = data["pos_a"]
        size = int(pos_a.shape[0])
        has_metadata = all(key in data for key in ("strand_a", "strand_b", "mapq_a", "mapq_b"))
        if include_metadata and has_metadata:
            strand_a = data["strand_a"]
            strand_b = data["strand_b"]
            mapq_a = data["mapq_a"]
            mapq_b = data["mapq_b"]
        else:
            strand_a = np.zeros(size, dtype=np.int8)
            strand_b = np.zeros(size, dtype=np.int8)
            mapq_a = np.zeros(size, dtype=np.int16)
            mapq_b = np.zeros(size, dtype=np.int16)
        return ContactIndex(
            chrom=str(data["chrom"]),
            pos_a=pos_a,
            pos_b=data["pos_b"],
            strand_a=strand_a,
            strand_b=strand_b,
            mapq_a=mapq_a,
            mapq_b=mapq_b,
        )


def load_npz_cache_manifest(cache_dir: str | Path, *, prefix: str = "contacts") -> dict[str, Path]:
    """Read a cache manifest and return `{chromosome: shard path}`."""
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / f"{prefix}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chromosomes = manifest.get("chromosomes", {})
    return {
        str(chrom): cache_dir / str(record["path"])
        for chrom, record in chromosomes.items()
        if isinstance(record, dict) and "path" in record
    }


def ensure_npz_cache(
    pairs_path: str | Path,
    *,
    cache_dir: str | Path,
    cache_prefix: str = "contacts",
    source: str = "auto",
    include_metadata: bool = True,
    require_cache: bool = False,
) -> None:
    """Build the NPZ cache at `cache_dir` if its manifest is missing, unless `require_cache` demands it exist."""
    manifest_path = Path(cache_dir) / f"{cache_prefix}.manifest.json"
    if manifest_path.exists():
        return
    if require_cache:
        raise FileNotFoundError(
            f"Required contact-index cache manifest is missing: {manifest_path}. "
            "Run `touche preprocess build-cache` first or disable require_cache."
        )
    build_npz_cache(
        pairs_path,
        cache_dir,
        source=source,
        prefix=cache_prefix,
        cis_only=True,
        include_metadata=include_metadata,
        index_strategy="chromosome",
    )


def load_cached_contact_indexes(
    pairs_path: str | Path,
    *,
    cache_dir: str | Path,
    cache_prefix: str = "contacts",
    source: str = "auto",
    include_metadata: bool = True,
    require_cache: bool = False,
) -> dict[str, ContactIndex]:
    """Load every chromosome's `ContactIndex` from an NPZ cache, building it first if missing.

    Unlike local-decay's `index_strategy="cache"` path (one chromosome shard
    loaded at a time, bounding memory to the largest chromosome), this loads
    every shard up front: APA/background already hold every chromosome's
    index in memory at once (their default `build_contact_indexes` call), so
    this only trades a raw-pairs-file parse for a faster NPZ-shard read, with
    no existing memory-bounded behavior to preserve.
    """
    ensure_npz_cache(
        pairs_path,
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
        source=source,
        include_metadata=include_metadata,
        require_cache=require_cache,
    )
    cache_paths = load_npz_cache_manifest(cache_dir, prefix=cache_prefix)
    return {
        chrom: load_npz_cache(path, include_metadata=include_metadata)
        for chrom, path in cache_paths.items()
    }


def build_npz_cache(
    pairs_path: str | Path,
    cache_dir: str | Path,
    *,
    source: str = "auto",
    prefix: str = "contacts",
    compressed: bool = False,
    cis_only: bool = True,
    include_metadata: bool = True,
    index_strategy: str = "chromosome",
    qc_out: str | Path | None = None,
    write_qc: bool = True,
) -> list[Path]:
    """Build a chromosome-sharded NPZ cache, dispatching to the `index_strategy` code path.

    `index_strategy="all"` reads the whole pairs file into memory at once
    (`_build_all_cache`); `"chromosome"` (the default) spools to a local
    Parquet file first and materializes one chromosome at a time
    (`_build_sharded_cache_from_spool`), bounding peak memory to the largest
    single chromosome instead of the whole file.
    """
    if index_strategy not in {"chromosome", "all"}:
        raise ValueError("index_strategy must be one of: chromosome, all")
    qc_path = _resolve_qc_out(cache_dir, prefix=prefix, qc_out=qc_out, write_qc=write_qc)
    if index_strategy == "all":
        return _build_all_cache(
            pairs_path,
            cache_dir,
            source=source,
            prefix=prefix,
            compressed=compressed,
            cis_only=cis_only,
            include_metadata=include_metadata,
            qc_out=qc_path,
        )

    return _build_sharded_cache_from_spool(
        pairs_path,
        cache_dir,
        source=source,
        prefix=prefix,
        compressed=compressed,
        cis_only=cis_only,
        include_metadata=include_metadata,
        qc_out=qc_path,
    )


def _build_all_cache(
    pairs_path: str | Path,
    cache_dir: str | Path,
    *,
    source: str,
    prefix: str,
    compressed: bool,
    cis_only: bool,
    include_metadata: bool,
    qc_out: str | Path | None,
) -> list[Path]:
    """`index_strategy="all"`: hold every chromosome in memory, then write shards."""
    indexes = build_contact_indexes(
        pairs_path,
        source=source,
        cis_only=cis_only,
        include_metadata=include_metadata,
    )
    written = write_npz_cache(
        indexes,
        cache_dir,
        prefix=prefix,
        compressed=compressed,
        source=pairs_path,
    )
    return written + _write_optional_cache_qc(qc_out, pairs_path=pairs_path, source=source)


def _resolve_qc_out(
    cache_dir: str | Path,
    *,
    prefix: str,
    qc_out: str | Path | None,
    write_qc: bool,
) -> Path | None:
    """Resolve the default `{prefix}.qc.json` path next to the cache, or None if QC is disabled."""
    if not write_qc:
        return None
    if qc_out is not None:
        return Path(qc_out)
    return Path(cache_dir) / f"{prefix}.qc.json"


def _build_sharded_cache_from_spool(
    pairs_path: str | Path,
    cache_dir: str | Path,
    *,
    source: str,
    prefix: str,
    compressed: bool,
    cis_only: bool,
    include_metadata: bool,
    qc_out: str | Path | None,
) -> list[Path]:
    """Build one NPZ shard per chromosome with bounded peak memory.

    The pairs file is scanned exactly once into a local Parquet spool (a
    streaming write, never holding the full file in the Python heap); every
    later pass (QC aggregation, per-chromosome shard writes) reads from that
    fast local spool instead of re-decompressing the source file, and each
    per-chromosome collect only ever materializes that one chromosome's rows.
    """

    with tempfile.TemporaryDirectory(prefix="touche-cache-spool-") as spool_dir:
        spool_path = Path(spool_dir) / "pairs.parquet"
        raw_columns = [
            "chrom_a",
            "chrom_b",
            "pos_a",
            "pos_b",
            "strand_a",
            "strand_b",
            "mapq_a",
            "mapq_b",
        ]
        scan_pairs(pairs_path, source=source).select(raw_columns).sink_parquet(spool_path)

        stats = None
        if qc_out is not None:
            stats = compute_pair_stats(pl.scan_parquet(spool_path))

        lf = pl.scan_parquet(spool_path)
        if cis_only:
            lf = lf.filter(pl.col("chrom_a") == pl.col("chrom_b"))
        if include_metadata:
            lf = lf.with_columns(_strand_code_expr("strand_a"), _strand_code_expr("strand_b"))
            columns = ["chrom_a", "pos_a", "pos_b", *_METADATA_COLUMNS]
        else:
            columns = ["chrom_a", "pos_a", "pos_b"]
        lf = lf.select(columns)

        chromosomes = lf.select("chrom_a").unique().collect(engine="streaming")["chrom_a"].to_list()

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        saver = np.savez_compressed if compressed else np.savez
        manifest = {
            "schema_version": 1,
            "touche_version": __version__,
            "source": str(pairs_path),
            "prefix": prefix,
            "compressed": compressed,
            "chromosomes": {},
        }
        written: list[Path] = []
        for chrom in sorted(chromosomes):
            chrom_df = lf.filter(pl.col("chrom_a") == chrom).collect(engine="streaming")
            index = _contact_index_from_frame(chrom, chrom_df, include_metadata=include_metadata)
            path = _write_npz_cache_shard(
                cache_dir,
                prefix=prefix,
                index=index,
                saver=saver,
                include_metadata=include_metadata,
            )
            manifest["chromosomes"][chrom] = {"path": path.name, "rows": index.size}
            written.append(path)

        manifest_path = cache_dir / f"{prefix}.manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append(manifest_path)

        if qc_out is not None:
            assert stats is not None
            write_qc_payload(qc_out, stats=stats, source=pairs_path)
            written.append(Path(qc_out))
        return written


def _write_optional_cache_qc(
    qc_out: str | Path | None, *, pairs_path: str | Path, source: str
) -> list[Path]:
    """`_build_all_cache`'s QC pass: re-scan `pairs_path` once more and write the QC JSON."""
    if qc_out is None:
        return []
    columns = ["chrom_a", "chrom_b", "pos_a", "pos_b", "mapq_a", "mapq_b"]
    lf = scan_pairs(pairs_path, source=source).select(columns)
    stats = compute_pair_stats(lf)
    write_qc_payload(qc_out, stats=stats, source=pairs_path)
    return [Path(qc_out)]


def _contact_index_from_frame(
    chrom: str, frame: pl.DataFrame, *, include_metadata: bool
) -> ContactIndex:
    """Convert one chromosome's collected rows into a `ContactIndex`, sorting by `pos_a` if needed."""
    pos_a = frame["pos_a"].to_numpy().astype(np.int64)
    pos_b = frame["pos_b"].to_numpy().astype(np.int64)
    size = pos_a.shape[0]
    if include_metadata:
        strand_a = frame["strand_a"].to_numpy().astype(np.int8)
        strand_b = frame["strand_b"].to_numpy().astype(np.int8)
        mapq_a = frame["mapq_a"].to_numpy().astype(np.int16)
        mapq_b = frame["mapq_b"].to_numpy().astype(np.int16)
    else:
        strand_a = np.zeros(size, dtype=np.int8)
        strand_b = np.zeros(size, dtype=np.int8)
        mapq_a = np.zeros(size, dtype=np.int16)
        mapq_b = np.zeros(size, dtype=np.int16)
    index = ContactIndex(
        chrom=chrom,
        pos_a=pos_a,
        pos_b=pos_b,
        strand_a=strand_a,
        strand_b=strand_b,
        mapq_a=mapq_a,
        mapq_b=mapq_b,
    )
    if size > 1 and np.any(pos_a[:-1] > pos_a[1:]):
        return index.sorted_by_pos_a()
    return index


def _write_npz_cache_shard(
    cache_dir: Path,
    *,
    prefix: str,
    index: ContactIndex,
    saver,
    include_metadata: bool,
) -> Path:
    """Write one chromosome's `ContactIndex` to `{cache_dir}/{prefix}.{safe_chrom}.npz`."""
    safe_chrom = re.sub(r"[^A-Za-z0-9_.-]+", "_", index.chrom)
    path = cache_dir / f"{prefix}.{safe_chrom}.npz"
    payload = {
        "chrom": np.asarray(index.chrom),
        "pos_a": index.pos_a,
        "pos_b": index.pos_b,
        "schema_version": np.asarray(1, dtype=np.int16),
        "touche_version": np.asarray(__version__),
        "include_metadata": np.asarray(include_metadata),
    }
    if include_metadata:
        payload.update(
            {
                "strand_a": index.strand_a,
                "strand_b": index.strand_b,
                "mapq_a": index.mapq_a,
                "mapq_b": index.mapq_b,
            }
        )
    saver(path, **payload)
    return path
