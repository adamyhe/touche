from __future__ import annotations

import json
import re
from array import array
from collections.abc import Collection
from pathlib import Path

import numpy as np

from touche import __version__
from touche.io import PairRecord, iter_pair_records
from touche.models import ContactIndex
from touche.preprocess import _freeze_stats, _new_stats, _observe_record, write_qc_payload


def build_contact_indexes(
    pairs_path: str | Path,
    *,
    source: str = "auto",
    cis_only: bool = True,
    include_metadata: bool = True,
    chromosomes: Collection[str] | None = None,
) -> dict[str, ContactIndex]:
    """Build chromosome-sharded numeric indexes from canonical/distiller pairs."""

    chromosome_filter = set(chromosomes) if chromosomes is not None else None
    builders: dict[str, dict[str, array]] = {}
    for _, record in iter_pair_records(pairs_path, source=source):
        chrom_a, pos_a, chrom_b, pos_b, strand_a, strand_b, mapq_a, mapq_b = record
        if cis_only and chrom_a != chrom_b:
            continue
        chrom = chrom_a
        if chromosome_filter is not None and chrom not in chromosome_filter:
            continue
        builder = builders.setdefault(chrom, _new_contact_builder())
        builder["pos_a"].append(pos_a)
        builder["pos_b"].append(pos_b)
        if include_metadata:
            builder["strand_a"].append(_strand_code(strand_a))
            builder["strand_b"].append(_strand_code(strand_b))
            builder["mapq_a"].append(mapq_a)
            builder["mapq_b"].append(mapq_b)

    indexes = {}
    for chrom, builder in builders.items():
        indexes[chrom] = _builder_to_index(chrom, builder, include_metadata=include_metadata)
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
        path = _write_npz_cache_shard(cache_dir, prefix=prefix, index=index, saver=saver)
        manifest["chromosomes"][chrom] = {"path": path.name, "rows": index.size}
        written.append(path)

    manifest_path = cache_dir / f"{prefix}.manifest.json"
    _write_cache_manifest(manifest_path, manifest)
    written.append(manifest_path)
    return written


def load_npz_cache(path: str | Path, *, include_metadata: bool = True) -> ContactIndex:
    with np.load(path, allow_pickle=False) as data:
        pos_a = data["pos_a"]
        size = int(pos_a.shape[0])
        if include_metadata:
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
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / f"{prefix}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chromosomes = manifest.get("chromosomes", {})
    return {
        str(chrom): cache_dir / str(record["path"])
        for chrom, record in chromosomes.items()
        if isinstance(record, dict) and "path" in record
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
) -> list[Path]:
    if index_strategy not in {"chromosome", "all"}:
        raise ValueError("index_strategy must be one of: chromosome, all")
    if index_strategy == "all":
        indexes = build_contact_indexes(
            pairs_path,
            source=source,
            cis_only=cis_only,
            include_metadata=include_metadata,
        )
        return write_npz_cache(
            indexes,
            cache_dir,
            prefix=prefix,
            compressed=compressed,
            source=pairs_path,
        ) + _write_optional_cache_qc(qc_out, pairs_path=pairs_path, source=source)

    builders: dict[str, dict[str, array]] = {}
    stats = _new_stats() if qc_out is not None else None
    for _, record in iter_pair_records(pairs_path, source=source):
        if stats is not None:
            _observe_record(stats, record)
        if not _cache_record(record, builders, cis_only=cis_only, include_metadata=include_metadata):
            continue

    return _write_builders_npz_cache(
        builders,
        cache_dir,
        prefix=prefix,
        compressed=compressed,
        source=pairs_path,
        include_metadata=include_metadata,
        qc_out=qc_out,
        stats=stats,
    )


def _cache_record(
    record: PairRecord,
    builders: dict[str, dict[str, array]],
    *,
    cis_only: bool,
    include_metadata: bool,
) -> bool:
    chrom_a, pos_a, chrom_b, pos_b, strand_a, strand_b, mapq_a, mapq_b = record
    if cis_only and chrom_a != chrom_b:
        return False
    builder = builders.setdefault(chrom_a, _new_contact_builder())
    builder["pos_a"].append(pos_a)
    builder["pos_b"].append(pos_b)
    if include_metadata:
        builder["strand_a"].append(_strand_code(strand_a))
        builder["strand_b"].append(_strand_code(strand_b))
        builder["mapq_a"].append(mapq_a)
        builder["mapq_b"].append(mapq_b)
    return True


def _write_builders_npz_cache(
    builders: dict[str, dict[str, array]],
    cache_dir: str | Path,
    *,
    prefix: str,
    compressed: bool,
    source: str | Path,
    include_metadata: bool,
    qc_out: str | Path | None,
    stats: dict[str, object] | None,
) -> list[Path]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if compressed else np.savez
    manifest = {
        "schema_version": 1,
        "touche_version": __version__,
        "source": str(source),
        "prefix": prefix,
        "compressed": compressed,
        "chromosomes": {},
    }
    written: list[Path] = []
    for chrom, builder in sorted(builders.items()):
        index = _builder_to_index(chrom, builder, include_metadata=include_metadata)
        path = _write_npz_cache_shard(cache_dir, prefix=prefix, index=index, saver=saver)
        manifest["chromosomes"][chrom] = {"path": path.name, "rows": index.size}
        written.append(path)

    manifest_path = cache_dir / f"{prefix}.manifest.json"
    _write_cache_manifest(manifest_path, manifest)
    written.append(manifest_path)
    if qc_out is not None:
        if stats is None:
            raise ValueError("stats are required when qc_out is provided")
        write_qc_payload(qc_out, stats=_freeze_stats(stats), source=source)
        written.append(Path(qc_out))
    return written


def _write_optional_cache_qc(
    qc_out: str | Path | None, *, pairs_path: str | Path, source: str
) -> list[Path]:
    if qc_out is None:
        return []
    stats = _new_stats()
    for _, record in iter_pair_records(pairs_path, source=source):
        _observe_record(stats, record)
    write_qc_payload(qc_out, stats=_freeze_stats(stats), source=pairs_path)
    return [Path(qc_out)]


def _new_contact_builder() -> dict[str, array]:
    return {
        "pos_a": array("q"),
        "pos_b": array("q"),
        "strand_a": array("b"),
        "strand_b": array("b"),
        "mapq_a": array("h"),
        "mapq_b": array("h"),
    }


def _builder_to_index(
    chrom: str, builder: dict[str, array], *, include_metadata: bool
) -> ContactIndex:
    pos_a = np.frombuffer(builder["pos_a"], dtype=np.int64)
    pos_b = np.frombuffer(builder["pos_b"], dtype=np.int64)
    size = int(pos_a.shape[0])
    if include_metadata:
        strand_a = np.frombuffer(builder["strand_a"], dtype=np.int8)
        strand_b = np.frombuffer(builder["strand_b"], dtype=np.int8)
        mapq_a = np.frombuffer(builder["mapq_a"], dtype=np.int16)
        mapq_b = np.frombuffer(builder["mapq_b"], dtype=np.int16)
    else:
        strand_a = np.zeros(size, dtype=np.int8)
        strand_b = np.zeros(size, dtype=np.int8)
        mapq_a = np.zeros(size, dtype=np.int16)
        mapq_b = np.zeros(size, dtype=np.int16)
    return ContactIndex(
        chrom=chrom,
        pos_a=pos_a,
        pos_b=pos_b,
        strand_a=strand_a,
        strand_b=strand_b,
        mapq_a=mapq_a,
        mapq_b=mapq_b,
    ).sorted_by_pos_a()


def _write_npz_cache_shard(cache_dir: Path, *, prefix: str, index: ContactIndex, saver) -> Path:
    path = cache_dir / f"{prefix}.{_safe_name(index.chrom)}.npz"
    saver(
        path,
        chrom=np.asarray(index.chrom),
        pos_a=index.pos_a,
        pos_b=index.pos_b,
        strand_a=index.strand_a,
        strand_b=index.strand_b,
        mapq_a=index.mapq_a,
        mapq_b=index.mapq_b,
        schema_version=np.asarray(1, dtype=np.int16),
        touche_version=np.asarray(__version__),
    )
    return path


def _write_cache_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _strand_code(value: str) -> int:
    return 1 if value == "+" else -1
