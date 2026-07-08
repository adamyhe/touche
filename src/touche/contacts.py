from __future__ import annotations

import json
import re
from array import array
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from touche import __version__
from touche.io import ParsedPairRecord, iter_pair_records
from touche.models import ContactIndex
from touche.pair_stats import (
    PairStatsAccumulator,
    freeze_pair_stats,
    new_pair_stats,
    observe_parsed_pair,
    write_qc_payload,
)

@dataclass
class ContactBuilder:
    pos_a: array = field(default_factory=lambda: array("q"))
    pos_b: array = field(default_factory=lambda: array("q"))
    strand_a: array = field(default_factory=lambda: array("b"))
    strand_b: array = field(default_factory=lambda: array("b"))
    mapq_a: array = field(default_factory=lambda: array("h"))
    mapq_b: array = field(default_factory=lambda: array("h"))
    is_sorted_by_pos_a: bool = True
    last_pos_a: int | None = None

    def append(
        self,
        record: ParsedPairRecord,
        *,
        include_metadata: bool,
    ) -> None:
        if self.last_pos_a is not None and record.pos_a < self.last_pos_a:
            self.is_sorted_by_pos_a = False
        self.last_pos_a = record.pos_a
        self.pos_a.append(record.pos_a)
        self.pos_b.append(record.pos_b)
        if include_metadata:
            self.strand_a.append(_strand_code(record.strand_a))
            self.strand_b.append(_strand_code(record.strand_b))
            self.mapq_a.append(record.mapq_a)
            self.mapq_b.append(record.mapq_b)


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
    builders: dict[str, ContactBuilder] = {}
    for _, record in iter_pair_records(pairs_path, source=source):
        if cis_only and record.chrom_a != record.chrom_b:
            continue
        chrom = record.chrom_a
        if chromosome_filter is not None and chrom not in chromosome_filter:
            continue
        builder = builders.setdefault(chrom, _new_contact_builder())
        builder.append(record, include_metadata=include_metadata)

    indexes = {}
    for chrom, builder in builders.items():
        indexes[chrom] = _contact_index_from_builder(chrom, builder, include_metadata=include_metadata)
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
    _write_cache_manifest(manifest_path, manifest)
    written.append(manifest_path)
    return written


def load_npz_cache(path: str | Path, *, include_metadata: bool = True) -> ContactIndex:
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
    write_qc: bool = True,
) -> list[Path]:
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

    return _build_sharded_cache_from_stream(
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
    if not write_qc:
        return None
    if qc_out is not None:
        return Path(qc_out)
    return Path(cache_dir) / f"{prefix}.qc.json"


def _build_sharded_cache_from_stream(
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
    builders: dict[str, ContactBuilder] = {}
    stats = new_pair_stats() if qc_out is not None else None
    for _, record in iter_pair_records(pairs_path, source=source):
        if stats is not None:
            observe_parsed_pair(stats, record)
        _add_record_to_cache_builder(
            record,
            builders,
            cis_only=cis_only,
            include_metadata=include_metadata,
        )

    return _write_cache_shards(
        builders,
        cache_dir,
        prefix=prefix,
        compressed=compressed,
        source=pairs_path,
        include_metadata=include_metadata,
        qc_out=qc_out,
        stats=stats,
    )


def _add_record_to_cache_builder(
    record: ParsedPairRecord,
    builders: dict[str, ContactBuilder],
    *,
    cis_only: bool,
    include_metadata: bool,
) -> None:
    if cis_only and record.chrom_a != record.chrom_b:
        return
    builder = builders.setdefault(record.chrom_a, _new_contact_builder())
    builder.append(record, include_metadata=include_metadata)


def _write_cache_shards(
    builders: dict[str, ContactBuilder],
    cache_dir: str | Path,
    *,
    prefix: str,
    compressed: bool,
    source: str | Path,
    include_metadata: bool,
    qc_out: str | Path | None,
    stats: PairStatsAccumulator | None,
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
        index = _contact_index_from_builder(chrom, builder, include_metadata=include_metadata)
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
    _write_cache_manifest(manifest_path, manifest)
    written.append(manifest_path)
    if qc_out is not None:
        if stats is None:
            raise ValueError("stats are required when qc_out is provided")
        write_qc_payload(qc_out, stats=freeze_pair_stats(stats), source=source)
        written.append(Path(qc_out))
    return written


def _write_optional_cache_qc(
    qc_out: str | Path | None, *, pairs_path: str | Path, source: str
) -> list[Path]:
    if qc_out is None:
        return []
    stats = new_pair_stats()
    for _, record in iter_pair_records(pairs_path, source=source):
        observe_parsed_pair(stats, record)
    write_qc_payload(qc_out, stats=freeze_pair_stats(stats), source=pairs_path)
    return [Path(qc_out)]


def _new_contact_builder() -> ContactBuilder:
    return ContactBuilder()


def _contact_index_from_builder(
    chrom: str, builder: ContactBuilder, *, include_metadata: bool
) -> ContactIndex:
    pos_a = np.frombuffer(builder.pos_a, dtype=np.int64)
    pos_b = np.frombuffer(builder.pos_b, dtype=np.int64)
    size = int(pos_a.shape[0])
    if include_metadata:
        strand_a = np.frombuffer(builder.strand_a, dtype=np.int8)
        strand_b = np.frombuffer(builder.strand_b, dtype=np.int8)
        mapq_a = np.frombuffer(builder.mapq_a, dtype=np.int16)
        mapq_b = np.frombuffer(builder.mapq_b, dtype=np.int16)
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
    if builder.is_sorted_by_pos_a:
        return index
    return index.sorted_by_pos_a()


def _write_npz_cache_shard(
    cache_dir: Path,
    *,
    prefix: str,
    index: ContactIndex,
    saver,
    include_metadata: bool,
) -> Path:
    path = cache_dir / f"{prefix}.{_safe_name(index.chrom)}.npz"
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


def _write_cache_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _strand_code(value: str) -> int:
    return 1 if value == "+" else -1
