from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from touche import __version__
from touche.io import iter_pairs
from touche.models import ContactIndex


def build_contact_indexes(
    pairs_path: str | Path,
    *,
    source: str = "auto",
    cis_only: bool = True,
) -> dict[str, ContactIndex]:
    """Build chromosome-sharded numeric indexes from canonical/distiller pairs."""

    builders: dict[str, dict[str, list]] = {}
    for _, pair, _ in iter_pairs(pairs_path, source=source):
        if cis_only and pair.chrom_a != pair.chrom_b:
            continue
        chrom = pair.chrom_a
        builder = builders.setdefault(
            chrom,
            {
                "pos_a": [],
                "pos_b": [],
                "strand_a": [],
                "strand_b": [],
                "mapq_a": [],
                "mapq_b": [],
            },
        )
        builder["pos_a"].append(pair.pos_a)
        builder["pos_b"].append(pair.pos_b)
        builder["strand_a"].append(pair.strand_a)
        builder["strand_b"].append(pair.strand_b)
        builder["mapq_a"].append(pair.mapq_a)
        builder["mapq_b"].append(pair.mapq_b)

    indexes = {}
    for chrom, builder in builders.items():
        index = ContactIndex(
            chrom=chrom,
            pos_a=np.asarray(builder["pos_a"], dtype=np.int64),
            pos_b=np.asarray(builder["pos_b"], dtype=np.int64),
            strand_a=np.asarray(builder["strand_a"], dtype="U1"),
            strand_b=np.asarray(builder["strand_b"], dtype="U1"),
            mapq_a=np.asarray(builder["mapq_a"], dtype=np.int16),
            mapq_b=np.asarray(builder["mapq_b"], dtype=np.int16),
        ).sorted_by_pos_a()
        indexes[chrom] = index
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
    written: list[Path] = []
    manifest = {
        "schema_version": 1,
        "touche_version": __version__,
        "source": str(source) if source is not None else None,
        "prefix": prefix,
        "compressed": compressed,
        "chromosomes": {},
    }
    saver = np.savez_compressed if compressed else np.savez
    for chrom, index in sorted(indexes.items()):
        path = cache_dir / f"{prefix}.{_safe_name(chrom)}.npz"
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
        manifest["chromosomes"][chrom] = {
            "path": path.name,
            "rows": index.size,
        }
        written.append(path)

    manifest_path = cache_dir / f"{prefix}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written.append(manifest_path)
    return written


def load_npz_cache(path: str | Path) -> ContactIndex:
    with np.load(path, allow_pickle=False) as data:
        return ContactIndex(
            chrom=str(data["chrom"]),
            pos_a=data["pos_a"],
            pos_b=data["pos_b"],
            strand_a=data["strand_a"],
            strand_b=data["strand_b"],
            mapq_a=data["mapq_a"],
            mapq_b=data["mapq_b"],
        )


def build_npz_cache(
    pairs_path: str | Path,
    cache_dir: str | Path,
    *,
    source: str = "auto",
    prefix: str = "contacts",
    compressed: bool = False,
    cis_only: bool = True,
) -> list[Path]:
    indexes = build_contact_indexes(pairs_path, source=source, cis_only=cis_only)
    return write_npz_cache(
        indexes,
        cache_dir,
        prefix=prefix,
        compressed=compressed,
        source=pairs_path,
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
