# Changelog

All notable changes to `touche` (PyPI: `ep-touche`) are recorded here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/). While the version is
`0.x`, minor releases may change defaults; any such change is listed under
**Changed** with the options that restore the previous behavior.

## [Unreleased]

### Documentation

- Final legacy-vs-`touche` timing numbers in
  [docs/performance.md](docs/performance.md), produced by
  `scripts/legacy_timing_comparison.py`.
- `preprocess build-cache --compressed` help and docs now state its measured
  cost: ~3.4x smaller shards, but ~40x slower to write and ~5x slower to load.
- This changelog and a [roadmap](ROADMAP.md), linked from the package
  metadata alongside the documentation index, CLI reference, and citation.

## [0.1.4] - 2026-08-21

### Added

- `--index-strategy cache` (with `--cache-dir`, `--require-cache`) for
  `apa aggregate`/`run` and `background count`/`run`, so one NPZ cache can be
  shared across commands on the same pairs file. Run wrappers namespace a
  shared cache directory per sample.
- `CONTRIBUTING.md` and `CITATION.cff`.
- [docs/performance.md](docs/performance.md) comparing `touche` with the
  reference implementation.

### Changed

- Pair QC aggregations are fused into a single scan (~3x faster on gzipped
  input).
- `scripts/reference_replication.py --jobs` defaults to the available cores.

## [0.1.3] - 2026-07-31

### Changed

- CI now tests Python 3.10 through 3.14.

## [0.1.2] - 2026-07-31

### Removed

- The unused scalar `fisher_greater`; `fisher_greater_batch` remains.

### Fixed

- Stale docstrings in the background counting and statistics modules.

## [0.1.1] - 2026-07-09

### Added

- Numba-accelerated APA, background, and local-decay kernels, verified
  exact-equivalent to the NumPy implementations they replace. Numba is a
  core dependency.
- Progress and profiling instrumentation (`--progress`, `--profile`).
- Docstrings across the package, improved CLI help, and rendered demo images
  in the replication guide.
- Documentation of the `polars-lts-cpu` workaround for older CPUs and Rosetta.

### Changed

- `polars` replaces `pandas` as the only dataframe library, internally and at
  the public `touche.api` surface. CSV outputs no longer carry an index
  column.
- NPZ cache building spools to Parquet and writes one chromosome at a time,
  bounding peak memory by the largest chromosome.
- Faster EP/background counting with sorted-window searches.
- Dependency lower bounds added.

### Fixed

- Python 3.10 compatibility.

## [0.1.0] - 2026-07-07

### Added

- Initial Python port of the Danko Lab
  [E-P_contacts](https://github.com/Danko-Lab/E-P_contacts) workflows: pairs
  preprocessing and QC, local-decay contact calling, APA aggregation and
  comparison, EP/background counting and comparison, and `run` pipeline
  wrappers with JSON manifests.

[Unreleased]: https://github.com/adamyhe/touche/compare/f6c6b18...HEAD
[0.1.4]: https://github.com/adamyhe/touche/commit/f6c6b18
[0.1.3]: https://github.com/adamyhe/touche/commit/83f7210
[0.1.2]: https://github.com/adamyhe/touche/commit/b81584d
[0.1.1]: https://github.com/adamyhe/touche/commit/3012e0b
[0.1.0]: https://github.com/adamyhe/touche/commit/337ff1c
