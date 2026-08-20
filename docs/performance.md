# Performance vs. the reference implementation

`touche` reimplements the workflows from
[Danko-Lab/E-P_contacts](https://github.com/Danko-Lab/E-P_contacts) (bash/awk
scripts calling per-locus Python/R). This guide compares the two along three
axes: runtime, intermediate/temp files written, and memory shape. It's based
on reading the reference scripts directly (cited by file/function below), not
guesswork.

End-to-end wall-clock and peak-RSS numbers from running both pipelines on the
same real data are pending a controlled run (see
[`scripts/reference_replication.py`](../scripts/reference_replication.py)) and
will be added here once collected. Everything below is a structural
comparison: what each pipeline actually does per bait/prey/chromosome, and the
isolated, already-verified speedups from `touche`'s own optimization history
(`notes/numba-implementation-plan.md`).

## Runtime

### Local-decay

The reference `ContactCaller_microC.bsh` spawns one OS process per bait:
`call_loop_bg` invokes `python ContactCaller_microC.py` once per bait, gated
by a `wait_a_second` bash loop that busy-polls `jobs -p` to cap concurrency at
30 processes. Each of those processes:

- re-imports pandas/statsmodels/scipy/rpy2/`fast_histogram` from scratch;
- re-reads a bait-specific gzip temp file the shell script split out for it;
- runs LOWESS twice via `statsmodels`, with the zero-inflation density loop
  (`optimize_lowess`) written as an explicit Python `for k in range(...)`
  loop — one Python-level iteration and one `np.sum` slice per position in
  the window, not vectorized — and the chunked-fit merge (`optimize_lowess2`)
  rebuilding `bgModel` as a plain Python list via repeated `.extend()`, one
  chunk (~200 per bait at the reference's default `dist=1_000_000`) at a
  time;
- calls R's `fisher.test` through `rpy2` inside a Python `for` loop, one R
  round-trip per prey — no batching.

`touche local-decay call` runs as one process for the whole bait set. Each
chromosome's contacts are loaded once into a `ContactIndex` and sorted once
(`_ordered_cis_index`, `src/touche/local_decay.py`); each bait's window is
then a `np.searchsorted` slice (O(log n)) instead of a per-bait disk
read+`awk` re-derivation. LOWESS runs through a batched, parallel Numba
kernel; Fisher testing has a Numba-vectorized batch path
(`fisher_backend="numba"`) that scores every prey for a bait in one call
instead of one R round-trip per prey. Baits are optionally fanned out across
a thread pool (`--jobs`) with each worker capping its own Numba thread budget
to avoid oversubscription — one bounded thread pool, not up to 30 forked
processes each paying Python/pandas/rpy2 import startup.

The reference Python script's chunk-merge loop has the same
list-rebuild-per-chunk shape that `touche`'s own Numba implementation
initially had and then fixed: replacing a per-chunk `np.concatenate` rebuild
with a pre-allocated buffer measured **~5.8x** on the isolated function at
`dist=1_000_000` (0.094s → 0.016s) and **~8.2x** end-to-end on top of LOWESS
batching, on a 40-bait synthetic benchmark (`notes/numba-implementation-plan.md`).
The reference implementation still has this cost, in pure Python, paid fresh
in every one of its per-bait processes.

### APA and EP/background

`MicroC_Stranded_Aggregation_pipeline_with_1D_signal.bsh` and
`MicroC_EP_and_BG_contacts.bsh` both `zcat` the *entire* compressed pairs file
once per chromosome (20–24 full decompress+scan passes of the same file,
one per mouse/human chromosome) just to split it by chromosome. Every bait
and every prey then gets its own `awk` pass over that chromosome's full
contact list to carve out its window (`baits_IN_contacts_folder/*.bed`,
`preys_IN_contacts_folder/*.bed` — one file, one full linear scan, per
locus). APA then spawns one Python subprocess per bait for the per-bait
matrix step (`MicroC_Stranded_Aggregation_pipeline_get_bait_matrix.py`); the
background script's final counting step spawns one Python subprocess per
**bait-prey pair** (nested loop over `preys_IN_baits_folder`, calling
`single_pair_contacts_and_background_calculation.py`).

`touche apa aggregate` and `touche background count` read the pairs file
once (optionally through the persistent NPZ `ContactIndex` cache — see
`touche preprocess build-cache`), bound each locus's window with
`np.searchsorted` against sorted position arrays (`src/touche/background.py:184-190`),
and hand each chromosome's full contact arrays to one
`@njit(parallel=True)` kernel call (`apa_matrix_numba`, `apa_anchor_signal_numba`
in `touche.numba.apa`; `count_ep_background_pairs_numba` in
`touche.numba.background`) that counts every locus's window in a single
pass — O(n_contacts) total per chromosome, not O(n_loci × n_contacts), and no
subprocess-per-locus fan-out. `reference_replication.py`'s own benchmark runs
these workflows at real scale (10,530 baits / 27,900 preys), where the
reference's per-locus process/file fan-out is largest.

### Bonus: `preprocess qc`/cache building

Unrelated to any reference counterpart (the reference pipeline has no
equivalent QC pass), `touche`'s own `compute_pair_stats` used to run three
separate `.collect()` calls against the same lazy Polars pipeline, re-scanning
and re-decompressing the source pairs file three times. Fusing them into one
`pl.collect_all()` call measured **~3x** faster on a benchmarked 2M-row
gzipped pairs file, and this scan is shared between `preprocess qc`,
`preprocess summarize`, and cache building, so the fix benefits all three.

## Intermediate / temp files

The reference scripts write a full directory tree of intermediate files that
outlive no step of the run — they're read back by the next `awk`/`cat`/`zcat`
in the pipeline, then left on disk:

| Workflow | Intermediate directories (reference) | Roughly how many files |
| --- | --- | --- |
| local-decay | `baits_by_chrom/`, `preys_by_chrom/` (1/chromosome), `preys_in_baits/` (1/bait), `confile_by_chrom/` (1/chromosome), `conFile/temp_*.gz` (1/bait), per-bait `.txt` before the final `cat` | ~3-4 per bait + a handful per chromosome — 1,500+ on a ~500-bait real run |
| APA | `contacts_shifted_by_chrom/`, `baits_by_chrom/`, `preys_by_chrom/` (1/chromosome each), `preys_IN_baits_folder/`, `baits_IN_contacts_folder/`, `per_bait_matrices_folder/` (1/bait each), `preys_IN_contacts_folder/` (1/prey) | tens of thousands at the benchmark's 10,530-bait/27,900-prey scale |
| EP/background | `contacts_by_chrom/`, `baits_by_chrom/`, `preys_by_chrom/` (1/chromosome each), `preys_IN_baits_folder/`, `baits_IN_contacts_folder/`, `preys_IN_contacts_folder/` (1/locus each) | same order of magnitude as APA |

`touche` writes zero disposable intermediate files for any of these three
workflows. The only files written beyond the final requested output(s) are
the *optional*, explicitly-requested, persistent NPZ `ContactIndex` cache
(`touche preprocess build-cache`) — a reusable artifact at a location you
choose, not disk litter the pipeline generates and never cleans up.

## Memory

The reference pipeline's per-bait-process model bounds *that process's* peak
memory naturally (each one only ever holds a single bait's contact slice),
but every process pays for its own embedded R runtime (`rpy2` starts a full R
interpreter inside each forked Python process) and represents contacts as a
Python list-of-lists-of-strings before a `numpy` object-array conversion
(`contacts = np.array([l.strip().split() for l in contact_fp.readlines()])`
in `ContactCaller_microC.py`) rather than typed arrays. APA/background hold
one chromosome's contact list on disk and re-scan it per locus rather than
indexing it once; concurrency is capped by OS process count (30–60 via
`wait_a_second`), not by measured memory, so peak memory scales with however
many of those processes the scheduler happens to overlap.

`touche`'s `ContactIndex` stores each chromosome's positions/strand/mapq as
compact typed NumPy arrays (`int64`/`int8`). The default `cache`
index-strategy (see the main README's "Contact indexing strategies") loads
one chromosome's NPZ shard into memory at a time, bounding peak memory to the
largest single chromosome rather than the whole genome; the `--jobs` thread
pool shares that one read-only `ContactIndex` across bait workers instead of
each worker holding its own copy. Numba kernels operate in place on those
arrays — no embedded R interpreter, no per-locus subprocess memory overhead.

## `touche`'s own numbers on the real example

[`scripts/reference_replication.py`](../scripts/reference_replication.py) ran
the full example end to end on a 16-core node against the real Danko-Lab
inputs (K562 local-decay at 8.9GB of pairs; DMSO/FLV/TRP APA and
EP/background at 433MB/2.8GB/3.15GB): **~1,318 seconds (~22 minutes)** total
across all 13 profiled steps, from already-downloaded raw pairs to every
local-decay/APA/background output and comparison plot.

| Step | Elapsed (s) | Peak RSS (MB) | CPU % |
| --- | ---: | ---: | ---: |
| preprocess-cache-k562 | 288.1 | 11,813 | 653% |
| local-decay-call | 244.2 | 12,294 | 964% |
| apa-aggregate-trp | 161.6 | 25,127 | 406% |
| apa-aggregate-flv | 146.9 | 22,296 | 405% |
| background-compare | 133.8 | 281 | 123% |
| background-count-trp | 71.2 | 22,202 | 467% |
| preprocess-cache-trp | 79.5 | 4,431 | 708% |
| preprocess-cache-flv | 71.7 | 3,912 | 699% |
| background-count-flv | 62.6 | 19,732 | 467% |
| apa-aggregate-dmso | 19.3 | 3,348 | 443% |
| local-decay-plot | 11.2 | 211 | 40% |
| preprocess-cache-dmso | 11.5 | 942 | 632% |
| background-count-dmso | 9.5 | 2,733 | 467% |
| apa-compare-{flv,trp}-vs-dmso | 2.8 each | 212 | 110% |
| local-decay-assign-pair-types | 1.8 | 105 | 90% |

The `preprocess-cache-dmso/flv/trp` rows above no longer exist as of this
writing: they cost 162.7s (12% of that run's total) building NPZ caches that
nothing downstream reads, since `apa aggregate`/`background count` have no
cache-consuming path (unlike `local-decay call`'s `--index-strategy
cache`/`--cache-dir`/`--require-cache`) — confirmed by reading
`aggregate_apa`/`count_ep_and_background`, which call `build_contact_indexes`
directly on the raw pairs file every time. `reference_replication.py` now
only builds the K562 cache `local-decay-call` actually consumes. The same gap
means `apa-aggregate-flv` and `background-count-flv` each separately re-parse
the same FLV pairs file from scratch (~65s + ~57s); adding cache support to
`apa`/`background` themselves would remove that duplication too, but is a
public CLI/API change scoped separately from this benchmark script fix.

CPU% above 100% reflects multi-core parallelism (Polars' scan engine, Numba's
`parallel=True` kernels, and local-decay's `--jobs` thread pool) — e.g.
`local-decay-call` averaged ~9.6 of the node's 16 cores over its 244s wall
time.

## What isn't measured yet

The comparisons above explain *why* `touche` should be faster and lighter on
disk/memory than the reference implementation, verified where isolated fixes
have before/after numbers, and the table above gives `touche`'s real
standalone cost. What's still missing is the other half: a matching
from-scratch run of the reference pipeline on the same inputs, to turn the
structural comparisons above into a real side-by-side wall-clock/peak-RSS
number.
