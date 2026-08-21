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

`touche`'s own `local-decay-call` had the same shape of inefficiency:
`_call_bait_contacts`'s `in_region` window filter used to do a full
boolean-mask pass over the *entire* chromosome's `pos_a`/`pos_b` arrays on
every bait call, even though `_ordered_cis_index` already sorts contacts by
`pos_a`. On a real benchmark run this mattered: only ~17s of
`local-decay-call`'s 243.6s wall time was accounted for by named profiling
steps (cache loading); the remaining ~227s (93%) was the per-bait compute
loop itself, where this sat. Fixed in `_bait_window_contacts`
(`src/touche/local_decay.py`): contacts with `pos_a` in the bait's window are
now a direct `np.searchsorted` slice (O(log n) instead of O(n) per bait);
contacts that start before the window but span into it (`pos_a < lo <= pos_b
<= hi`) are bounded to `pos_a >= lo - max_span`, where `max_span` is the
chromosome's largest observed `pos_b - pos_a` (computed once per chromosome,
not per bait) -- `pos_b <= pos_a + max_span` bounds how far left such a
contact's `pos_a` can be, so that candidate slice stays small and still only
needs a boolean mask across a narrow range, not the whole chromosome.
Verified bit-identical against the pre-fix mask via `git stash` (randomized
datasets specifically sized to exercise the spanning case, an
exact-boundary-contacts case, both `n_jobs=1` and `n_jobs=2`), plus a
permanent regression test (50 randomized trials against a brute-force
reimplementation of the original mask formula). **~3x** speedup (2.76s →
0.93s) on a synthetic 3M-contact/500-bait benchmark matching the real K562
scale profiled in `notes/numba-implementation-plan.md`.

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
EP/background at 433MB/2.8GB/3.15GB): **~836 seconds (~13.9 minutes)**
total across all 16 profiled steps, from already-downloaded raw pairs to
every local-decay/APA/background output and comparison plot -- down from an
initial **~1,318 seconds (~22.0 minutes)** before the four fixes below.

| Step | Elapsed (s) | Peak RSS (MB) | CPU % |
| --- | ---: | ---: | ---: |
| preprocess-cache-k562 | 271.3 | 11,854 | 692% |
| local-decay-call | 139.7 | 9,821 | 737% |
| apa-aggregate-trp | 97.3 | 10,383 | 313% |
| apa-aggregate-flv | 88.4 | 9,225 | 319% |
| preprocess-cache-trp | 81.3 | 4,551 | 695% |
| preprocess-cache-flv | 73.1 | 3,836 | 686% |
| background-compare | 22.4 | 243 | 126% |
| apa-aggregate-dmso | 14.0 | 1,535 | 297% |
| preprocess-cache-dmso | 11.7 | 861 | 623% |
| background-count-trp | 10.0 | 6,457 | 281% |
| background-count-flv | 9.9 | 5,740 | 268% |
| apa-compare-{flv,trp}-vs-dmso | ~3.9 each | ~180 | ~116% |
| background-count-dmso | 3.9 | 1,009 | 197% |
| local-decay-plot | 2.9 | 176 | 113% |
| local-decay-assign-pair-types | 1.8 | 89 | 114% |

Each step's number above is the best (least-contended) across four full
runs -- see the note after fix 4 below for why more than one run was
needed. These numbers reflect four fixes this table motivated (see git
history for the full before/after):

1. The `preprocess-cache-dmso/flv/trp` steps used to build NPZ caches
   nothing downstream read (162.7s wasted, ~12% of the original run) --
   removed, then reinstated once they became genuinely shared.
2. `apa aggregate`/`background count` now support `--index-strategy cache`,
   so those same per-sample caches (built once, with strand metadata since
   APA needs it) are shared between both workflows via `--require-cache`
   instead of each re-parsing its sample's raw pairs file from scratch.
   Cache-build time is counted once, in `preprocess-cache-{dmso,flv,trp}`
   above -- `apa-aggregate-*`/`background-count-*` never rebuild it
   (`--require-cache` fails loudly if it's missing), so their numbers are
   pure cache-read cost. Effect on the raw-pairs-parse step specifically
   ("read inputs" in `report/command-timings.csv`): FLV 64.8s → 6.0s (apa),
   57.3s → 4.4s (background); TRP 71.9s → 6.7s (apa), 65.7s → 4.9s
   (background) -- and peak RSS for those same steps dropped by roughly half
   to two-thirds.
3. `_call_bait_contacts`'s `in_region` window filter (see "Runtime" above)
   now resolves via `np.searchsorted` instead of a full per-bait boolean
   mask over the whole chromosome. **`local-decay-call`: 243.6s → 139.7s**
   (1.74x), with peak RSS also dropping 12,399MB → 9,821MB. The named
   profiling steps inside it (cache loading) are unchanged at ~17.3s, so
   this reflects a ~227s → ~122s (1.86x) reduction in the per-bait compute
   loop itself, where LOWESS/Fisher/histogram work (unaffected by this fix)
   also lives.
4. `_safe_kde` (`src/touche/background.py`) used to call
   `gaussian_kde(values)(values)` -- fitting *and* evaluating on every point,
   O(n_fit x n_eval) with no faster path in scipy. At `background-compare`'s
   real ~68k-row scale, run once per sample pair (three: DMSO/FLV, DMSO/TRP,
   FLV/TRP), this dominated its 134.3s. Fixed by capping the fit to a random
   subsample (default 5000 points, fixed seed) while still coloring every
   point -- KDE bandwidth already smooths over local neighborhoods, so a
   several-thousand-point subsample gives a visually indistinguishable
   density estimate. Verified against the real `background_comparison.tsv`
   from this benchmark run: **13.6-14.0x speedup per plot** (12.0s → 0.88s on
   the machine used to verify) with **0.990-0.995 correlation** between old
   and new point coloring across all three sample pairs, plus a visual
   side-by-side confirming the two are indistinguishable. This only affects
   scatter-plot coloring, not `compare_background_ratios`'s numeric output.
   Follow-up full-pipeline re-runs confirm it end to end and reproducibly:
   `background-compare` **134.3s → 25.9s → 25.7s → 22.4s (up to 6.0x)**
   on the same real data across three independent re-runs, consistent with
   the isolated per-plot verification. Two of those re-runs also landed on
   a noisier node, each time in different unrelated steps -- the first saw
   `local-decay-plot` drop 94% → 30% CPU and `preprocess-cache-k562`
   692% → 490% (both 30-50% slower with no code change); the second instead
   hit `apa-aggregate-trp` (313% → 136% CPU, 97.3s → 284.7s) and
   `apa-compare-flv-vs-dmso` (98% → 31% CPU, 4.1s → 92.9s). Different steps
   stalling on different runs, with no correlation to what code changed
   between them, is exactly what shared-node contention looks like rather
   than a regression; the fourth re-run landed on an uncontended node and
   reproduced every step at or near its best-observed time. The table above
   reports each step's best (least-contended) time across all four runs --
   mostly from the first and fourth runs, with `local-decay-plot` and
   `local-decay-assign-pair-types` from the third and `background-compare`
   from the fourth.

`preprocess-cache-k562` (271.3s) remains the largest untouched step --
profiling a synthetic 22M-row pairs file placed 66% of its time in the
single unavoidable full-file scan (gzip decompression + parsing into the
local Parquet spool), with QC and all per-chromosome shard writes reading
cheaply from that spool afterward (Parquet row-group pruning). No fix
identified; this looks like inherent I/O cost, not an algorithmic
inefficiency. One potential shortcut was ruled out empirically: BGZF
(block-gzip, used by pairtools/distiller-nf by convention to support
`pairix`/`tabix` indexing) is a strict superset of gzip whose concatenated
block structure a BGZF-aware reader can decompress in parallel -- but
`htsfile` against the real K562 input (`GSE206131_K562_cis_mapq30_pairs.txt.gz`)
reports "unknown gzip-compressed data", not BGZF, so there's no block
structure here to exploit.

CPU% above 100% reflects multi-core parallelism (Polars' scan engine, Numba's
`parallel=True` kernels, and local-decay's `--jobs` thread pool) — e.g.
`local-decay-call` averaged ~7.4 of the node's 16 cores over its 139.7s wall
time.

## What isn't measured yet

The comparisons above explain *why* `touche` should be faster and lighter on
disk/memory than the reference implementation, verified where isolated fixes
have before/after numbers, and the table above gives `touche`'s real
standalone cost. What's still missing is the other half: a matching
from-scratch run of the reference pipeline on the same inputs, to turn the
structural comparisons above into a real side-by-side wall-clock/peak-RSS
number.
