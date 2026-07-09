# Numba implementation plan

## Goal

Add optional Numba acceleration for the counting-heavy parts of `touche` without
changing default install requirements, public output schemas, or reference
compatibility.

The pure NumPy/Pandas path remains the canonical correctness path. Numba kernels
should be selected explicitly at first, tested against the NumPy path, and only
considered as a default after chromosome-scale benchmarks show clear wins.

## Dependency policy

**Update:** Numba is now a core dependency (`pyproject.toml`'s `dependencies`,
no more `[project.optional-dependencies] fast` extra), and `backend` now
defaults to `"numba"` everywhere it's exposed (see "Decide default behavior"
below for the rationale and evidence). Base installs still do not import
Numba during package import -- only when a numba backend is actually
requested, unchanged from the original policy. `lowess_backend` remains a
separate, independent default (`"statsmodels"`) pending a real-data parity
check -- see the note at the end of "Decide default behavior".

Original policy (superseded, kept for history):

`pyproject.toml` used to expose:

```toml
[project.optional-dependencies]
fast = ["numba"]
```

Keep Numba optional. Base installs should not import Numba during package import.

Implementation pattern:

- Put Numba code in a dedicated module such as `src/touche/kernels.py` or
  `src/touche/numba_kernels.py`.
- Use an availability helper, for example `touche.backends.get_backend(...)`, so
  domain modules do not each own optional-import logic.
- Raise a clear error if the user requests `backend="numba"` without installing
  `ep-touche[fast]`.
- Keep `backend="numpy"` as the default for the first release containing Numba
  kernels.

## Backend API shape

Add a shared backend option to in-memory compute functions first:

```python
compute_apa(..., backend="numpy")
compute_ep_and_background(..., backend="numpy")
compute_local_decay(..., backend="numpy")
```

Then expose the same option in CLI commands whose compute path supports it:

```bash
touche apa aggregate --backend numpy
touche background count --backend numpy
touche local-decay call --backend numpy
```

Allowed values:

- `numpy`: current implementation.
- `numba`: use Numba kernels, error if unavailable.
- `auto`: optional later; use Numba if installed and the input shape is supported.

Do not add `auto` until benchmark coverage is good enough to avoid surprise
compile-time overhead on small jobs.

## Candidate kernels

### 1. EP/background counts

Priority: high.

Current hot functions:

- `touche.background._count_between_windows`
- `touche.background._count_anchor_to_background`
- outer bait/prey loops in `compute_ep_and_background`

Why this should be first:

- The counting logic is integer-only and easy to validate exactly.
- It does not depend on LOWESS, SciPy, Pandas indexing, or plotting.
- It has nested loops over candidate bait/prey pairs and full contact arrays.

Target kernel:

```python
count_ep_background_pairs_numba(
    pos_a,
    pos_b,
    bait_centers,
    prey_centers,
    pair_bait_index,
    pair_prey_index,
    window,
    min_bg_distance,
    max_bg_distance,
) -> tuple[ep_counts, bg_counts]
```

Python remains responsible for:

- grouping by chromosome
- identifying candidate bait/prey pairs by distance
- assembling the output dataframe

Numba owns:

- EP window counts
- bait-to-prey background counts
- prey-to-bait background counts

Validation:

- exact equality with `compute_ep_and_background(..., backend="numpy")`
- test empty chromosomes, no candidate pairs, zero counts, and contacts exactly
  on window boundaries

### 2. APA matrix and 1D signal counts

Priority: high, after EP/background.

Current hot functions:

- `touche.apa._add_anchor_signal`
- `touche.apa._add_pair_matrix`
- outer bait/candidate-prey loops in `compute_apa`

Why this is promising:

- APA is repeated bounded-window counting over numeric arrays.
- The current Python/Pandas matrix updates add overhead inside nested loops.

Target kernels:

```python
anchor_signal_numba(pos_a, pos_b, centers, strands, mask, window, pixels) -> signal
apa_matrix_numba(pos_a, pos_b, bait_centers, bait_strands, prey_centers, prey_strands, mask, window, pixels) -> matrix
```

Likely implementation details:

- Convert strands to small integers before entering Numba, e.g. `+ -> 1`,
  `- -> -1`.
- Return plain NumPy arrays, then wrap them in the existing Pandas result objects.
- Avoid Pandas operations inside kernels.
- Keep the current `_pixel_labels(...)` behavior in Python so output labels do
  not drift.

Validation:

- exact equality for `ApaResult.matrix`, `bait_signal`, and `prey_signal`
- tests for both positive and negative strand orientation
- tests for shifted positions and boundary contacts

### 3. Local-decay neighborhood counts

Priority: medium.

Current hot function:

- `touche.local_decay._call_bait_contacts`

Potential Numba pieces:

- region filtering around one bait
- distance histogram construction
- plus/minus contact-position extraction
- observed contact counts for candidate preys

Keep out of Numba initially:

- LOWESS calls through statsmodels
- Fisher exact p-value through SciPy
- output record construction

Target split:

```python
local_decay_counts_numba(
    pos_a,
    pos_b,
    bait_center,
    prey_centers,
    dist,
    cap,
    min_distance,
) -> observed, strand_distances, plus_count, minus_count, distances
```

The Python layer can continue to run LOWESS and Fisher tests. This avoids a
large compatibility risk while still removing the most repetitive observed-count
work.

Validation:

- exact equality for observed counts and directional distances
- tolerance comparison for expected values and p-values after the unchanged
  Python LOWESS/Fisher steps

### 4. Preprocessing and cache IO

Priority: low.

Do not start here. Text parsing, gzip IO, and dataframe construction are not
ideal first Numba targets. Revisit only after profiling shows CPU-bound parsing
that cannot be addressed with chunking or format changes.

## Benchmark plan

Add a benchmark script under `notes/benchmarks/` or a lightweight non-test module
that is not run by CI by default.

Benchmark levels:

1. Synthetic unit-size fixtures, mainly for compile sanity.
2. Synthetic chromosome-scale arrays with configurable contacts and anchors.
3. One real chromosome from a processed Micro-C pairs file when available.
4. Full workflow benchmark when large reference-sized inputs are available.

Track:

- wall time including and excluding first-call JIT compilation
- peak RSS
- contacts processed per second
- candidate bait/prey pairs processed per second
- output equality versus NumPy
- compile cache behavior across repeated calls

Suggested commands:

```bash
uv run python notes/benchmarks/benchmark_numba_kernels.py --workflow background
uv run python notes/benchmarks/benchmark_numba_kernels.py --workflow apa
uv run python notes/benchmarks/benchmark_numba_kernels.py --workflow local-decay
```

The benchmark script can require `ep-touche[fast]` and should skip gracefully if
Numba is unavailable.

## Implementation milestones

1. Add backend plumbing:
   - `backend` argument on in-memory compute functions.
   - CLI `--backend` option where supported.
   - optional import helper and clear missing-extra error.
   - Status: first slice done for EP/background counting.

2. Add EP/background Numba kernel:
   - implement kernel over arrays.
   - keep dataframe assembly in Python.
   - add paired NumPy/Numba tests.
   - benchmark synthetic chromosome-scale data.
   - Status: done for `compute_ep_and_background`, `background count`, and
     `background run`.

3. Add APA Numba kernels:
   - implement 1D anchor signal kernel.
   - implement 2D APA accumulation kernel.
   - add exact equality tests against the current implementation.
   - benchmark with realistic `window`/`pixels` values.
   - Status: done for `compute_apa`, `apa aggregate`, and `apa run`.

4. Add local-decay count helpers:
   - accelerate observed counts and distance histogram pieces only.
   - leave LOWESS and Fisher exact in Python.
   - add equality/tolerance tests.
   - Status: observed-count helper done for `compute_local_decay`,
     `local-decay call`, and `local-decay run`; distance histogram remains NumPy.

5. Decide default behavior:
   - keep default `numpy` unless repeated real-data benchmarks show meaningful
     speedups after accounting for JIT compile time.
   - consider `backend="auto"` only after there are input-size heuristics.
   - **Status: done.** The large-scale synthetic benchmark (100k contacts,
     300 baits/preys, 20 Mb genome) and the "Large-scale local-decay
     benchmark with numba LOWESS" run above showed correctness-verified
     speedups (APA ~80-400x, background ~7-22x, local-decay ~3.8x with
     `lowess_backend="numba"`) with matching row-count/matrix-sum
     fingerprints and no regressions -- the bar set above is met. `backend`
     now defaults to `"numba"` (via `touche.backends.DEFAULT_BACKEND`) at
     every compute function, pipeline, and CLI flag; `numba` is now a core
     dependency rather than the `fast` extra, since defaulting to it while
     it stayed optional would break a fresh install's first default use.
     `backend="numpy"` remains fully supported as an explicit opt-out.
     `lowess_backend` is a separate, independent decision and stays
     `"statsmodels"`-default for now -- a real-data parity check against
     `lowess_backend="numba"` is in progress to decide whether a "legacy"
     statsmodels fallback is needed for anything beyond what already exists
     (the statsmodels path itself, unconditionally available either way).
     `backend="auto"` remains not implemented, unchanged.

## Risks and guardrails

- Numba compile overhead may make small jobs slower. Keep explicit selection.
- Numba support across Python, NumPy, and NumPy scalar dtypes can be fussy. Keep
  kernels narrow and typed around `int64`, `float64`, and small integer strand
  codes.
- Do not move Pandas dataframes or object arrays into kernels.
- Do not duplicate algorithm semantics. The Python path should remain readable
  and serve as the reference implementation.
- Require exact equality for integer count outputs before trusting speedups.
- Avoid whole-genome monolithic arrays. Keep chromosome-level processing and
  bounded candidate-pair chunks.

## Initial EP/background benchmark

Environment:

- macOS arm64 development machine.
- Python 3.12 via `uv run --extra fast`.
- Numba 0.66.0 from the `fast` extra.
- Synthetic one-chromosome background-count benchmark.

Command:

```bash
UV_CACHE_DIR=/private/tmp/touche-uv-cache \
  uv run --extra fast python notes/benchmarks/benchmark_numba_kernels.py \
  --workflow background \
  --contacts 100000 \
  --baits 300 \
  --preys 300 \
  --repeats 5
```

Initial result after first-call JIT warmup:

```text
numpy: min=0.4704s median=0.5598s mean=0.5596s repeats=5
numba: min=0.0868s median=0.0891s mean=0.0942s repeats=5
median_speedup_excluding_compile=6.28x
rows=1171
```

After replacing eager background membership checks with conditional checks that
only test background windows when one contact endpoint is already inside the
bait/prey window:

```text
numpy: min=0.4543s median=0.4957s mean=0.5001s repeats=5
numba: min=0.0564s median=0.0648s mean=0.0655s repeats=5
median_speedup_excluding_compile=7.65x
rows=1171
kernel_eager: min=0.0754s median=0.0785s mean=0.0790s repeats=5
kernel_optimized: min=0.0479s median=0.0523s mean=0.0511s repeats=5
kernel_optimized_speedup=1.50x
```

Thread sensitivity for the optimized kernel:

```text
NUMBA_NUM_THREADS=1: median_speedup_excluding_compile=1.62x
NUMBA_NUM_THREADS=4: median_speedup_excluding_compile=4.60x
NUMBA_NUM_THREADS=8: median_speedup_excluding_compile=7.88x
```

Interpretation:

- Numba is significantly faster for this EP/background kernel on the tested
  synthetic workload.
- Most of the win comes from parallel execution; single-threaded Numba is only
  modestly faster than NumPy for this benchmark.
- Conditional background checks improve the Numba kernel by about 1.5x compared
  with the original eager-check kernel while preserving exact integer counts.
- Keep Numba optional for now because only one workflow has an accelerated
  backend, small jobs pay JIT compile overhead, and package install simplicity is
  still useful.
- Reconsider requiring Numba after APA and local-decay kernels are implemented
  and full-workflow benchmarks show consistent wins.

## Initial APA benchmark

Command:

```bash
UV_CACHE_DIR=/private/tmp/touche-uv-cache \
  uv run --extra fast python notes/benchmarks/benchmark_numba_kernels.py \
  --workflow apa \
  --contacts 50000 \
  --baits 200 \
  --preys 200 \
  --repeats 3
```

Result after first-call JIT warmup:

```text
numpy: min=10.8232s median=10.8299s mean=11.0516s repeats=3
numba: min=0.2755s median=0.3535s mean=0.3276s repeats=3
median_speedup_excluding_compile=30.64x
matrix_sum=2228
```

Interpretation:

- Numba is very effective for APA aggregation despite the initial 2D accumulator
  being sequential, because the old path paid heavy Python/Pandas overhead in
  nested matrix updates.
- APA and EP/background now both show strong enough speedups that `ep-touche[fast]`
  should be recommended for large analyses.
- Keep Numba optional until full workflow benchmarks include installation,
  compile warmup, and repeated sample processing costs.

## Initial local-decay benchmark

Command:

```bash
UV_CACHE_DIR=/private/tmp/touche-uv-cache \
  uv run --extra fast python notes/benchmarks/benchmark_numba_kernels.py \
  --workflow local-decay \
  --contacts 1000 \
  --baits 5 \
  --preys 10 \
  --genome-size 1000000 \
  --repeats 2
```

Result after first-call JIT warmup:

```text
numpy: min=2.0764s median=2.0785s mean=2.0785s repeats=2
numba: min=2.0874s median=2.0881s mean=2.0881s repeats=2
median_speedup_excluding_compile=1.00x
rows=50
```

Interpretation:

- The narrow observed-count helper is exact but does not materially improve
  local-decay runtime on this workload.
- LOWESS fitting dominates local-decay runtime. Further local-decay acceleration
  should target smoothing/model fitting or reduce how often LOWESS is called,
  rather than adding more small observed-count kernels.

## Experimental Numba LOWESS benchmark

Implemented an opt-in `lowess_backend="numba"` path for local-decay smoothing.
This is a specialized evenly spaced local-linear LOWESS implementation with
tricube weights and robust bisquare residual reweighting. The default remains
`lowess_backend="statsmodels"` because exact parity is not guaranteed.

The first implementation drifted because it ignored statsmodels' `delta`
interpolation. Direct LOWESS calls matched statsmodels when `delta=0`, while
`delta=16` diverged because statsmodels skips nearby regressions and linearly
interpolates. Adding delta anchors/interpolation fixed the production chunked
local-decay wrappers.

Initial synthetic drift check against statsmodels zero-inflation smoothing before
delta interpolation:

```text
max_abs=0.203125
mean_abs=0.0032500000000002505
corr=0.9589578882978257
statsmodels_range=0.24999999999999278..0.5
numba_range=0.25..0.5
```

After delta interpolation:

```text
fit_zero_inflation_model delta=16: max_abs=7.216e-15, mean_abs=2.504e-16, corr=1.0
fit_distance_decay_model delta=16: max_abs=6.830e-18, mean_abs=5.885e-19, corr=1.0
```

Local-decay benchmark with the experimental smoother after delta interpolation:

```bash
UV_CACHE_DIR=/private/tmp/touche-uv-cache \
  uv run --extra fast python notes/benchmarks/benchmark_numba_kernels.py \
  --workflow local-decay \
  --contacts 1000 \
  --baits 5 \
  --preys 10 \
  --genome-size 1000000 \
  --repeats 2 \
  --lowess-backend numba
```

```text
numpy: min=0.6839s median=0.6927s mean=0.6927s repeats=2
numba: min=0.6961s median=0.7255s mean=0.7255s repeats=2
median_speedup_excluding_compile=0.95x
rows=50
```

Compared with the same benchmark using statsmodels LOWESS:

```text
statsmodels smoother median ~= 2.1377s
numba smoother median ~= 0.6927s
```

Interpretation:

- The Numba smoother can reduce smoothing runtime substantially for this small
  synthetic benchmark and now matches the chunked smoothing wrappers on
  regression fixtures.
- It should remain opt-in until we validate biological conclusions and
  thresholded call sets against the reference smoother on real/reference data.
- The remaining small NumPy-vs-Numba difference inside the Numba smoother run
  confirms observed-count acceleration is not the main local-decay bottleneck.

Native statsmodels optimizations:

- `is_sorted=True` is now used because local-decay smoothing arrays are already sorted.
- `missing="none"` is now used because the generated arrays are finite and dense.
- `lowess_iterations` is exposed; statsmodels' robust reweighting passes are
  expensive, and lowering this from the reference default `3` can be a useful
  speed/compatibility tradeoff.
- `lowess_delta` was already exposed and remains the main statsmodels speed knob.

## Large-scale local-decay benchmark with numba LOWESS

Re-ran the local-decay benchmark at a much larger, more realistic synthetic
scale (100,000 contacts, 300 baits, 300 preys, 20 Mb genome, 5 repeats) on a
multi-core HPC node, this time passing `--lowess-backend numba` to both legs
of the `backend=numpy` vs `backend=numba` comparison. The earlier large-scale
run (see "Initial local-decay benchmark") only swapped the counting kernel and
left the LOWESS smoother on `statsmodels` for both legs, which is why it only
saw ~1.03x.

Correctness held throughout (`rows=1794`, matching row counts and no mismatch
warnings across every leg).

```text
numpy counting + statsmodels LOWESS (earlier run): median=451.0s
numpy counting + numba LOWESS:                     median=119.2s   (3.8x)
numba counting + numba LOWESS:                     median=123.3s   (0.966x vs the row above)
```

Interpretation:

- The ~3.8x win comes entirely from the LOWESS kernel, not the counting
  kernel. Once `lowess_backend="numba"` is set, swapping the counting
  `backend` between `numpy` and `numba` makes no meaningful difference --
  confirming LOWESS fitting, not observed/expected counting, is local-decay's
  real bottleneck at realistic scale.
- CPU utilization moved from ~1.2 cores (both `statsmodels`-LOWESS legs) to
  ~11.2-11.4 cores with `lowess_backend="numba"`, matching
  `lowess_evenly_spaced_numba`'s internal `prange` over anchor points doing
  real work on a many-core node.
- Peak RSS actually dropped slightly (421 -> 393 MiB) when also switching
  counting to numba, because the numpy-counting leg already pays numba's
  import cost once `lowess_backend="numba"` is set -- there's no separate
  "numba tax" left for the counting leg to add on top.
- Separately, background's numba wall time dropped a lot between the two
  large-scale runs (28.0s -> 7.3s) purely from numba's on-disk `cache=True`
  reusing a previously-compiled kernel -- a reminder that a single cold run
  measures compile time bundled into wall time, and is pessimistic versus
  steady-state usage.

This closes the LOWESS half of the "Future work" question below: getting
local-decay's big real-world speedup does not require the speculative
nopython-port-for-`prange`-across-baits work -- defaulting `lowess_backend`
to `numba` gets most of the win today, with much less engineering cost. Exact
parity with `statsmodels` is still unvalidated on real/reference data (see
"Experimental Numba LOWESS benchmark" above), so `lowess_backend="numba"`
should stay opt-in until that's checked, independent of whatever the counting
`backend` default ends up being.

## Fisher exact microbenchmark

Replaced `scipy.stats.fisher_exact(..., alternative="greater")` with the
equivalent direct hypergeometric survival function:

```python
hypergeom.sf(a - 1, total, row_1, col_1)
```

This preserves exact one-sided p-values for the rounded 2x2 tables used by the
reference local-decay workflow.

Benchmark:

```text
scipy_fisher_exact: min=1.7876s median=1.8119s mean=1.8708s repeats=3
direct_hypergeom_sf: min=1.3367s median=1.4653s mean=1.4361s repeats=3
median_speedup=1.24x
```

## Future work: parallelizing the remaining heavy bits

**Update:** the "tabled" decision below (whether `backend`/`lowess_backend="numba"`
becomes default) is now resolved -- see "Decide default behavior" above.
`backend` defaults to `numba` as of this update; `lowess_backend` stays
`statsmodels`-default pending a real-data parity check, with the "Large-scale
local-decay benchmark with numba LOWESS" section above already showing that
switching `lowess_backend` to `numba` (independent of `backend`) is what
actually fixes local-decay's real bottleneck (~3.8x), which supersedes the
nopython-port-for-`prange`-across-baits option originally discussed here --
that speedup is achievable today without it. The rest of this section is
rewritten as a prioritized roadmap for what's left, now that LOWESS is no
longer the dominant local-decay cost once `lowess_backend="numba"` is used.

**Kernel-level `prange` vs. outer-level (chromosome/bait) threading, under a
fixed core budget:** these two levels are not additive -- they compete for
the same cores, and combining them without capping one causes
oversubscription (see "Risks and guardrails" below). The deciding factor is
whether a *single* kernel call's internal parallelism degree already
approaches the core count:

- **Local-decay's LOWESS kernel** (`lowess_evenly_spaced_numba`) parallelizes
  over delta-spaced anchor points within *one bait's* histogram -- at
  realistic settings (`dist=1_000_000`, `delta=16`) that's on the order of
  tens of thousands of anchors, enough to saturate almost any realistic core
  count by itself. Kernel-level parallelism should stay primary here; item 3
  below (outer chromosome/bait threading) is **not** a good fit for
  local-decay and should not be added on top of it -- doing so would mostly
  just oversubscribe, since numba is already using every core for whichever
  bait is in flight.
- **APA/background's per-chromosome kernels** have a parallelism degree
  bounded by baits/preys active in that chromosome (APA's anchor kernel) or
  candidate pairs after distance filtering (background) -- both shrink a lot
  for smaller/sparser chromosomes, leaving cores idle that a concurrent
  second chromosome could otherwise use. Item 3 below designs outer
  chromosome-level threading for this case, but it's deferred for now in
  favor of staying sequential and letting `NUMBA_NUM_THREADS` control core
  usage -- see item 3's "Status: deferred" note for why.

1. **Vectorize local-decay's per-prey Fisher-exact loop.** **Status: done,
   but only a partial fix -- see caveat below.** `_call_bait_contacts`'s tail
   loop previously computed `exp_prob` via a fresh
   `bg_pdf[exp_start:exp_stop+1].sum()` slice-sum per prey, then called
   `fisher_greater` (pure scipy `hypergeom.sf`) once per prey -- both scalar,
   entirely single-threaded and GIL-bound, running once per (bait, prey)
   candidate. Replaced with: a cumulative-sum lookup for `exp_prob`
   (`cumsum = np.concatenate([[0.0], np.cumsum(bg_pdf)])`, indexed once per
   prey instead of re-summing a window each time) and a new vectorized
   `fisher_greater_batch(a1, a2, b1, b2) -> np.ndarray` in `stats.py`
   (`hypergeom.sf` broadcasts over array args), both applied across all of a
   bait's preys at once instead of one Python-level iteration per prey.
   Verified via exact equality against the scalar loop on a synthetic
   fixture and via the existing `test_numba_compute_local_decay_matches_numpy`
   parity test.

   **Caveat (partially resolved by the numba `fisher_backend` prototype
   below) -- vectorizing alone only partially addressed low core
   utilization on local-decay.** Microbenchmarking (5000 synthetic preys,
   ~2M-bin background histogram) found the windowed-sum-to-cumsum change is
   a large win in isolation (~13x on that piece alone) but negligible
   against the total, because `fisher_greater`/`hypergeom.sf` completely
   dominates runtime -- vectorizing it via `fisher_greater_batch` only gave
   ~1.5x (scipy's hypergeometric survival function has substantial fixed
   per-element cost even when called as one batched array op instead of N
   scalar calls; it isn't a thin wrapper over cheap arithmetic). After
   `backend="numba"` and `lowess_backend="numba"` made the counting/LOWESS
   kernels fast and fully parallel, this comparatively-slow, still
   single-threaded loop became a larger share of each bait's wall time than
   before -- which is why the real-data benchmark showed only ~2-3 cores
   active on local-decay rather than saturating the core budget.

   **Update: prototyped an opt-in `fisher_backend="numba"` that replaces
   `hypergeom.sf` itself.** `numba_kernels.py`'s `hypergeom_sf_numba` (with
   scalar helpers `_log_binom`/`_hypergeom_sf_scalar`) computes the
   hypergeometric survival function via a `prange`-parallel log-space PMF
   tail sum -- `lgamma`-based binomial-coefficient differences (avoiding
   naive factorials, which overflow immediately at this scale), summing
   whichever side of the k-split is smaller (direct tail vs. `1 -`
   complement) to avoid precision loss when the true answer is close to 1.
   Exposed as `fisher_backend: Literal["scipy", "numba"] = "scipy"` --
   mirroring the `lowess_backend` pattern exactly (own `DEFAULT_FISHER_BACKEND`
   constant and `validate_fisher_backend()` in `backends.py`, threaded through
   every `call_local_decay`/`compute_local_decay`/`_call_bait_contacts` layer,
   `run_local_decay_pipeline`, and a `--fisher-backend {scipy,numba}` CLI flag
   on `local-decay call`/`run`) -- `fisher_backend="scipy"` (exact, scipy-backed)
   stays the default; `numba` is opt-in.

   Validated against `scipy.stats.hypergeom.sf` across 40,000 randomized
   table shapes (uniform-random `(k, M, n, N)`, plus tables shaped like
   local-decay's actual usage: `M`/`N` in the millions from a genome-scale
   background histogram with `n` in the tens from observed/expected contact
   counts) -- max absolute error ~1.7e-8, zero NaN mismatches (both sides
   return `nan` for the degenerate `total<=0` table, which
   `_call_bait_contacts` cannot produce since its histograms always have at
   least one bin). This is negligible for p-value thresholding at any
   normal significance level. `tests/test_stats.py::test_fisher_greater_batch_numba_matches_scipy`
   and `tests/test_local_decay.py::test_fisher_backend_numba_matches_scipy`
   encode this as a closeness check (`atol=1e-6`), not exact equality --
   unlike the integer-count numba/numpy backends elsewhere in this file,
   this is a different floating-point algorithm for the same quantity, so
   bit-identical output was never the goal.

2. **Parallelize `apa_matrix_numba`.** **Status: done.**
   `apa_matrix_numba` (`src/touche/numba_kernels.py:207`) previously ran
   `@njit(cache=True)` only -- no `parallel=True`/`prange`, unlike its sibling
   `apa_anchor_signal_numba` (`:168`, `parallel=True` with `prange` over
   `center_index`). Naively adding `prange` to the outer
   `for pair_index in range(...)` loop would introduce a real data race --
   multiple pairs write `matrix[prey_bin, bait_bin] += count` into the *same*
   shared cell (unlike `apa_anchor_signal_numba`, where each `prange`
   iteration owns an exclusive output row indexed by the loop variable
   itself), so `matrix` is now `thread_matrices` of shape
   `(n_threads, bins, bins)`: each thread accumulates into its own exclusive
   slice via `get_thread_id()`, and the caller sums over `axis=0` after the
   parallel loop -- the standard thread-local-buffer pattern for
   scatter-add reductions numba can't auto-parallelize. `n_threads` is
   computed via `numba.get_num_threads()` in the Python wrapper
   (`apa.py`'s `_apa_matrix_numba`) and passed in as a plain argument rather
   than called inside the kernel -- calling `get_num_threads()` from inside
   an `@njit(cache=True)` function makes numba treat it as a dynamic global
   and silently disables on-disk caching (confirmed via
   `NumbaWarning: Cannot cache compiled function ... as it uses dynamic
   globals`); `get_thread_id()` alone does not have this problem and stays
   inside the kernel. Verified via the existing exact-equality test
   (`tests/test_apa_aggregate.py::test_numba_compute_apa_matches_numpy`) and
   an ad hoc synthetic benchmark (100k contacts, 300 baits/preys, 20 Mb
   chromosome, matching the scale used elsewhere in this file) showing
   ~4.3x speedup over the old sequential kernel with bit-identical output.

3. **Opt-in cross-chromosome thread-pool parallelism for APA/background only**
   (biggest lift, needs its own review before implementing; per the framing
   above, deliberately scoped to exclude `compute_local_decay`).
   **Status: deferred.** For the sake of simplicity, `compute_apa` and
   `compute_ep_and_background`'s outer per-chromosome loops stay strictly
   sequential for now; core usage for a whole run is controlled via the
   `NUMBA_NUM_THREADS` environment variable (or `numba.set_num_threads()` in
   a notebook) instead of adding a second, outer-level parallelism knob.
   Given item 2 already gives every numba kernel in the codebase
   kernel-level `prange` parallelism, the added API surface and complexity
   below (new `n_jobs`/`--jobs` parameter, worker-thread error propagation,
   progress-bar interaction, work-stealing scheduler) isn't worth it
   right now. The design is kept below in case per-chromosome kernel
   parallelism is later shown (via profiling on real multi-core hardware)
   to leave cores idle for long stretches -- e.g. many small chromosomes
   trailing behind one large one -- at which point this becomes worth
   revisiting rather than re-deriving from scratch.

   `compute_apa` and `compute_ep_and_background`'s outer per-chromosome loops
   (`apa.py:116`, `background.py:106`) are strictly sequential today, but
   each chromosome's `ContactIndex` and bait/prey subset is fully independent
   -- the only shared state to reconcile is each function's own accumulator
   (`apa.py`'s `matrix_arr`/`bait_signal_arr`/`prey_signal_arr`, accumulated
   via `+=`; `background.py`'s plain `rows` list, built via `.append`). With
   `backend="numba"` now the default, numba-jitted kernels release the GIL
   while running, so a `concurrent.futures.ThreadPoolExecutor` across
   chromosomes can get real parallelism for the compute-heavy portion
   without multiprocessing's pickling/startup overhead (the process-based
   option originally proposed here) -- each worker thread processes one
   chromosome (building its own local accumulator/list), the main thread
   reduces (`sum` the arrays / concatenate the lists, in a fixed chromosome
   order, not completion order) after all futures complete.
   - **Must cap numba's thread budget per worker** (e.g.
     `numba.set_num_threads(max(1, cores // n_jobs))` before each worker's
     kernel call) -- otherwise each of `n_jobs` outer threads spawns its own
     numba thread pool sized to all cores, oversubscribing badly once more
     than one chromosome is in flight.
   - **Must not statically assign one thread per chromosome.** Chromosomes
     vary hugely in size (chr1 vs. chrY), so a naive 1-thread-per-chromosome
     assignment wastes cores once the small chromosomes finish and sit idle
     while the biggest one grinds on -- wall clock ends up capped by the
     single biggest chromosome regardless of `n_jobs`. Use a bounded
     work-stealing pool (submit one task per chromosome to a fixed-size
     executor) and schedule biggest-chromosome-first, so the long-pole task
     starts immediately and smaller ones backfill the remaining workers as
     they free up.
   - Scope as a new opt-in `n_jobs: int = 1` parameter on `compute_apa`/
     `compute_ep_and_background` plus a `--jobs`/`-j` CLI flag, defaulting to
     sequential (1) until validated on real multi-core hardware -- this is
     genuinely new public API surface (return-value ordering, error
     propagation from worker threads, interaction with the existing progress
     bars) that deserves its own dedicated plan/review rather than being
     bundled into a default-flip change.
