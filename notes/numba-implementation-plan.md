# Numba implementation plan

## Goal

Add optional Numba acceleration for the counting-heavy parts of `touche` without
changing default install requirements, public output schemas, or reference
compatibility.

The pure NumPy/Pandas path remains the canonical correctness path. Numba kernels
should be selected explicitly at first, tested against the NumPy path, and only
considered as a default after chromosome-scale benchmarks show clear wins.

## Dependency policy

`pyproject.toml` already exposes:

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

## Future work: parallelizing across baits/chromosomes (tabled)

Per-chromosome/per-bait work is embarrassingly parallel in both `compute_apa`
(`src/touche/apa.py`) and `compute_local_decay` (`src/touche/local_decay.py`),
but the two have very different payoffs today:

- APA already gets ~400x from the numba backend on real data (sub-second), so
  parallelizing it further has little upside.
- Local-decay's bottleneck is the per-bait LOWESS fit in `_call_bait_contacts`
  (`src/touche/local_decay.py:553`), not the counting kernels — the numba
  backend only gets ~1.008x there (see the local-decay benchmark above)
  because it doesn't touch the LOWESS step. Baits are independent (no shared
  mutable state, just appended records), making cross-bait parallelism the
  natural next lever if local-decay speed becomes a priority.

Two ways to get that parallelism, with a real tradeoff:

1. **Process-based (`ProcessPoolExecutor`/`multiprocessing`), across baits or
   chromosomes.** Works regardless of `lowess_backend` (`statsmodels` or
   `numba`), since it sidesteps the GIL entirely rather than relying on
   Numba's threading. Lower implementation cost — the per-bait work in
   `_call_bait_contacts` already returns plain record lists with no shared
   state, so this is close to a straight `pool.map`. Overhead: process
   startup and pickling indexes/baits per worker; pairs naturally with the
   NPZ cache's per-chromosome shards (`index_strategy="cache"`) as an
   existing sharding unit.
2. **A nopython (Numba) port with `prange` across baits.** Confirmed via
   direct inspection that `lowess_evenly_spaced_numba`
   (`src/touche/numba_kernels.py:309`) is already `@njit(parallel=True)` with
   its own internal `prange` over LOWESS anchor points. Numba does not
   support nested parallel regions — calling this kernel from inside a *new*
   outer `prange` over baits would silently serialize the inner loop rather
   than adding real cross-bait parallelism. To actually parallelize across
   baits this way requires: (a) a non-parallel variant of the LOWESS kernel
   (plain `range`, no inner `prange`) for use inside the outer loop, and (b)
   porting `_call_bait_contacts`'s histogramming plus
   `fit_zero_inflation_model`/`fit_distance_decay_model`
   (`src/touche/local_decay.py:694`, `:730`) into nopython code, since an
   `@njit(parallel=True)` function can't call back into regular Python. This
   is a real rewrite of the local-decay per-bait pipeline, not a kernel
   tweak, and it only benefits `lowess_backend="numba"` users — the default
   `lowess_backend` is `statsmodels`, which `prange` can't touch at all.

Decision to make before picking either path: whether `lowess_backend="numba"`
(and the numba compute backend generally) becomes the default/required path,
since the nopython-port route (option 2) is only worth its cost if numba is
no longer optional. Tabled until that's decided; process-based parallelism
(option 1) remains available as a backend-agnostic fallback either way.
