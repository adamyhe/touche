# API documentation

The command line remains the best entrypoint for reproducible batch runs, but
the core APIs can also be used interactively from Python notebooks or folded
into custom analysis scripts.

The notebook-friendly pattern is:

1. read pairs and anchors once
2. compute with in-memory objects
3. display figures directly
4. write outputs only when needed

This API is provisional while the result-object layer matures.

## Imports

```python
import touche.api as tt
```

## Reuse contact indexes

Build chromosome-sharded contact indexes once and reuse them across analyses:

```python
indexes = tt.build_contact_indexes("sample.nodups_30_intra.pairs.gz", source="touche")
```

## Progress and Profiling

Compute functions are quiet by default. Pass `progress=True` to show a `tqdm`
progress bar in notebooks or terminals:

```python
counts = tt.compute_ep_and_background(
    indexes,
    baits,
    preys,
    min_distance=25_000,
    max_distance=150_000,
    window=2_500,
    min_bg_distance=10_000,
    max_bg_distance=150_000,
    progress=True,
)
```

For reusable profiling, create an instrumentation object and pass it through one
or more calls:

```python
instrument = tt.Instrumentation(progress=True, profile=True)

apa = tt.compute_apa(..., progress=instrument)
calls = tt.compute_local_decay(..., progress=instrument)

instrument.timings
```

File-backed helpers such as `tt.aggregate_apa`, `tt.call_local_decay`, and
`tt.count_ep_and_background` also accept `progress` and `profile`.

## APA

```python
baits = tt.read_bed_anchors("promoters.bed")
preys = tt.read_bed_anchors("enhancers.bed")

apa = tt.compute_apa(
    indexes,
    baits,
    preys,
    min_distance=25_000,
    max_distance=150_000,
    window=10_000,
    pixels=50,
)

fig = apa.plot()
```

APA counting defaults to `backend="numba"`. Pass `backend="numpy"` for the
plain NumPy reference implementation instead.

Write reference-style outputs only when needed:

```python
paths = apa.write("results/apa")
```

## Local Decay

```python
baits = tt.read_center_anchors("enhancer_baits.tsv")
preys = tt.read_center_anchors("promoter_preys.tsv")

calls = tt.compute_local_decay(
    indexes,
    baits,
    preys,
    dist=1_000_000,
    cap=2_000,
)
```

`compute_local_decay` defaults to `backend="numba"`, which accelerates only
the observed-count helper. LOWESS fitting (`lowess_backend`, default
`"statsmodels"`) is separate and often dominates runtime -- pass
`backend="numpy"` for the plain NumPy reference implementation instead.

`compute_local_decay(..., lowess_backend="numba")` enables an experimental
Numba LOWESS implementation for evenly spaced local-decay smoothing arrays. Use
`lowess_iterations=0` or `1` to reduce robust reweighting work when that drift is
acceptable; the reference-compatible default is `3`.

`compute_local_decay(..., fisher_backend="numba")` enables an experimental,
`prange`-parallel hypergeometric survival function for the per-prey Fisher
exact test, in place of the default `fisher_backend="scipy"`
(`scipy.stats.hypergeom.sf`). This step is single-threaded regardless of
`backend`/`lowess_backend`, and with those set to `"numba"` it becomes the
main reason local-decay doesn't saturate available cores; `fisher_backend="numba"`
addresses that at the cost of p-values that match scipy to within ~1e-8
absolute error rather than exactly (see
`notes/numba-implementation-plan.md` for the validation methodology).

`compute_local_decay(..., n_jobs=N)` processes up to `N` baits concurrently
in a thread pool instead of one at a time (default `n_jobs=1`, sequential).
Baits are independent, so this is exact -- not an approximation -- as long
as `N` doesn't oversubscribe available cores: each worker's own numba
thread budget is capped to `cores // n_jobs` automatically, but kernel-level
`prange` parallelism (from `backend`/`lowess_backend`/`fisher_backend`) and
this bait-level parallelism are not additive, they compete for the same
cores. Worth combining with the `numba` backends above once per-bait
overhead (contact filtering, histogram construction) is a meaningful share
of runtime relative to the kernels themselves -- see
`notes/numba-implementation-plan.md` for when that's the case.

Pair-type plotting accepts an in-memory dataframe or an assignments file:

```python
plot_data, fig = tt.plot_pair_type_distribution(assignments_df)
```

## EP/Background Counts

```python
baits = tt.read_bed_anchors("promoters.bed")
preys = tt.read_bed_anchors("enhancers.bed")

counts = tt.compute_ep_and_background(
    indexes,
    baits,
    preys,
    min_distance=25_000,
    max_distance=150_000,
    window=2_500,
    min_bg_distance=10_000,
    max_bg_distance=150_000,
)
```

EP/background counting defaults to the Numba backend:

```python
counts = tt.compute_ep_and_background(..., backend="numba")  # the default
```

Pass `backend="numpy"` for the plain NumPy reference implementation instead.

## Saving Figures

Plot functions return `matplotlib.figure.Figure`. Save figures using either the
plot function's output path argument or Matplotlib directly:

```python
fig = tt.plot_apa_change(matrix)
fig.savefig("change.svg")
```

The CLI wrappers still save files and close figures automatically.
