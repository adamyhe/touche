# API documentation

The command line remains the best entrypoint for reproducible batch runs, but
the core APIs can also be used interactively from Python notebooks or folded
into custom analysis scripts.

The notebook-friendly pattern is:

1. read pairs and anchors once
2. compute with in-memory objects
3. display figures directly
4. write outputs only when needed

This API is provisional while the result-object layer matures. Statistical
functions already return the `StatResult` objects described under
[Statistics](#statistics).

## CPU usage

Compute-heavy functions use Numba kernels. Set `NUMBA_NUM_THREADS` before
starting Python to control the thread budget:

```bash
NUMBA_NUM_THREADS=8 python analysis.py
NUMBA_NUM_THREADS=8 jupyter lab
```

In an already-running Python process, advanced users can also call
`numba.set_num_threads(N)` before running compute functions.

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

`lowess_backend` and `fisher_backend` default to `"numba"`. Use
`lowess_backend="statsmodels"` and `fisher_backend="scipy"` for conservative
reference comparisons; statsmodels requires the optional `legacy` extra
(`pip install ep-touche[legacy]` / `uv sync --extra legacy`).

Use `lowess_iterations` to change the number of robust residual reweighting
passes. Lower values are faster but can change expected-contact estimates.
With the default `decay_model="normalized"` it applies only to the
zero-inflation fit; the decay fit uses none, because reweighting biases a
sparse count histogram downward.

`compute_local_decay(..., n_jobs=N)` processes up to `N` baits concurrently.
Keep `n_jobs=1` when Numba is already using the available cores. Increase it
only after profiling shows idle CPU outside the kernels.

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

## Pair lists

`tt.build_pair_table` materializes the bait/prey universe that `compute_apa`
and `compute_ep_and_background` otherwise build implicitly, so it can be
inspected, filtered, and passed back in:

```python
pairs = tt.build_pair_table(baits, preys, min_distance=25_000, max_distance=150_000)
pairs = pairs.filter(pl.col("distance") > 50_000)

apa = tt.compute_apa(indexes, baits, preys, pairs=pairs, window=10_000, pixels=50,
                     min_distance=25_000, max_distance=150_000)
```

`tt.read_bedpe` reads an explicit list from disk and `tt.write_bedpe` writes
one. Every row carries a `pair_id` that is center-based and
anchor-order-invariant, so the same pair joins across local-decay, APA,
background, and imported external calls.

## Statistics

Every statistical function returns a `StatResult`: a polars table plus a
`MethodInfo` describing what was tested and what kind of claim the answer
supports. Read `result.info` before quoting `result.table`.

```python
result = tt.test_contacts(calls, method="binomial", fdr="bh")

result.info.inference_class   # "technical": about read sampling in this library
result.info.fdr_family        # "all called pairs"
result.info.warnings          # assumption violations, untestable rows, ...
result.table                  # pair_id, p_value, q_value, n_trials, p_null, ...
result.write("calls.tested.tsv")   # writes calls.tested.tsv.meta.json too
```

`compute_local_decay(..., method="binomial", schema="tidy")` computes the
same p-values during calling. `tt.assess_calibration` checks whether a
p-value column is actually uniform on a null pair set.

Group comparisons take a `cluster_by` column and resample whole clusters for
their confidence interval, because pairs sharing a promoter or enhancer are
not independent observations:

```python
comparison = tt.compare_groups(
    assignments,
    value_col="log2_oe",
    group_col="PosNeg",
    cluster_by="bait_id",
    bootstrap=1000,
)
```

`tt.compare_paired`, `tt.correlate`, `tt.match_pairs`, and
`tt.balance_table` round out the descriptive layer;
`tt.test_background_change` is the zero-safe EP-versus-background comparison
between two libraries.

Quantitative APA scores need per-chromosome pileups for their intervals,
since the resampling unit is the chromosome and never the pixel:

```python
apa = tt.compute_apa(..., keep_chromosomes=True)
summary = tt.summarize_apa(apa, bootstrap=1000)
```

The [statistics guide](statistics.md) documents each method's null, unit of
replication, assumptions, FDR family, and failure modes.

## External calls

```python
calls = tt.read_loop_calls("fithic2.significances.txt", format="fithic2")
annotated = tt.annotate_pairs(pairs, calls, slop=2000)
tt.write_loop_calls(pairs, "pairs.bedpe")
```

Supported formats are in `tt.LOOP_FORMATS`. Matching is by anchor
containment in both anchor orders, not `pair_id` equality, because `touche`
anchors are points and external callers report bins.

## Saving Figures

Plot functions return `matplotlib.figure.Figure`. Save figures using either the
plot function's output path argument or Matplotlib directly:

```python
fig = tt.plot_apa_change(matrix)
fig.savefig("change.svg")
```

The CLI wrappers still save files and close figures automatically.
