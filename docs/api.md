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

## Saving Figures

Plot functions return `matplotlib.figure.Figure`. Save figures using either the
plot function's output path argument or Matplotlib directly:

```python
fig = tt.plot_apa_change(matrix)
fig.savefig("change.svg")
```

The CLI wrappers still save files and close figures automatically.
