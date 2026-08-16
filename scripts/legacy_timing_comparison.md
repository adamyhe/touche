# touche vs. legacy timing comparison

This benchmark is agent-facing and is not run by normal tests. It runs one
representative real-data command from each `touche` workflow family
(local-decay, APA, EP/background) against its original Danko-Lab
`E-P_contacts` `.bsh`/`.py` counterpart, on identical inputs and parameters,
and profiles both with the same wall-time/peak-RSS/CPU harness
`reference_replication.py` uses.

**Run this on a remote server, not your laptop.** The legacy pipeline spawns
many per-bait/per-chromosome Python and R subprocesses over the full real
K562/mESC pairs files; a single legacy step can take hours. See "Running on a
remote server" below.

## Prerequisites

1. A local clone of <https://github.com/Danko-Lab/E-P_contacts> (pass
   `--reference-dir`, or `--clone-reference` to have the script clone it for
   you).
2. For the local-decay comparison only: the legacy repo's own R (>=4.1) +
   `rpy2` environment (its `ContactCaller_microC.py` calls R's
   `fisher.test` via `rpy2` -- this is exactly the dependency `touche` was
   built to remove, see `docs/reproducing-reference-plots.md`). Build it from
   the reference repo's `environment.yml`:

   ```bash
   cd /path/to/E-P_contacts
   conda env create -f environment.yml
   ```

   Then point every legacy subprocess at that environment with
   `--legacy-shell-prefix`:

   ```bash
   --legacy-shell-prefix "conda run -n EP-contacts --no-capture-output"
   ```

   APA and EP/background have no R/rpy2 dependency (pure Python + awk/bash),
   so `--legacy-shell-prefix` only matters for `--workflows local-decay`.
3. `touche`'s own dependencies (`uv sync --dev`), same as
   `reference_replication.py`.

`ContactCaller_microC.bsh` also hardcodes
`export LD_LIBRARY_PATH=/programs/R-4.0.5/lib/` (a Cornell HPC path). This
script does not patch that -- `_reference/E-P_contacts`-style checkouts are
read-only prior art, not something to edit. A nonexistent directory in
`LD_LIBRARY_PATH` is normally harmless as long as your R install's shared
libraries are otherwise on the default linker search path; if the local-decay
legacy step fails with an R/library loading error, this is the first thing to
check.

## What It Compares

Per workflow family, one touche command vs. its legacy equivalent, same
parameters:

| Family | touche step | Legacy step |
| --- | --- | --- |
| local-decay | `local-decay call` (K562, `--dist 1000000 --cap 2000`) | `ContactCaller_microC.bsh` + concatenating its per-bait outputs |
| APA | `apa aggregate` (`--apa-sample`, default DMSO; `25-150kb`, `10kb` window, `50` pixels) | `MicroC_Stranded_Aggregation_pipeline_with_1D_signal.bsh` |
| background | `background count` (`--background-sample`, default DMSO; `25-150kb`, `2.5kb` window, `10-150kb` background) | `MicroC_EP_and_BG_contacts.bsh` |

Only the compute-heavy call/aggregate/count steps are timed -- not the
downstream assign/plot/compare steps, which are comparably cheap
post-processing on both sides and aren't where the performance gap is.

Select a subset with `--workflows local-decay apa` etc. Choose which mESC
treatment APA/background compare against with `--apa-sample`/
`--background-sample` (`dmso`/`flv`/`trp`).

## Core Usage

The two sides don't use cores the same way, and the legacy scripts aren't even
consistent with each other -- worth knowing before reading the speedup
numbers as a clean "touche is Nx faster on the same hardware" claim.

**touche** (all three workflows): governed by `NUMBA_NUM_THREADS`, which numba
defaults to every logical core on the machine. Nothing in
`legacy_timing_comparison.py` caps it.

`local-decay call` is the one workflow where `NUMBA_NUM_THREADS` alone is not
enough to saturate a many-core box. Only its LOWESS kernel is
`prange`-parallel; the rest of each bait's work (`_call_bait_contacts`'s
region filtering, histogram construction, `fit_zero_inflation_model`'s own
glue code) is single-threaded Python/NumPy. At real-data scale that
single-threaded glue code is the majority of wall time (~66% in a
500-bait/3M-contact profile, see `notes/numba-implementation-plan.md`), and
only `local-decay call --jobs` (a bait-level thread pool; baits are
independent, so this is exact, not approximate) parallelizes it.

This script's own `--jobs`/`-j` forwards to that flag and **defaults to
`available_cores()`** (touche's own CLI defaults `--jobs` to `1` instead --
deliberately conservative there, since it's a general-purpose library/CLI
that also needs to serve small inputs where thread-pool setup outweighs the
benefit; this script always runs against the full real Gasperini bait set, so
that concern doesn't apply and a "use what you have" default serves the
comparison's purpose better). `available_cores()` prefers `NUMBA_NUM_THREADS`
if you've set it, else `os.sched_getaffinity` (respects scheduler/cgroup CPU
pinning on Linux, unlike `os.cpu_count()`, which reports the whole node
regardless of your actual allocation), else `os.cpu_count()`.

Each worker thread still caps its own numba thread budget to
`get_num_threads() // n_jobs` before running kernels (see
`_call_bait_contacts_threaded` in `src/touche/local_decay.py`), so this
doesn't oversubscribe *relative to what `get_num_threads()` reports* -- but
that's only accurate if `NUMBA_NUM_THREADS` (or the detected affinity) matches
what your scheduler actually gave you. On a shared HPC node, explicitly set
`NUMBA_NUM_THREADS` to your real per-job core allocation (e.g. `export
NUMBA_NUM_THREADS=$NSLOTS` for SGE, `$SLURM_CPUS_PER_TASK` for SLURM) before
running, so both the default `--jobs` and numba's own kernels agree with your
actual allocation instead of the whole node's core count.

Pass `--jobs 1` explicitly to see the fully-sequential baseline instead (e.g.
to reproduce the earlier low-utilization behavior, or to isolate how much of
the speedup vs. legacy comes from `--jobs` specifically vs. the numba
kernels alone). Returns diminish past `n_jobs≈4` on a 10-core machine in
touche's own benchmark, so the default isn't guaranteed optimal on every
machine -- if you want to squeeze out more, try a couple of explicit values
and compare `report/summary.md`'s wall time for `local-decay-call`.

**Legacy**, per `.bsh` script:

| Workflow | Legacy parallelism | Cores used |
| --- | --- | --- |
| local-decay (`ContactCaller_microC.bsh`) | `wait_a_second()` throttles backgrounded `&` jobs at a **hardcoded 30** | up to 30 single-core processes |
| APA (`MicroC_Stranded_Aggregation_pipeline_with_1D_signal.bsh`) | same throttle pattern, capped at **60** | up to 60 single-core processes |
| background (`MicroC_EP_and_BG_contacts.bsh`) | setup/splitting stages are capped at 60, but the main EP/background counting loop (the dominant cost) has no `&` and no throttle at all | **1 core** for the dominant cost |

Two consequences:

- `--legacy-cpu` is passed through faithfully as `ContactCaller_microC.bsh`'s
  documented positional `[CPU.threads]` argument, but that script only
  **echoes** the value for its startup diagnostics -- it never uses it to set
  the concurrency cap, which stays hardcoded at 30 regardless. This script
  doesn't patch that (the legacy `.bsh` is read-only prior art), so
  `--legacy-cpu` doesn't actually change legacy core usage today.
- Because background's real counting loop is single-threaded, its
  touche-vs-legacy speedup will scale up with core count on a many-core box
  in a way local-decay's and APA's won't (both cap at 30/60 regardless of
  how many cores are available) -- that's a genuine property of the legacy
  code, not an artifact of this comparison.

`local-decay`'s touche side is reported two ways in `report/speedup.md`:
warm (`local-decay call` reusing a persistent NPZ cache) and cold (adding the
one-time `preprocess build-cache` cost), since the legacy pipeline has no
persistent-cache equivalent and re-parses the raw pairs file every run.

## Running on a remote server

Because a legacy step can run for hours, run it inside `tmux` so an SSH
disconnect doesn't kill it, and keep results incremental so a crash doesn't
lose already-measured (possibly hours-long) steps.

Start a session and launch the benchmark inside it:

```bash
tmux new -s touche-legacy-bench

# inside the tmux session:
uv run python scripts/legacy_timing_comparison.py \
  --reference-dir /path/to/E-P_contacts \
  --legacy-shell-prefix "conda run -n EP-contacts --no-capture-output" \
  --work-dir benchmark/legacy-timing-comparison \
  --progress
```

Detach with `Ctrl-b` `d` and disconnect freely. Reattach later to watch
`--progress` output live:

```bash
tmux ls
tmux attach -t touche-legacy-bench
```

`legacy_timing_comparison.py` writes each step's result to
`benchmark-results.jsonl` as soon as that step finishes (not just at the end),
so if the session or box dies partway through, resume in a fresh session
without re-running already-completed steps:

```bash
tmux new -s touche-legacy-bench

# inside the tmux session:
uv run python scripts/legacy_timing_comparison.py \
  --reference-dir /path/to/E-P_contacts \
  --legacy-shell-prefix "conda run -n EP-contacts --no-capture-output" \
  --work-dir benchmark/legacy-timing-comparison \
  --resume-from benchmark/legacy-timing-comparison/benchmark-results.jsonl
```

Once finished, pull back just the report (small) rather than the full
`data/`/`outputs/` trees (large, mostly downloaded pairs files and legacy
intermediate `.bed`/`.gz` scratch files):

```bash
rsync -av remote:touche/benchmark/legacy-timing-comparison/report/ ./legacy-timing-report/
```

## Outputs

Default work directory: `benchmark/legacy-timing-comparison/`.

- `data/`: downloaded pairs files and anchor/CSV inputs (only what the
  selected `--workflows`/`--apa-sample`/`--background-sample` need).
- `outputs/`: touche command outputs, NPZ cache, and legacy outputs under
  `outputs/legacy/`.
- `logs/*.stdout`/`*.stderr`: per-step subprocess output.
- `benchmark-results.jsonl`: one profiled result per step, written
  incrementally.
- `benchmark-manifest.json`: settings and every step result.
- `report/summary.md`, `report/summary.csv`, `report/index.html`: the same
  step-by-step report shape `reference_replication.py` writes (wall time,
  peak RSS, CPU, output sizes).
- `report/speedup.md`, `report/speedup.csv`, `report/speedup.svg`: the
  touche-vs-legacy wall-time comparison, per workflow family.

## Preview Without Running Anything

```bash
uv run python scripts/legacy_timing_comparison.py \
  --reference-dir /path/to/E-P_contacts --dry-run
```

Prints the planned downloads and step commands (including the exact legacy
`bash -c` invocations) as JSON without downloading or running anything.
