# touche documentation

This directory contains human-facing usage notes, tutorials, and reproducibility
guides for `touche`.

Implementation plans, agent logs, experiments, and design sketches belong in
`notes/` instead.

## Running Commands

If `touche` is installed with `pip`, run commands directly with `touche`.
When working from a development checkout or with a `uv` managed project,
use `uv run touche` as shown in the examples.

For routine analyses, start with the [CLI reference](cli.md)'s main workflow
section. It shows when to preprocess pairs, when local-decay caches are useful,
which `run` wrappers to use, and how to set `NUMBA_NUM_THREADS` for long jobs.
The same guide also documents the `polars-lts-cpu` install workaround for older
CPUs and x86-64 Python on Apple Silicon under Rosetta.

## Guides

- [CLI reference](cli.md): command groups, common options, examples, outputs,
  and run-wrapper manifests.
- [Micro-C preprocessing](micro-c-preprocessing.md): how `touche` hands off
  from distiller-nf or equivalent pairs-producing workflows, then filters,
  converts, QC-summarizes, and caches pairs.
- [Reproducing reference plots](reproducing-reference-plots.md): end-to-end
  commands for reproducing the local-decay, APA, and EP/background plots from
  the upstream [Danko-Lab/E-P_contacts](https://github.com/Danko-Lab/E-P_contacts)
  workflows.
- [Performance vs. the reference implementation](performance.md): how
  `touche`'s runtime, intermediate-file, and memory behavior compares to the
  reference bash/R/Python scripts, and why.
- [Notebook API](api.md): provisional in-memory APIs for notebooks,
  reusable contact indexes, and figure-returning plot helpers.
- [Testing and publishing](testing-and-publishing.md): CI, local checks, PyPI
  trusted publishing, and release checklist for the `ep-touche` distribution.
