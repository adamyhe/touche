# touche documentation

This directory contains human-facing usage notes, tutorials, and reproducibility
guides for `touche`.

Implementation plans, agent logs, experiments, and design sketches belong in
`notes/` instead.

## Running Commands

If `touche` is installed with `pip`, run commands directly with `touche`.
When working from a development checkout, use `uv run touche` as shown in the
examples.

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
- [Notebook API](api.md): provisional in-memory APIs for notebooks,
  reusable contact indexes, and figure-returning plot helpers.
- [Testing and publishing](testing-and-publishing.md): CI, local checks, PyPI
  trusted publishing, and release checklist for the `ep-touche` distribution.
