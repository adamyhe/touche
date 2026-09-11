"""Curated notebook-facing re-export surface: `import touche.api as tt`.

Every name here is public and intended for interactive use; add new
compute/plot/statistics functions to both the imports and `__all__` below
when they're meant to be reachable this way.

Roughly grouped:

- pair universe -- `read_bed_anchors`, `build_pair_table`, `read_bedpe`,
  `write_bedpe`, `split_pair_anchors`;
- contact indexes -- `build_contact_indexes` and the NPZ cache helpers;
- domain compute -- `compute_local_decay`, `compute_apa`,
  `compute_ep_and_background`, plus their file-driven wrappers;
- statistics -- `test_contacts`, `assess_calibration`, `summarize_apa`,
  `test_background_change`, `compare_groups`, `compare_paired`,
  `correlate`, `match_pairs`, `balance_table`, and the `touche.stats`
  primitives underneath them;
- interoperability -- `read_loop_calls`, `write_loop_calls`,
  `annotate_pairs`.

Every statistical function returns a `StatResult`: a polars table plus the
`MethodInfo` describing what was tested, over what universe, under what
assumptions, and whether the answer is descriptive, technical-sampling, or
biological inference. Read `result.info` before quoting `result.table`.
"""

from __future__ import annotations

from touche.adapters import LOOP_FORMATS, annotate_pairs, read_loop_calls, write_loop_calls
from touche.anchors import read_bed_anchors
from touche.apa import (
    ApaResult,
    aggregate_apa,
    compare_apa_change,
    compute_apa,
    plot_apa_change,
    plot_raw_apa_heatmap,
    read_apa_matrix,
    write_apa_result,
)
from touche.apa_masks import (
    ApaScore,
    MaskSpec,
    build_masks,
    default_masks,
    read_masks,
    summarize_apa,
    write_masks,
)
from touche.background import (
    compare_background_ratios,
    compute_ep_and_background,
    count_ep_and_background,
    merge_background_counts,
    plot_background_scatter,
    read_background_counts,
)
from touche.compare import balance_table, compare_groups, compare_paired, correlate, match_pairs
from touche.contacts import build_contact_indexes, build_npz_cache, load_npz_cache, write_npz_cache
from touche.differential import test_background_change, zero_class_counts
from touche.instrumentation import Instrumentation, make_instrumentation
from touche.local_decay import (
    assign_pair_types,
    call_local_decay,
    compute_local_decay,
    plot_pair_type_distribution,
    read_center_anchors,
    to_tidy_calls,
)
from touche.metadata import METHOD_REGISTRY, MethodInfo, StatResult, describe_method
from touche.pairs import (
    PAIR_COLUMNS,
    attach_pair_ids,
    build_pair_table,
    make_pair_ids,
    pair_table_from_centers,
    read_bedpe,
    split_pair_anchors,
    write_bedpe,
)
from touche.significance import assess_calibration, read_local_decay_calls, test_contacts
from touche.stats import (
    adjust_pvalue_column,
    adjust_pvalues,
    binom_sf_greater,
    bootstrap_ci,
    bootstrap_difference_ci,
    cliffs_delta,
    fisher_greater_batch,
    log2_fold_change,
    log_odds_ratio,
    median_difference,
    poisson_sf_greater,
    rank_biserial,
)

__all__ = [
    "LOOP_FORMATS",
    "METHOD_REGISTRY",
    "PAIR_COLUMNS",
    "ApaResult",
    "ApaScore",
    "Instrumentation",
    "MaskSpec",
    "MethodInfo",
    "StatResult",
    "adjust_pvalue_column",
    "adjust_pvalues",
    "aggregate_apa",
    "annotate_pairs",
    "assess_calibration",
    "assign_pair_types",
    "attach_pair_ids",
    "balance_table",
    "binom_sf_greater",
    "bootstrap_ci",
    "bootstrap_difference_ci",
    "build_contact_indexes",
    "build_masks",
    "build_npz_cache",
    "build_pair_table",
    "call_local_decay",
    "cliffs_delta",
    "compare_apa_change",
    "compare_background_ratios",
    "compare_groups",
    "compare_paired",
    "compute_apa",
    "compute_ep_and_background",
    "compute_local_decay",
    "correlate",
    "count_ep_and_background",
    "default_masks",
    "describe_method",
    "fisher_greater_batch",
    "load_npz_cache",
    "log2_fold_change",
    "log_odds_ratio",
    "make_instrumentation",
    "make_pair_ids",
    "match_pairs",
    "median_difference",
    "merge_background_counts",
    "pair_table_from_centers",
    "plot_apa_change",
    "plot_background_scatter",
    "plot_pair_type_distribution",
    "plot_raw_apa_heatmap",
    "poisson_sf_greater",
    "rank_biserial",
    "read_apa_matrix",
    "read_background_counts",
    "read_bed_anchors",
    "read_bedpe",
    "read_center_anchors",
    "read_local_decay_calls",
    "read_loop_calls",
    "read_masks",
    "split_pair_anchors",
    "summarize_apa",
    "test_background_change",
    "test_contacts",
    "to_tidy_calls",
    "write_apa_result",
    "write_bedpe",
    "write_loop_calls",
    "write_masks",
    "write_npz_cache",
    "zero_class_counts",
]
