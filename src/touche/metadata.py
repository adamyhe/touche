"""Self-describing metadata envelope carried by every `touche` statistical result.

Public API: `MethodInfo` (what test was run, on what, under what
assumptions), `StatResult` (a polars table plus its `MethodInfo`), and
`METHOD_REGISTRY`/`describe_method` (the documented hypothesis, null, and
inference class for each named method).

Every inferential function in `touche` returns a `StatResult` rather than a
bare frame, because a p-value column without its test universe, unit of
replication, and inference class is not interpretable and not reproducible.
Writing a `StatResult` always writes a `.meta.json` sidecar next to the
table for exactly that reason.

`inference_class` is the field to read first:

- `descriptive` -- summarizes the data at hand; no population claim.
- `technical` -- inference about read sampling within the observed
  libraries, conditional on those libraries. Not evidence about biological
  variation between samples.
- `biological` -- inference that generalizes across biological replicates.
  Requires replicate libraries as the unit of analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from touche import __version__

INFERENCE_CLASSES = {"descriptive", "technical", "biological"}

METHOD_REGISTRY: dict[str, dict[str, str]] = {
    "legacy_fisher": {
        "inference_class": "descriptive",
        "hypothesis": (
            "One-sided Fisher exact test on a 2x2 table of (observed, fitted expected) "
            "against (histogram bins - observed, histogram bins - fitted expected)."
        ),
        "null": (
            "Not a generative null. The table's second column is a fitted expectation "
            "rather than an observed count, and its trial total is the number of "
            "one-base distance-histogram bins rather than a contact total, so the "
            "resulting quantity is a reproducibility-preserving score, not a calibrated "
            "p-value. Retained to reproduce published reference output."
        ),
        "unit_of_analysis": "bait-prey pair",
        "reference": "Danko Lab E-P_contacts ContactCaller_microC",
    },
    "binomial": {
        "inference_class": "technical",
        "hypothesis": (
            "K ~ Binomial(N, p0), where K is the observed contacts between the bait and "
            "prey windows, N is the number of contacts anchored at the bait in the "
            "prey's direction, and p0 is the local distance-decay probability that such "
            "a contact falls in the prey window."
        ),
        "null": "The bait's contacts are distributed according to its fitted local distance decay alone.",
        "unit_of_analysis": "bait-prey pair",
        "reference": "FitHiC2-style distance-aware binomial contact significance",
    },
    "poisson": {
        "inference_class": "technical",
        "hypothesis": "K ~ Poisson(N * p0), the rare-event limit of the binomial model.",
        "null": "The bait's contacts are distributed according to its fitted local distance decay alone.",
        "unit_of_analysis": "bait-prey pair",
        "reference": "FitHiC2-style distance-aware contact significance",
    },
    "mannwhitney": {
        "inference_class": "descriptive",
        "hypothesis": "Two independent groups of pair-level values are drawn from the same distribution.",
        "null": "Group membership is exchangeable with respect to the compared value.",
        "unit_of_analysis": "bait-prey pair",
        "reference": "Mann-Whitney U",
    },
    "wilcoxon": {
        "inference_class": "descriptive",
        "hypothesis": "Paired differences are symmetric about zero.",
        "null": "The pairing carries no systematic shift.",
        "unit_of_analysis": "matched pair of observations",
        "reference": "Wilcoxon signed-rank",
    },
    "pearson": {
        "inference_class": "descriptive",
        "hypothesis": "Two pair-level values are linearly uncorrelated.",
        "null": "rho = 0.",
        "unit_of_analysis": "bait-prey pair",
        "reference": "Pearson product-moment correlation",
    },
    "spearman": {
        "inference_class": "descriptive",
        "hypothesis": "Two pair-level values are monotonically unrelated.",
        "null": "rho_s = 0.",
        "unit_of_analysis": "bait-prey pair",
        "reference": "Spearman rank correlation",
    },
    "odds_ratio": {
        "inference_class": "technical",
        "hypothesis": (
            "The enhancer-promoter versus matched-background odds are equal between two "
            "libraries (a per-pair 2x2 table)."
        ),
        "null": "log odds ratio = 0, conditional on the observed libraries.",
        "unit_of_analysis": "bait-prey pair within one library pair",
        "reference": "Per-pair contingency comparison; not a biological-replicate test",
    },
    "ranking_evaluation": {
        "inference_class": "descriptive",
        "hypothesis": (
            "How well each score ranks validated functional enhancer-promoter pairs above "
            "non-functional ones, measured by average precision and ROC AUC."
        ),
        "null": (
            "A score with no information ranks at the positive-class prevalence (AUPRC) and "
            "at 0.5 (ROC AUC). Compare against those, and against a distance-only baseline."
        ),
        "unit_of_analysis": "labelled bait-prey pair",
        "reference": "Enhancer-prediction evaluation practice; AUPRC primary for imbalanced labels",
    },
    "apa_mask": {
        "inference_class": "descriptive",
        "hypothesis": "None; a summary of aggregated pileup signal over named submatrix masks.",
        "null": "Not applicable unless a null model is attached.",
        "unit_of_analysis": "aggregated pileup over a pair set",
        "reference": "coolpup.py-style pileup scoring (P2LL, P2M)",
    },
}


def describe_method(method: str) -> dict[str, str]:
    """Registry entry for `method`, or a minimal stub for a method not yet registered."""
    return dict(
        METHOD_REGISTRY.get(
            method,
            {
                "inference_class": "descriptive",
                "hypothesis": "Not documented.",
                "null": "Not documented.",
                "unit_of_analysis": "unspecified",
                "reference": "unspecified",
            },
        )
    )


@dataclass(frozen=True, slots=True)
class MethodInfo:
    """Everything needed to interpret and reproduce one statistical result.

    `n_input`/`n_tested` make the test universe explicit: the difference is
    how many rows were excluded, and `filtered` says why. `fdr_family`
    records what a `q_value` column was corrected over -- `"all"` for one
    global family, or the grouping columns for a stratified one.
    """

    method: str
    inference_class: str
    hypothesis: str
    null: str
    alternative: str = "two-sided"
    unit_of_analysis: str = "bait-prey pair"
    cluster_unit: str | None = None
    n_input: int = 0
    n_tested: int = 0
    fdr_method: str | None = None
    fdr_family: str | None = None
    seed: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    filtered: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    reference: str = "unspecified"

    def write_sidecar(self, table_path: str | Path, *, extra: dict[str, Any] | None = None) -> Path:
        """Write this metadata as `<table_path>.meta.json` and return that path.

        Split out from `StatResult.write` because a result whose table is in
        a fixed legacy layout -- which has nowhere to record a method -- still
        needs one. `extra` merges in fields only the caller knows, such as the
        row count.
        """
        table_path = Path(table_path)
        meta_path = table_path.with_suffix(table_path.suffix + ".meta.json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.to_dict(), **(extra or {})}
        meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return meta_path

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, with the `touche` version and a UTC timestamp attached."""
        return {
            "schema_version": 1,
            "touche_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "method": self.method,
            "inference_class": self.inference_class,
            "hypothesis": self.hypothesis,
            "null": self.null,
            "alternative": self.alternative,
            "unit_of_analysis": self.unit_of_analysis,
            "cluster_unit": self.cluster_unit,
            "n_input": int(self.n_input),
            "n_tested": int(self.n_tested),
            "fdr_method": self.fdr_method,
            "fdr_family": self.fdr_family,
            "seed": self.seed,
            "parameters": self.parameters,
            "filtered": self.filtered,
            "warnings": list(self.warnings),
            "reference": self.reference,
        }


def method_info(method: str, **overrides: Any) -> MethodInfo:
    """Build a `MethodInfo` from `METHOD_REGISTRY`, overriding any field.

    Keeps call sites from restating a method's fixed hypothesis/null text
    every time they report a result, while still letting a caller record the
    run-specific fields (counts, seed, parameters, warnings).
    """
    fields: dict[str, Any] = dict(describe_method(method))
    fields.update(overrides)
    return MethodInfo(method=method, **fields)


@dataclass(frozen=True, slots=True)
class StatResult:
    """A statistical result table paired with the `MethodInfo` describing it."""

    table: pl.DataFrame
    info: MethodInfo

    @property
    def height(self) -> int:
        """Number of result rows."""
        return self.table.height

    def to_dict(self) -> dict[str, Any]:
        """Metadata dict plus the result rows, for JSON summaries and manifests."""
        return {**self.info.to_dict(), "rows": self.table.height}

    def write(self, path: str | Path, *, separator: str = "\t") -> dict[str, Path]:
        """Write the table to `path` and its metadata to a `.meta.json` sidecar.

        The sidecar is not optional: a q-value column whose FDR family and
        inference class live only in a terminal log is not reproducible.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.write_csv(path, separator=separator)
        return {"table": path, "metadata": self.info.write_sidecar(path, extra={"rows": self.table.height})}
