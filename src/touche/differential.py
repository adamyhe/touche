"""Zero-safe differential enhancer-promoter versus background analysis between libraries.

Public API: `test_background_change` (per-pair EP/background odds-ratio
comparison between two libraries, with BH-adjusted q-values and explicit
zero-class accounting) and `zero_class_counts` (the four-way zero/positive
breakdown on its own).

This is deliberately labelled **technical** inference, not biological. With
one library per condition, the only randomness a per-pair 2x2 table can
speak to is read sampling within the libraries observed. It cannot
distinguish a condition effect from between-animal, between-passage, or
between-prep variation, because those were never sampled. Reporting it as
evidence of a biological difference is the single most common way this
analysis goes wrong, so `test_background_change` refuses to call itself
anything else and says so in its warnings.

The other thing this module refuses to do is drop zeros. A pair with
contacts in one library and none in the other is a complete gain or loss --
usually the effect of greatest interest -- and filtering it because a ratio
is undefined removes exactly the strongest observations. Raw infinite
log ratios are reported as infinities; a separate pseudocount column exists
only so a plot has something finite to draw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from touche.background import PAIR_COLUMNS, merge_background_counts
from touche.metadata import StatResult, method_info
from touche.models import NamedPath
from touche.pairs import make_pair_ids
from touche.stats import ADJUST_METHODS, adjust_pvalue_column, log2_fold_change, log_odds_ratio

TEST_METHODS = {"wald", "fisher"}


def test_background_change(
    control: NamedPath,
    treatment: NamedPath,
    *,
    method: str = "wald",
    fdr: str = "bh",
    correction: float = 0.5,
    display_pseudocount: float = 1.0,
    confidence: float = 0.95,
    out_path: str | Path | None = None,
) -> StatResult:
    """Compare one treatment library against one control on the EP-versus-background contrast.

    For each pair the 2x2 table is

    ```text
                    enhancer-promoter    matched background
        control          EP_C                   BG_C
        treatment        EP_T                   BG_T
    ```

    and the estimand is its log odds ratio: does enhancer-promoter signal
    change *more than its own local background does*. That contrast is what
    makes the comparison robust to a global difference in library depth or
    coverage, which a bare EP-count fold change is not.

    `method="wald"` uses the Woolf standard error on the log odds ratio --
    vectorized, and the default. `method="fisher"` runs an exact two-sided
    conditional test per pair, which is better behaved at small counts and
    substantially slower. `correction` is the Haldane-Anscombe continuity
    correction applied to the tested table; the reported raw log2 ratio
    change is uncorrected, so complete gains and losses stay infinite.

    Every pair present in both inputs is tested, zeros included. Pairs
    present in only one input are not tested -- absence from a file means
    the pair was outside that sample's candidate universe, not that it had
    zero contacts -- and their count is reported.
    """

    if method not in TEST_METHODS:
        raise ValueError(f"method must be one of: {', '.join(sorted(TEST_METHODS))}")
    if fdr not in ADJUST_METHODS:
        raise ValueError(f"fdr must be one of: {', '.join(sorted(ADJUST_METHODS))}")
    if correction < 0:
        raise ValueError("correction must be non-negative")

    shared = merge_background_counts([control, treatment], how="inner")
    union = merge_background_counts([control, treatment], how="full")
    not_shared = union.height - shared.height

    ep_c = shared[f"EP_contacts_{control.name}"].cast(pl.Float64).to_numpy()
    bg_c = shared[f"BG_contacts_{control.name}"].cast(pl.Float64).to_numpy()
    ep_t = shared[f"EP_contacts_{treatment.name}"].cast(pl.Float64).to_numpy()
    bg_t = shared[f"BG_contacts_{treatment.name}"].cast(pl.Float64).to_numpy()

    estimate, standard_error = log_odds_ratio(ep_t, bg_t, ep_c, bg_c, correction=correction)
    if method == "wald":
        p_values, ci_low, ci_high = _wald_test(estimate, standard_error, confidence)
    else:
        p_values = _fisher_two_sided(ep_t, bg_t, ep_c, bg_c)
        _, ci_low, ci_high = _wald_test(estimate, standard_error, confidence)

    table = shared.select(PAIR_COLUMNS).with_columns(
        make_pair_ids("chr", "promoter", "chr", "enhancer"),
        pl.Series(f"EP_{control.name}", ep_c, dtype=pl.Int64),
        pl.Series(f"BG_{control.name}", bg_c, dtype=pl.Int64),
        pl.Series(f"EP_{treatment.name}", ep_t, dtype=pl.Int64),
        pl.Series(f"BG_{treatment.name}", bg_t, dtype=pl.Int64),
        pl.Series("log2_ratio_change", log2_fold_change(ep_t * bg_c, ep_c * bg_t)),
        pl.Series(
            "log2_ratio_change_display",
            log2_fold_change(
                (ep_t + display_pseudocount) * (bg_c + display_pseudocount),
                (ep_c + display_pseudocount) * (bg_t + display_pseudocount),
            ),
        ),
        pl.Series("log_odds_ratio", estimate),
        pl.Series("log_odds_ratio_se", standard_error),
        pl.Series("ci_low", ci_low),
        pl.Series("ci_high", ci_high),
        pl.Series("p_value", p_values),
        pl.lit(f"odds_ratio_{method}").alias("method"),
    )
    table = adjust_pvalue_column(table, method=fdr)

    zeros = zero_class_counts(ep_c, ep_t)
    n_tested = int(np.isfinite(p_values).sum())
    warnings = [
        "This is technical inference: it is conditional on the two observed libraries and is not "
        "evidence of biological variation between conditions. A biological claim needs replicate "
        "libraries as the unit of analysis.",
    ]
    if zeros["zero_zero"]:
        warnings.append(
            f"{zeros['zero_zero']} pairs have zero enhancer-promoter contacts in both libraries. "
            "They are retained and tested (the test is uninformative for them) rather than "
            "filtered, so the FDR family matches the declared pair universe."
        )
    if not_shared:
        warnings.append(
            f"{not_shared} pairs are present in only one input and were not tested; absence from a "
            "counts file means the pair was outside that sample's candidate universe."
        )

    info = method_info(
        "odds_ratio",
        alternative="two-sided",
        unit_of_analysis="bait-prey pair within one library pair",
        cluster_unit=None,
        n_input=union.height,
        n_tested=n_tested,
        fdr_method=fdr,
        fdr_family="all pairs shared by both libraries",
        parameters={
            "control": control.name,
            "treatment": treatment.name,
            "test": method,
            "correction": correction,
            "display_pseudocount": display_pseudocount,
            "confidence": confidence,
            **{f"n_{key}": value for key, value in zeros.items()},
        },
        filtered={"not_shared": not_shared, "untestable": shared.height - n_tested},
        warnings=warnings,
    )
    result = StatResult(table=table, info=info)
    if out_path is not None:
        result.write(out_path)
    return result


def zero_class_counts(control_counts: np.ndarray, treatment_counts: np.ndarray) -> dict[str, int]:
    """Four-way zero/positive breakdown of paired count vectors.

    `zero_positive` is a complete gain and `positive_zero` a complete loss.
    Reporting these separately is what stops a pipeline from quietly
    discarding its largest effects as "undefined ratios".
    """

    control_zero = np.asarray(control_counts) == 0
    treatment_zero = np.asarray(treatment_counts) == 0
    return {
        "zero_zero": int(np.sum(control_zero & treatment_zero)),
        "zero_positive": int(np.sum(control_zero & ~treatment_zero)),
        "positive_zero": int(np.sum(~control_zero & treatment_zero)),
        "positive_positive": int(np.sum(~control_zero & ~treatment_zero)),
    }


def _wald_test(
    estimate: np.ndarray, standard_error: np.ndarray, confidence: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided normal-approximation p-values and interval for a log odds ratio."""
    from scipy.stats import norm

    critical = float(norm.ppf(0.5 + confidence / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = estimate / standard_error
        p_values = 2.0 * norm.sf(np.abs(z))
        low = estimate - critical * standard_error
        high = estimate + critical * standard_error
    usable = np.isfinite(estimate) & np.isfinite(standard_error) & (standard_error > 0)
    return (
        np.where(usable, p_values, np.nan),
        np.where(usable, low, np.nan),
        np.where(usable, high, np.nan),
    )


def _fisher_two_sided(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Exact two-sided conditional p-value per 2x2 table.

    Looped rather than vectorized: the two-sided exact p-value sums
    hypergeometric probabilities no more likely than the observed table,
    which has no batch form in scipy. Budget roughly a minute per million
    pairs and prefer `method="wald"` when counts are large.
    """
    from scipy.stats import fisher_exact

    p_values = np.empty(a.shape[0], dtype=np.float64)
    for i in range(a.shape[0]):
        table = [[int(a[i]), int(b[i])], [int(c[i]), int(d[i])]]
        p_values[i] = fisher_exact(table).pvalue
    return p_values
