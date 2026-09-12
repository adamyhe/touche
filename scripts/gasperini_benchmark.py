#!/usr/bin/env python3
"""Benchmark touche's per-pair contact scores against the Gasperini K562 functional set.

Answers the question the `binomial` default was changed on faith: does a
calibrated test actually rank validated enhancer-promoter pairs better than
the reference workflow's Fisher score -- and does either beat the trivial
baselines?

Three things are measured, matching the benchmark plan in
`notes/touche-literature-statistics-integration-plan.md`:

1. **Null calibration.** Distance-preserving random-shift pairs give a set
   that should be null. A correctly specified test produces uniform
   p-values there; `assess_calibration` reports KS uniformity and empirical
   type-I error, stratified by distance and by coverage, because a method
   can look uniform overall and be badly anticonservative for exactly the
   short-range, low-coverage pairs of interest.
2. **Functional prediction.** AUPRC (primary, because the labels are
   heavily imbalanced) and ROC AUC (secondary) at separating functional from
   non-functional pairs, with chromosome-held-out means and a bootstrap
   interval over chromosomes.
3. **Matched prediction.** The same, after matching positives and negatives
   on genomic distance and coverage, so a method is not credited for
   rediscovering that functional pairs are closer together.

Two baselines are scored alongside the methods and are the point of the
exercise: raw `observed` contact count, and distance alone. A contact score
that does not beat `neg_log10_distance` has not been shown to add anything.

`--demo` runs the whole pipeline on a small synthetic dataset with a planted
signal, in seconds and with no downloads. Use it to exercise the code path;
use the real mode to answer the question.

See `gasperini_benchmark.md` for usage and how to read the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference_replication import (  # noqa: E402
    LOCAL_DECAY_BAITS,
    LOCAL_DECAY_FUNCTIONAL,
    LOCAL_DECAY_NONFUNCTIONAL,
    LOCAL_DECAY_PREYS,
    REFERENCE_RAW_BASE,
    Download,
    available_cores,
    download_reference_inputs,
)

from touche import __version__  # noqa: E402
from touche.compare import balance_table, match_pairs  # noqa: E402
from touche.evaluate import evaluate_scores, precision_recall_curve  # noqa: E402
from touche.local_decay import call_local_decay, read_center_anchors  # noqa: E402
from touche.significance import assess_calibration, test_contacts  # noqa: E402
from touche.stats import log2_fold_change, pearson_dispersion  # noqa: E402

K562_PAIRS = "GSE206131_K562_cis_mapq30_pairs.txt.gz"
GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE206nnn/GSE206131/suppl"

METHODS = ("binomial", "poisson", "legacy_fisher", "negative_binomial")
BASELINES = ("log2_oe", "observed", "neg_log10_distance", "log2_oe_global")

# Keys of the reference functional/nonfunctional CSVs. In the reference
# workflow baits are TREs and preys are promoters, so `target_site` is the
# bait and `target_promoter` is the prey.
LABEL_KEYS = ("target_site.chr", "target_promoter.center", "target_site.center")

# Covariates the reference label files already carry, and the confounders the
# benchmark plan asks to control for: enhancer accessibility and promoter
# transcription. Carried through when present so the matched comparison can
# balance on them instead of only on distance and coverage.
LABEL_COVARIATES = {"mean_ATAC_RPM": "enhancer_atac", "PROseq_GB_RPKM": "promoter_proseq"}


@dataclass(frozen=True, slots=True)
class Inputs:
    """Everything the benchmark reads, whether downloaded or generated."""

    pairs: Path
    baits: Path
    preys: Path
    functional: Path
    nonfunctional: Path


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = build_demo_inputs(args.work_dir) if args.demo else fetch_inputs(args)
    cache_dir = args.work_dir / "contact_cache"
    call_kwargs = dict(
        cache_dir=cache_dir,
        decay_model=args.decay_model,
        dist=args.dist,
        cap=args.cap,
        min_distance=args.min_distance,
        n_jobs=args.jobs,
        index_strategy="all" if args.demo else "cache",
        progress=args.progress,
    )

    log("calling contacts on the real anchor pairs")
    real_calls = call_tidy(inputs, args.work_dir / "real", "real", **call_kwargs)

    log(f"calling contacts on {len(args.shifts)} distance-preserving shifted anchor sets")
    null_calls = pl.concat(
        [
            call_tidy(
                shifted_inputs(inputs, shift, args.work_dir / f"shift{shift}"),
                args.work_dir / f"shift{shift}",
                f"shift{shift}",
                **call_kwargs,
            ).with_columns(pl.lit(shift).alias("shift"))
            for shift in args.shifts
        ]
    )
    # The dispersion belongs to the null, not to the pairs being tested.
    dispersion = float(
        pearson_dispersion(
            null_calls["observed"].to_numpy().astype(float),
            null_calls["expected"].to_numpy().astype(float),
        )
    )
    log(f"  dispersion estimated on the null pairs: {dispersion:.3f}")
    null = score_all_methods(null_calls, dispersion=dispersion)
    real = add_abc_score(
        attach_labels(
            add_global_decay_score(
                score_all_methods(real_calls, dispersion=dispersion), null
            ),
            inputs.functional,
            inputs.nonfunctional,
        )
    )
    null, contaminated = drop_contaminated_null(null, inputs, tolerance=args.null_exclusion)
    log(f"  dropped {contaminated} shifted pairs that landed back on real anchors")
    if null.is_empty():
        raise SystemExit(
            "Every shifted pair landed back on a real candidate pair, so there is no null set. "
            "Choose --shifts that are not commensurate with the anchor spacing."
        )

    outputs: dict[str, Path] = {}
    outputs["scored"] = write_table(real, out_dir / "scored_pairs.tsv")
    outputs["null_scored"] = write_table(null, out_dir / "null_pairs.tsv")

    log("assessing null calibration")
    calibration = calibration_report(null)
    outputs["calibration"] = write_table(calibration, out_dir / "calibration.tsv")
    expected_bias = expected_bias_report(null)
    outputs["expected_bias"] = write_table(expected_bias, out_dir / "expected_bias.tsv")
    decay_comparison = decay_model_comparison(null)
    if not decay_comparison.is_empty():
        outputs["decay_comparison"] = write_table(
            decay_comparison, out_dir / "decay_model_comparison.tsv"
        )

    log("evaluating functional prediction")
    prediction = evaluate_scores(
        real,
        label_col="label",
        score_cols=score_columns(real),
        positive_label="positive",
        held_out_col="chrom",
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    prediction.write(out_dir / "prediction.tsv")
    outputs["prediction"] = out_dir / "prediction.tsv"

    log("evaluating functional prediction on distance- and coverage-matched pairs")
    matched, balance = matched_evaluation(real, args)
    outputs["balance"] = write_table(balance, out_dir / "balance.tsv")
    if matched is None:
        log("  too few matched pairs to evaluate; skipping", level="warn")
        matched_prediction = None
    else:
        matched_prediction = evaluate_scores(
            matched,
            label_col="label",
            score_cols=score_columns(matched),
            positive_label="positive",
            held_out_col="chrom",
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        matched_prediction.write(out_dir / "prediction_matched.tsv")
        outputs["prediction_matched"] = out_dir / "prediction_matched.tsv"

    if not args.no_plots:
        log("writing figures")
        outputs.update(write_figures(real, null, out_dir))

    manifest = {
        "schema_version": 1,
        "touche_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "demo": args.demo,
        "argv": sys.argv,
        "parameters": {
            "decay_model": args.decay_model,
            "dispersion": dispersion,
            "dist": args.dist, "cap": args.cap, "min_distance": args.min_distance,
            "shifts": args.shifts, "bootstrap": args.bootstrap, "seed": args.seed,
            "match_distance_caliper": args.match_distance_caliper,
            "match_coverage_caliper": args.match_coverage_caliper,
            "match_activity_caliper": args.match_activity_caliper,
        },
        "counts": {
            "real_pairs": real.height,
            "null_pairs": null.height,
            "null_pairs_dropped_as_contaminated": contaminated,
            "positive": int((real["label"] == "positive").sum()),
            "negative": int((real["label"] == "negative").sum()),
            "unlabelled": int(real["label"].is_null().sum()),
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = write_summary(
        out_dir / "summary.md",
        manifest=manifest,
        calibration=calibration,
        expected_bias=expected_bias,
        decay_comparison=decay_comparison,
        prediction=prediction,
        matched_prediction=matched_prediction,
    )
    log(f"wrote {summary}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work-dir", type=Path, default=Path("benchmark/gasperini/work"))
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark/gasperini/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark/gasperini/report"))
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on a small synthetic dataset with a planted signal instead of downloading.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Require inputs to already exist.")
    parser.add_argument(
        "--decay-model",
        choices=["legacy", "normalized", "anchored"],
        default="anchored",
        help="Background-density scaling passed to local-decay; see touche's --decay-model.",
    )
    parser.add_argument("--dist", type=int, default=1_000_000)
    parser.add_argument("--cap", type=int, default=2_000)
    parser.add_argument("--min-distance", type=int, default=5_000)
    parser.add_argument(
        "--shifts",
        type=int,
        nargs="+",
        default=[1_000_000, -1_000_000, 2_000_000],
        help=(
            "Offsets (bp) applied to both anchor sets to build the null. Shifting both "
            "preserves each pair's distance exactly while moving it off its real locus."
        ),
    )
    parser.add_argument(
        "--null-exclusion",
        type=int,
        default=5_000,
        help=(
            "Drop a shifted pair when both of its anchors land within this many bases of a "
            "real candidate anchor. Without it a shift can map the anchor set back onto "
            "itself and the 'null' quietly contains real signal."
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--match-distance-caliper",
        type=float,
        default=0.05,
        help="Caliper on log10 genomic distance when matching positives to negatives.",
    )
    parser.add_argument(
        "--match-coverage-caliper",
        type=float,
        default=0.10,
        help="Caliper on log10 n_trials (per-bait coverage) when matching.",
    )
    parser.add_argument(
        "--match-activity-caliper",
        type=float,
        default=0.15,
        help=(
            "Caliper on log10 enhancer accessibility and log10 promoter transcription, when "
            "the label files supply them. Ignored otherwise."
        ),
    )
    parser.add_argument("--jobs", type=int, default=available_cores())
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.demo:
        # The synthetic genome is ~2 Mb, so real-data windows would swallow it.
        args.dist = min(args.dist, 200_000)
        # Not a multiple of the 50 kb anchor spacing, so the shift cannot
        # map the (jittered) anchor set back onto itself.
        args.shifts = [337_000, -337_000]
    return args


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def fetch_inputs(args: argparse.Namespace) -> Inputs:
    """Download (or locate) the real Gasperini K562 pairs, anchors, and labels."""
    data_dir = args.data_dir
    inputs = Inputs(
        pairs=data_dir / K562_PAIRS,
        baits=data_dir / "Input_files" / LOCAL_DECAY_BAITS,
        preys=data_dir / "Input_files" / LOCAL_DECAY_PREYS,
        functional=data_dir / "Input_files" / LOCAL_DECAY_FUNCTIONAL,
        nonfunctional=data_dir / "Input_files" / LOCAL_DECAY_NONFUNCTIONAL,
    )
    if args.skip_download:
        # `Inputs` uses slots, so it has no __dict__ to enumerate.
        missing = [
            str(getattr(inputs, field.name))
            for field in fields(inputs)
            if not Path(getattr(inputs, field.name)).exists()
        ]
        if missing:
            raise SystemExit("--skip-download given but these inputs are missing:\n  " + "\n  ".join(missing))
        return inputs

    downloads = [
        Download(K562_PAIRS, f"{GEO_BASE}/{K562_PAIRS}", inputs.pairs),
        *[
            Download(name, f"{REFERENCE_RAW_BASE}/{name}", data_dir / "Input_files" / name)
            for name in (
                LOCAL_DECAY_BAITS, LOCAL_DECAY_PREYS, LOCAL_DECAY_FUNCTIONAL, LOCAL_DECAY_NONFUNCTIONAL
            )
        ],
    ]
    log("fetching reference inputs (several GB on a first run)")
    download_reference_inputs(downloads)
    return inputs


def build_demo_inputs(work_dir: Path) -> Inputs:
    """Synthesize a small dataset with a planted enhancer-promoter signal.

    Half the candidate pairs get extra contacts injected between their two
    anchors and are labelled functional, so the benchmark should recover an
    AUPRC clearly above the prevalence baseline while the distance-only
    baseline sits at chance. A demo that only proved the code runs would not
    catch a metric wired up backwards.

    The injected signal is deliberately modest and variable, so the classes
    overlap and the methods are not all pinned at a perfect score -- a
    saturated demo cannot tell two methods apart either.

    Anchor positions are jittered rather than evenly spaced. On a regular
    grid a shift that is a multiple of the spacing maps the anchor set onto
    itself, and the random-shift null silently becomes the real data.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    chroms = ("chr1", "chr2", "chr3", "chr4")
    span = 2_000_000
    separation = 120_000

    rows: list[str] = []
    anchors: dict[str, list[tuple[int, int, bool]]] = {}
    for chrom in chroms:
        # Background: a decaying cloud of cis contacts across the chromosome.
        starts = rng.integers(0, span, 120_000)
        distances = np.clip(rng.exponential(60_000, starts.size).astype(int), 1_000, 400_000)
        rows += [
            f"{chrom}\t{a}\t{chrom}\t{min(a + d, span)}\t+\t-\tUU\t30\t30"
            for a, d in zip(starts, distances)
        ]

        grid = np.arange(100_000, span - separation - 100_000, 50_000)
        jittered = grid + rng.integers(-18_000, 18_000, grid.size)
        chrom_anchors = []
        for index, bait in enumerate(sorted(int(value) for value in jittered)):
            prey = bait + separation
            is_functional = index % 2 == 0
            chrom_anchors.append((bait, prey, is_functional))
            if not is_functional:
                continue
            # Modest, variable focal signal so the classes overlap.
            injected = int(rng.integers(4, 16))
            rows += [
                f"{chrom}\t{bait + j}\t{chrom}\t{prey + k}\t+\t-\tUU\t30\t30"
                for j, k in zip(
                    rng.integers(-400, 400, injected), rng.integers(-400, 400, injected)
                )
            ]
        anchors[chrom] = chrom_anchors

    inputs = Inputs(
        pairs=work_dir / "demo.pairs",
        baits=work_dir / "demo_baits.tsv",
        preys=work_dir / "demo_preys.tsv",
        functional=work_dir / "demo_functional.csv",
        nonfunctional=work_dir / "demo_nonfunctional.csv",
    )
    inputs.pairs.write_text("\n".join(rows) + "\n", encoding="utf-8")
    inputs.baits.write_text(
        "".join(f"{c}\t{b}\n" for c, values in anchors.items() for b, _, _ in values), encoding="utf-8"
    )
    inputs.preys.write_text(
        "".join(f"{c}\t{p}\n" for c, values in anchors.items() for _, p, _ in values), encoding="utf-8"
    )
    for path, wanted in ((inputs.functional, True), (inputs.nonfunctional, False)):
        labelled = [
            (chrom, prey, bait)
            for chrom, values in anchors.items()
            for bait, prey, is_functional in values
            if is_functional is wanted
        ]
        pl.DataFrame(labelled, schema=list(LABEL_KEYS), orient="row").write_csv(path)
    return inputs


def shifted_inputs(inputs: Inputs, shift: int, work_dir: Path) -> Inputs:
    """Anchor sets translated by `shift`, for the distance-preserving null.

    Both baits and preys move by the same offset, so every pair keeps its
    exact genomic distance and its exact anchor spacing while landing
    somewhere the real regulatory contact is not. Shifting only one set
    would change the distance distribution and make the null incomparable.

    Shifted anchors outside the original per-chromosome anchor span are
    dropped, so the null stays inside assayed territory rather than running
    off a chromosome end into a region with no coverage at all.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    shifted = Inputs(
        pairs=inputs.pairs,
        baits=work_dir / "baits.tsv",
        preys=work_dir / "preys.tsv",
        functional=inputs.functional,
        nonfunctional=inputs.nonfunctional,
    )
    baits = read_center_anchors(inputs.baits)
    preys = read_center_anchors(inputs.preys)
    bounds = (
        pl.concat([baits, preys])
        .group_by("chr")
        .agg(pl.col("center").min().alias("low"), pl.col("center").max().alias("high"))
    )
    for source, path in ((baits, shifted.baits), (preys, shifted.preys)):
        moved = (
            source.with_columns((pl.col("center") + shift).alias("center"))
            .join(bounds, on="chr", how="inner")
            .filter((pl.col("center") >= pl.col("low")) & (pl.col("center") <= pl.col("high")))
            .select("chr", "center")
        )
        moved.write_csv(path, separator="\t", include_header=False)
    return shifted


def drop_contaminated_null(
    null: pl.DataFrame, inputs: Inputs, *, tolerance: int
) -> tuple[pl.DataFrame, int]:
    """Remove shifted pairs that landed back on the real candidate anchors.

    A random shift is only a null if it moves pairs somewhere the real
    regulatory contact is not. When the shift happens to be commensurate
    with structure in the anchor set, it can map the set onto itself and the
    "null" silently becomes the real data again -- which makes every method
    look anticonservative for a reason that has nothing to do with the
    method. A pair is dropped when *both* anchors land within `tolerance` of
    a real anchor, since either alone is common and harmless.

    Returns the filtered frame and the number dropped, so the report can say
    how much of the null survived.
    """

    if tolerance <= 0 or null.is_empty():
        return null, 0
    baits = read_center_anchors(inputs.baits)
    preys = read_center_anchors(inputs.preys)
    keep = np.zeros(null.height, dtype=bool)
    chroms = null["chrom"].to_numpy()
    for chrom in np.unique(chroms):
        selected = chroms == chrom
        real_baits = np.sort(baits.filter(pl.col("chr") == chrom)["center"].to_numpy())
        real_preys = np.sort(preys.filter(pl.col("chr") == chrom)["center"].to_numpy())
        near_bait = _near(null["bait_center"].to_numpy()[selected], real_baits, tolerance)
        near_prey = _near(null["prey_center"].to_numpy()[selected], real_preys, tolerance)
        keep[selected] = ~(near_bait & near_prey)
    return null.filter(pl.Series(keep)), int((~keep).sum())


def _near(values: np.ndarray, reference: np.ndarray, tolerance: int) -> np.ndarray:
    """Whether each value is within `tolerance` of any entry of a sorted reference array."""
    if reference.size == 0:
        return np.zeros(values.shape, dtype=bool)
    index = np.searchsorted(reference, values)
    left = reference[np.clip(index - 1, 0, reference.size - 1)]
    right = reference[np.clip(index, 0, reference.size - 1)]
    return np.minimum(np.abs(values - left), np.abs(values - right)) <= tolerance


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def call_tidy(
    inputs: Inputs, work_dir: Path, tag: str, *, cache_dir: Path, **kwargs: Any
) -> pl.DataFrame:
    """Call local-decay contacts once, on the tidy schema.

    One call is enough for every method: the tidy output carries `observed`,
    `n_trials`, and `p_null`, so each null can be applied afterwards with
    `test_contacts`. That also guarantees all methods are compared on
    identical counts rather than on separate runs that could drift.

    `cache_dir` is shared across every call in a run. The NPZ contact cache
    depends only on the pairs file, and the real and shifted calls all read
    the same one -- giving each its own would rebuild a multi-gigabyte cache
    once per shift, which is the slowest step in the whole benchmark.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    return call_local_decay(
        inputs.baits,
        inputs.preys,
        inputs.pairs,
        work_dir / f"{tag}_calls.tsv",
        schema="tidy",
        cache_dir=cache_dir,
        **kwargs,
    )


def score_all_methods(calls: pl.DataFrame, *, dispersion: float | str = "pearson") -> pl.DataFrame:
    """Add one `p_<method>`/`q_<method>`/`neg_log10_p_<method>` triple per method, plus baselines.

    `dispersion` is only consumed by `negative_binomial`. Pass the value
    estimated on the *null* pairs: estimating it from the pairs being scored
    folds real signal into the variance and makes the test conservative by
    an unknown amount.
    """
    scored = calls
    for method in METHODS:
        result = test_contacts(calls, method=method, fdr="bh", dispersion=dispersion)
        p_values = result.table["p_value"].to_numpy()
        scored = scored.with_columns(
            pl.Series(f"p_{method}", p_values),
            pl.Series(f"q_{method}", result.table["q_value"].to_numpy()),
            pl.Series(f"neg_log10_p_{method}", -np.log10(np.clip(p_values, 1e-300, 1.0))),
        )
    scored = scored.with_columns(
        (-pl.col("distance").cast(pl.Float64).log10()).alias("neg_log10_distance"),
        pl.col("distance").cast(pl.Float64).log10().alias("log10_distance"),
        (pl.col("n_trials").cast(pl.Float64) + 1).log10().alias("log10_n_trials"),
    )
    # Every method now has its own explicit column, so the call's single
    # unlabelled `p_value`/`method` pair would only be ambiguous.
    return scored.drop("p_value", "method", strict=False)


def score_columns(table: pl.DataFrame) -> list[str]:
    """Every column `evaluate_scores` ranks: the methods, then the baselines.

    Contact is only one determinant of a CRISPRi outcome. Whether perturbing
    an enhancer measurably changes a gene also depends on the enhancer's own
    activity, the promoter's activity, and how responsive that promoter is
    to added input -- none of which `touche` measures. So the activity
    columns the label files carry are scored as baselines in their own right,
    and an ABC-style combination is scored too. Without them, a contact
    score's AUPRC is being read against an implicit ceiling nobody has
    measured.
    """

    wanted = [*[f"neg_log10_p_{method}" for method in METHODS], *BASELINES]
    wanted += [f"log10_{name}" for name in LABEL_COVARIATES.values() if f"log10_{name}" in table.columns]
    if "abc_score" in table.columns:
        wanted.append("abc_score")
    return [column for column in wanted if column in table.columns]


def add_global_decay_score(real: pl.DataFrame, null: pl.DataFrame, *, bins: int = 80) -> pl.DataFrame:
    """Score pairs against a single genome-wide `P(s)` instead of a per-bait fit.

    This is the control for `touche`'s most expensive component. The per-bait
    LOWESS fit dominates local-decay's runtime; the cheap alternative is one
    distance curve for the whole genome, scaled by each bait's own coverage.
    Without this baseline there is no way to tell whether the per-bait
    modelling earns its cost.

    The curve is `mean(observed / n_trials)` in distance bins, fitted on the
    *null* pairs so it is not contaminated by the signal being scored, and
    applied as `n_trials * g(distance)`.
    """

    usable = null.filter((pl.col("n_trials") > 0) & (pl.col("distance") > 0))
    if usable.is_empty():
        return real
    distance = usable["distance"].to_numpy().astype(float)
    rate = usable["observed"].to_numpy().astype(float) / usable["n_trials"].to_numpy().astype(float)
    edges = np.unique(np.quantile(distance, np.linspace(0.0, 1.0, bins)))
    if edges.size < 3:
        return real
    index = np.clip(np.searchsorted(edges, distance) - 1, 0, edges.size - 2)
    curve = np.array(
        [rate[index == k].mean() if np.any(index == k) else np.nan for k in range(edges.size - 1)]
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    known = np.isfinite(curve)
    if not known.any():
        return real

    expected = real["n_trials"].to_numpy().astype(float) * np.interp(
        real["distance"].to_numpy().astype(float), centers[known], curve[known]
    )
    return real.with_columns(
        pl.Series("expected_global", expected),
        pl.Series("log2_oe_global", log2_fold_change(real["observed"].to_numpy(), expected)),
    )


def add_abc_score(table: pl.DataFrame) -> pl.DataFrame:
    """Activity-by-Contact style score, if the label files supplied an activity term.

    `A_e * C_ep / sum over that promoter's candidate enhancers`, following the
    ABC model's form: enhancer activity times contact, normalized within the
    promoter. Activity is `mean_ATAC_RPM` and contact is the raw observed
    count.

    This exists to answer one question -- does contact add anything on top of
    activity? -- and not as an ABC implementation. The denominator sums only
    over *labelled* pairs sharing a promoter, which is a restricted candidate
    universe, and the activity term is a single accessibility measure rather
    than the model's enhancer-activity definition.
    """

    if "log10_enhancer_atac" not in table.columns:
        return table
    activity = (10 ** pl.col("log10_enhancer_atac")) - 0.01
    numerator = activity.clip(lower_bound=0.0) * pl.col("observed").cast(pl.Float64)
    return table.with_columns(
        (numerator / numerator.sum().over("prey_id")).alias("abc_score")
    )


def attach_labels(calls: pl.DataFrame, functional: Path, nonfunctional: Path) -> pl.DataFrame:
    """Join the reference functional/nonfunctional labels onto a tidy call table.

    Mirrors `touche.local_decay.assign_pair_types`, which keys on
    `(target_site.chr, target_promoter.center, target_site.center)` -- bait
    is the TRE and prey the promoter in the reference anchor files. Pairs in
    neither list are left null (the reference's "other" class) and are
    excluded from evaluation rather than treated as negatives, since they
    were never tested for function.
    """

    labelled = calls
    for path, label in ((functional, "positive"), (nonfunctional, "negative")):
        source = pl.read_csv(path)
        available = {
            column: name for column, name in LABEL_COVARIATES.items() if column in source.columns
        }
        keys = source.select(
            pl.col(LABEL_KEYS[0]).cast(pl.Utf8).alias("chrom"),
            pl.col(LABEL_KEYS[1]).cast(pl.Int64).alias("prey_center"),
            pl.col(LABEL_KEYS[2]).cast(pl.Int64).alias("bait_center"),
            pl.lit(label).alias(f"_{label}"),
            *[pl.col(column).cast(pl.Float64).alias(f"_{name}_{label}") for column, name in available.items()],
        )
        labelled = labelled.join(keys, on=["chrom", "prey_center", "bait_center"], how="left")

    labelled = labelled.with_columns(
        pl.coalesce(pl.col("_positive"), pl.col("_negative")).alias("label")
    ).drop("_positive", "_negative")
    for name in LABEL_COVARIATES.values():
        columns = [f"_{name}_positive", f"_{name}_negative"]
        if all(column in labelled.columns for column in columns):
            labelled = labelled.with_columns(
                # log10 with an offset: both are RPM/RPKM-like and can be zero.
                (pl.coalesce(pl.col(columns[0]), pl.col(columns[1])) + 0.01).log10().alias(f"log10_{name}")
            ).drop(columns)
    return labelled


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def calibration_report(null: pl.DataFrame) -> pl.DataFrame:
    """KS uniformity and empirical type-I error per method, overall and per stratum.

    Stratified by distance and coverage quintiles because a method can be
    uniform overall while being anticonservative for short-range or
    low-coverage pairs -- which are precisely the pairs an enhancer-promoter
    study cares about.
    """

    binned = null.with_columns(
        pl.col("distance").qcut(5, labels=[f"d{i}" for i in range(5)], allow_duplicates=True).alias("distance_bin"),
        pl.col("n_trials").qcut(5, labels=[f"c{i}" for i in range(5)], allow_duplicates=True).alias("coverage_bin"),
    )
    frames = []
    for method in METHODS:
        table = binned.select(
            pl.col(f"p_{method}").alias("p_value"), "n_trials", "p_null", "distance_bin", "coverage_bin"
        )
        for strata, name in ((None, "overall"), ("distance_bin", "distance"), ("coverage_bin", "coverage")):
            report = assess_calibration(
                table, strata=strata, trials_col="n_trials", probability_col="p_null"
            )
            frames.append(
                report.with_columns(
                    pl.lit(method).alias("method"),
                    pl.lit(name).alias("stratification"),
                    pl.col(strata).cast(pl.Utf8).alias("stratum") if strata else pl.lit("all").alias("stratum"),
                ).drop(strata if strata else [])
            )
    combined = pl.concat(frames, how="diagonal_relaxed")
    lead = ["method", "stratification", "stratum", "n_tested", "fraction_at_one"]
    rates = [c for c in combined.columns if c.startswith("reject_rate")]
    attainable = [c for c in combined.columns if c.startswith("attainable_rate")]
    return combined.select(*lead, *rates, *attainable, "ks_statistic", "ks_p_value")


def decay_model_comparison(null: pl.DataFrame, *, bins: int = 80) -> pl.DataFrame:
    """Per-bait versus global expectations on null pairs, out of sample.

    The global curve is fitted on one shift and evaluated on the others, so
    the comparison does not flatter it. Reports the bias
    (`observed_over_expected`, want 1) and the dispersion (want low -- it
    sets how much variance a calibrated test must inflate, and therefore how
    much power it gives up).
    """

    shifts = sorted(set(null["shift"].to_list()))
    if len(shifts) < 2:
        return pl.DataFrame()
    fit = null.filter((pl.col("shift") == shifts[0]) & (pl.col("n_trials") > 0) & (pl.col("distance") > 0))
    held = null.filter(pl.col("shift") != shifts[0])
    if fit.is_empty() or held.is_empty():
        return pl.DataFrame()

    distance = fit["distance"].to_numpy().astype(float)
    rate = fit["observed"].to_numpy().astype(float) / fit["n_trials"].to_numpy().astype(float)
    edges = np.unique(np.quantile(distance, np.linspace(0.0, 1.0, bins)))
    index = np.clip(np.searchsorted(edges, distance) - 1, 0, edges.size - 2)
    curve = np.array(
        [rate[index == k].mean() if np.any(index == k) else np.nan for k in range(edges.size - 1)]
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    known = np.isfinite(curve)

    held = held.with_columns(
        pl.Series(
            "expected_global",
            held["n_trials"].to_numpy().astype(float)
            * np.interp(held["distance"].to_numpy().astype(float), centers[known], curve[known]),
        ),
        pl.col("distance").qcut(4, labels=[f"d{i}" for i in range(4)], allow_duplicates=True).alias("stratum"),
    )

    rows = []
    for label, frame in [("all", held)] + [
        (s, held.filter(pl.col("stratum") == s)) for s in ("d0", "d1", "d2", "d3")
    ]:
        observed = frame["observed"].to_numpy().astype(float)
        row: dict[str, Any] = {"stratum": label, "n": frame.height}
        for name, column in (("per_bait", "expected"), ("global", "expected_global")):
            expected = frame[column].to_numpy().astype(float)
            row[f"{name}_observed_over_expected"] = float(observed.sum() / expected.sum())
            row[f"{name}_dispersion"] = pearson_dispersion(observed, expected)
        rows.append(row)
    return pl.DataFrame(rows)


def expected_bias_report(null: pl.DataFrame) -> pl.DataFrame:
    """Is `expected` an unbiased prediction of `observed` on null pairs?

    This is the most interpretable single diagnostic in the benchmark, and
    it is method-independent: every test consumes the same `expected`, so if
    the ratio is not ~1 the problem is the distance-decay model, not the
    choice of null. A ratio of `r` means the expectation is `r` times too
    small, and any test built on it will reject roughly `r` times too often.

    `dispersion` is the Pearson dispersion of the observed counts about
    `expected`. The binomial and Poisson nulls both assume it is 1. A value
    above 1 means the counts are more variable than the model allows, which
    inflates the tails and over-rejects *even when the mean is exactly
    right* -- so read it together with `observed_over_expected`. A
    dispersion of `phi` over-rejects by roughly `phi`, and no improvement to
    the decay fit can repair it; that needs a null with a dispersion
    parameter.
    """

    binned = null.with_columns(
        pl.col("distance").qcut(5, labels=[f"d{i}" for i in range(5)], allow_duplicates=True).alias("stratum")
    )
    overall = binned.select(pl.lit("all").alias("stratum")).head(1).with_columns(
        pl.lit(binned.height).alias("n"),
        pl.lit(binned["distance"].median()).alias("median_distance"),
        pl.lit(binned["observed"].mean()).alias("mean_observed"),
        pl.lit(binned["expected"].mean()).alias("mean_expected"),
    )
    binned = binned.with_columns(
        pl.when(pl.col("expected") > 0)
        .then((pl.col("observed") - pl.col("expected")) ** 2 / pl.col("expected"))
        .otherwise(None)
        .alias("_pearson")
    )
    overall = overall.with_columns(pl.lit(binned["_pearson"].mean()).alias("dispersion"))
    per_stratum = (
        binned.group_by("stratum")
        .agg(
            pl.len().alias("n"),
            pl.col("distance").median().alias("median_distance"),
            pl.col("observed").mean().alias("mean_observed"),
            pl.col("expected").mean().alias("mean_expected"),
            pl.col("_pearson").mean().alias("dispersion"),
        )
        .sort("stratum")
        .with_columns(pl.col("stratum").cast(pl.Utf8))
    )
    return pl.concat([overall, per_stratum], how="diagonal_relaxed").with_columns(
        (pl.col("mean_observed") / pl.col("mean_expected")).alias("observed_over_expected")
    )


def calibration_verdict(calibration: pl.DataFrame) -> list[str]:
    """Flag anticonservative methods loudly, so a bad null cannot pass unnoticed."""
    overall = calibration.filter(pl.col("stratification") == "overall")
    lines = []
    for row in overall.iter_rows(named=True):
        rate = row["reject_rate_at_0.05"]
        if rate is None or not np.isfinite(rate):
            continue
        if rate > 0.075:
            lines.append(
                f"- **{row['method']} is anticonservative**: it rejects {rate:.1%} of null pairs at "
                f"a nominal 5%. Its q-values do not control FDR at the stated level."
            )
        elif rate < 0.02:
            lines.append(
                f"- {row['method']} is conservative ({rate:.1%} rejected at a nominal 5%). Check its "
                "AUPRC before reading this as a good property -- a test that never rejects is "
                "trivially calibrated and useless."
            )
        else:
            lines.append(f"- {row['method']} is approximately calibrated ({rate:.1%} at a nominal 5%).")
    return lines


def matched_evaluation(
    real: pl.DataFrame, args: argparse.Namespace
) -> tuple[pl.DataFrame | None, pl.DataFrame]:
    """Match positives to negatives on every available confounder, and report balance either way.

    Distance and coverage always; enhancer accessibility and promoter
    transcription too when the label files supply them, since those are
    strong confounders of functional status and leaving them unbalanced
    credits a contact score for rediscovering them. Each added covariate
    costs matched pairs, so the retained count is reported.
    """
    calipers = {
        "log10_distance": args.match_distance_caliper,
        "log10_n_trials": args.match_coverage_caliper,
    }
    for name in LABEL_COVARIATES.values():
        column = f"log10_{name}"
        if column in real.columns:
            calipers[column] = args.match_activity_caliper
    covariates = list(calipers)

    labelled = real.filter(pl.col("label").is_not_null()).drop_nulls(covariates)
    before = balance_table(labelled, group_col="label", covariates=covariates, groups=("positive", "negative"))
    matched = match_pairs(
        labelled,
        group_col="label",
        covariates=calipers,
        groups=("positive", "negative"),
        seed=args.seed,
    )
    if matched.height < 20:
        return None, before.with_columns(pl.lit("before").alias("stage"))
    after = balance_table(matched, group_col="label", covariates=covariates, groups=("positive", "negative"))
    balance = pl.concat(
        [before.with_columns(pl.lit("before").alias("stage")), after.with_columns(pl.lit("after").alias("stage"))]
    )
    return matched, balance


def write_figures(real: pl.DataFrame, null: pl.DataFrame, out_dir: Path) -> dict[str, Path]:
    """QQ plot of null p-values and precision-recall curves for every score."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {}

    figure, axes = plt.subplots(1, len(METHODS), figsize=(4 * len(METHODS), 4), sharey=True)
    for axis, method in zip(np.atleast_1d(axes), METHODS):
        observed = np.sort(null[f"p_{method}"].drop_nulls().drop_nans().to_numpy())
        if observed.size:
            expected = (np.arange(observed.size) + 0.5) / observed.size
            axis.plot([0, 1], [0, 1], "k--", lw=1)
            axis.plot(expected, observed, lw=2)
        axis.set_title(f"{method}\n(null pairs)")
        axis.set_xlabel("expected uniform quantile")
    np.atleast_1d(axes)[0].set_ylabel("observed p-value")
    figure.suptitle("Null calibration: points on the diagonal mean a correctly specified null")
    figure.tight_layout()
    paths["qq_plot"] = out_dir / "null_pvalue_qq.svg"
    figure.savefig(paths["qq_plot"])
    plt.close(figure)

    labelled = real.filter(pl.col("label").is_not_null())
    labels = (labelled["label"] == "positive").to_numpy()
    figure, axis = plt.subplots(figsize=(6, 5))
    for name in score_columns(labelled):
        scores = labelled[name].cast(pl.Float64).to_numpy()
        usable = np.isfinite(scores)
        precision, recall, _ = precision_recall_curve(labels[usable], scores[usable])
        if precision.size:
            axis.plot(recall, precision, lw=2, label=name)
    if labels.size:
        axis.axhline(labels.mean(), color="k", ls="--", lw=1, label="no-skill (prevalence)")
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_title("Functional enhancer-promoter pairs")
    axis.legend(fontsize=7)
    figure.tight_layout()
    paths["pr_curves"] = out_dir / "precision_recall.svg"
    figure.savefig(paths["pr_curves"])
    plt.close(figure)
    return paths


def write_summary(
    path: Path,
    *,
    manifest: dict[str, Any],
    calibration: pl.DataFrame,
    expected_bias: pl.DataFrame,
    decay_comparison: pl.DataFrame,
    prediction: Any,
    matched_prediction: Any,
) -> Path:
    """Render the headline tables as markdown, with the caveats attached."""
    lines = [
        "# Gasperini benchmark",
        "",
        f"- touche {manifest['touche_version']}, generated {manifest['created_at']}",
        f"- decay model: **{manifest['parameters']['decay_model']}**",
        f"- demo mode: **{manifest['demo']}**"
        + ("  (synthetic data with a planted signal -- not evidence about real data)" if manifest["demo"] else ""),
        f"- labelled pairs: {manifest['counts']['positive']} functional, "
        f"{manifest['counts']['negative']} non-functional, "
        f"{manifest['counts']['unlabelled']} unlabelled (excluded)",
        f"- null pairs: {manifest['counts']['null_pairs']} from shifts {manifest['parameters']['shifts']}"
        f" ({manifest['counts']['null_pairs_dropped_as_contaminated']} dropped for landing back "
        "on real anchors)",
        "",
        "## Verdict",
        "",
        *calibration_verdict(calibration),
        "",
        "## Is `expected` unbiased?",
        "",
        "Every method consumes the same `expected`, so this table is about the distance-decay",
        "model rather than the choice of null. On null pairs `observed_over_expected` should be",
        "~1; a ratio of `r` means the expectation is `r` times too small and any test built on",
        "it rejects roughly `r` times too often.",
        "",
        "`dispersion` is the Pearson dispersion, which the binomial and Poisson nulls both",
        "assume is 1. Above 1 the counts are more variable than the model allows, so the tails",
        "are too heavy and the test over-rejects **even with the mean exactly right**. No",
        "improvement to the decay fit can repair that -- it needs a null carrying a dispersion",
        "parameter.",
        "",
        _markdown_table(expected_bias),
        "",
        "## Does the per-bait fit earn its cost?",
        "",
        "`touche`'s per-bait LOWESS fit dominates local-decay's runtime. The cheap alternative",
        "is a single genome-wide P(s) curve scaled by each bait's coverage. This compares the",
        "two on held-out null pairs -- the global curve is fitted on one shift and evaluated on",
        "the others -- so the question is answerable rather than assumed. Lower dispersion means",
        "a calibrated test has to inflate the variance less, and so gives up less power.",
        "",
        _markdown_table(decay_comparison),
        "",
        "`log2_oe_global` in the prediction tables below is the same global curve used as a",
        "ranking score, against `log2_oe` from the per-bait fit.",
        "",
        "## Null calibration",
        "",
        "**Compare each rejection rate against `attainable_rate_at_*`, not against its",
        "nominal level.** These tests are discrete, so the attainable size is well below",
        "alpha -- around half of it on real data. A rate at the attainable rate is correctly",
        "calibrated; one far below it is losing power to a biased expectation; one far above",
        "it is anticonservative.",
        "",
        "**Read the rejection rates, not the KS statistic.** An upper-tail test on small",
        "counts is discrete: every pair with zero observed contacts gets p = 1 exactly, which",
        "on real data is about half of them. That atom drives the KS statistic to ~0.5 with",
        "p = 0 however well calibrated the test is -- `fraction_at_one` shows how much of it",
        "is discreteness. A rejection rate materially above its nominal level is the real",
        "sign of an anticonservative test.",
        "",
        _markdown_table(calibration.filter(pl.col("stratification") == "overall")),
        "",
        "Worst distance/coverage stratum per method (largest rejection rate at 0.05):",
        "",
        _markdown_table(
            calibration.filter(pl.col("stratification") != "overall")
            .sort("reject_rate_at_0.05", descending=True)
            .group_by("method", maintain_order=True)
            .head(1)
        ),
        "",
        "## Functional prediction",
        "",
        "`touche` measures contact and nothing else, while a CRISPRi outcome also depends on",
        "enhancer activity, promoter activity, and promoter responsiveness. The activity",
        "columns are scored as baselines so that gap is visible rather than assumed: compare",
        "the contact scores against `log10_enhancer_atac` and `log10_promoter_proseq`, and read",
        "`abc_score` as the question of whether contact adds anything on top of activity.",
        "",
        "Check `balance.tsv` before interpreting any of it. If the label sets are already",
        "balanced on a covariate, that covariate cannot predict here and its baseline will sit",
        "at the prevalence -- which says nothing about whether it matters biologically, only",
        "that this comparison controlled it by construction. Conversely the *unmatched* table",
        "rewards whichever score best proxies the covariate the label sets differ on most.",
        "",
        "Compare each AUPRC against `baseline_auprc` (the prevalence), not against 0.5.",
        "`held_out_auprc_mean` averages per-chromosome AUPRC with a bootstrap interval over",
        "chromosomes -- the honest uncertainty, since pairs within a chromosome are not independent.",
        "",
        _markdown_table(prediction.table),
        "",
        *_warning_lines(prediction),
    ]
    if matched_prediction is not None:
        lines += [
            "### Distance- and coverage-matched",
            "",
            "Positives and negatives matched on log10 distance and log10 coverage, so a method",
            "is not credited for rediscovering that functional pairs are closer together.",
            "",
            _markdown_table(matched_prediction.table),
            "",
            *_warning_lines(matched_prediction),
        ]
    lines += [
        "## Caveats",
        "",
        "- The null is distance-preserving random-shift pairs. A shifted pair can land on a real",
        "  contact by chance, which inflates the extreme tail slightly; the calibration check is",
        "  therefore mildly pessimistic, not optimistic.",
        "- `p_null` is fitted from the same bait window each pair is tested in. The leakage is",
        "  small but not zero, and this benchmark does not remove it.",
        "- AUPRC differences within the bootstrap interval are not evidence of a better method.",
        "- Everything here is technical inference from one library. It says nothing about",
        "  biological reproducibility across replicates.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _warning_lines(result: Any) -> list[str]:
    """Render a result's warnings into the summary.

    Without this they live only in the `.meta.json` sidecar, which is where
    the least-read and most-important part of a statistical result should
    not be.
    """
    if not result.info.warnings:
        return []
    return ["**Warnings**", "", *[f"- {warning}" for warning in result.info.warnings], ""]


def _markdown_table(frame: pl.DataFrame) -> str:
    """Render a small polars frame as a markdown table, rounding floats for readability."""
    if frame.is_empty():
        return "_(no rows)_"
    rounded = frame.with_columns(
        [pl.col(name).round(4) for name, dtype in zip(frame.columns, frame.dtypes) if dtype.is_float()]
    )
    header = "| " + " | ".join(rounded.columns) + " |"
    divider = "| " + " | ".join("---" for _ in rounded.columns) + " |"
    body = [
        "| " + " | ".join("" if value is None else str(value) for value in row) + " |"
        for row in rounded.iter_rows()
    ]
    return "\n".join([header, divider, *body])


def write_table(frame: pl.DataFrame, path: Path) -> Path:
    """Write a benchmark table as TSV, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path, separator="\t")
    return path


def log(message: str, *, level: str = "info") -> None:
    """Progress logging to stderr, so stdout stays a clean JSON manifest."""
    print(f"[{level}] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
