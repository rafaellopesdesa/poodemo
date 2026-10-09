"""Append notebook 12's diagnostic follow-ups without touching executed cells.

Run from the repository root: python scripts/build_toy_diagnostics_notebook.py
An existing follow-up is left untouched. Use --clean only for a fresh notebook
without saved outputs; all other notebooks are always left untouched.
"""
import argparse
import json

from build_notebooks import BOOTSTRAP, COMMON_IMPORTS, REPO, code, md, write


SETUP = r"""
### Reuse the completed notebook 11 run

Keep **`RUN_NAME = "paper-distinct-s-v4"`**, the same `MODE`, and the same
`CONFIG_OVERRIDES` used in notebook 11. No regeneration, retraining, or rerunning
01–11 is required. This notebook reads notebook 11's completed toy records and
frozen model, then writes its own diagnostics to a separate directory. It never
extends or overwrites the notebook 11 ensemble.

Start a fresh Colab runtime so the setup loads the updated source. If reusing a
runtime, refresh its temporary `/content/poodemo` checkout and restart first;
your saved Google Drive run is separate. The setup reuses the repository's
normal private-repository authentication and Drive mount. Select a GPU runtime
for the later stages: neural preselection and ratio evaluation benefit from it,
while the small likelihood fits run on the CPU.

The **first stage uses saved tables only**, without loading trained networks or
creating new events. Set `RUN_INTEGRATION = False` and `RUN_REFITS = False` for
that inexpensive first look. The default enables all stages; the settings below
control their computational budgets independently. `MODE="smoke"` refers to an
existing smoke run with its own completed notebook 11 outputs.
"""

STYLE = r'''
import mplhep as hep
hep.style.use("ATLAS")
# Typography and ticks only; no ATLAS label.
plt.rcParams.update({"axes.grid": False, "figure.dpi": 110})
'''

SETTINGS = r'''
from poodemo.toy_diagnostics import (
    load_toy_study, analyze_paired_toys, plot_paired_toys,
    bootstrap_coverage, plot_bootstrap_coverage,
)

SOURCE_TAG = "toy_study_direct"  # Completed notebook 11 ensemble.
OUTPUT_TAG = "toy_diagnostics"  # Separate from notebook 11 outputs.
BRANCH_SPLIT = 1.0  # Operational low/high-mu branch definition, not a fit constraint.
COVERAGE_LEVELS = (0.68, 0.95)
N_BOOTSTRAP = 100 if MODE == "smoke" else 1000
BOOTSTRAP_SEED = 120424

RUN_INTEGRATION = True
BASE_GENERATED_PER_SOURCE = None  # None uses the original quadrature manifest.
BANK_MULTIPLES = (1, 2, 5)
BANK_REPLICAS = 2
INTEGRATION_SEED = 120524
GENERATION_CHUNK_SIZE = 100_000

RUN_REFITS = True
N_RANDOM_TOYS = 3 if MODE == "smoke" else 200  # Per tested hypothesis.
MAX_FLAGGED_TOYS = None  # None keeps every flagged toy, in addition to the random sample.
REFIT_SELECTION_SEED = 120624
WIDE_MU_BOUNDS = (0.0, 3.0)
DENSE_FIT_GRID_SIZE = 129  # Grid in kappa=sqrt(mu), plus local refinement.
BIN_COUNTS = (12, 36)  # 36 subdivides each of the original 12 bins into three.
N_LIKELIHOOD_SCANS = 3
PROGRESS_EVERY = 10

DIAGNOSTIC_DIR = ROOT / "results" / "12_toy_diagnostics" / OUTPUT_TAG
FIGURE_DIR = ROOT / "plots" / "12_toy_diagnostics" / OUTPUT_TAG
DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def show_figures(figures):
    for name, figure in figures.items():
        print(name)
        display(figure)
        plt.close(figure)


def show_tables(result, *, max_rows=20):
    for name, frame in result.items():
        if isinstance(frame, pd.DataFrame):
            print(f"{name}: {len(frame):,} rows")
            display(frame.head(max_rows))


print("Source ensemble:", SOURCE_TAG)
print("Diagnostic tables:", DIAGNOSTIC_DIR)
print("Integration stage:", RUN_INTEGRATION, "| Refit stage:", RUN_REFITS)
'''

LOAD = r'''
source = load_toy_study(run, source_tag=SOURCE_TAG)
print("Notebook 11 provenance and settings:")
display(pd.Series(source["metadata"]))
print("Completeness by hypothesis and case (including missing and failed records):")
display(source["completeness"])
source["completeness"].to_csv(DIAGNOSTIC_DIR / "source_completeness.csv", index=False)
print("Failed fits remain in the diagnostics; they are not silently treated as valid.")
with (DIAGNOSTIC_DIR / "saved_table_analysis.json").open("w") as handle:
    json.dump({"source_tag": SOURCE_TAG,
               "source_results_sha256": source["results_sha256"],
               "source_config_sha256": source["config_sha256"],
               "branch_split": BRANCH_SPLIT,
               "n_bootstrap": N_BOOTSTRAP,
               "bootstrap_seed": BOOTSTRAP_SEED,
               "coverage_levels": COVERAGE_LEVELS}, handle, indent=2)
'''

PAIRED = r'''
paired = analyze_paired_toys(source, branch_split=BRANCH_SPLIT)
for name, frame in paired.items():
    if isinstance(frame, pd.DataFrame):
        frame.to_csv(DIAGNOSTIC_DIR / f"paired_{name}.csv", index=False)
show_tables(paired)
show_figures(plot_paired_toys(paired, output=FIGURE_DIR / "paired"))
'''

BOOTSTRAP_RUN = r'''
coverage = bootstrap_coverage(
    source, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
    levels=COVERAGE_LEVELS,
)
coverage.to_csv(DIAGNOSTIC_DIR / "coverage_bootstrap.csv", index=False)
display(coverage)
show_figures(plot_bootstrap_coverage(coverage, output=FIGURE_DIR / "coverage"))
'''

CONTEXT = r'''
ctx = None
integration = None
refits = None
if RUN_INTEGRATION or RUN_REFITS:
    from poodemo.toy_diagnostic_context import prepare_toy_diagnostic_context
    ctx = prepare_toy_diagnostic_context(
        run, source_tag=SOURCE_TAG, output_tag=OUTPUT_TAG,
    )
    original_n = int(ctx["original_n_per_source"])
    generated_per_source = (
        original_n if BASE_GENERATED_PER_SOURCE is None
        else int(BASE_GENERATED_PER_SOURCE)
    )
    print("Original generated proposals per source:", original_n)
    print("Diagnostic base proposals per source:", generated_per_source)
else:
    print("Saved-table analysis complete. Enable a later stage to load the frozen models.")
'''

INTEGRATION_RUN = r'''
if RUN_INTEGRATION:
    from poodemo.toy_integration_diagnostics import (
        run_integration_diagnostics, plot_integration_diagnostics,
    )
    integration = run_integration_diagnostics(
        ctx, generated_per_source=generated_per_source,
        multiples=BANK_MULTIPLES, replicas=BANK_REPLICAS,
        seed=INTEGRATION_SEED, n_bins=BIN_COUNTS,
        chunk_size=GENERATION_CHUNK_SIZE,
    )
    for name in ("bin_closure", "rate_closure", "convergence", "score_closure"):
        print(name)
        display(integration[name])
    show_figures(plot_integration_diagnostics(
        integration, output=FIGURE_DIR / "integration",
    ))
else:
    print("Independent integration study skipped by RUN_INTEGRATION=False.")
'''

SELECTION = r'''
if RUN_REFITS:
    from poodemo.toy_refit_diagnostics import (
        select_refit_toys, run_refit_diagnostics, plot_refit_diagnostics,
    )
    selection = select_refit_toys(
        source, n_random=N_RANDOM_TOYS, max_flagged=MAX_FLAGGED_TOYS,
        seed=REFIT_SELECTION_SEED, branch_split=BRANCH_SPLIT,
    )
    selection.to_csv(DIAGNOSTIC_DIR / "refit_selection.csv", index=False)
    print("Selected experiments (each is reconstructed with its original RNG stream):")
    display(selection)
    print("Use random-subset summaries for population comparisons; flagged toys are enriched.")
else:
    print("Event reconstruction and likelihood refits skipped by RUN_REFITS=False.")
'''

REFITS_RUN = r'''
if RUN_REFITS:
    refits = run_refit_diagnostics(
        ctx, selection, integration=integration,
        mu_bounds=WIDE_MU_BOUNDS, grid_size=DENSE_FIT_GRID_SIZE,
        bin_counts=BIN_COUNTS, n_scans=N_LIKELIHOOD_SCANS,
        progress_every=PROGRESS_EVERY,
    )
    original_checks = refits["comparisons"].loc[
        refits["comparisons"].variant.eq("original")
    ]
    print("Reproduction of notebook 11 under its original settings:")
    display(original_checks)
    mismatches = original_checks.loc[
        original_checks.saved_valid.astype(bool)
        & ~original_checks.reproduction_pass.astype(bool)
    ]
    if len(mismatches):
        display(mismatches)
        raise RuntimeError(
            "Original-settings fits did not reproduce saved valid notebook 11 results. "
            "Inspect these records before interpreting numerical or statistical differences."
        )
    print("Original valid-source fits reproduced. Refit summary:")
    display(refits["summary"])
    print("Paired comparisons (first 30 rows):")
    display(refits["comparisons"].head(30))
    show_figures(plot_refit_diagnostics(refits, output=FIGURE_DIR / "refits"))
'''


def diagnostic_cells():
    return [
        md(r"""
        # 12. Why do the toy test-statistic distributions differ?

        Diagnose the notebook 11 results at **\(\mu=0\) and \(1.4\)** without
        changing the physics model or trained networks. We separate competing
        likelihood minima, numerical minimization, integration error, and
        finite-bin compression. All fits remain **stat-only** and use the same
        two-sided \(q_\mu=-2\log[L(\mu)/L(\hat\mu)]\).

        | Stage | Question | Additional cost |
        |---|---|---|
        | Saved-record analysis | Does a different minimum explain the paired \(q_\mu\) differences? | Tables and plots only |
        | Coverage bootstrap | How uncertain are the empirical calibration thresholds and their transfer? | Resampling existing toy records |
        | Independent integration | Do simulator bin means and expected scores agree with the frozen model? | New integration proposals and frozen preselection |
        | Paired refits | Does the result change with finer minimization, wider bounds, more bins, or new integrals? | Reconstructed selected experiments and fits |

        A different test-statistic distribution from the analytical benchmark
        is not by itself a calibration failure. For calibration, compare each
        fitted model's own-toy distribution with its simulator-toy distribution.
        """),
        md(SETUP),
        code(BOOTSTRAP),
        code(COMMON_IMPORTS + "\n" + STYLE),
        md(r"""
        ### Settings and computational budgets

        The original exposure, tested hypotheses, observable power, random seed,
        and 12-bin definition are read from the saved notebook 11 metadata.
        The \(\eta=\mu_{\rm test}\) observable and its normalization constants
        stay frozen during every fit and integration comparison.

        Defaults use 1,000 coverage bootstrap replicates; two independent banks
        at each of 1, 2, and 5 times the original generated sample size; and a
        random sample of 200 experiments per hypothesis, plus all flagged toys.
        A flag selects an experiment for which at least one fit reaches the
        original upper bound. The flagged collection can substantially exceed
        200 experiments. Set `MAX_FLAGGED_TOYS` to a finite number if necessary;
        the selected toy IDs are saved for reproducibility.

        Matching heavy-stage checkpoints are reused. If changing integration
        settings, refit selection, fit bounds, or whether independent templates
        are used, choose a new `OUTPUT_TAG`; incompatible cached calculations
        are rejected rather than mixed.

        `BRANCH_SPLIT=1` is a descriptive threshold separating the low- and
        high-\(\mu\) solutions. It is not a new physical bound or an optimizer
        setting. At truth 1.4 the high branch is the branch containing the truth.
        """),
        code(SETTINGS),
        md(r"""
        ### 1. Pair the existing simulator experiments

        The analytical, learned, and binned fits share each physical simulator
        experiment. Pair by **tested hypothesis and toy ID**, retaining failure
        flags, rather than comparing independent histogram fluctuations.

        Inspect \(\hat\mu_{\rm binned}\) versus \(\hat\mu_{\rm analytic}\), then
        \(\Delta q=q_{\rm binned}-q_{\rm analytic}\) within each pair of branches.
        A concentration of large differences in branch-disagreeing experiments
        would support the competing-minimum explanation. Comparisons within
        the same branch check whether that explanation is sufficient.
        """),
        code(LOAD),
        code(PAIRED),
        md(r"""
        ### 2. Include calibration-threshold uncertainty

        Recompute the model-toy critical values and the simulator acceptance in
        each bootstrap replicate. Resample **whole toy IDs**, sharing a draw
        across methods whose experiments are paired, while keeping notebook 11's
        disjoint calibration and evaluation partitions. This incorporates the
        finite ensemble uncertainty of the critical value as well as the
        evaluation fraction.

        Interpret the results as uncertainty conditional on the saved model and
        integration bank. They do not include finite-MC uncertainty in its
        templates or in the trained ratios. At \(\mu=0\), the point mass at
        \(q=0\) makes empirical quantiles discontinuous and nonrandomized
        acceptance can be conservative; bootstrap intervals are a diagnostic,
        not an exact coverage guarantee. The extreme log-scale tail remains
        limited by the number of available toys.
        """),
        code(BOOTSTRAP_RUN),
        md(r"""
        ### Load the frozen model only for the remaining stages

        The next stages verify the source run before reconstructing any events.
        Network parameters, learned-ratio normalization, preselection, physical
        rates, and observable settings come from the completed source run.
        A diagnostic output tag identifies these comparisons separately.
        """),
        code(CONTEXT),
        md(r"""
        ### 3. Independent bin-yield and integration convergence checks

        Generate fresh, independent proposal banks, apply the same frozen
        preselection, and integrate the analytical physical intensities in the
        **unchanged** observable bins. Compare the resulting bin means and total
        rates with notebook 11's original predictions. This directly tests the
        assertion that simulator histograms and model Poisson draws should
        share the same bin means.

        The base sample size counts proposals **before preselection, per
        proposal source**. It is read from the original quadrature manifest;
        accepted counts differ. Bank sizes and independent seeds are recorded.
        Increasing this size changes numerical precision, not physical exposure.
        Compare the independent replicas and their MC uncertainties; a single
        random realization need not improve monotonically with size. Plot error
        bars represent the fresh validation bank uncertainty conditional on the
        original frozen model. Fixed-partition uncertainty estimates for the
        original bank are also tabulated, but its observable normalizers depend
        on that bank: a naive old-plus-new quadrature error is not an exact pull.

        At the boundary, use \(\kappa=\sqrt\mu\). For fixed bin means
        \(\nu_i^{\rm model}(\kappa)\), the expected score is
        \[
        E_{\rm true}[U_\kappa]
        =\sum_i\left(\frac{\nu_i^{\rm true}}{\nu_i^{\rm model}}-1\right)
        \frac{\partial\nu_i^{\rm model}}{\partial\kappa},\qquad
        I_{\kappa\kappa}
        =\sum_i\frac{(\partial_\kappa\nu_i^{\rm model})^2}
        {\nu_i^{\rm model}}.
        \]
        Inspect \(E[U_\kappa]/\sqrt{I_{\kappa\kappa}}\), with its integration
        uncertainty. A negative expected score can leave the constrained Asimov
        MLE at zero while changing how often fluctuated experiments land on the
        boundary. This local score does not predict which distant minimum wins;
        the paired branch analysis addresses that separate question. The
        validation expectation here is independently estimated,
        not an exact infinite-MC truth.

        The 36-bin templates below subdivide the original 12 bins into three;
        they preserve their boundaries and keep 0.5 inside a bin. They use the
        **same power \(a\)**. No spline interpolation is introduced.
        """),
        code(INTEGRATION_RUN),
        md(r"""
        ### 4. Reconstruct a random sample and the flagged experiments

        The original seed, tested hypothesis, toy ID, and simulator stream
        reconstruct the same experiments used in notebook 11. First check the
        original fit settings; then compare a denser minimization on the same
        range and a fit extended to \(0\leq\mu\leq3\). The notebook checks
        reproduced counts, \(q\), and \(\hat\mu\) before showing the diagnostic
        summary; a mismatch for an originally valid fit stops interpretation.

        The random subset supports comparisons representative of the saved
        ensemble. The additional upper-bound toys are intentionally enriched
        for difficult cases. Representative likelihood scans prioritize branch
        disagreements among the selected experiments. **Do not interpret the
        pooled selected-sample fraction as the population probability.**
        """),
        code(SELECTION),
        md(r"""
        ### 5. Separate optimizer, fit-range, integration, and binning effects

        Compare one controlled change at a time. Dense fits at the original
        upper bound check numerical minimization; extending only the bound
        checks truncation. With the same range and minimizer, compare the
        original 12 bins against their nested 36-bin refinement, using common
        integration information. When the independent integration stage was
        run, the refits also compare original and independently estimated
        template/rate predictions while keeping the observable fixed.

        Likelihood scans for representative selected experiments expose both
        local minima and the tested value. Inspect failed fits and repeated
        upper-bound hits before drawing conclusions about tails. Reaching 3
        would mean the wider range still needs investigation; 3 is a diagnostic
        choice, not a proof that the global minimum has been found.

        More bins need not move every individual \(q_\mu\) monotonically toward
        the unbinned value: both the numerator and fitted denominator change.
        Use paired differences, branch changes, and the random-subset summaries
        together. We keep \(a\) fixed so a separate future transformation study
        does not become confounded with this bin-count comparison.
        """),
        code(REFITS_RUN),
        md(r"""
        ### Reading the result

        - **Different winning minima dominate the paired discrepancy:** the
          compression changes discrimination between interference solutions.
        - **Dense fits change results at the same bound:** improve minimization
          before interpreting the ensemble statistically.
        - **The wider range changes boundary toys:** the original scan domain
          truncated relevant alternatives.
        - **Independent bin means or expected scores disagree with the original
          templates:** investigate integration precision before attributing all
          nonclosure to neural ratios or compression.
        - **The 36-bin refinement reduces branch disagreements after those
          checks:** this supports finite-bin information loss as a contributor.

        None of these diagnostics changes the learned-model toy interpretation
        from notebook 11: those toys and their likelihood share a finite bank.
        An independent learned-model sampling test remains a separate question.
        This notebook adds no nuisance parameters and does not propagate finite
        template statistics as a profiled uncertainty.
        """),
        code(r'''
        print("Notebook 11 source records were preserved.")
        print("Diagnostic tables and checkpoints:", DIAGNOSTIC_DIR)
        print("Figures:", FIGURE_DIR)
        '''),
    ]


CONVERGENCE_MARKER = "integration-convergence-v1"

CONVERGENCE_SETUP = r'''
# This follow-up is independent of the earlier analysis-stage variables.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from IPython.display import display

from poodemo.toy_diagnostic_context import prepare_toy_diagnostic_context
from poodemo.toy_convergence_banks import prepare_convergence_banks
from poodemo.toy_convergence_refits import (
    load_convergence_selection, run_convergence_refits, plot_convergence_refits,
)

hep.style.use("ATLAS")  # Typography and ticks; no experiment label.
plt.rcParams.update({"axes.grid": False, "figure.dpi": 110})

SOURCE_TAG = "toy_study_direct"
OUTPUT_TAG = "toy_diagnostics"  # Must match the completed stages above.
CONVERGENCE_TAG = "bank_convergence"  # New checkpoints live below this tag.
EXTRA_BANK_MULTIPLES = (10,)  # Extend the saved 1x/2x/5x banks, for both replicas.
CONVERGENCE_BIN_COUNTS = (12, 36)
CONVERGENCE_MU_BOUNDS = (0.0, 3.0)
CONVERGENCE_GRID_SIZE = 129
CONVERGENCE_PROGRESS_EVERY = 10

convergence_context = prepare_toy_diagnostic_context(
    run, source_tag=SOURCE_TAG, output_tag=OUTPUT_TAG,
)
convergence_selection = load_convergence_selection(convergence_context)
convergence_figure_dir = (
    Path(ROOT) / "plots" / "12_toy_diagnostics" / OUTPUT_TAG
    / "convergence" / CONVERGENCE_TAG
)
convergence_figure_dir.mkdir(parents=True, exist_ok=True)


def show_convergence_figures(figures):
    for name, figure in figures.items():
        print(name)
        display(figure)
        plt.close(figure)


print("Verified saved toy selection; no new random subset was drawn:")
display(convergence_selection.groupby("mu_test").agg(
    selected=("toy_id", "size"),
    random=("is_random", "sum"),
    flagged=("is_flagged", "sum"),
))
print("New bank multiples:", EXTRA_BANK_MULTIPLES)
print("Population summaries use only the saved random subset.")
print("New diagnostic tag:", CONVERGENCE_TAG)
'''

CONVERGENCE_BANKS = r'''
convergence_banks = prepare_convergence_banks(
    convergence_context,
    extra_multiples=EXTRA_BANK_MULTIPLES,
    bin_counts=CONVERGENCE_BIN_COUNTS,
    convergence_tag=CONVERGENCE_TAG,
)
print("Integration provenance:")
display(pd.Series(convergence_banks["metadata"]))
print("Independent bank checkpoints available for refitting:")
display(pd.DataFrame([
    {"replica": replica, "generated_per_source": size}
    for replica, size in sorted(convergence_banks["banks"])
]))
'''

CONVERGENCE_FITS = r'''
convergence_study = run_convergence_refits(
    convergence_context, convergence_selection, convergence_banks,
    mu_bounds=CONVERGENCE_MU_BOUNDS,
    grid_size=CONVERGENCE_GRID_SIZE,
    bin_counts=CONVERGENCE_BIN_COUNTS,
    progress_every=CONVERGENCE_PROGRESS_EVERY,
    convergence_tag=CONVERGENCE_TAG,
)

count_mismatches = convergence_study["refits"].loc[
    ~convergence_study["refits"].count_reproduced.astype(bool)
]
if len(count_mismatches):
    display(count_mismatches)
    raise RuntimeError(
        "Reconstructed event counts differ from the saved experiments. "
        "Inspect these records before interpreting convergence results."
    )

convergence_summary = convergence_study["summary"]
bank_columns = ["mu_test", "bank_kind", "replica", "generated_per_source", "model"]
print("Random-subset fit checks, including original-bank controls:")
display(convergence_summary[bank_columns + [
    "n_random", "n_valid", "n_failed", "upper_boundary_fraction",
]])
print("Mean q and its toy-sampling standard error, conditional on each bank:")
display(convergence_summary[bank_columns + ["mean_q", "mean_q_se"]])
print("Binned minus matched analytical fit on the same toys and integration bank:")
display(convergence_summary.loc[convergence_summary.model.ne("analytic"), bank_columns + [
    "n_paired_valid", "n_paired_failed", "mean_delta_q", "mean_delta_q_se",
    "branch_disagreement_fraction", "branch_disagreement_se",
]])
print("Paired changes in the gap with bank size and between independent replicas:")
display(convergence_study["convergence"].loc[
    convergence_study["convergence"].metric.eq("gap")
])
show_convergence_figures(plot_convergence_refits(
    convergence_study, output=convergence_figure_dir,
))
print("New tables and checkpoints:", convergence_study["metadata"]["output_dir"])
print("New figures:", convergence_figure_dir)
'''


def convergence_cells():
    cells = [
        md(r"""
        ### 6. Follow-up: repeat the paired fits across independent, larger banks

        **Start here for the new tests.** In a fresh runtime, execute the first
        two code cells of this notebook: runtime setup, then common imports and
        `create_run`. Keep the same v4 run name, mode, and configuration. Then
        jump directly to this section. The completed stages above do not need
        to run again, and their saved results remain available for comparison.

        We now refit the **exact same selected simulator experiments** with
        both independent integration replicas, at every saved bank size and
        at a new, larger size. The default extends each 1x/2x/5x bank to 10x.
        The analytical benchmark receives rate integrals from the same bank
        used for the corresponding binned predictions. Its event densities
        remain analytical; its selected-rate integrals are numerical.

        This addresses two questions separately: whether the result changes
        when replacing replica 0 by replica 1, and whether those changes shrink
        as each bank grows. We compare mean \(q_\mu\), the paired
        binned-minus-analytical difference, the zero-\(q\) fraction, and branch
        disagreement. The 12-bin model and its nested 36-bin refinement use the
        same events, fit bounds, minimizer, and integration proposals.

        The observable, original normalization constants, networks, physical
        exposure, and preselection stay frozen. These templates use the
        **analytical ratio observable**, as in the completed refit stage; the
        bank extension evaluates the frozen selector and analytical densities,
        without retraining networks or evaluating new learned density ratios.

        **Keep `OUTPUT_TAG` unchanged:** it locates the completed integration
        checkpoints and saved toy selection. New outputs live under
        `convergence/CONVERGENCE_TAG`. Matching checkpoints resume; if changing
        the new settings, use a new `CONVERGENCE_TAG` (for example, to request
        `EXTRA_BANK_MULTIPLES=(10, 20)`). The original stages are preserved.
        """),
        code(CONVERGENCE_SETUP),
        md(r"""
        #### Recover both replicas and extend their integration prefixes

        The saved 1x/2x/5x component sums are reused. Only the additional proposals
        needed for the larger banks are generated and preselected. Sample sizes
        count generated events **per proposal source before preselection**;
        multiplying the bank size does not multiply the experiment's luminosity.

        Within a replica, larger banks contain the smaller prefixes and are
        therefore correlated. Replica 0 and replica 1 have independent proposal
        streams. The helper verifies the saved configuration and retains this
        distinction in its provenance. This stage can take time; intermediate
        checkpoints allow it to resume after interruption.
        """),
        code(CONVERGENCE_BANKS),
        md(r"""
        #### Refit the same experiments and measure paired changes

        The saved `refit_selection.csv` is checked against the original refit
        configuration. No toy IDs are redrawn. Both the random controls and
        flagged experiments are retained, with their original membership flags.
        Only the **random subset** enters population summaries; the additional
        flagged toys remain useful individual diagnostics.

        For each bank, compare the binned result with the analytical fit using
        **that same bank's selected rates**. The convergence tables then compare
        each experiment with itself across sizes and replicas. Their paired
        standard errors measure variation across the sampled experiments,
        conditional on the banks. They are not errors on the MC integration
        itself. Failed fits and upper-bound hits must be inspected alongside
        the means; they are not silently counted as successful closure.
        """),
        code(CONVERGENCE_FITS),
        md(r"""
        #### How to interpret this follow-up

        - If both replicas approach similar means and paired gaps as their
          sizes grow, the original discrepancy was partly integration error.
        - If the analytical mean moves as well, the earlier analytical curve
          was not yet an integration-independent reference.
        - If the binned-minus-analytical gap remains stable across the larger
          independent banks, inspect the paired branch changes and 12-to-36-bin
          comparison before attributing it to finite-MC noise.
        - If replicas still disagree, increase the integration budget before
          drawing a conclusion about the remaining compression effect.

        Two independent replicas provide a sensitivity check, not a precise
        estimate of MC uncertainty and not a profiled finite-MC statistical
        model. Shared toy events make comparisons more precise; they do not
        remove bank uncertainty. Prefix growth need not improve every result
        monotonically. The fixed-reference ratio preserves a particular
        hypothesis comparison, not necessarily the full likelihood family
        explored by the freely fitted denominator, so agreement of all test
        statistic distributions is not guaranteed even after convergence.
        """),
    ]
    cells[0].metadata["poodemo_append_section"] = CONVERGENCE_MARKER
    return cells


DECOMPOSITION_MARKER = "likelihood-decomposition-v1"

DECOMPOSITION_SETUP = r'''
# Standalone follow-up: only the first two notebook setup/import cells are needed.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from IPython.display import display

from poodemo.toy_diagnostic_context import prepare_toy_diagnostic_context
from poodemo.toy_likelihood_decomposition import (
    run_likelihood_decomposition, plot_likelihood_decomposition,
)

hep.style.use("ATLAS")  # Typography and ticks; no experiment label.
plt.rcParams.update({"axes.grid": False, "figure.dpi": 110})

SOURCE_TAG = "toy_study_direct"
OUTPUT_TAG = "toy_diagnostics"
CONVERGENCE_TAG = "bank_convergence"  # The completed section 6 run.
DECOMPOSITION_TAG = "fixed_pair"  # New results, separate from section 6 fits.
LARGEST_PREFIXES = 2  # Normally 5x and 10x, using both saved replicas.
DECOMPOSITION_PROGRESS_EVERY = 10

decomposition_context = prepare_toy_diagnostic_context(
    run, source_tag=SOURCE_TAG, output_tag=OUTPUT_TAG,
)
decomposition_figure_dir = (
    Path(ROOT) / "plots" / "12_toy_diagnostics" / OUTPUT_TAG
    / "convergence" / CONVERGENCE_TAG / "decomposition" / DECOMPOSITION_TAG
)
print("Reading completed section 6:", CONVERGENCE_TAG)
print("Largest independent prefixes per replica:", LARGEST_PREFIXES)
print("The original bank is also retained as a control.")
print("Reference hypothesis: mu=1; observable frozen at eta=mu_test.")
'''

DECOMPOSITION_RUN = r'''
decomposition_study = run_likelihood_decomposition(
    decomposition_context,
    convergence_tag=CONVERGENCE_TAG,
    decomposition_tag=DECOMPOSITION_TAG,
    largest_prefixes=LARGEST_PREFIXES,
    progress_every=DECOMPOSITION_PROGRESS_EVERY,
)
print("Provenance:")
display(pd.Series({key: value for key, value in decomposition_study["metadata"].items()
                   if key != "configuration"}))
decomposition_records = decomposition_study["records"]
failed_decompositions = decomposition_records.loc[~decomposition_records.decomposition_valid]
if len(failed_decompositions):
    print("Failed source fits or decomposition checks (complete records also saved):")
    display(failed_decompositions[[
        "mu_test", "toy_id", "bank_kind", "replica", "generated_per_source", "model",
        "valid", "decomposition_message",
    ]])
print("Maximum absolute q reproduction error:",
      decomposition_records.q_reproduction_error.abs().max())
if (failed_decompositions.valid).any():
    raise RuntimeError(
        "A saved valid fit failed the fixed-reference likelihood checks. "
        "Inspect the displayed failures before interpreting the decomposition."
    )
print("Maximum absolute paired decomposition identity error:",
      decomposition_study["paired"].reconstruction_error.abs().max())

decomposition_summary = decomposition_study["summary"]
decomposition_overall = decomposition_summary.loc[decomposition_summary.branch_pair.eq("all")]
decomposition_keys = ["mu_test", "bank_kind", "replica", "generated_per_source", "model"]
print("Random-control counts; means and RMS condition on jointly valid fits:")
with pd.option_context("display.max_rows", None):
    display(decomposition_overall[decomposition_keys + [
        "n_random", "n_paired_valid", "n_paired_failed",
    ]])
for statistic in ("mean", "rms"):
    print(statistic, "of each signed term: delta_q = delta_pair - delta_min")
    columns = [f"{statistic}_{term}{suffix}"
               for term in ("delta_pair", "delta_min", "delta_q") for suffix in ("", "_se")]
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(decomposition_overall[decomposition_keys + columns])
print("Branch strata, ordered analytical / binned; conditional on branch membership:")
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    display(decomposition_summary.loc[decomposition_summary.branch_pair.ne("all"),
        decomposition_keys + ["branch_pair", "n_stratum", "branch_fraction",
                              "mean_delta_pair", "mean_delta_min", "mean_delta_q"]])
print("Changes between bank prefixes and independent replicas on the same toys:")
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    display(decomposition_study["changes"].loc[
        decomposition_study["changes"].comparison_type.ne("original"),
        ["mu_test", "model", "term", "comparison_type", "from_replica", "to_replica",
         "from_size", "to_size", "n_valid", "n_failed", "mean_change", "paired_se", "rms_change"],
    ])
print("Complete per-toy tables are saved as CSVs alongside the new results.")
'''

DECOMPOSITION_PLOTS = r'''
decomposition_figures = plot_likelihood_decomposition(
    decomposition_study, output=decomposition_figure_dir,
)
for name, figure in decomposition_figures.items():
    print(name)
    display(figure)
    plt.close(figure)
print("New tables:", decomposition_study["metadata"]["output_dir"])
print("New figures:", decomposition_figure_dir)
'''


def decomposition_cells():
    cells = [
        md(r"""
        ### 7. Follow-up: fixed-pair comparison versus the freely fitted minimum

        **Run only this new section.** In a fresh runtime with the updated source,
        execute the first two notebook code cells (setup, then common imports and
        `create_run`), keep the same run name, mode and configuration, and jump
        here. Section 6 must already be complete on disk; it does not need to run
        again. Keep its `SOURCE_TAG`, `OUTPUT_TAG` and `CONVERGENCE_TAG` below.

        We separate two reasons why the binned and analytical test statistics
        can differ. With the observable frozen at \(\eta=\mu_{\rm test}\), define
        for either likelihood \(m\)

        \[
        D_{m,\eta}(\mu)=-2\log\frac{L_{m,\eta}(\mu)}{L_{m,\eta}(1)},\qquad
        q_m(\eta)=D_{m,\eta}(\eta)-D_{m,\eta}(\hat\mu_m).
        \]

        The unbinned analytical likelihood does not itself depend on the
        observable partition. The subscript \(\eta\) records the matched bank
        and frozen partition used in its comparison with the binned model.
        Writing \(\Delta=\mathrm{binned}-\mathrm{analytical}\),

        \[
        \boxed{\Delta q=\underbrace{\Delta D(\eta)}_{\text{fixed-pair term}}
        -\underbrace{\Delta D(\hat\mu)}_{\text{minimum term}}.}
        \]

        Here the minimum term uses **each model's own saved best fit**. It is
        not the difference evaluated at one common best-fit parameter. Its
        contribution to \(\Delta q\) has a minus sign.

        The fixed-pair term tests the comparison between \(\eta\) and the
        reference \(1\) that the ratio observable is designed to preserve
        before binning. The minimum term tests the change in improvement of
        fit when the denominator is free to explore the full parameter range.
        Finite bins and finite integration samples can affect both terms.
        """),
        code(DECOMPOSITION_SETUP),
        md(r"""
        #### Reuse the completed fits and reconstruct their exact experiments

        Defaults use the two largest completed prefixes (normally 5x and 10x),
        both independent replicas, 12 and 36 bins, and the original-bank control.
        No integration samples or new toy IDs are added. The saved simulator
        experiments are reconstructed once each, passed through the same frozen
        preselection, and shared by all comparisons. No ratio-network evaluation
        or optimization is required in this stage.

        We evaluate \(D(\eta)\) and \(D(\hat\mu)\) at the stored fit result and
        check that their difference reproduces the saved \(q\). The context,
        predictions, fit settings, selected toy IDs and reconstructed event
        coordinates are checked against section 6. Missing or incompatible
        inputs fail visibly rather than triggering new fits or bank generation.
        Each analytical/binned pair uses the same selected-rate integrals, so
        its fixed-pair Poisson rate contribution cancels up to numerical
        precision; the remaining difference probes the event/bin log ratios.

        Population summaries use only the original **random controls**. Failed
        fits are explicitly counted; valid-only summaries are conditional on
        joint validity. The additional flagged experiments remain in the
        per-toy CSVs. Error bars are paired toy-sampling standard errors
        conditional on the banks, not finite-MC integration uncertainties.
        Matching new checkpoints resume under `DECOMPOSITION_TAG`.
        """),
        code(DECOMPOSITION_RUN),
        md(r"""
        #### Plot both terms, their spread, and their bank dependence

        Inspect the signed mean **and RMS** of each term: a small signed mean
        can conceal cancellation between large positive and negative errors.
        The per-toy plots identify whether the two fitted minima occupy the
        same or different branches, using the saved split at \(\mu=1\).
        This branch label is an operational diagnostic, not a fit constraint.

        All signed decomposition axes are **linear**, because these quantities
        can be negative. Logarithmic test-statistic distributions in earlier
        sections serve a different purpose. Compare each experiment across
        bank sizes and replicas before interpreting a residual as compression.
        """),
        code(DECOMPOSITION_PLOTS),
        md(r"""
        #### What would identify the next step?

        - **Small fixed-pair error, large minimum contribution:** the intended
          hypothesis comparison is preserved more accurately than the full
          family explored by the fitted denominator. Inspect branch changes;
          adding bins to the same one-dimensional ratio may not cure this.
        - **Terms change materially between large banks or their replicas:**
          integration precision remains a limiting uncertainty. Two replicas
          diagnose sensitivity but do not establish a finite-MC error model.
        - **Stable banks and smaller fixed-pair RMS with 36 bins:** finite-bin
          coarsening contributes to the discrepancy. Check the minimum term
          separately; the total test statistic need not improve monotonically.
        - **Failed reproduction or invalid fits:** resolve those records before
          interpreting the corresponding conditional means or tails.

        A possible subsequent control is the two-dimensional analytical
        statistic \((u,v)\), with the event count, where
        \(u=I/(B+\mathrm{NI})\), \(v=S/(B+\mathrm{NI})\) and
        \(I=\mathrm{SBI}-S-B\). For \(\kappa=\sqrt\mu\), the stat-only intensity
        is \(\lambda_\kappa=(B+\mathrm{NI})(1+\kappa u+\kappa^2v)\).
        Before binning this retains the parameter dependence of the full
        family wherever the reference intensity is positive. A finite 2D
        histogram would still have binning and integration errors. This section
        does not implement that separate control.

        These tests do not recalibrate confidence intervals. Once the numerical
        and compression contributions are understood, assess coverage using
        each binned model's own toys and physical simulator toys. Agreement with
        the analytical distribution of \(q\) is not itself a coverage criterion.
        """),
    ]
    cells[0].metadata["poodemo_append_section"] = DECOMPOSITION_MARKER
    return cells


def append_section_cells(destination, marker, factory, id_prefix):
    """Append only new cells, preserving the original notebook text verbatim."""
    original = destination.read_text()
    notebook = json.loads(original)
    if any(cell.get("metadata", {}).get("poodemo_append_section") == marker
           for cell in notebook["cells"]):
        print(f"{id_prefix.capitalize()} section already present; preserved {destination}")
        return False

    # Locate the end of the top-level cells array without normalizing any old
    # cell, output, execution count, or notebook metadata through nbformat.
    cells_key = original.index('"cells"')
    cells_start = original.index("[", cells_key)
    parsed_cells, cells_end = json.JSONDecoder().raw_decode(original, cells_start)
    if parsed_cells != notebook["cells"]:
        raise ValueError("Could not identify the notebook's top-level cells array")
    extra = factory()
    used_ids = {cell.get("id") for cell in notebook["cells"]}
    for index, cell in enumerate(extra):
        cell.id = f"{id_prefix}-{index:03d}"
        if cell.id in used_ids:
            raise ValueError(f"Notebook cell ID already exists: {cell.id}")
        cell.source = cell.source.splitlines(keepends=True)
    serialized = json.dumps(extra, indent=2, ensure_ascii=False)
    inner = serialized[2:-2]  # Strip only the new array's opening/closing lines.
    inner = "\n".join("  " + line for line in inner.splitlines())
    insertion = ",\n" if notebook["cells"] else "\n"
    updated = original[:cells_end - 1].rstrip() + insertion + inner + "\n  " + original[cells_end - 1:]
    parsed_updated = json.loads(updated)
    if parsed_updated["cells"][:len(notebook["cells"])] != notebook["cells"]:
        raise AssertionError("Appending changed an existing notebook cell")
    for key in notebook:
        if key != "cells" and parsed_updated[key] != notebook[key]:
            raise AssertionError("Appending changed notebook metadata")
    destination.write_text(updated)
    print(f"Appended {len(extra)} cells to {destination}; existing cells preserved")
    return True


def append_convergence_cells(destination):
    return append_section_cells(destination, CONVERGENCE_MARKER, convergence_cells, "convergence")


def build_toy_diagnostics_notebook(*, clean=False):
    destination = REPO / "notebooks" / "12_toy_diagnostics.ipynb"
    if destination.exists() and not clean:
        append_convergence_cells(destination)
        append_section_cells(destination, DECOMPOSITION_MARKER, decomposition_cells, "decomposition")
    else:
        write(destination.name, diagnostic_cells() + convergence_cells() + decomposition_cells())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true",
                        help="Regenerate notebook 12 without saved outputs")
    build_toy_diagnostics_notebook(clean=parser.parse_args().clean)
