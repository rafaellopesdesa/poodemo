"""Build notebook 12 alone, preserving every existing notebook and its outputs.

Run from the repository root: python scripts/build_toy_diagnostics_notebook.py
"""
from build_notebooks import BOOTSTRAP, COMMON_IMPORTS, code, md, write


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


def build_toy_diagnostics_notebook():
    write("12_toy_diagnostics.ipynb", diagnostic_cells())


if __name__ == "__main__":
    build_toy_diagnostics_notebook()
