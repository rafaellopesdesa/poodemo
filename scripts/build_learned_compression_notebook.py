"""Build notebook 13 without rewriting any earlier notebook or saved output.

Run from the repository root: python scripts/build_learned_compression_notebook.py
"""
from build_notebooks import BOOTSTRAP, COMMON_IMPORTS, code, md, write


SETUP_INTRO = r"""
### Reuse the trained v4 models

Use the same **`RUN_NAME = "paper-distinct-s-v4"`**, `MODE`, and
`CONFIG_OVERRIDES` as your completed notebooks 01–03. Run **13 only** after
updating the source and restarting the runtime; no retraining or rerunning
08–12 is needed. The saved preselection and ratio networks are frozen.

The bootstrap below reuses an existing `/content/poodemo` checkout. In Colab,
delete that temporary checkout and restart if it contains older source; the
saved Drive run is separate. Private-repository access follows the earlier
notebooks: a `GITHUB_TOKEN` Colab secret with Contents: read, the hidden prompt,
or `/content/poodemo_source.zip`. `JAX_BACKEND="auto"` uses an available GPU.

This notebook generates **two new, independent simulation integration banks**
with deterministic, separate random streams. Each bank has two million
generated events per S, B, and NI proposal stratum in production, before the
existing preselection; smoke mode uses 20,000 per stratum. These are integration
events, not new training samples and not pseudo-experiments. The physical
expected yields and experimental exposure remain those of the saved run.
The two banks require time and disk space; progress is printed and their
verified cache is reused when settings and source artifacts match.
"""

STYLE = r'''
import mplhep as hep
hep.style.use("ATLAS")
# Typography/ticks only; no ATLAS label.
plt.rcParams.update({"axes.grid": False, "figure.dpi": 110})
'''

SETTINGS = r'''
from poodemo.learned_compression_banks import prepare_compression_banks
from poodemo.learned_compression_study import run_learned_compression_study
from poodemo.learned_compression_plots import (
    plot_compression_scans, plot_compression_information,
    plot_compression_distributions, plot_compression_closure,
)

N_BINS = 36
REFERENCE_RATIO_A = 100.0
REFERENCE_RATIO_ETA_MIN = 0.2
N_PER_SOURCE = 20_000 if MODE == "smoke" else 2_000_000
SEED = 131013
CHUNK_SIZE = 100_000
OUTPUT_TAG = "13_learned_compression_36_floor02"
SYSTEMATICS = (False, True)
MU_GRID = None  # None uses the saved physical scan grid; no physical floor.
DISTRIBUTION_ETAS = (0.0, 0.3, 0.6, 1.0, 1.5)

# Display limits only: the complete scan is calculated and saved.
SCAN_XLIM = (0.02, 1.15)
SCAN_YLIM = (0.0, 8.0)
INFORMATION_XLIM = None

FIGURE_DIR = ROOT / "plots" / OUTPUT_TAG
print(f"{N_BINS} bins; a={REFERENCE_RATIO_A:g}; numerator eta_min={REFERENCE_RATIO_ETA_MIN:g}")
print(f"Two banks × three strata × {N_PER_SOURCE:,} events before preselection")
print("Physical Asimov truth:", run.config["asimov_mu"])
print("Physical free-fit bounds:", run.config["mu_fit_bounds"])
print("Results:", ROOT / "results" / OUTPUT_TAG)
'''

BANKS = r'''
banks = prepare_compression_banks(
    run, n_per_source=N_PER_SOURCE, seed=SEED,
    chunk_size=CHUNK_SIZE, output_tag=OUTPUT_TAG,
)

bank_rows = []
for bank_name in ("template", "validation"):
    meta = banks[bank_name]["metadata"]
    for source in ("S", "B", "NI"):
        bank_rows.append({
            "bank": bank_name, "source": source,
            "generated": meta["generated_counts"][source],
            "selected": meta["selected_counts"][source],
            "stream_id": meta["stream_id"],
        })
display(pd.DataFrame(bank_rows))
print("Frozen rates from the original saved integration bank:")
display(pd.Series(banks["rates"], index=("S", "SBI", "B", "NI"), name="selected_rate"))
print("Frozen NN normalizers from the original saved integration bank:")
display(pd.Series(banks["metadata"]["frozen_normalizers"]))
print("Source and bank fingerprint:", banks["metadata"]["fingerprint"])
print("Cache:", banks["directory"])
'''

RUN_STUDY = r'''
study = run_learned_compression_study(
    run, banks, n_bins=N_BINS, reference_power=REFERENCE_RATIO_A,
    reference_eta_floor=REFERENCE_RATIO_ETA_MIN,
    scan_grid=MU_GRID, systematics=SYSTEMATICS,
    distribution_etas=DISTRIBUTION_ETAS, output_tag=OUTPUT_TAG,
)

display(pd.Series(study["metadata"], name="study metadata"))
diagnostics = study["map_diagnostics"]
print("Map validity summary (one row per bank, map, and construction point):")
display(diagnostics.groupby(["bank", "model"], dropna=False).agg(
    construction_points=("valid", "size"),
    valid_points=("valid", "sum"),
    maximum_invalid_fraction=("invalid_fraction", "max"),
    maximum_invalid_expected_fraction=("invalid_expected_fraction", "max"),
))
invalid_maps = diagnostics.loc[~diagnostics.valid.fillna(False)]
if len(invalid_maps):
    print("Invalid learned intensities prevent a valid log-ratio map at these points; inspect before interpreting:")
    display(invalid_maps)
invalid_fits = study["scans"].loc[~study["scans"].valid.fillna(False)]
if len(invalid_fits):
    print("Unsuccessful or unavailable fits are retained in the tables and omitted from plotted lines:")
    display(invalid_fits)
'''

DISPLAY_HELPER = r'''
def show_figures(figures):
    for name, figure in figures.items():
        print(name)
        display(figure)
        plt.close(figure)
'''


def cells():
    return [
        md(r"""
        # 13. Learned compression with independently simulated bin yields

        Repeat the principal comparison of notebook 08 using the **trained
        density-ratio networks to define the observable**. The likelihood in
        both binned constructions comes from physical simulator yields. Compare
        analytical unbinned inference, an analytical ratio map with 36 bins,
        and a learned ratio map with the same 36 bins. The numerator endpoint
        floor is 0.2 and the transformation power is 100, as in 08–11.

        Two complementary Asimov comparisons separate compression from
        finite-simulation closure: matched bank A data/templates, and independent
        bank B data with bank A templates. No network is retrained, no learned
        density weights define physical yields, and no spline approximation is
        used. This notebook does not measure repeated-experiment coverage.
        """),
        md(SETUP_INTRO),
        code(BOOTSTRAP),
        code(COMMON_IMPORTS + "\n" + STYLE),
        md(r"""
        ## 1. Freeze the observable separately from the physical likelihood

        The two observable choices are
        \[
        t_\eta^{\rm ana}(x)=\operatorname{sigmoid}
        \left[a\log\frac{r(x;\eta_{\rm eff})}{r(x;1)}\right],\qquad
        t_\eta^{\rm NN}(x)=\operatorname{sigmoid}
        \left[a\log\frac{\widehat r(x;\eta_{\rm eff})}{\widehat r(x;1)}\right],
        \quad \eta_{\rm eff}=\max(\eta,0.2).
        \]
        The saved process-ratio networks and their original integration-bank
        normalizers are frozen before evaluating either new bank. Analytical
        and learned map normalizations use the same fixed selected physical
        rates. We do not recalibrate either map on bank B or tune it for closure.

        At each tested physical \(\mu_0\), choose \(\eta=\mu_0\), fill the
        observed histogram, and freeze its partition for **both** the constrained
        and free fits. The denominator fit is repeated for each partition.
        Only the observable numerator is floored; the physical \(\mu\), yield
        coefficients, and free-fit domain retain their original definitions.
        All hypotheses below the floor share a partition, while their predicted
        yields remain different. At \(\eta=1\) both maps collapse to 0.5,
        contained inside a bin; event-count information remains.

        Change `MU_GRID` for the physical scan or the plotting limits below for
        display. Changing the simulation-bank settings or frozen input artifacts
        requires a distinct `OUTPUT_TAG`; prior results and earlier notebooks
        remain separate. The defaults preserve the completed v4 physics run.
        """),
        code(SETTINGS),
        md(r"""
        ## 2. Independent physical simulation banks

        Generate stratified S/B/NI proposals with independent deterministic
        streams for A and B, then apply the saved preselection. Evaluate the
        analytical component intensities and NI variations on the surviving
        events. The proposal/integration weights account for generated events
        that fail selection; S, SBI, B, and NI template integrals are estimates
        of absolute selected physical yields.

        The trained networks are evaluated on these same selected events **only
        to assign bins**. Negative or nonfinite combined learned intensities
        invalidate the corresponding ratio map and are reported explicitly;
        they are not hidden by dropping events or by treating negative density
        estimates as physical yields. Bank caches include source fingerprints
        and payload checksums.
        """),
        code(BANKS),
        md(r"""
        ## 3. Construct direct templates and fit both Asimov comparisons

        For map \(t_\eta\), the bin prediction is
        \[
        \Lambda_b(\mu,\alpha;\eta)=
        \int\mathbf{1}[t_\eta(x)\in B_b]\,
        \lambda(x;\mu,\alpha)\,dx.
        \]
        The simulation estimate of this integral uses physical component
        intensities. A neural approximation changes the partition, not the
        intensity used to predict its yield.

        | Comparison | Asimov event integration | Binned template integration | Purpose |
        |---|---|---|---|
        | Matched | Bank A | Bank A | Compression/binning comparison with a common numerical expectation |
        | Independent | Bank B | Bank A | Detect sensitivity to independent finite simulation |

        Both comparisons use the same physical Asimov truth and independently
        fitted analytical unbinned baseline. The baseline is recomputed for these
        banks rather than imported from notebook 03. The matched construction
        shares numerical fluctuations and therefore cannot establish independent
        closure; the second comparison addresses that limitation.

        NI fixed and NI profiled are both calculated. As in 09, the profiled
        binned model morphs nominal/down/up NI yields **after integration**.
        The analytical unbinned model morphs event-level intensities. Their
        difference can include nuisance-interpolation effects in addition to
        compression. The NI-fixed comparison isolates compression more directly.
        """),
        code(RUN_STUDY),
        code(DISPLAY_HELPER),
        md(r"""
        ## 4. Likelihood scans and differences from analytical unbinned

        Compare the two binned curves against the analytical unbinned curve,
        and against each other. The residual panel shows
        \(\Delta q_\mu=q_\mu^{\rm binned}-q_\mu^{\rm unbinned}\) for the
        **same** Asimov comparison and nuisance treatment. Inspect matched and
        independent results separately. A binned curve need not differ from the
        unbinned curve with one sign at every point of a profiled scan.

        These plots keep the familiar zoom from 08. Complete scan tables and
        the full-range plots below retain every calculated point. Failed fits
        and invalid maps remain visible in the diagnostic tables; their plotted
        values are gaps, not zero test statistics.
        """),
        code(r'''
        show_figures(plot_compression_scans(
            study, xlim=SCAN_XLIM, ylim=SCAN_YLIM, output=FIGURE_DIR / "zoom"))
        show_figures(plot_compression_scans(
            study, output=FIGURE_DIR / "full_range"))
        '''),
        md(r"""
        ## 5. Fitted values and local information

        Every scan point has its own frozen compression and therefore its own
        unrestricted fit. The fitted \(\widehat\mu\) values below are not
        interchangeable with the minimum of the assembled moving-observable
        scan. Compare matched and independent fits at the same construction
        point to assess finite-bank sensitivity.

        The local-width calculation differentiates the **physical** model at the
        Asimov truth with each map fixed. Thus the horizontal axis below labels
        the map, not the physical point at which the information is evaluated.
        These widths describe the nominal bank-A model; they do not replace
        the independent-bank closure test or a repeated-experiment study.
        """),
        code(r'''
        fit_columns = [c for c in (
            "comparison", "systematics", "model", "mu", "eta_effective",
            "mu_hat", "q", "delta_q", "valid", "fit_message",
        ) if c in study["scans"]]
        display(study["scans"][fit_columns])
        width_columns = [c for c in (
            "systematics", "model", "eta", "eta_effective", "information_mu",
            "sigma_mu", "information_fraction", "width_valid", "width_message",
        ) if c in study["information"]]
        display(study["information"][width_columns])
        show_figures(plot_compression_information(
            study, xlim=INFORMATION_XLIM, output=FIGURE_DIR / "information"))
        '''),
        md(r"""
        ## 6. Physical component shapes on the two maps

        Compare S, SBI, B, and NI on the analytical and learned axes for
        \(\eta=0,0.3,0.6,1,1.5\). The plotted component histograms are
        normalized separately **for display only**; the fits retain absolute
        yields. All histograms use the same 36 edges as the likelihood, including
        the convention that places 0.5 inside a bin. Logarithmic axes expose tails.
        """),
        code(r'''
        show_figures(plot_compression_distributions(
            study, etas=DISTRIBUTION_ETAS, output=FIGURE_DIR / "distributions"))
        '''),
        md(r"""
        ## 7. Independent bin-yield closure

        For the physical Asimov truth, compare bank A template expectations with
        bank B expectations in the identical frozen partition. There is **no
        overall or binwise rescaling to enforce closure**. The lower panels show
        B/A; error bars quantify independent integration-bank fluctuations,
        not experimental Poisson fluctuations. The ratio errors propagate
        numerator and denominator Monte Carlo uncertainties. Shared events can
        correlate different bins/partitions, so these panels are diagnostics,
        not a collection of independent significance tests.

        Empty A bins have an undefined ratio and are retained in the tables.
        Inspect effective sample sizes and pulls as well as visually smooth
        curves. Increasing generated simulation, rather than the number of toys,
        addresses this source of numerical uncertainty.
        """),
        code(r'''
        show_figures(plot_compression_closure(
            study, etas=DISTRIBUTION_ETAS, output=FIGURE_DIR / "closure"))
        closure = study["bin_closure"]
        display(closure.loc[np.isclose(closure.eta, DISTRIBUTION_ETAS[0])])
        print("All numerical results:", ROOT / "results" / OUTPUT_TAG)
        print("All figures (PDF and PNG):", FIGURE_DIR)
        '''),
        md(r"""
        ## Interpretation

        The matched comparison measures how the actual trained map changes the
        usefulness of a 36-bin partition, on top of analytical-map binning loss.
        The independent comparison tests whether those conclusions persist when
        the Asimov expectation does not share the template bank's fluctuations.
        These are numerical Asimov expectations integrated with physical
        simulator weights, not exact closed-form integrals.

        A learned map can be imperfect while its simulator-derived bin means
        define a valid binned model. The central question here is the sensitivity
        retained by that particular map and bin budget. Agreement in these tests
        does not establish coverage, validate a learned unbinned density, or
        eliminate finite-simulation uncertainties. Notebook 11 addresses the
        repeated-experiment question separately; its current toys use its own
        configured construction and are not rerun or overwritten here.

        If the learned and analytical partitions differ appreciably, inspect the
        map diagnostics, component tails, per-bin integration precision, and the
        NI-fixed/profiling comparison before attributing everything to network
        calibration. The saved outputs and frozen provenance allow a larger
        independent-bank run to be compared without changing the trained model.
        """),
    ]


def build():
    write("13_learned_ratio_compression.ipynb", cells())


if __name__ == "__main__":
    build()
