"""Build notebooks 10/11 without rewriting notebooks 01--09 or their outputs.

Run from the repository root: python scripts/build_comparison_notebooks.py
The shared runtime bootstrap is imported from build_notebooks.py; all new cell
sources live here so these two notebooks can be reviewed and regenerated alone.
"""
from build_notebooks import BOOTSTRAP, COMMON_IMPORTS, code, md, write


SETUP_INTRO = r"""
### Reuse the completed v4 run

Keep **`RUN_NAME = "paper-distinct-s-v4"`**, the same `MODE`, and the same
`CONFIG_OVERRIDES` used for your completed notebooks 01–03 and 08–09. These new
studies do not require regeneration, retraining, or notebooks 04–07. Notebook 10
uses the selected integration bank; notebook 11 also loads the trained ratios
and the **12-bin, a=100** spline model saved by notebook 09.

Start a fresh Colab runtime to load the updated source. The temporary checkout
is `/content/poodemo`; delete that checkout if reusing a runtime with old source,
then restart. Your saved run in Google Drive is separate. The setup downloads
the repository if needed and mounts Drive. For a private repository, give Colab
read access, save a fine-grained token with repository **Contents: read** access
as the `GITHUB_TOKEN` Colab secret, or use the hidden prompt. Alternatively,
upload `/content/poodemo_source.zip`. Tokens are not printed or saved.

`JAX_BACKEND="auto"` uses a Colab GPU when available. Choose `"cpu"` for an
intentional CPU run or `"gpu"` to require a GPU. Setup verifies double-precision
JAX calculations. `MODE="smoke"` uses a separate, already-created smoke run;
changing this notebook's mode does not create the prerequisite samples/models.
Only a migration from an older, incompatible run requires rerunning 01–03.
"""

STYLE = r'''
import mplhep as hep
hep.style.use("ATLAS")
# ATLAS-like typography/ticks only: no experiment label is added.
plt.rcParams.update({"axes.grid": False, "figure.dpi": 110})
'''


def start(title, introduction):
    return [md(f"# {title}\n\n{introduction}"), md(SETUP_INTRO),
            code(BOOTSTRAP), code(COMMON_IMPORTS + "\n" + STYLE)]


DISCRIMINATOR_SETTINGS = r'''
from poodemo.discriminator_study import run_discriminator_study, plot_discriminator_study

N_BINS = 12  # Per coordinate: 12, 12**2, and 12**3 cells.
POWERS = {"NI": 1.0, "SBI": 2.0, "B": 2.0}
REFERENCE_RATIO_A = 100.0  # Same moving physical-reference ratio as 08/09.
RATIO_SOURCE = "analytic"  # Optional alternative: "learned".
SYSTEMATICS = (False, True)  # Compare both fixed and profiled NI.
MORPH_ORDER = "after"  # Interpolate the binned NI templates, as in 09.
MU_GRID = None  # None uses the saved run scan; e.g. np.linspace(0.02, 2., 101).

print(f"Fixed grids: {N_BINS}, {N_BINS**2}, {N_BINS**3} cells")
print("Process powers:", POWERS, "| physical-reference power:", REFERENCE_RATIO_A)
print("Observable axes:", RATIO_SOURCE, "| Asimov truth:", run.config["asimov_mu"])
'''

DISCRIMINATOR_RUN = r'''
study = run_discriminator_study(
    run, n_bins=N_BINS, powers=POWERS, reference_power=REFERENCE_RATIO_A,
    ratio_source=RATIO_SOURCE, systematics=SYSTEMATICS,
    scan_grid=MU_GRID, morph_order=MORPH_ORDER,
)
display(study["axis_summary"])
display(study["occupancy"])
print("Full settings:")
display(pd.Series(study["config"]))
'''

DISCRIMINATOR_PLOTS = r'''
FIGURE_DIR = ROOT / "plots" / "10_discriminator_study"
figures = plot_discriminator_study(study, output=FIGURE_DIR)
for name, figure in figures.items():
    print(name)
    display(figure)
    plt.close(figure)
print("Figures:", FIGURE_DIR)
print("Tables:", ROOT / "results" / "10_discriminator_study")
'''


def discriminator_cells():
    cells = start("10. How much information do individual process discriminators retain?", r"""
    Compare the **12-bin moving likelihood-ratio observable** from notebook 08
    with fixed one-, two-, and three-dimensional process-ratio histograms.
    All methods use the same analytical Asimov expectation and keep the
    extended-likelihood event count. The analytical unbinned likelihood is the
    full-information benchmark. The comparison asks how much is retained by
    each compression, without assuming that any finite histogram is sufficient.
    """)
    cells += [
        md(r"""
        ### Four observable choices

        Write \(r_j(x)=p_j(x)/p_S(x)\), with each process density normalized
        after the same preselection. For the process axes use
        \[
        z_j(x)=\frac{r_j(x)^{a_j}}{1+r_j(x)^{a_j}}
        =\operatorname{sigmoid}\!\left(a_j\log r_j(x)\right).
        \]
        A finite positive \(a_j\) preserves ordering while redistributing bin
        resolution. The editable starting powers are \(a_{NI}=1\) and
        \(a_{SBI}=a_B=2\). Inspect the marginal plots and endpoint-occupation
        table before changing them; the powers are fixed for the whole study.

        | Observable | Coordinates | Total cells |
        |---|---|---:|
        | Fixed NI discriminator | \(z_{NI}\) | 12 |
        | Fixed NI and SBI discriminators | \((z_{NI},z_{SBI})\) | 144 |
        | Fixed NI, SBI and B discriminators | \((z_{NI},z_{SBI},z_B)\) | 1728 |
        | Moving physical-reference ratio | \(z_\eta=R_\eta^{100}/(1+R_\eta^{100}),\ R_\eta=p_\eta/p_1\) | 12 |

        The default **analytical axes** match the construction in 08–09 and
        isolate the information lost by compression and binning. Choosing
        `RATIO_SOURCE="learned"` uses trained ratios for both the fixed process
        axes and the moving physical-reference axis, adding their approximation
        errors. The physical generating
        model and analytical benchmark remain unchanged.

        For the moving observable, set \(\eta=\mu_0\) at each tested point,
        then **freeze the observable and the data** throughout its likelihood
        fit. Fixed process axes stay fixed across the entire scan. All use the
        same [0,1] edge convention that keeps 0.5 inside a bin. The likelihood
        keeps absolute expected counts. Only the marginal display plots are
        normalized per process to compare their shapes; fit templates retain yields.
        """),
        code(DISCRIMINATOR_SETTINGS),
        md(r"""
        ### Build the templates and scan the likelihood

        The physical \(\mu\) coefficients and selected process yields are
        identical across the histograms. Stat-only fixes the NI nuisance at
        zero; the profiled comparison includes its Gaussian constraint and
        the binned nuisance interpolation used in notebook 09. The diagnostic
        `MORPH_ORDER` setting can separately investigate interpolation versus
        integration. Default settings do not change the saved physics run.

        The occupancy table reports sparse and empty cells, together with
        integration-bank effective sample sizes. Going from 12 to 1728 cells
        increases the integration demands, even if the physical information
        improves. Sparse cells are reported, not filled with artificial events.
        """),
        code(DISCRIMINATOR_RUN),
        md(r"""
        ### Distributions, likelihood scans, and local information

        First inspect the one-dimensional axes and their joint distributions.
        Then compare the likelihood scans directly. The information panel
        measures **local precision at the fixed Asimov truth**, which is a
        different question from retaining a finite-separation test against 1.
        At \(\eta=1\), the moving observable collapses to 0.5 and contains only
        event-count information. Its test of the generating truth can still
        correctly give zero even though its frozen local width is poor.
        """),
        code(DISCRIMINATOR_PLOTS),
        code(r'''
        display(study["information"])
        scans = study["scans"]
        if "valid" in scans.columns:
            invalid = scans.loc[~scans["valid"].fillna(False)]
            if len(invalid):
                print("Inspect these unsuccessful fits before interpreting the curves:")
                display(invalid)
        display(scans.head(15))
        '''),
        md(r"""
        ### What this demonstrates

        With NI fixed, the nominal intensity is a known linear combination of
        S, SBI, B and NI. Consequently the **continuous vector**
        \((r_{NI},r_{SBI},r_B)\), together with the event count, contains the
        information needed for the nominal \(\mu\) family: the common S density
        cancels from likelihood ratios. Finite 12-by-12-by-12 bins still lose
        information, and sparse integration cells can limit the comparison.
        When NI is profiled, its variation ratios need not be functions of this
        nominal three-ratio vector, so this sufficiency argument is narrower.

        SBI and B are similar in this model, but that alone does not establish
        that the third axis is redundant: their difference helps reconstruct
        the interference contribution. Measure its incremental gain from the
        actual fits. The fixed grids are nested coordinate refinements, but
        this does not require every plotted profiled test statistic to increase
        monotonically. The comparison with the moving 12-bin ratio shows how efficiently a
        test-specific ordering can use a small bin budget. This does not prove
        that one fixed scalar retains the entire parameter family, nor that
        the profiled or learned cases inherit exact pairwise optimality.

        Next: **11_test_statistic_toys.ipynb**, which replaces Asimov
        expectations with repeated experiments and measures calibration.
        """),
    ]
    return cells


TOY_SETTINGS = r'''
from poodemo.toy_study import run_toy_study, plot_toy_study

MU_VALUES = (0.0, 1.4)  # In each ensemble, mu_true = mu_test.
N_TOYS = 12 if MODE == "smoke" else 500
N_BINS = 12
REFERENCE_RATIO_A = 100.0  # Must match the 12-bin notebook 09 spline file.
EXPOSURE = 1.0  # Scale all selected expected yields together.
SEED = 110923
MU_BOUNDS = (0.0, 2.0)  # 1.4 is an interior tested point, not a fit boundary.
FIT_GRID_SIZE = 65  # Grid in kappa=sqrt(mu), followed by local refinement.
OUTPUT_TAG = "toy_study"  # Change for a study with different core settings.
PROGRESS_EVERY = 10

print(f"{N_TOYS} toys per case at each of {MU_VALUES}; five comparisons")
print(f"Exposure={EXPOSURE:g}; mu fit range={MU_BOUNDS}; stat-only")
print("Matching completed rows are reused. Increase N_TOYS to extend an ensemble.")
'''

TOY_RUN = r'''
toys = run_toy_study(
    run, n_toys=N_TOYS, mu_values=MU_VALUES, n_bins=N_BINS,
    reference_power=REFERENCE_RATIO_A, exposure=EXPOSURE, seed=SEED,
    mu_bounds=MU_BOUNDS, grid_size=FIT_GRID_SIZE,
    output_tag=OUTPUT_TAG, progress_every=PROGRESS_EVERY,
)
print("Generation, fit, and numerical settings:")
display(pd.Series(toys["metadata"]))
print("Direct-versus-spline template closure at the tested anchors:")
display(toys["template_closure"])
'''

TOY_PLOTS = r'''
FIGURE_DIR = ROOT / "plots" / "11_test_statistic_toys" / OUTPUT_TAG
figures = plot_toy_study(toys, output=FIGURE_DIR)
for name, figure in figures.items():
    print(name)
    display(figure)
    plt.close(figure)
print("Figures:", FIGURE_DIR)
print("Toy records:", ROOT / "results" / f"{OUTPUT_TAG}_results.csv")
'''


def toy_cells():
    cells = start("11. Test-statistic distributions and calibration with toys", r"""
    Measure the **stat-only** sampling distribution at the two requested
    hypotheses, \(\mu=0\) and \(\mu=1.4\). In each ensemble the tested value is
    also the generating truth. Compare analytical and learned unbinned fits
    with the 12-bin spline model from notebook 09, using both physical
    simulator toys and each approximate model's own toys.

    Run **01–03 and 09** first in the same v4 run; 09 itself uses the completed
    08 study. Notebook 10 and notebooks 04–07 are not prerequisites. No training
    is performed here. The full production toy study is intentionally left for
    you to run; saved progress makes it resumable.
    """)
    cells += [
        md(r"""
        ### Define one statistic and five comparisons

        At each tested \(\mu_0\), use the **two-sided** likelihood-ratio statistic
        \[
        t_{\mu_0}=-2\log\frac{L(\mu_0)}{L(\widehat\mu)},
        \qquad \widehat\mu=\underset{0\leq\mu\leq2}{\arg\max}\ L(\mu).
        \]
        All NI nuisances are fixed at nominal. This is not the one-sided
        upper-limit statistic that sets the result to zero when
        \(\widehat\mu>\mu_0\). The exact zero boundary is included: minimization
        uses \(\kappa=\sqrt\mu\), avoiding differentiation of \(\sqrt\mu\) at zero,
        and searches the grid's candidate minima before local refinement.
        The point 1.4 lies inside the fit range, allowing fluctuations on both
        sides. Inspect upper-bound fits before trusting a truncated fit range.

        | Comparison | Toy-generating distribution | Likelihood fitted |
        |---|---|---|
        | Analytical / simulator | Physical simulator | Analytical unbinned |
        | Learned / simulator | Same physical simulator experiments | Learned unbinned |
        | Learned / learned | Learned model on the integration bank | Learned unbinned |
        | Binned / simulator | Same simulator experiments, histogrammed | 12-bin model from 09 |
        | Binned / binned | Independent Poisson counts from the binned model | Same 12-bin model |

        The simulator rows deliberately share each experiment across fit
        methods. For the binned fits, build the observable at \(\eta=\mu_0\)
        and **hold it fixed while fitting every alternative \(\mu\)**.
        The reference in the observable remains \(p(x;1)\).
        """),
        code(TOY_SETTINGS),
        md(r"""
        ### What is drawn, and what is held fixed?

        **Physical simulator toys:** draw a Poisson event count with the
        selected physical mean, then independent selected event coordinates
        from the coherent interference model plus NI. These are fresh simulator
        events, independent of the training and integration banks. Evaluate
        the analytical and learned likelihoods on those same coordinates.
        Histogramming that Poisson point-process sample already gives Poisson
        bin counts. **Do not fluctuate those counts a second time.**

        **Learned-model toys:** use Poisson sampling from the learned intensity
        represented on the saved integration bank. This is a finite-bank
        approximation to sampling the continuous learned model, and it uses
        the same numerical normalization as the learned likelihood. Increasing
        the toy count cannot remove that integration approximation. It tests
        internal sampling calibration of the learned model; agreement here
        does not establish correctness for physical simulator data.

        **Binned-model toys:** draw independent Poisson counts directly from
        the frozen binned model's predicted means, without using the analytical
        simulator as the toy source. The physical simulator and binned-model
        rows therefore expose the effect of template approximation separately.

        Notebook 09's saved eta grid starts above zero. At **eta=0**, this
        notebook builds the direct analytical boundary template using the same
        12 bins and observable, rather than extrapolating its spline. At 1.4
        it uses the saved spline interpolation. The template-closure table
        makes this distinction visible. The model at each frozen eta is then
        used for the entire physical-mu fit and for its own Poisson toys.
        """),
        md(r"""
        ### Run or resume the experiment ensembles

        The default 500 toys per case and hypothesis are a first calibration
        study, not a precise tail measurement. `MODE="smoke"` uses 12 toys to
        check execution only. Increase `N_TOYS` in this same study to extend
        it; deterministic seeds and saved rows preserve completed toys.
        Changing a core setting such as exposure, the fit range, or observable
        requires a different `OUTPUT_TAG`. Periodic progress messages identify
        completed experiments. The full sample of test statistics and fitted
        values is saved, including fit diagnostics; failed fits must be
        inspected rather than silently counted as successful calibration.
        """),
        code(TOY_RUN),
        md(r"""
        ### Empirical distributions and calibrated acceptance

        Compare the test-statistic histograms, empirical survival curves,
        quantiles, and fitted-\(\mu\) distributions. Similar Asimov scans do
        not force these sampling distributions to agree. In particular,
        the zero boundary and the second interference minimum can invalidate
        a simple common asymptotic reference. The gray dashed asymptotic
        curves are visual guides only. We **measure** critical values from toys
        instead of imposing a \(\chi^2_1\) or boundary-mixture approximation.

        The cross-calibration table applies a model's own empirical critical
        value to simulator-generated test statistics. Even-numbered toy IDs
        calibrate the critical values; disjoint odd-numbered IDs measure
        acceptance, including in the model's own validation sample. Acceptance
        at the tested truth is the pointwise coverage of the corresponding
        inverted test. Compare this external acceptance to the nominal level.
        The quoted conditional binomial uncertainty describes the evaluation
        toys; it does **not** include uncertainty in the finite calibration
        sample's critical value. Ties, especially at the zero boundary, can
        make a non-randomized acceptance rule conservative. Inspect the failed-fit
        counts and bounds reported with each comparison. These two truth points
        do not establish coverage throughout the whole parameter range.
        """),
        code(TOY_PLOTS),
        code(r'''
        print("Empirical test-statistic and fit summaries:")
        display(toys["summary"])
        print("Calibration/coverage comparisons:")
        display(toys["coverage"])
        print("Individual toy records (first rows; complete table is saved):")
        display(toys["results"].head(20))
        failed = toys["results"].loc[~toys["results"]["valid"].astype(bool)]
        if len(failed):
            print("Failed fits: inspect these before interpreting calibration:")
            display(failed)
        '''),
        md(r"""
        ### How to interpret agreement or disagreement

        - **Analytical / simulator** establishes the physical sampling
          benchmark, including boundaries and interference ambiguities.
        - **Learned / simulator versus learned / learned** tests whether
          self-calibration transfers to the real toy generator.
        - **Binned / simulator versus binned / binned** tests the corresponding
          transfer for the 12-bin template likelihood.
        - **Binned versus unbinned** quantifies how much the economical
          histogram changes the test-statistic distribution and fitted values.

        If the upper fit boundary is frequently reached, widen `MU_BOUNDS`
        and use a new output tag before interpreting quantiles. If the
        empirical tails remain noisy, increase `N_TOYS`. A larger ensemble
        reduces toy uncertainty; it does not fix a learned-density error,
        spline mismatch, or insufficient numerical integration.
        """),
    ]
    return cells


def build_comparison_notebooks():
    write("10_discriminator_histograms.ipynb", discriminator_cells())
    write("11_test_statistic_toys.ipynb", toy_cells())


if __name__ == "__main__":
    build_comparison_notebooks()
