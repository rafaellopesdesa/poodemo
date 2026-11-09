"""Rebuild notebook 09 with a fresh pinned checkout; preserve study controls."""
import ast
import textwrap

import nbformat
from build_notebooks import BOOTSTRAP, REPO

# Tested empirical-CDF refinement and direct floor controls.
# Pin code independently of notebook-output commits for reproducible reruns.
SOURCE_COMMIT = "69e8fdcff7e761bbd843418a1e10f779ffba77c7"


def fresh_bootstrap(previous):
    """Use the shared installer with an isolated source download on every run."""
    source = BOOTSTRAP.strip()
    # Retain user settings, including multiline CONFIG_OVERRIDES dictionaries.
    for node in ast.parse(previous).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        name = getattr(node.targets[0], 'id', None)
        if name not in ('RUN_NAME', 'MODE', 'CONFIG_OVERRIDES', 'JAX_BACKEND'):
            continue
        replacement = previous.splitlines()[node.lineno - 1:node.end_lineno]
        target = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and getattr(n.targets[0], 'id', None) == name)
        lines = source.splitlines()
        lines[target.lineno - 1:target.end_lineno] = replacement
        source = '\n'.join(lines)
    source = source.replace('\ntry:\n    import google.colab',
                            f'\nSOURCE_COMMIT = "{SOURCE_COMMIT}"\n\ntry:\n    import google.colab', 1)
    source = source.replace('    REPO_DIR = Path("/content/poodemo")', '''    import tempfile
    # Download afresh even in a reused runtime; do not alter older checkouts.
    REPO_DIR = Path(tempfile.mkdtemp(prefix="poodemo-09-", dir="/content"))
    print(f"Notebook 09 source: {SOURCE_COMMIT} (fresh download)")''')
    # An older uploaded ZIP must not override this notebook's tested revision.
    start = source.index('        uploaded_zip = ')
    end = source.index('\n\n        # Validate archive paths', start)
    download = source[source.index('            archive_url = ', start):end]
    download = textwrap.indent(textwrap.dedent(download), '        ')
    download = download.replace('"https://api.github.com/repos/rafaellopesdesa/poodemo/tarball/main"',
                                'f"https://api.github.com/repos/rafaellopesdesa/poodemo/tarball/{SOURCE_COMMIT}"')
    download = download.replace('Upload /content/poodemo_source.zip or provide read access.',
                                'Provide a GitHub token with read access to download the pinned notebook 09 source.')
    source = source[:start] + download + source[end:]
    # Only tar downloads remain in this bootstrap, so remove the ZIP branch.
    start = source.index('            if archive_kind == "zip":')
    end = source.index('            candidates = ', start)
    extraction = source[source.index('                with tarfile.open', start):end]
    source = source[:start] + textwrap.indent(textwrap.dedent(extraction), '            ') + source[end:]
    source = source.replace('        import zipfile\n', '').replace('        archive_kind = "tar"\n', '')
    source = source.replace('        import tempfile\n', '')
    source = source.replace('The uploaded archive must contain', 'The downloaded archive must contain')
    start = source.index('    cached_package = sys.modules.get("poodemo")')
    end = source.index('\nelse:\n    # Locally:', start)
    source = source[:start] + '''    # Run from the top: subsequent cells recreate objects from this checkout.
    # Clear all poodemo submodules, including old plotting and fit functions.
    for module_name in list(sys.modules):
        if module_name == "poodemo" or module_name.startswith("poodemo."):
            del sys.modules[module_name]
''' + source[end:]
    return source


SOURCE_CHECK = '''import poodemo.reference_splines as _reference_splines

_expected_methods = ("ecdf_u",)
_loaded_methods = tuple(getattr(_reference_splines, "METHODS", ()))
if _loaded_methods != _expected_methods or getattr(_reference_splines, "STUDY_VERSION", 0) != 3:
    raise RuntimeError(
        f"Old interpolation code is loaded: {_loaded_methods}. "
        "Run this notebook from the setup cell at the top before starting the study.")
if IN_COLAB:
    _expected_source = (REPO_DIR / "poodemo" / "reference_splines.py").resolve()
    if Path(_reference_splines.__file__).resolve() != _expected_source:
        raise RuntimeError("The interpolation module was not loaded from this run's fresh checkout.")
print("Verified notebook 09 study version:", _reference_splines.STUDY_VERSION)
print("Verified interpolation methods:", ", ".join(_loaded_methods))
print("Interpolation module:", _reference_splines.__file__)'''

DESCRIPTION = r'''# 9. Empirical CDF closure and direct floor controls

Keep the **exact weighted empirical CDF in $u$** as the sole interpolation
method. First refine it using both template and likelihood accuracy, then
validate at reserved points. Next compare **direct fills** with and without the
observable floor, at standard and finer binning, and with both nuisance-morph
orders. These are separate tests of interpolation and compression.

The baseline uses the same filtered Unity evaluation, floor, power and bin count
as notebook 08. No new samples or network training are needed. Numerical limits,
invalid fits and failed closure checks stay visible; a method is not accepted
merely because it reproduces its anchors.'''

THEORY = r'''### Exact weighted empirical CDF in the rescaled coordinate

Remove the known contraction around $\eta=1$:
$$
u_\eta(x)=\frac{\log R_\eta(x)}{\eta_{\mathrm{eff}}-1},\qquad
u_1(x)=\left.\partial_\mu\log p(x;\mu,0)\right|_{\mu=1},\qquad
z_\eta=\operatorname{sigmoid}\!\left[a(\eta_{\mathrm{eff}}-1)u_\eta\right].
$$
Here $\eta_{\mathrm{eff}}=\max(\eta,\eta_{\min})$ selects the observable only;
physical $\mu$, its yields, and the generating truth are never floored. Stable
`log1p` algebra avoids cancellation near the reference point. Original $z$ edges map to
$$
u_b(\eta)=\frac{\operatorname{logit}(z_b)}{a(\eta_{\mathrm{eff}}-1)}.
$$
The ordering reverses below $1$; endpoint bins retain infinite limits in $u$.
At $\eta=1$ we retain the exact rate-only collapse into the bin containing $0.5$.

At each anchor $\eta_k$, for each component and nominal/NI variation, retain
every weighted event, group tied coordinates, and evaluate raw prefix sums:
$$
\widehat F_{j,k}(u)=\frac{1}{N_j}\sum_i w_{ij}
\mathbf{1}\!\left[u_{\eta_k}(x_i)\leq u\right],\qquad N_j=\sum_i w_{ij}.
$$
Binary searches integrate these empirical CDFs over the **query $\eta$'s original
bin boundaries**. The endpoint convention is $[u_{\rm lo},u_{\rm hi})$ above
$1$ and $(u_{\rm lo},u_{\rm hi}]$ below $1$. All tied events stay together.
Convex linear interpolation between adjacent $\eta$ anchors preserves
nonnegative component bins, totals and coordinate monotonicity.

There is no smoothing in $u$, endpoint adjustment, positive bin floor, or
renormalization. This is exact for the finite bank at anchors, up to roundoff;
interpolation in $\eta$ remains approximate. Its step structure is allowed:
inner fits vary physical $\mu$ and the nuisance with $\eta$ frozen. Only two
full anchor CDFs are cached. The former smooth-CDF alternatives are removed.'''

VALIDATION = r'''### Likelihood-sensitive refinement, followed by independent validation

Start with moderate anchors, focused around $\eta=1$. Each round checks template
midpoints using component errors scaled by $\sqrt{H+1}$ and Poisson deviances of
physical mixtures. The $+1$ is only a diagnostic scale, never a modified mean.
Signed physical coefficients and support mismatches are retained.

**New:** the displayed scan and additional predeclared quarter-interval points
also check $\Delta q$ against direct fits, with NI fixed and profiled. Failing
likelihood points request new neighboring anchors, without turning the checked
point itself into a knot. Direct reference fits are cached across rounds.
`max_anchors` and `max_rounds` bound the work; the history records template and
likelihood failures separately. The displayed scan now participates in refinement
and must not be described as independent validation.

After refinement stops, reserved irrational-fraction points in the final
intervals test templates; selected reserved points also test fits. These values
are never used to refine the grid. The summary reports `heldout_fit_checks` and
`heldout_q_failures` separately. Additional diagnostics retain sparse bins,
common-parameter objectives, full $(\mu,\alpha_{NI})$ surfaces and MLE shifts.
All fits use directly refilled Asimov counts and a fixed observable in numerator
and denominator. A numerical optimizer failure is accepted only if a feasible
candidate certifies the nonnegative NLL bound within $10^{-10}$; raw statuses
are saved. Other failures remain invalid.

The baseline floor, power and binning are fixed during this interpolation test.
Nuisance morphing is applied after bin integration, with the original Gaussian
constraint. No method is automatically promoted to downstream inference.'''

CONTROLS = '''from poodemo.reference_splines import SplineStudySettings, run_reference_spline_comparison

SPLINE_BINS = N_BINS  # Must have been tested in the matching notebook 08 run.
SPLINE_SETTINGS = SplineStudySettings(
    initial_anchors=81,
    focus_anchors=81,
    focus_halfwidth=0.1,
    max_anchors=4801,
    max_rounds=12,
    max_bin_pull=0.05,
    max_deviance=0.005,
    q_tolerance=0.01,
    mle_tolerance=0.02,
    likelihood_refine_points=32,
    validation_intervals=96,
    validation_fit_points=8,
    surface_points=3,
    surface_mu_points=81,
    surface_alpha_points=41,
    cdf_cache_size=2,
)
print("CPU interpolation/fit study; no training. Full-event CDFs use bounded memory.")
print("Refinement checks templates and q; the scan is not a held-out validation set.")
print("Reaching a refinement limit is reported as a failed check.")'''

PLOTS = '''from poodemo.reference_spline_plots import plot_reference_spline_comparison

SCAN_XLIM = (0.02, 1.15)
SCAN_YLIM = (0, 8)
FOCUS_XLIM = (0.94, 1.06)
figures = plot_reference_spline_comparison(
    run, splines, scan_xlim=SCAN_XLIM, scan_ylim=SCAN_YLIM, focus_xlim=FOCUS_XLIM)
plt.show()'''


def spline_cells(cells):
    def md(source, ident):
        return nbformat.v4.new_markdown_cell(source.strip(), id=ident)
    def code(source, ident):
        return nbformat.v4.new_code_cell(source.strip(), id=ident)
    by_id = {c.id: c for c in cells}
    runtime = next(c for c in cells if c.source.startswith('### Runtime and persistent run'))
    bootstrap = next(c for c in cells if 'RUN_NAME =' in c.source and 'IN_COLAB' in c.source)
    bootstrap.source = fresh_bootstrap(bootstrap.source)
    setup = next(c for c in cells if 'run = create_run(ROOT,' in c.source)
    definition = next(c for c in cells if c.source.startswith('### Ratio to the fixed physical reference'))
    definition.source = definition.source.replace(
        'The matched no-floor study below measures local widths at fixed truth $\\mu_*=1$;',
        'Notebook 08\'s matched no-floor study measures local widths at fixed truth $\\mu_*=1$;')
    definition.source = definition.source.replace(
        '**`REFERENCE_RATIO_A = 100`** throughout notebooks 08–11.',
        '**`REFERENCE_RATIO_A = 100`** for the baseline comparison. The direct controls below vary the floor and bin count explicitly.')
    definition.source = definition.source.replace(
        'These settings\nrequire rerunning the downstream studies, without sample generation, retraining\nor changing the saved physics run.',
        'The baseline must match the completed notebook 08 evaluation. The controls below reuse that bank and do not change the baseline configuration.')
    runtime.source = r'''### Runtime and persistent run

Run **all cells from the top**. Notebook 09 downloads the tested source revision
into a fresh temporary Colab directory on every setup run and clears cached
`poodemo` imports. The next cell must print **Verified interpolation methods:
ecdf_u**, with **study version 3** before model loading or interpolation begins. Existing temporary
checkouts and previously uploaded ZIPs are not used by this notebook.

If GitHub requires authentication, setup uses the Colab secret `GITHUB_TOKEN`
or a hidden token prompt. Only repository Contents read access is needed.
The token is sent in an HTTP header and is not saved or printed.

Keep the existing `RUN_NAME`, `MODE`, and Drive model directory. Samples,
models and results remain under the selected Drive evaluation. This update
does not require sample generation, retraining, or a new physics run name.
The source revision and module path are printed so each run can be identified.

For this interpolation comparison, notebook 08 must already have
completed with the same Unity directory, filtering policy, floor, power and bin
count. Its saved unbinned curves are reused from that evaluation. The interpolation and direct floor controls run on CPU and can take appreciable time; progress
and bounded refinement budgets are printed. No rerun of 08 is needed when those
settings match. Local execution uses the local checkout and checks the methods
before proceeding.'''
    controls = by_id.get('spline-controls', code(CONTROLS, 'spline-controls'))
    if 'likelihood_refine_points=' not in controls.source:
        controls.source = controls.source.replace('max_anchors=2401', 'max_anchors=4801').replace('max_rounds=8', 'max_rounds=12')
        controls.source = controls.source.replace('    mle_tolerance=0.02,', '    mle_tolerance=0.02,\n    likelihood_refine_points=32,')
        if 'likelihood_refine_points=' not in controls.source:
            controls.source = controls.source.replace('SplineStudySettings(', 'SplineStudySettings(likelihood_refine_points=32,', 1)
    controls.source = controls.source.replace(
        'Each method has its own refinement limit; reaching it is reported as a failed check.',
        'Template and likelihood checks drive refinement; reaching a limit is a failed check.')
    return [md(DESCRIPTION, 'spline-title'), runtime, bootstrap,
            code(SOURCE_CHECK, 'spline-source-check'), setup,
            by_id['reference-model-intro'], by_id['reference-model-source'], definition,
            md(THEORY, 'spline-theory'), md(VALIDATION, 'spline-validation-intro'),
            controls,
            code('''splines = run_reference_spline_comparison(run, n_bins=SPLINE_BINS, settings=SPLINE_SETTINGS)
display(splines["summary"])
print("Results:", splines["output"])''', 'spline-run'),
            code('''print("Refinement history (the displayed scan is part of refinement):")
display(splines["refinement"])
print("Reserved likelihood checks, never used for refinement:")
comparison = splines["comparison"]
reserved = comparison.loc[~comparison.used_for_refinement & ~comparison.on_scan & ~comparison.is_anchor]
display(reserved.assign(abs_delta_q=reserved.delta_q.abs()).nlargest(15, "abs_delta_q"))
print("Independent template checks with the largest physical-probe deviance:")
display(splines["validation"].query("role == 'heldout'").nlargest(15, "max_deviance"))
print("Coordinate checks at eta anchors, shown separately for each method:")
for method in splines["models"]:
    print(method)
    display(splines["validation"].query(
        "role == 'anchor_coordinate_check' and method == @method").nlargest(10, "max_deviance"))
print("Off-anchor fit discrepancies (invalid fits are listed separately):")
comparison = splines["comparison"]
off_anchor = comparison.loc[~comparison.is_anchor]
display(off_anchor.assign(abs_delta_q=off_anchor.delta_q.abs()).nlargest(15, "abs_delta_q"))
display(comparison.loc[~comparison.valid | ~comparison.valid_direct | comparison.mle_shift_flag])
print("Same-parameter objective errors, including nuisance excursions:")
fixed = splines["fixed_checks"]
display(fixed.assign(abs_delta_nll=fixed.delta_nll.abs()).nlargest(15, "abs_delta_nll"))
display(fixed.loc[fixed.finite_direct != fixed.finite_interpolated])''', 'spline-tables'),
            code('''# Exact mathematical limits, independently of any fit.
for name, interpolator in splines["models"].items():
    if REFERENCE_RATIO_ETA_MIN > 0:
        plateau = interpolator(np.array([0., REFERENCE_RATIO_ETA_MIN / 2, REFERENCE_RATIO_ETA_MIN]))
        np.testing.assert_allclose(plateau, np.broadcast_to(plateau[-1], plateau.shape), rtol=0., atol=1e-10)
    collapsed = interpolator(1.)
    assert np.count_nonzero(collapsed.sum(axis=(0, 1))) == 1
    np.testing.assert_allclose(collapsed.sum(axis=-1), splines["bank"].totals, rtol=1e-12)
print("Exact floor plateau, equality collapse and component totals verified.")''', 'spline-exact-limits'),
            by_id.get('spline-plots', code(PLOTS, 'spline-plots')),
            md(r'''### Reading interpolation closure

Passing requires refinement convergence, reserved template and fit checks,
anchor closure, valid fits and matching physical domains. A finite validation
is not a coverage claim. Inspect invalid fits and MLE flags even if $q$ closes.
Near $\eta=1$, all events may occupy one bin over a small interval. Distinct
physical strengths with the same total rate can then be equally good MLEs;
compare objectives before interpreting a branch change as numerical failure.

Results and the empirical-CDF recipe are saved in
`results/reference_ratio_interpolation/`; figures are in `plots/09_interpolation/`.
The existing quadrature is fingerprinted and reused, not duplicated. Historical
spline products and notebook 08 results are not overwritten.

Direct-versus-interpolated fits test interpolation. The next study uses **only
direct fills**, so any remaining differences there cannot be blamed on CDF
interpolation. Analytical-versus-learned unbinned curves remain a separate test
of density-ratio learning.''', 'spline-interpretation'),
            code('''display(splines["morph_comparison"])
display(splines["summary"])
if not splines["summary"].passed.all():
    print("Some methods did not pass: inspect failures/limits before using their templates in inference.")
else:
    print("All methods passed the configured finite closure checks; inspect the MLE diagnostics as well.")
print("Saved tables:", splines["output"])
print("Saved figures:", run.path("plots", "09_interpolation"))''', 'spline-finish'), *floor_cells(by_id)]


FLOOR_CONTROLS = """from poodemo.reference_floor import FloorStudySettings, run_reference_floor_study, plot_reference_floor_study

FLOOR_SETTINGS = FloorStudySettings(
    resolution_multiplier=4,  # 36 and 144 bins at the notebook default.
    alpha_grid_points=17,     # Refine each resolved conditional-NI basin.
    alpha_xtol=1e-7,
)
SECONDARY_XLIM = (0.02, 0.30)
print("Direct fills: floor 0 and the baseline floor; standard/finer bins; both morph orders.")
print("This full-scan CPU control can take appreciable time, especially morphing before binning.")"""


def floor_cells(by_id):
    def md(source, ident):
        return nbformat.v4.new_markdown_cell(source.strip(), id=ident)
    def code(source, ident):
        return nbformat.v4.new_code_cell(source.strip(), id=ident)
    return [md(r"""### Direct controls: floor, bin resolution and nuisance morph order

The visible secondary-minimum discrepancy lies below the baseline floor, where
the interpolation already returns exact direct fills. We therefore compare the
full physical scan at floor $0$ and the baseline floor (normally $0.2$), with
$36$ and $144$ bins by default. At every point $\eta=\mu_0$ selects and freezes
the observable; the fit varies the original physical parameters.

For every setting compare **morph then bin** with **bin then morph**. The former
is a direct compression of the analytic unbinned model. The latter also tests
noncommutativity of binning and the exponential nuisance morph. All variants use
the same event bank, true $\mu_*=1$, physical fit bounds, and Gaussian constraint.
This is a scan-wide test, not only a local-width comparison at the truth.

For these direct Asimov models we check that the feasible generating point
attains the nonnegative KL-plus-constraint lower bound. This certifies the
unrestricted denominator without repeating an expensive two-parameter fit.
The conditional NI profile searches every minimum resolved by a coarse grid,
refines each basin, and checks endpoints. Invalid searches and boundaries are
reported. This finite search is not a proof that no narrower basin exists.
The certification is **not** applied to imperfect interpolated templates.

The first plots compare both resolutions and floors with the saved analytic
unbinned curve, for fixed and profiled NI, including a secondary-region zoom.
The final plots show conditional nuisance shifts and the magnitude of the
morph-order effect. No interpolation is used anywhere in these controls.
""", 'floor-intro'),
            by_id.get('floor-controls', code(FLOOR_CONTROLS, 'floor-controls')),
            code('''floor_study = run_reference_floor_study(
    run, n_bins=SPLINE_BINS, quadrature=splines["bank"].quad, settings=FLOOR_SETTINGS)
display(floor_study["summary"])
display(floor_study["scans"].loc[~floor_study["scans"].valid | floor_study["scans"].alpha_boundary])
print("Results:", floor_study["output"])''', 'floor-run'),
            code('''floor_figures = plot_reference_floor_study(
    run, floor_study, scan_xlim=SCAN_XLIM, scan_ylim=SCAN_YLIM, secondary_xlim=SECONDARY_XLIM)
plt.show()
print("Largest profiled effects of removing the floor:")
floor_effect = floor_study["floor_comparison"].query("systematics")
display(floor_effect.assign(abs_delta_q=floor_effect.delta_q_no_floor_minus_floor.abs()).nlargest(15, "abs_delta_q"))
print("Largest profiled effects of morph order:")
morph_effect = floor_study["morph_comparison"].query("systematics")
display(morph_effect.assign(abs_delta_q=morph_effect.delta_q_after_minus_before.abs()).nlargest(15, "abs_delta_q"))
print("Direct controls all valid:", floor_study["metadata"]["all_fits_valid"])
print("Saved figures:", run.path("plots", "09_floor_study"))''', 'floor-plots'),
            md(r"""### What these comparisons establish

- **Interpolation:** use independent empirical-CDF closure checks to judge the remaining approximation in $\eta$.
- **Floor:** compare the matched direct scans with and without it, particularly below $0.2$.
- **Resolution:** compare standard and finer bins at fixed floor and morph order. These grids need not be nested, so pointwise monotonic improvement is not guaranteed.
- **Morph order:** compare the two direct implementations before attributing a gap to information loss alone.

A nuisance-sensitive likelihood need not be fully preserved by a one-dimensional
nominal-density-ratio observable, even with fine bins and no floor. These controls
can identify a residual compression problem; they do not assume its cure.
Inspect fit statuses before interpreting any discrepancy. The controls are saved
separately in `results/reference_ratio_floor_study/` and `plots/09_floor_study/`;
the selected baseline configuration and notebook 08 products are unchanged.
""", 'floor-interpretation')]

if __name__ == '__main__':
    from build_reference_notebooks import configure_reference_cells
    path = REPO / 'notebooks/09_reference_ratio_spline_templates.ipynb'
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = configure_reference_cells(notebook.cells, 9)
    nbformat.write(notebook, path)
    print(path)
