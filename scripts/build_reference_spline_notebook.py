"""Rebuild notebook 09 only; preserve bootstrap, model selection and controls."""
import nbformat
from build_notebooks import REPO

DESCRIPTION = r'''# 9. Reference-ratio interpolation and likelihood closure

Compare direct **36-bin** histograms with three template interpolators:
independent-bin PCHIP with adaptive $\eta$ anchors, a joint cumulative
representation in $(z,\eta)$, and a cumulative representation using the
rescaled coordinate $(u,\eta)$. Keep the observable floor at $0.2$ and power
at $100$, matching notebook 08 and its selected, filtered Unity ensembles.

The direct fills remain the reference. A method is **not accepted merely
because it is smooth or reproduces its anchors**: reserved template checks,
profiled likelihoods, fitted branches and common-parameter objectives are
reported separately. No interpolation is automatically selected for downstream
inference. This notebook requires no new samples or network training.'''

THEORY = r'''### Interpolate probability across bins and construction points

For each component $j$ and nominal/NI variation, the cumulative representation
predicts original-bin yields as
$$
H_{jb}(\eta)=N_j\,[F_j(z_{b+1},\eta)-F_j(z_b,\eta)].
$$
Monotonicity in the coordinate and fixed cumulative endpoints preserve
nonnegative bins and the component total. We use **convex linear interpolation
in $\eta$**, a degree-one spline; unconstrained bicubic interpolation would not
preserve those guarantees. In the raw-$z$ control, coordinate-spline knots
include every output edge, so the resulting counts are algebraically equal
to linear interpolation of bin yields. Coupling bins alone is not a guarantee
of improved closure.

Near $\eta=1$, remove the known contraction from the interpolation coordinate:
$$
u_\eta(x)=\frac{\log R_\eta(x)}{\eta_{\mathrm{eff}}-1},\qquad
u_1(x)=\left.\partial_\mu\log p(x;\mu,0)\right|_{\mu=1},\qquad
z_\eta=\operatorname{sigmoid}\!\left[a(\eta_{\mathrm{eff}}-1)u_\eta\right].
$$
The implementation uses `log1p` and the analytic
interference coefficients to avoid cancellation near the reference point.
The **original $z$ bin edges** map to
$$
u_b(\eta)=\frac{\operatorname{logit}(z_b)}{a(\eta_{\mathrm{eff}}-1)}.
$$
Their ordering reverses below $1$. Endpoint bins retain infinite limits in $u$.
At $\eta=1$, the histogram is filled exactly into the bin containing $0.5$;
we do not replace this rate-only experiment with fixed score bins. The exact
floor plateau is also preserved. Physical $\mu$ is never floored.

The $u$ CDF uses monotone PCHIP through weighted mid-mass values at **all distinct
event coordinates**. Only two anchor CDFs are kept in memory. Its continuous
approximation to the finite-bank empirical CDF introduces a coordinate-smoothing
error, which is tested at $\eta$ anchors as well as between them. More $\eta$
anchors cannot remove that error. Structural support endpoints and equality
collapse remain exact; no positive bin floors or pseudo-counts are introduced.'''

VALIDATION = r'''### Refine, then validate on reserved points

Each method starts with the same moderate grid, with extra anchors around
$\eta=1$ and exact anchors at the floor and equality collapse. Its own failing
midpoints select additional anchors, bounded by `max_anchors` and `max_rounds`.
The study prints progress and records the stopping reason for every method.

Refinement checks the largest component error scaled by $\sqrt{H+1}$ and
Poisson deviances for physical mixtures at several $\mu$ values and NI anchors.
The $+1$ is **only a diagnostic scale**, never an alteration of a Poisson mean.
Signed physical coefficients are retained, so unphysical interpolated mixtures
are detected. These tolerances are computational controls, not physics errors.

After adaptation stops, the grids are frozen. Reserved points at an irrational
fraction of selected final intervals test interpolation independently of the
midpoint decisions. The likelihood scan and extra reserved fits use directly
refilled Asimov counts and a frozen $\eta$ in both fits. Additional diagnostics
include small/empty bins, fixed-parameter NLL comparisons, both best-fit
parameters, boundaries and full $(\mu,\alpha_{NI})$ surfaces at the worst points.
The $u$ method also reports coordinate-smoothing errors at its own anchors.

The surface plots share the direct model's likelihood offset. They are not
independently shifted to their minima. A good $q$ comparison can coexist with a
moving MLE in a nearly flat likelihood; the MLE flag is reported separately.
At a numerical line-search failure, a feasible candidate reaching the known
nonnegative NLL lower bound within $10^{-10}$ may certify the minimum. The raw
optimizer status and this certificate are both saved. Other fit failures are
reported and fail closure; they are not assigned a test statistic.
Nuisance morphing still occurs **after** evaluating the nominal and $\pm1$
templates, with the same Gaussian NI constraint. Morph-before/bin-before
noncommutativity is measured separately using direct templates.

The floor, power and binning are held fixed throughout this comparison.
Limits or failed checks remain visible in the final summary; there is no
silent fallback to direct templates at difficult reserved points.'''

CONTROLS = '''from poodemo.reference_splines import SplineStudySettings, run_reference_spline_comparison

SPLINE_BINS = N_BINS  # Must have been tested in the matching notebook 08 run.
SPLINE_SETTINGS = SplineStudySettings(
    initial_anchors=81,
    focus_anchors=81,
    focus_halfwidth=0.1,
    max_anchors=2401,
    max_rounds=8,
    max_bin_pull=0.05,
    max_deviance=0.005,
    q_tolerance=0.01,
    mle_tolerance=0.02,
    validation_intervals=96,
    validation_fit_points=8,
    surface_points=3,
    surface_mu_points=81,
    surface_alpha_points=41,
    cdf_cache_size=2,
)
print("CPU interpolation/fit study; no training. Full-event CDFs use bounded memory.")
print("Each method has its own refinement limit; reaching it is reported as a failed check.")'''

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
    setup = next(c for c in cells if 'run = create_run(ROOT,' in c.source)
    definition = next(c for c in cells if c.source.startswith('### Ratio to the fixed physical reference'))
    definition.source = definition.source.replace(
        'The matched no-floor study below measures local widths at fixed truth $\\mu_*=1$;',
        'Notebook 08\'s matched no-floor study measures local widths at fixed truth $\\mu_*=1$;')
    runtime.source = runtime.source.split('For the 36-bin/floored-observable update')[0].rstrip()
    note = 'For this interpolation comparison'
    runtime.source = runtime.source.split(note)[0].rstrip() + '\n\n' + r'''For this interpolation comparison, refresh the temporary checkout or start a
fresh Colab runtime, then run **09 from the top**. Notebook 08 must already have
completed with the same Unity directory, filtering policy, floor, power and bin
count. Its saved unbinned curves are reused from that evaluation. The three
interpolation comparisons run on CPU and can take appreciable time; progress
and bounded refinement budgets are printed. No training or regeneration is needed.'''
    return [md(DESCRIPTION, 'spline-title'), runtime, bootstrap, setup,
            by_id['reference-model-intro'], by_id['reference-model-source'], definition,
            md(THEORY, 'spline-theory'), md(VALIDATION, 'spline-validation-intro'),
            by_id.get('spline-controls', code(CONTROLS, 'spline-controls')),
            code('''splines = run_reference_spline_comparison(run, n_bins=SPLINE_BINS, settings=SPLINE_SETTINGS)
display(splines["summary"])
print("Results:", splines["output"])''', 'spline-run'),
            code('''print("Independent template checks with the largest physical-probe deviance:")
display(splines["validation"].query("role == 'heldout'").nlargest(15, "max_deviance"))
print("CDF coordinate-smoothing checks at eta anchors:")
display(splines["validation"].query("role == 'anchor_coordinate_check'").nlargest(10, "max_deviance"))
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
            md(r'''### Reading the result

The summary accepts a method only when refinement converged, the reserved
and coordinate checks passed, and all tested off-anchor likelihood comparisons
passed. This is a finite numerical validation, not a coverage claim. Inspect
MLE flags and invalid fits even when the test statistic closes. At $\eta=1$,
multiple physical strengths may have the same total rate.

The tables and interpolation recipes are saved in
`results/reference_ratio_interpolation/` under the printed filtered evaluation;
figures are in `plots/09_interpolation/`. Each method has its own anchor archive.
The $u$ CDF recipe references the existing quadrature, whose fingerprint is
saved; no large event bank is duplicated. Legacy spline products and notebook
08 outputs are not replaced by this benchmark.

Analytical versus learned unbinned curves still test density-ratio learning.
Direct versus interpolated bins test interpolation. The saved morph-order
table tests the separate noncommutativity of binning and nuisance morphing.
Choose a downstream interpolation only after examining these comparisons.''', 'spline-interpretation'),
            code('''display(splines["morph_comparison"])
display(splines["summary"])
if not splines["summary"].passed.all():
    print("Some methods did not pass: inspect failures/limits before using their templates in inference.")
else:
    print("All methods passed the configured finite closure checks; inspect the MLE diagnostics as well.")
print("Saved tables:", splines["output"])
print("Saved figures:", run.path("plots", "09_interpolation"))''', 'spline-finish')]


if __name__ == '__main__':
    from build_reference_notebooks import configure_reference_cells
    path = REPO / 'notebooks/09_reference_ratio_spline_templates.ipynb'
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = configure_reference_cells(notebook.cells, 9)
    nbformat.write(notebook, path)
    print(path)
