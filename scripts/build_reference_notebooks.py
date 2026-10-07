"""Build 08/09 from the current 06/07 notebooks, preserving their study settings.

Only new notebooks are written; source notebooks and saved outputs are untouched.
"""
import copy
import nbformat
from build_notebooks import REPO, md, code, write

DEFINITION = r"""
### Ratio to the fixed physical reference

At nominal NI nuisance, use the analytical **selected, normalized** densities:
\[
R_\eta(x)=\frac{r(x;\eta)}{r(x;1)}
=\frac{p(x;\eta,0)}{p(x;1,0)}
=\frac{D_\eta(x)/\Lambda_\eta}{D_1(x)/\Lambda_1},
\qquad z_\eta(x)=\frac{R_\eta(x)}{1+R_\eta(x)}.
\]
The S reference cancels. Both normalizations use the same final quadrature;
this observable is analytical, not learned. The denominator is always at
**mu=1 and alpha_NI=0**, independent of the fit, anchor and Asimov truth settings.
The physical component coefficients, selected yields and Gaussian NI constraint
are unchanged. The event count retains the Poisson rate information.

For fixed NI and Asimov truth 1, the continuous ratio retains the full comparison
between eta and 1 at any separation. Uniform finite bins can still lose
information. This pairwise statement does not imply sufficiency for all fitted
mu values or for NI profiling. Compare to **analytical unbinned**; the learned
references retain their separate closure interpretations.

At **eta=1**, R=1 and z=1/2 exactly for every event. The code preserves this
collapse, without jitter, score substitution, adaptive bins, or rescaling.
That frozen histogram measures only the total rate. Its local shape information
vanishes, even though the Asimov test statistic for testing 1 is correctly zero.
Near 1 the distribution contracts, so uniform bins can remain very coarse.
"""
SPLINE_WARNING = r"""
### Interpolation at the collapsed anchor

Use **20 uniform bins**, as in notebook 7, with the same PCHIP method and
anchor settings. The grid additionally includes exactly eta=1, even if a
changed Asimov truth differs from 1. No process yields or data are modified.

Because 1/2 is a bin edge with this even bin count, events can exchange the two
central bins when eta crosses 1. At exactly 1 they all enter the bin containing
1/2 under NumPy's histogram convention. Individual bin yields therefore need
not be continuous at this anchor. **A smooth spline cannot represent that
collapse and its one-sided limits exactly.** We intentionally keep the same
spline construction to expose this limitation, rather than replacing the
observable. Data are always refilled directly at each eta.

Inspect the midpoint template errors and off-anchor delta_q values, especially
near 1. Agreement at eta=1 itself only verifies reproduction of a stored anchor.
Spline discrepancies here are interpolation errors, not evidence against the
pairwise optimality of the continuous ratio. A spline fit need not have its
minimum at the generating value when its templates differ from the exact data.
"""


def transformed_cells(name, old_prefix, new_prefix):
    cells = copy.deepcopy(nbformat.read(REPO / 'notebooks' / name, as_version=4).cells)
    for cell in cells:
        cell.source = (cell.source.replace('"ratio"', '"reference_ratio"')
                       .replace("'ratio'", "'reference_ratio'")
                       .replace('direct ratio histogram', 'direct reference_ratio histogram')
                       .replace('fixed ratio histogram', 'fixed reference_ratio histogram')
                       .replace('spline ratio histogram', 'spline reference_ratio histogram')
                       .replace('ratio_', 'reference_ratio_')
                       .replace('reference_reference_ratio', 'reference_ratio')
                       .replace(old_prefix + '_', new_prefix + '_')
                       .replace("'" + old_prefix + "'", "'" + new_prefix + "'")
                       .replace('"' + old_prefix + '"', '"' + new_prefix + '"')
                       .replace('r_\\eta/(1+r_\\eta)', 'R_\\eta/(1+R_\\eta)'))
        if cell.cell_type == 'code':
            if 'OBSERVABLE_LABEL =' in cell.source:
                cell.source = '\n'.join(
                    'OBSERVABLE_LABEL = ' + repr(r'Physical-reference ratio $z_\eta=R_\eta/(1+R_\eta)$')
                    if line.startswith('OBSERVABLE_LABEL =') else line
                    for line in cell.source.splitlines())
            cell.outputs = []
            cell.execution_count = None
    return [c for c in cells if c.source.strip()]


def build_reference_notebooks():
    cells = transformed_cells('06_ratio_histograms.ipynb', '06', '08')
    replacements = {
        '# 6.': r'''# 8. Histograms of the ratio to mu=1

Repeat notebook 6 with R=r_eta/r_1 and z=R/(1+R), retaining its bin counts,
scan curves, axis controls, distributions and estimator diagnostics. Reuse the
completed run from notebooks 1–3: no regeneration or training is needed.
Results use `reference_ratio_` and figures `08_`, preserving studies 04–07.''',
        '### The ratio of normalized': DEFINITION,
        '### Freeze the observable': r'''### Freeze the observable inside each fit

For a test of mu_0, construct z at eta=mu_0 and nominal NI, then freeze event
assignments, edges and Asimov counts while fitting mu and, where enabled, NI.
The denominator density stays at (1,0); it is never replaced by the best fit.
The binning study fixes NI, as in notebook 6. At Asimov truth 1, infinitely fine
ratio bins recover the pairwise unbinned test, but not necessarily local Fisher
information at the tested eta. The two diagnostics answer different questions.''',
        '### Distinguish ratio compression': r'''### Finite-separation fidelity versus local information

The likelihood scan tests eta against the fixed Asimov truth 1. Its continuous
nominal ratio is sufficient for that pair. The Fisher table instead evaluates
local information at mu=eta: pairwise sufficiency does not guarantee retaining
that derivative. In particular, the frozen eta=1 observable has zero shape
information and one occupied bin. The extended information still includes rate.

Non-nested uniform bin counts need not improve monotonically. Near eta=1,
finite binning can obscure the continuous-ratio advantage. Compare the scan
with `analytic unbinned`, separately from the learned references.''',
        'This estimator study': r'''Keep the same estimator resolution as notebook 6: the finest bin count in
the saved run configuration. The BIN_COUNTS override only affects the coarsening
study. The truth remains fixed while eta changes. At eta=1 this is a rate-only
measurement; multiple mu values may have equal rates, so a different fitted
branch need not indicate a failed fit. Local widths do not describe that global
ambiguity. Invalid widths are reported rather than floored.''',
        'Next:': 'Next: **09_reference_ratio_spline_templates.ipynb**. Earlier outputs remain intact.',
    }
    for cell in cells:
        if cell.cell_type == 'markdown':
            for key, value in replacements.items():
                if cell.source.startswith(key): cell.source = md(value).source; break
    # A diagnostic of the exact collapse and finite-bin occupation, not a
    # substitute observable. Runs independently of the fits after setup.
    cells.insert(7, code('''
from poodemo.pipeline import observable_values, prepare_quadrature
quad = prepare_quadrature(run)
collapse_rows = []
for eta in [0.99, 0.999, 1.0, 1.001, 1.01]:
    z = observable_values(run, quad, eta, "reference_ratio")
    occupied = np.count_nonzero(np.histogram(z, np.linspace(0, 1, 21))[0])
    collapse_rows.append(dict(eta=eta, z_min=z.min(), z_max=z.max(), occupied_bins_20=occupied))
collapse = pd.DataFrame(collapse_rows)
display(collapse)
assert np.all(observable_values(run, quad, 1., "reference_ratio") == .5)
collapse.to_csv(ROOT / "results" / "reference_ratio_collapse.csv", index=False)
'''))
    write('08_reference_ratio_histograms.ipynb', cells)

    cells = transformed_cells('07_ratio_spline_templates.ipynb', '07', '09')
    replacements = {
        '# 7.': r'''# 9. Physical-reference ratio splines and profiled fits

Repeat notebook 7 with R=r_eta/r_1 and z=R/(1+R), using 20 bins and the same
nominal NI construction, physical model and Gaussian constraint. Run notebook
8 first using the same completed Drive run. Tables use `reference_ratio_` and
figures `09_`; earlier studies are preserved.''',
        'Here the frozen scalar': DEFINITION,
        '### Hold the spline': SPLINE_WARNING,
        '### Read the comparisons': r'''### Read the comparisons separately

- Direct versus spline tests template interpolation, including the collapse.
- Coarse versus fine direct histograms tests finite binning.
- NI fixed versus profiled tests nuisance effects; the nominal scalar is not
  guaranteed sufficient for profiling. The Gaussian auxiliary term remains.
- The morph-order diagnostic separates bin-level nuisance interpolation from
  integrating the unbinned nuisance morph.
- Analytical unbinned is the physical reference; the two learned curves test
  external closure and internal model-Asimov consistency separately.

At truth 1 the continuous nominal ratio preserves the fixed-NI pairwise test.
It does not guarantee full local Fisher information at every eta or sufficient
information for all nuisance alternatives. The physical-yield plot at frozen
eta=1 contains only one occupied bin: it shows the rate-only experiment.''',
        '### Further experiments': r'''### Interpretation

Compare notebooks 06–09 at matching resolutions. In particular, inspect the
contracting distributions, rate-only local width at eta=1, and spline errors
near that anchor. These are consequences of using a physical reference that
coincides with one construction point. Coverage still requires calibrated
pseudo-experiments beyond the Asimov comparisons.''',
    }
    for cell in cells:
        if cell.cell_type == 'markdown':
            for key, value in replacements.items():
                if cell.source.startswith(key): cell.source = md(value).source; break
    cells.insert(8, code('''
# Explicitly surface the singular-anchor interpolation diagnostic.
print("Largest off-anchor bin-fraction errors (inspect eta near 1):")
display(splines["validation"].nlargest(12, "max_fraction_error"))
print("Direct-versus-spline likelihood checks near eta=1:")
display(splines["comparison"].loc[abs(splines["comparison"]["mu"] - 1.) < .05])
'''))
    write('09_reference_ratio_spline_templates.ipynb', cells)


if __name__ == '__main__':
    build_reference_notebooks()
