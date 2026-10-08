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
\qquad z_\eta(x)=\frac{R_\eta(x)^a}{1+R_\eta(x)^a}
=\operatorname{sigmoid}\!\left(a\log R_\eta(x)\right).
\]
The S reference cancels. Both normalizations use the same final quadrature;
this observable is analytical, not learned. The denominator is always at
**mu=1 and alpha_NI=0**, independent of the fit, anchor and Asimov truth settings.
The physical component coefficients, selected yields and Gaussian NI constraint
are unchanged. The event count retains the Poisson rate information.

Set **`REFERENCE_RATIO_A`** in the configuration cell; the default is **100**.
Any finite positive value preserves the ordering and the range [0,1]. Values
above 1 spread ratios close to 1 over more of the range; **a=1** recovers the
original transformation. Use the same value in 08 and 09. This is an observable
setting, so changing it requires rerunning 08–09, without retraining or changing
the saved physics run. It stays fixed across all eta values and likelihood fits.

For fixed NI and Asimov truth 1, the continuous ratio retains the full comparison
between eta and 1 at any separation. Finite bins can still lose
information. This pairwise statement does not imply sufficiency for all fitted
mu values or for NI profiling. Compare to **analytical unbinned**; the learned
references retain their separate closure interpretations.

At **eta=1**, R=1 and z=1/2 exactly for every event, for every allowed a.
The code preserves this collapse without jitter or score substitution.
That frozen histogram measures only the total rate. Its local shape information
vanishes, even though the Asimov test statistic for testing 1 is correctly zero.
Near 1 the distribution still contracts for any finite a.

Every histogram uses `observable_bin_edges`: **0.5 is strictly inside a bin**.
Odd bin counts use uniform edges. For even counts, the internal edges shift
left by half a nominal bin width, so the central bins keep equal widths and
0.5 is a bin center (for at least four bins). Only the endpoint bins have widths
0.5 and 1.5 times the usual width. The count and endpoints 0 and 1 are preserved.
The same edges are used in plots, direct fits, estimator diagnostics and splines.
These are event-yield plots per bin, not densities divided by bin width.
"""
SPLINE_WARNING = r"""
### Interpolation at the collapsed anchor

Use the editable **`SPLINE_BINS`** setting (default 60), with the same power
`REFERENCE_RATIO_A` as notebook 08. The saved 08 study is checked for a matching
power and bin-edge policy; rerun 08 first if either has changed. The PCHIP
method and anchor settings are retained. The grid additionally includes eta=1, even if a
changed Asimov truth differs from 1. No process yields or data are modified.

The shared bin edges place 1/2 strictly inside a bin, avoiding a numerical split
of the degenerate eta=1 events. For a finite bank, sufficiently nearby anchors
also occupy that bin: a local rate-only plateau is a real finite-binning effect.
Stretching can sharpen the template changes outside this plateau. Smooth
interpolation of finite-bank histograms can still need additional eta knots.
Data are always refilled directly at each eta.

Inspect the midpoint template errors and off-anchor delta_q values, especially
near 1. Agreement at eta=1 itself only verifies reproduction of a stored anchor.
Spline discrepancies here are interpolation errors, not evidence against the
pairwise optimality of the continuous ratio. A spline fit need not have its
minimum at the generating value when its templates differ from the exact data.
"""

REFERENCE_SETUP = '''
# Observable-only setting: keep this identical in notebooks 08 and 09.
# a=1 reproduces R/(1+R); a>1 spreads values away from 0.5.
REFERENCE_RATIO_A = 100.0
from dataclasses import replace
if not np.isfinite(REFERENCE_RATIO_A) or REFERENCE_RATIO_A <= 0:
    raise ValueError("REFERENCE_RATIO_A must be finite and positive")
run = replace(run, config={**run.config, "reference_ratio_power": float(REFERENCE_RATIO_A)})
print(f"Reference-ratio power a={REFERENCE_RATIO_A:g}; 0.5 is inside a bin")
'''

COLLAPSE_CODE = '''
from poodemo.pipeline import observable_values, prepare_quadrature, observable_bin_edges
quad = prepare_quadrature(run)
COLLAPSE_BINS = 20
collapse_edges = observable_bin_edges(COLLAPSE_BINS, "reference_ratio")
central_bin = np.searchsorted(collapse_edges, .5, side="right") - 1
assert collapse_edges[central_bin] < .5 < collapse_edges[central_bin + 1]
collapse_rows = []
for eta in [0.99, 0.999, 1.0, 1.001, 1.01]:
    z = observable_values(run, quad, eta, "reference_ratio")
    occupied = np.count_nonzero(np.histogram(z, collapse_edges)[0])
    collapse_rows.append(dict(eta=eta, a=REFERENCE_RATIO_A, n_bins=COLLAPSE_BINS,
                              z_min=z.min(), z_max=z.max(), occupied_bins=occupied,
                              central_low=collapse_edges[central_bin],
                              central_high=collapse_edges[central_bin + 1]))
collapse = pd.DataFrame(collapse_rows)
display(collapse)
assert np.all(observable_values(run, quad, 1., "reference_ratio") == .5)
assert collapse.loc[collapse.eta.eq(1.), "occupied_bins"].item() == 1
collapse.to_csv(ROOT / "results" / "reference_ratio_collapse.csv", index=False)
'''


def configure_reference_cells(cells):
    """Apply shared 08/09 controls while retaining user-edited plot settings."""
    for cell in cells:
        source = cell.source
        if cell.cell_type == 'code':
            if 'run = create_run(ROOT, mode=MODE, overrides=CONFIG_OVERRIDES)' in source:
                line = 'run = create_run(ROOT, mode=MODE, overrides=CONFIG_OVERRIDES)'
                if 'REFERENCE_RATIO_A =' not in source:
                    source = source.replace(line, line + '\n' + REFERENCE_SETUP.strip())
            if 'OBSERVABLE_LABEL =' in source:
                source = '\n'.join(
                    r'OBSERVABLE_LABEL = rf"$z_\eta=R_\eta^a/(1+R_\eta^a),\quad a={REFERENCE_RATIO_A:g}$"'
                    if line.startswith('OBSERVABLE_LABEL =') else line
                    for line in source.splitlines())
                source = source.replace('from poodemo.pipeline import prepare_quadrature, observable_values\n',
                                        'from poodemo.pipeline import prepare_quadrature, observable_values, observable_bin_edges\n')
                begin = source.find('selected_s, selected_sbi, selected_b, _ =')
                end = source.find('HIST_BINS =', begin)
                if begin >= 0 and end > begin:
                    source = source[:begin] + 'ETA_VALUES = [0.0, 0.3, 0.6, 1.0, 1.5]\n' + source[end:]
                source = source.replace('edges = np.linspace(0., 1., HIST_BINS + 1)',
                                        'edges = observable_bin_edges(HIST_BINS, OBSERVABLE)')
            if 'collapse_rows = []' in source:
                source = COLLAPSE_CODE.strip()
            if 'splines = run_spline_study(run, n_bins=' in source:
                import re
                match = re.search(r'n_bins=(\d+)', source)
                if match and 'SPLINE_BINS =' not in source:
                    source = source.replace('splines = run_spline_study',
                                            f'SPLINE_BINS = {match.group(1)}  # Must appear in notebook 08 BIN_COUNTS.\n'
                                            'splines = run_spline_study')
                    source = source.replace(f'n_bins={match.group(1)}', 'n_bins=SPLINE_BINS')
            cell.outputs = []
            cell.execution_count = None
        elif source.startswith('### Ratio to the fixed physical reference'):
            source = DEFINITION.strip()
        elif source.startswith('### Interpolation at the collapsed anchor'):
            source = SPLINE_WARNING.strip()
        elif source.startswith('### Runtime and persistent run'):
            source = source.replace(
                'For the enlarged NI bank and wider ratio classifiers, use the new default\n'
                '`RUN_NAME = "paper-distinct-s-v4"`. Rerun notebooks 01–03; then rerun 04–09',
                'When migrating from a pre-v4 run to the enlarged NI bank and wider ratio\n'
                'classifiers, use `RUN_NAME = "paper-distinct-s-v4"` and rerun 01–03, then 04–09')
            note = ('\n\nFor the reference-ratio stretch/bin-edge update alone, keep your completed\n'
                    'v4 run and rerun **08, then 09** with the same `REFERENCE_RATIO_A`.\n'
                    'No sample generation, retraining or new run name is required.')
            if 'For the reference-ratio stretch/bin-edge update alone' not in source:
                source += note
        elif source.startswith('### How the observable changes with eta'):
            source = source.replace(
                'The default anchors include the generating value, the nominal total-rate\n'
                'turnover, and the other equal-rate solution when those lie in the scan range.\n'
                'These rate landmarks need not coincide with full-likelihood extrema.',
                'The displayed anchors are **0.0, 0.3, 0.6, 1.0 and 1.5**, including\n'
                'the background-only coherent limit and a point above the reference.\n'
                'Eta=0 is supported for these distributions; score/Fisher scans stay above zero.')
        cell.source = source
    return cells


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

Repeat notebook 6 with R=r_eta/r_1 and z=R^a/(1+R^a), retaining its bin counts,
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

Non-nested bin counts need not improve monotonically. Near eta=1,
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
    cells.insert(7, code(COLLAPSE_CODE))
    write('08_reference_ratio_histograms.ipynb', configure_reference_cells(cells))

    cells = transformed_cells('07_ratio_spline_templates.ipynb', '07', '09')
    replacements = {
        '# 7.': r'''# 9. Physical-reference ratio splines and profiled fits

Repeat notebook 7 with R=r_eta/r_1 and z=R^a/(1+R^a), using editable binning and the same
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
    write('09_reference_ratio_spline_templates.ipynb', configure_reference_cells(cells))


if __name__ == '__main__':
    build_reference_notebooks()
