"""Build 04--07 from the current 08--11 sources, with a selected-S denominator.

Only reference-specific definitions/checks, power controls, source pins and output
identities differ. Original notebook cells and saved outputs are never modified.
"""
from copy import deepcopy
from pathlib import Path
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "6ef001cb22c678f8fbf8f82f6290344d82d4b6d7"  # Validated backend; independent of later output commits.
NAMES = {
    4: '04_signal_ratio_histograms.ipynb',
    5: '05_signal_ratio_spline_templates.ipynb',
    6: '06_signal_discriminator_histograms.ipynb',
    7: '07_signal_test_statistic_toys.ipynb',
}

DEFINITION = r'''### Ratio to the selected signal density

At nominal NI nuisance, define the learned, selected and normalized ratio
$$
R^S_\eta(x)=\frac{p(x;\eta_{\mathrm{eff}},0)}{p_S(x)},\qquad
\eta_{\mathrm{eff}}=\max(\eta,\eta_{\min}),\qquad
z^S_\eta(x)=\operatorname{sigmoid}\!\left(a\log R^S_\eta(x)\right).
$$
The baseline is **no floor**, **$a=2$**, and **36 bins**, so
$z^S_\eta=(R^S_\eta)^2/[1+(R^S_\eta)^2]$. The denominator is the selected pure-signal
shape, not the physical mixture at $\mu=1$. It stays fixed during every fit.

Use the same filtered learned component ratios $\hat r_j$, frozen normalizers
$Z_j=\langle\hat r_j\rangle_{S,\mathrm{MC}}$, and selected rates $\nu_j$:
$$
\hat f_\eta(x)=\sum_j c_j(\eta)\nu_j\hat r_j(x)/Z_j,\qquad
\nu_\eta=\sum_j c_j(\eta)\nu_j,\qquad
R^S_\eta=\frac{\hat f_\eta/\nu_\eta}{\hat f_S/\nu_S}.
$$
Here $\hat r_S=Z_S=1$, hence $\hat f_S/\nu_S=1$; the unknown common
$p_S(x)$ cancels. No analytical density is evaluated to define the observable.
Selected rates and normalizers never depend on data, toys or fitted parameters.
The extended Poisson term retains the event-rate information.

There is **no forced collapse at $\eta=1$**: the physical mixture generally
has a different shape from pure signal. Changing the reference can change
which information the scalar compression retains, but does not guarantee
sufficiency for a global fit over all physical $\mu$ or for NI profiling.
The sigmoid is invertible before binning; finite bins can lose information.

Retain the exact same half-step bin edges as 08--11, including 0.5 inside a
bin, for a controlled parallel. Physical coefficients, nuisance morphing,
fit bounds, MC prefixes and bounded total-yield zeroing are unchanged.
The explicit 0.2-floor control remains a separate study, not the baseline.
'''

THEORY = r'''### Exact weighted empirical CDF in the signal log ratio

For the signal reference use the regular coordinate
$$
u^S_\eta(x)=\log R^S_\eta(x),\qquad
z^S_\eta=\operatorname{sigmoid}(a u^S_\eta),\qquad
u_b=\frac{\operatorname{logit}(z_b)}{a}.
$$
There is no division by $\eta-1$, score-limit substitution, reversal below 1,
or equality-collapse rule. Unlike notebook 09, the log ratio does not vanish
identically at 1. The original sigmoid edges map to fixed log-ratio boundaries.

At every anchor $\eta_k$, retain the exact weighted empirical CDF for each
component and NI variation:
$$
\widehat F_{j,k}(u)=\frac{\sum_i w_{ij}\mathbf{1}[u^S_{\eta_k}(x_i)\leq u]}
{\sum_i w_{ij}}.
$$
Use binary searches with the same half-open bin convention as direct fills;
endpoint bins include the full support. Ties stay together. Interpolate
convexly between neighboring $\eta$ anchors, preserving nonnegative component
bins and their totals. With fixed $u$ boundaries, this is also linear
interpolation of the anchor bin yields. It is exact at anchors on this finite
bank, while off-anchor accuracy still needs validation.

Keep the same refinement, held-out checks, nuisance treatments and two-anchor
memory cache as 09. The extra anchors near 1 are retained for the parallel
comparison, although there is no special contraction there in this family.
'''

LIMITS = '''# Verify anchor closure and component totals, including the regular eta=1 map.
for name, interpolator in splines["models"].items():
    for eta in (0., 1., float(interpolator.etas[-1])):
        direct = splines["bank"].exact(eta)
        np.testing.assert_allclose(interpolator(eta), direct, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(direct.sum(axis=-1), splines["bank"].totals, rtol=1e-12)
    if REFERENCE_RATIO_ETA_MIN > 0:
        plateau = interpolator(np.array([0., REFERENCE_RATIO_ETA_MIN / 2, REFERENCE_RATIO_ETA_MIN]))
        np.testing.assert_allclose(plateau, np.broadcast_to(plateau[-1], plateau.shape), rtol=0., atol=1e-10)
print("Direct/ECDF anchors, component totals and any enabled floor plateau verified.")'''


def references(source):
    """Rewrite notebook references, never numerical settings or random seeds."""
    source = re.sub(r'(?i)(notebooks?\s+)(08|09|10|11)\b',
                    lambda m: m[1]+f'{int(m[2])-4:02}', source)
    for old, new in [('08/09', '04/05'), ('08/10', '04/06'), ('03/08/10', '03/04/06'),
                     ('08–11', '04–07'), ('08 first, then 09', '04 first, then 05'),
                     ('08 and 09', '04 and 05'), ('08 and 10', '04 and 06'),
                     ("10's fixed-grid", "06's shared fixed-grid")]:
        source = source.replace(old, new)
    # Bare 08/09 refer to notebook numbers in these sources; do not replace 10M.
    source = re.sub(r'(?<![\w.])0([89])(?!\w|\.\d)', lambda m: '0'+str(int(m[1])-4), source)
    return source


def signal_cells(number):
    original = number + 4
    path = next((ROOT/'notebooks').glob(f'{original:02}_*.ipynb'))
    cells = deepcopy(nbformat.read(path, as_version=4).cells)
    for c in cells:
        s = c.source
        if c.cell_type == 'code' and 'SOURCE_COMMIT = ' in s:
            # Avoid touching bootstrap behavior, controls or package installation.
            s = re.sub(r'SOURCE_COMMIT = [\'"][0-9a-f]+[\'"]', f'SOURCE_COMMIT = "{SOURCE_COMMIT}"', s)
            s = s.replace(f'poodemo-{original:02}-', f'poodemo-{number:02}-')
            s = s.replace(f'Notebook {original:02}', f'Notebook {number:02}')
            s = s.replace(f'notebook {original:02} source', f'notebook {number:02} source')
        else:
            s = references(s)
            s = s.replace('REFERENCE_RATIO_A = 100.0', 'REFERENCE_RATIO_A = 2.0')
            s = s.replace('$a=100$', '$a=2$').replace('**$a=100$**', '**$a=2$**')
            s = s.replace('samples_08', 'samples_04').replace('normalizer_08', 'normalizer_04')
            s = s.replace('poodemo-09-ratios', 'poodemo-05-ratios').replace('poodemo-10-ratios', 'poodemo-06-ratios').replace('poodemo-11-ratios', 'poodemo-07-ratios')
            s = s.replace('from poodemo.colab_reference import load_08_baseline, ColabTemplateBank',
                          'from poodemo.signal_reference import load_signal_baseline\nfrom poodemo.colab_reference import ColabTemplateBank')
            s = s.replace('from poodemo.colab_reference import load_08_baseline',
                          'from poodemo.signal_reference import load_signal_baseline')
            s = s.replace('load_08_baseline(', 'load_signal_baseline(')
            s = s.replace('from poodemo.colab_reference import publish_08_baseline',
                          'from poodemo.signal_reference import publish_signal_baseline')
            s = s.replace('publish_08_baseline(', 'publish_signal_baseline(')
            for old,new in [('08_', '04_'), ('09_interpolation', '05_interpolation'),
                            ('10_colab_10M_precision_v1','06_signal_colab_10M_precision_v1'),
                            ('11_own_model_statonly_10M_36_9_v1','07_signal_own_model_statonly_10M_36_9_v1')]:
                s = s.replace(old,new)
            if c.cell_type == 'code' and 'REFERENCE_RATIO_A = ' in s:
                s = s.replace('run = run.with_config(', 'run = run.with_config(reference_ratio_denominator="S", ', 1)
            if c.cell_type == 'markdown' and s.startswith('# '):
                s = re.sub(r'^# \d+\.', f'# {number:02}.', s)
            if c.id == 'mc-review-policy':
                s = s.replace('This policy does not freeze $\\eta$ or remove its exact $\\eta=1$ collapse.',
                              'This policy does not freeze $\\eta$; the signal reference has no forced collapse at 1.')
        c.source = s
        if c.cell_type == 'code':
            c.outputs = []
            c.execution_count = None
        c.metadata = {}  # Drop Colab execution/output IDs and widget state.
    by_id = {c.id:c for c in cells}
    def put(ident, source): by_id[ident].source = source.strip()
    def change(ident, old, new):
        assert old in by_id[ident].source, (number, ident, old)
        by_id[ident].source = by_id[ident].source.replace(old,new)
    if number in (4,5):
        put('cell-004' if number == 4 else 'cell-005', DEFINITION)
    if number == 4:
        configure_04_review(by_id)
        change('cell-000', 'Learned reference-ratio', 'Learned signal-reference ratio')
        change('cell-000', 'Generate fresh independent simulation **in Colab**',
               'Reuse the matching completed 08 event/prediction cache, or generate simulation **in Colab**')
        put('cell-007', '''ranges = templates.collapse_table(CHECK_ETAS, N_BINS)
display(ranges)
# eta=1 is a regular signal-reference observable; do not force one occupied bin.
np.testing.assert_allclose(templates.anchors(1.,N_BINS)[0].sum(axis=-1), samples['rates'][:4], rtol=1e-12)
ranges.to_csv(run.path('results','signal_ratio_ranges.csv'),index=False)''')
        change('cell-005', 'The separate exact collapse at $\\eta=1$ remains.',
               'At $\\eta=1$ the observable remains the physical mixture divided by pure signal.')
        change('cell-014', 'At $\\eta=1$ the histogram retains only rate information.',
               'There is no imposed loss of all shape information at $\\eta=1$.')
        change('cell-017', 'At $\\eta=1$ this is a rate-only measurement: multiple physical $\\mu$ values\ncan have equal rates.',
               'The signal reference has no forced rate-only point. Multiple physical $\\mu$ values\ncan nevertheless compete in a coarse or information-losing compression.')
        change('mc-review-power', 'POWER_VALUES = [50., 100., 150.]', 'POWER_VALUES = [1.0, 2.0, 3.0]')
        change('mc-review-power-note', '$a=50,100,150$', '$a=1,2,3$')
        change('colab-sampling-setup', 'mc_run = run',
               'assert 2 <= SELECTED_EVENTS_PER_SOURCE <= 10_000_000\nmc_run = run')
        for ident in ('cell-006','cell-015','cell-019'):
            by_id[ident].source = by_id[ident].source.replace("PLOT_PREFIX = '08'", "PLOT_PREFIX = '04'")
    elif number == 5:
        put('spline-theory', THEORY)
        put('spline-exact-limits', LIMITS)
        change('spline-interpretation',
               'Near $\\eta=1$, all events may occupy one bin over a small interval. Distinct\nphysical strengths with the same total rate can then be equally good MLEs;\ncompare objectives before interpreting a branch change as numerical failure.',
               'There is no forced collapse near $\\eta=1$. Compression and finite MC can\nstill create competing minima; compare objectives before interpreting a\nbranch change as numerical failure.')
        by_id['spline-source-check'].source += '\nfrom poodemo.signal_reference import signal_log_ratio\nprint("Signal-reference coordinate available:", signal_log_ratio.__module__)'
    elif number == 7:
        change('prepare', 'from poodemo.own_model_toys import prepare_own_model_toys, run_own_model_toys',
               'from poodemo.own_model_toys import run_own_model_toys\nfrom poodemo.signal_reference_toys import prepare_signal_model_toys')
        change('prepare', 'prepare_own_model_toys(', 'prepare_signal_model_toys(')
        change('runtime', 'Keep the same `OUTPUT_TAG`: the completed 1,000-toy baseline ensembles are\nreused and extended. Only the three additional resolutions start new ensembles.',
               'Use the separate signal-reference `OUTPUT_TAG`. These ensembles start new\ncheckpoints; rerunning resumes only matching 07 results. Existing 11 toys are preserved.')
        change('execution-note', 'The original four-case backend and input fingerprints are preserved, so the\nprevious 1,000 toys resume under the same tag.',
               'The original four-case fitting and sampling engines are reused, with a new\nsignal-reference model fingerprint and output directory.')
        change('resolution-note', 'The scalar ratio at $\\eta=\\mu_0$ is designed to compare the tested point with\n$\\mu=1$.',
               'The scalar ratio at $\\eta=\\mu_0$ compares the tested physical shape with\nthe selected pure-signal shape.')
    overview = r'''### Parallel signal-reference study

This notebook parallels **SOURCE**. It uses $p(x;\eta)/p_S(x)$ with **$a=2$**
in place of $p(x;\eta)/p(x;1)$ with $a=100$. Both are normalized selected
shapes. The fixed process-ratio axes already use $p_S(x)$ and are unchanged.
The scalar observable remains frozen throughout each physical fit.

Keep the same run name, filtered ensembles, MC campaign, bin edges, physical
model, fit settings and study sequence. No additional reference events are
needed when the matching 10M cache exists. Notebook 04 fills new histograms
from those cached predictions and publishes `colab_04_baseline.json`;
05--07 require that completed manifest. Outputs and checkpoints are separate
from 08--11. Run **04, then 05, 06 and 07**; 07 uses direct templates and does
not require successful interpolation in 05.
'''.replace('SOURCE',f'{original:02}')
    if number == 4:
        overview += '\nPower controls compare $a=1,2,3$, retaining the same relative\nvariations around the new baseline. The event-cache seed and chunk settings\nmatch 08; keep them unchanged to reuse exactly the same events.\n'
    cells.insert(1, nbformat.v4.new_markdown_cell(overview.strip(), id='signal-parallel'))
    return cells



def configure_04_review(cells):
    cells['colab-sampling-setup'].source += """
# Additional resolutions ONLY for the information plot at eta=mu=1.
# Keep the primary 36-bin fit and the full likelihood-scan bin counts unchanged.
INFORMATION_BIN_COUNTS = [4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 48, 60, 72, 96, 108, 144]
INFORMATION_ETA = 1.0
"""
    cells['cell-009'].source = r"""### Analytical and learned unbinned benchmarks

Retain the analytical unbinned reference and add notebook 03's **learned
unbinned curve on the physical Asimov sample**, using the matching filtered
ensemble. Both appear in the likelihood comparisons, including the secondary
minimum zoom. The saved learned-model self-Asimov curve is a different study.

The binned curves use same-template MC Asimov counts and their frozen selected
rates; the learned reference retains 03's quadrature normalization. Thus these
curves diagnose different sources of disagreement but do not form an exact
decomposition into learning and binning error. Neither reference is used to
choose bin edges or fit $a$ to the analytical answer. No networks are retrained.
""".strip()
    cells['cell-010'].source += """
from poodemo.discriminator_unbinned import learned_reference
learned_unbinned, learned_widths, learned_provenance = learned_reference(run, scan_grid)
learned_unbinned.to_csv(run.path('results','signal_learned_unbinned_reference.csv'),index=False)
save_json(run.path('results','signal_learned_unbinned_reference.json'),learned_provenance)
display(learned_unbinned)
print('Learned comparison: notebook 03, physical Asimov; original frozen normalization.')
"""
    # Use the same reference styles in all likelihood comparisons.
    cells['cell-012'].source = cells['cell-012'].source.replace(
        '        ax.plot(group[x_column], group[y_column].where(group["valid"]) if "valid" in group else group[y_column], marker=".", ms=3, label=" | ".join(parts))',
        '        model_label = str(group["model"].iloc[0]) if "model" in group else ""\n'
        '        style = ({"color":"black", "linestyle":"--", "linewidth":1.8} if model_label == "analytic unbinned"\n'
        '                 else {"color":"#CC79A7", "linestyle":"-.", "linewidth":1.8} if model_label.startswith("learned unbinned") else {})\n'
        '        ax.plot(group[x_column], group[y_column].where(group["valid"]) if "valid" in group else group[y_column], marker=".", ms=3, label=" | ".join(parts), **style)')
    # Leave the analytical-only saved unbinned table intact: downstream helpers
    # explicitly consume it as the oracle reference. Append only in comparisons.
    cells['cell-013'].source = cells['cell-013'].source.replace(
        'scans = binning["scans"]',
        'scans = pd.concat([binning["scans"], learned_unbinned.loc[~learned_unbinned.systematics]], ignore_index=True)\n'
        'scans.to_csv(run.path("results","signal_likelihood_comparison.csv"),index=False)')
    cells['cell-013'].source += r"""
# Same curves, with a dedicated view of the secondary minimum.
plot_scans(shown_scans, scan_title + ': secondary minimum', '04_binning_scans_secondary.pdf',
           xlim=(0.02,0.30), ylim=None);
"""
    for ident in ('colab-convergence','mc-review-power'):
        source=cells[ident].source
        needle="    ax.plot(ref.mu,ref.q.where(ref.valid),'k--',label='Analytical unbinned')"
        assert needle in source
        selection='learned_unbinned.systematics.eq(syst)'
        if ident=='mc-review-power':selection+=' & learned_unbinned.mu.le(.3)'
        source=source.replace(needle,needle+"\n    learned_ref=learned_unbinned.loc["+selection+"].sort_values('mu')\n"
            "    ax.plot(learned_ref.mu,learned_ref.q.where(learned_ref.valid),color='#CC79A7',ls='-.',\n"
            "            label='Learned unbinned (03, physical Asimov)')")
        cells[ident].source=source
    cells['cell-014'].source += r"""

The next plot has a separate dense sweep of **18 bin counts, from 4 to 144**,
at fixed $\eta=\mu=1$, with NI fixed. It reads the same cached predictions;
there are no new events and no extra likelihood scans at these resolutions.
Nested MC prefixes expose sensitivity to rare sampled bins.

For each model, the extended information separates into
$$
I=\sum_b\frac{(\partial_\mu\nu_b)^2}{\nu_b},\qquad
I_{\rm rate}=\frac{(\partial_\mu\sum_b\nu_b)^2}{\sum_b\nu_b},\qquad
I_{\rm shape}=I-I_{\rm rate}.
$$
Each model subtracts its own rate information before the shape comparison.
The denominators of the plotted fractions remain analytical unbinned Fisher
information. The learned physical-Asimov curvature is not automatically Fisher
information for an imperfect model and is not substituted for it.

An empty bin with zero derivative contributes zero. A zero mean with nonzero
derivative makes the usual Fisher expression undefined; bounded zeroing does
not restore regularity. Such resolutions appear as red crosses at the bottom,
with source counts in the diagnostic table, rather than disappearing silently
or being assigned an artificial information value. No means or derivatives are
floored. These local quantities do not determine the global secondary minimum.
"""
    cells['cell-015'].source = """from poodemo.signal_information import run_signal_information_study, plot_signal_information
information_study = run_signal_information_study(run, samples,
    bin_counts=INFORMATION_BIN_COUNTS, eta=INFORMATION_ETA, prefixes=MC_PREFIXES)
information = information_study['information']
display(information)
display(information.groupby('selected_per_source').information_valid.agg(['size','sum']))
invalid_information = information_study['diagnostics'].query('nonregular or raw_mean < 0')
print('Bins responsible for undefined Fisher points (no artificial floors):')
display(invalid_information)
figure = plot_signal_information(information_study,
    selected_per_source=SELECTED_EVENTS_PER_SOURCE, output=run.path('plots'))
display(figure)
plt.close(figure)
print('Full information and per-source bin diagnostics:', information_study['output'])
""".strip()

def build():
    for number,name in NAMES.items():
        nb = nbformat.v4.new_notebook(cells=signal_cells(number), metadata={
            'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'},
            'language_info': {'name':'python'},
        })
        nbformat.validate(nb)
        nbformat.write(nb, ROOT/'notebooks'/name)
        print(name)


if __name__ == '__main__':
    build()
