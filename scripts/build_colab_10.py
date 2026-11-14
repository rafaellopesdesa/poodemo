"""Build only notebook 10: shared <=10M bank and diagnostic MC precision."""
import re
from pathlib import Path

import nbformat

from build_notebooks import BOOTSTRAP, COMMON_IMPORTS
from build_reference_notebooks import MODEL_SOURCE_CODE
from build_reference_spline_notebook import fresh_bootstrap

SOURCE_COMMIT = 'f7de151b4bfdad94cd4e144bcbdb2997ce8206f1'


def discriminator_cells():
    md = lambda text, name: nbformat.v4.new_markdown_cell(text.strip(), id=name)
    code = lambda text, name: nbformat.v4.new_code_cell(text.strip(), id=name)
    bootstrap = fresh_bootstrap(BOOTSTRAP).replace('poodemo-09-', 'poodemo-10-').replace(
        'Notebook 09', 'Notebook 10').replace('notebook 09 source', 'notebook 10 source')
    bootstrap = re.sub(r'SOURCE_COMMIT = "[0-9a-f]+"', f'SOURCE_COMMIT = "{SOURCE_COMMIT}"', bootstrap)
    model = MODEL_SOURCE_CODE.split('# Cap selected integration events per source; all events estimate acceptance.')[0]
    model += '''\n_filtered_run = run
run = use_practical_run(run)
print('Practical evaluation:', run.evaluation_root)'''
    return [md(r'''# 10. Fixed learned discriminators on the shared 10M bank

Compare fixed 1D, 2D and 3D learned discriminators with the **36-bin moving
reference ratio**, using the exact selected-event bank completed in 08.
The fixed grids contain $36$, $36^2=1296$, and $36^3=46656$ cells. A recap
compares moving 1D histograms with **36, 72, 108 and 144 bins**.
A separate factorial study compares **9, 18 and 36 bins per fixed 3D axis**
with **2.5M, 5M and 10M events per source** from the same bank.
The process-axis powers remain $a_{NI}=1$ and $a_{SBI}=a_B=2$; the moving
observable uses $a=100$ and no floor. Likelihood and information comparisons
show both analytical unbinned and notebook 03's learned unbinned reference.

**No events are generated, and no source may exceed 10M selected events.**
MC precision is measured and displayed. It never triggers more events,
adaptive rebinning, smoothing, a change of powers, or a new nuisance parameter.
''', 'title'), md('''### Run from the top after the completed 08

Use the same run and filtered Unity ensembles as 08/09. This notebook loads
08's final baseline manifest, verifies cached ratios and stages about 1.4 GB
on Colab local disk. Fixed histograms are filled in chunks and checkpointed.
There is no generation, NN evaluation or training. The calculations here are
CPU work; the runtime setup can also use an available GPU for the learned
unbinned curvature calculation.

The moving 36-bin histograms are read directly from 08, then refitted and
checked against its saved scans. The finer recap histograms read those same
cached ratios; they do not require more events. Notebook 09's interpolator is
not required. The learned unbinned scan/workspace are reused from the same
filtered notebook 03 evaluation, without evaluating the networks again.
New results have a separate output tag; completed 08/09 results are preserved.
''', 'runtime'), code(bootstrap, 'bootstrap'), code(COMMON_IMPORTS, 'imports'),
        code('''N_BINS = 36
RECAP_BIN_COUNTS = [36, 72, 108, 144]
CONVERGENCE_3D_BINS = [9, 18, 36]
CONVERGENCE_PREFIXES = [2_500_000, 5_000_000, 10_000_000]
CONVERGENCE_PROBES = [0.02, 0.054, 0.122, 0.344, 1.4]
POWERS = {"NI": 1.0, "SBI": 2.0, "B": 2.0}
REFERENCE_RATIO_A = 100.0
REFERENCE_RATIO_ETA_MIN = 0.0
MAX_SELECTED_PER_SOURCE = 10_000_000  # Hard ceiling, not a generation target.
MC_ZERO_MAX_BIN = 0.01
MC_ZERO_MAX_TOTAL = 0.05
OUTPUT_TAG = "10_colab_10M_precision_v1"
PRECISION_MUS = [0.02, 0.054, 0.122, 1.0]  # Low strength, both secondary regions, truth.
SOURCE_3D_PLOTS = []  # Optional additional atlases, e.g. ["NI", "NI_up", "NI_down"].
run = run.with_config(reference_ratio_power=REFERENCE_RATIO_A,
    reference_ratio_eta_floor=REFERENCE_RATIO_ETA_MIN,
    mc_zero_max_bin=MC_ZERO_MAX_BIN, mc_zero_max_total=MC_ZERO_MAX_TOTAL)
''', 'settings'), md('''### Reuse notebook 03's frozen ensemble selection

The same retained Unity models define every observable. The completed 08
manifest must match their identity, the power, bin count and zeroing policy.
''', 'model-note'), code(model, 'model-source'),
        code('''from poodemo.colab_reference import load_08_baseline
run, samples_08 = load_08_baseline(
    run, N_BINS, max_selected=MAX_SELECTED_PER_SOURCE,
    local_cache=Path('/content/poodemo-10-ratios') if IN_COLAB else None,
)
print('Selected events per source:', samples_08['target'])
print('Shared 08 baseline:', run.evaluation_root)
print('No generation, retraining or enlargement of this bank.')
''', 'load-bank'), md(r'''### Direct histograms and likelihoods

Freeze all fixed process coordinates and bin assignments for the full scan.
For the moving observable, freeze its partition at $\eta=\mu_0$ during each
fit. Selected rates and learned-ratio normalizers remain those used in 08.
NI is morphed after binning, with the same sparse-anchor convention as 08/09.

After combining signed components, tiny negative **total** means may be set
to zero only if no bin needs more than $0.01$ events and the summed correction
is at most $0.05$. Larger deficits remain invalid. This is the shared 08/09
policy, not a remedy for inadequate MC precision. Corrections are audited;
component templates and all uncertainty diagnostics retain raw values.
''', 'likelihood-note'), code('''from poodemo.colab_discriminators import run_colab_discriminator_study
study = run_colab_discriminator_study(run, samples_08, n_bins=N_BINS,
    powers=POWERS, recap_bins=RECAP_BIN_COUNTS, output_tag=OUTPUT_TAG)
display(study['axis_summary'])
display(study['occupancy'])
display(study['scans'].groupby(['model','systematics']).valid.agg(['size','sum']))
display(study['scans'].loc[~study['scans'].valid])
display(study['baseline_checks'].groupby('systematics').agg(
    points=('mu','size'), passed=('passed','sum'), max_abs_delta_q=('delta_q',lambda x:x.abs().max())))
print('Raw minima and zeroing at conditional fits:')
display(study['scans'].filter(regex='^(model|mu|systematics|valid|raw_minimum|negative_total|zeroed_bins|zeroed_total|rejected_negative)$'))
''', 'study'), code('''from poodemo.discriminator_study import plot_discriminator_study
FIGURE_DIR = run.path('plots', OUTPUT_TAG)
figures = plot_discriminator_study(study, output=FIGURE_DIR)
for name, figure in figures.items():
    print(name)
    display(figure)
    plt.close(figure)
display(study['information'])
print('Invalid Fisher widths remain visible even when likelihood fits are valid.')
''', 'comparison-plots'), md(r'''### Recap of 08: moving 1D bin resolution

The 36/72/108/144-bin scans use direct fills of the same 10M bank, the same
$a=100$, no floor and the same bounded-zeroing policy. The primary baseline
remains **36 bins**. The recap includes a zoom on the secondary minima and
both fixed-NI and profiled-NI comparisons. These are fixed diagnostic choices,
not an adaptive binning procedure.

The **learned unbinned** curve is explicitly the physical (analytical) Asimov
comparison saved by 03, not its learned-model self-Asimov curve. It retains
03's quadrature normalizations and original selected rates; the histogram
curves retain 08's frozen MC rates and same-template Asimov observations.
Thus it is a closure reference, not a claim of identical finite-MC experiments.
Its local-information line is half the Hessian of its actual physical-Asimov
objective at $\mu=1,\alpha_{NI}=0$, checked at two step sizes. For an imperfect
model this is local curvature, not a coverage guarantee or a Fisher identity.
No analytical density defines any learned observable or histogram yield.
''', 'recap-note'), code('''from poodemo.discriminator_precision import plot_resolution_recap
display(study['resolution_scans'].groupby(['model','systematics']).valid.agg(['size','sum']))
display(study['resolution_occupancy'])
for name, figure in plot_resolution_recap(study, output=FIGURE_DIR):
    print(name)
    display(figure)
    plt.close(figure)
''', 'recap-plots'), md(r'''### Fixed 3D binning versus finite MC statistics

Cross **9, 18 and 36 bins per coordinate** (729, 5,832 and 46,656 cells)
with **2.5M, 5M and 10M selected events per source**. These are prefixes of
the existing cached streams. No events are generated or networks evaluated.
The smaller prefixes have resumable count caches; the full prefix reuses the
existing 36-bin cache. Frozen rates, powers, ratio normalizers and member
selection are identical in every case. Changing MC size does not change the
physical exposure. Each case uses its own templates to construct its Asimov
sample at $\mu=1$, so this tests template precision, not data fluctuations.

The coarser grids merge adjacent cells of the **existing** 36-bin grid:
two fine cells per axis for 18 bins and four for 9 bins. We retain its exact
edges; generating fresh shifted grids would not produce nested partitions.
All six sources are merged before NI morphing, preserving every source total.
The primary 36-bin analysis and the moving 1D recap stay as above.

Compare the fixed-NI scans first. Exact binning loses information, while
finite-template noise can inflate apparent discrimination; agreement with the
analytical curve can result from cancellation. Profiled scans additionally
probe NI-anchor precision and the effect of morphing after merging. Nonlinear
morphing need not commute with merging. The bounded-zeroing policy stays at
$0.01$ events per bin and $0.05$ in total; invalid fits and corrections remain
visible rather than being repaired by changing these thresholds.

Merging raw components and zeroing their total do not commute. A coarse bin
can have a positive Asimov count but a nonpositive predicted mean, even when
the permitted correction is tiny. Its stat-only Poisson penalty is infinite;
a profiled fit may move NI sharply to avoid that boundary. The tables count
these **nominal-NI support mismatches**, including their Asimov yield. Crosses
mark invalid fits; hollow diamonds mark valid fits whose nominal-NI prediction
has such a mismatch. These are support artifacts to inspect, not extra physics
information. No additional floor or smoothing is applied.

Full scans, secondary-region zooms and differences from the analytical curve
show both unbinned references. Sample-size panels show $q$ against
$N_{\max}/N_{MC}$ at saved scan points nearest the requested probes. Prefixes
overlap and are **correlated**: connecting the points is descriptive, not an
independent-point fit or an extrapolation to infinite MC.
''', 'convergence-note'), code('''from poodemo.discriminator_convergence import run_convergence_study, plot_convergence_study
convergence_3d = run_convergence_study(run, study,
    bin_counts=CONVERGENCE_3D_BINS, prefixes=CONVERGENCE_PREFIXES,
    probes=CONVERGENCE_PROBES)
display(convergence_3d['probe_mapping'])
display(convergence_3d['scans'].groupby(
    ['n_bins', 'selected_per_source', 'systematics']).valid.agg(['size','sum']))
display(convergence_3d['scans'].loc[~convergence_3d['scans'].valid])
print('Nonpositive nominal prediction with positive Asimov yield:')
display(convergence_3d['scans'].loc[convergence_3d['scans'].nominal_support_mismatch_bins > 0,
    ['n_bins','selected_per_source','mu','systematics','valid','q','alpha_NI',
     'nominal_support_mismatch_bins','nominal_support_mismatch_asimov']])
display(convergence_3d['occupancy'])
display(convergence_3d['baseline_checks'].groupby('systematics').agg(
    points=('mu','size'), passed=('passed','sum'),
    max_abs_delta_q=('delta_q',lambda x:x.abs().max())))
display(convergence_3d['scans'].groupby(['n_bins','selected_per_source','systematics']).agg(
    max_zeroed_total=('zeroed_total','max'), max_zeroed_bins=('zeroed_bins','max'),
    rejected_points=('rejected_negative','sum')))
print('3D convergence tables and count archives:', convergence_3d['output'])
''', 'convergence-study'), md(r'''### Precision of the difference between hypotheses

Good relative precision on a large total yield need not imply good precision
on the much smaller change between hypotheses. At nominal NI define
$$
\Delta\nu_b=\nu_b(\mu)-\nu_b(1),\qquad
\sigma^2_{\Delta,b}=\sum_j[c_j(\mu)-c_j(1)]^2
\widehat{\mathrm{Var}}(\widehat\nu_{jb}).
$$
This includes the correlation from using the same component templates in both
hypotheses. The per-cell tables retain the raw contrast, its MC uncertainty,
and the raw Poisson contribution to $q$. The scatter panels compare
$|\Delta\nu_b|/\nu_b(1)$ with $\sigma_{\Delta,b}/\nu_b(1)$ near the secondary
region; points above the diagonal have more estimated MC noise than contrast.
Cells with undefined quantities remain explicitly recorded in the tables.

For strictly positive test and truth yields $x_b,y_b$, a second-order estimate
of the upward MC bias is also tabulated:
$$
\widehat B=\sum_{b:x_b,y_b>0}y_b\sum_j
\widehat{\mathrm{Var}}(\widehat\nu_{jb})
\left[\frac{c_j(\mu)}{x_b}-\frac{c_j(1)}{y_b}\right]^2.
$$
This is a diagnostic using estimated quantities. It can be unreliable in sparse
bins, excludes support boundaries and rate/network uncertainty, and applies
only with NI fixed. It is **never subtracted** from a scan and is not used to
select binning. No MC statistical nuisance parameters are introduced here.
''', 'contrast-note'), code('''display(convergence_3d['contrast_summary'])
CONVERGENCE_FIGURE_DIR = FIGURE_DIR / '3d_convergence_v1' / convergence_3d['identity'][:20]
for name, figure in plot_convergence_study(convergence_3d, output=CONVERGENCE_FIGURE_DIR):
    print(name)
    display(figure)
    plt.close(figure)
''', 'convergence-plots'), md(r'''### Relative MC statistical uncertainty in every cell

For independent source $j$, with $N_j$ selected events and frozen yield $\nu_j$,
let $n_{jb}$ be its count in cell $b$ and $\widehat p_{jb}=n_{jb}/N_j$. We use
the fixed-sample multinomial marginal variance, as in 08:
$$
\widehat\nu_{jb}=\nu_j\widehat p_{jb},\qquad
\widehat{\mathrm{Var}}(\widehat\nu_{jb})=
\frac{\nu_j^2\widehat p_{jb}(1-\widehat p_{jb})}{N_j-1}.
$$
At nominal NI, the physical coefficients in $(S,SBI,B,NI)$ order are
$$
c(\mu)=(\mu-\sqrt\mu,\sqrt\mu,1-\sqrt\mu,1).
$$
Independent source variances add with **squared** coefficients:
$$
\widehat\nu_b(\mu)=\sum_j c_j(\mu)\widehat\nu_{jb},\qquad
\sigma^2_{b,MC}(\mu)=\sum_j c_j(\mu)^2
\widehat{\mathrm{Var}}(\widehat\nu_{jb}).
$$
This retains the correlations implied by constructing interference as
$SBI-S-B$. The displayed relative uncertainty is
$\sigma_{b,MC}/\widehat\nu_b$ for a **positive raw** total. A cancellation can
therefore give a large relative uncertainty even with many MC events.

Gray marks no sampled support: zero observed MC events do **not** establish
zero uncertainty. Purple marks a nonpositive raw total, whose relative
uncertainty is undefined. Finite color scales have under/over-range colors;
the saved tables retain every numeric value. The 10% line and summary counts
are descriptive guides, not selection or rebinning rules.

These are conditional shape-statistics diagnostics. They exclude acceptance/
rate uncertainty, NN uncertainty, and the covariance between different bins.
They are not data Poisson errors and are not added to the likelihood. Total
plots use nominal NI; separate source tables/plots report all six independent
sources, including NI-up and NI-down. They do not claim propagated MC errors
for an arbitrary interpolated nuisance value.
''', 'precision-definition'), code('''from poodemo.discriminator_precision import save_precision_tables, plot_mc_precision
precision = save_precision_tables(study, mus=PRECISION_MUS)
display(precision)
print('Full per-cell CSV tables:', study['precision_output'])
for name, figure in plot_mc_precision(study, source_3d=SOURCE_3D_PLOTS, output=FIGURE_DIR):
    print(name)
    display(figure)
    plt.close(figure)
''', 'precision-plots'), md('''### Reading this comparison

All fixed 3D cells are displayed as labelled slices in the B coordinate; they
are not projected or merged. Moving-observable heat maps show each bin's
precision at every tested strength for all four resolutions. Compressed CSV tables contain raw yields,
absolute/relative uncertainties, source counts and explicit support status.
The six-source count archive permits additional diagnostic plots without
reading events again.

Compare valid likelihood scans first. Inspect sparse-cell precision, NI anchor
support and zeroing alongside them. A valid Asimov fit is not independent MC
closure, and an invalid Fisher width does not itself invalidate a finite
likelihood-ratio evaluation. The sample size and binning stay fixed regardless
of these diagnostic results.
''', 'interpretation')]


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1]/'notebooks/10_discriminator_histograms.ipynb'
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = discriminator_cells()
    nbformat.write(notebook, path)
