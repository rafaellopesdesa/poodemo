"""Build notebook 11: four own-model null ensembles, 1000 toys each."""
import re
from pathlib import Path

import nbformat

from build_notebooks import BOOTSTRAP, COMMON_IMPORTS
from build_reference_notebooks import MODEL_SOURCE_CODE
from build_reference_spline_notebook import fresh_bootstrap

SOURCE_COMMIT = '947d151df8a5a235894e82869087b6ad4c2298a6'  # Tested backend, independent of notebook output commits.
NOTEBOOK = Path(__file__).resolve().parents[1]/'notebooks/11_test_statistic_toys.ipynb'


def toy_cells():
    md = lambda text, name: nbformat.v4.new_markdown_cell(text.strip(), id=name)
    code = lambda text, name: nbformat.v4.new_code_cell(text.strip(), id=name)
    previous = nbformat.read(NOTEBOOK, as_version=4) if NOTEBOOK.exists() else None
    old_bootstrap = next(c.source for c in previous.cells if c.cell_type == 'code') if previous else BOOTSTRAP
    bootstrap = fresh_bootstrap(old_bootstrap).replace('poodemo-09-', 'poodemo-11-').replace(
        'Notebook 09', 'Notebook 11').replace('notebook 09 source', 'notebook 11 source')
    bootstrap = re.sub(r'SOURCE_COMMIT = "[0-9a-f]+"', f'SOURCE_COMMIT = "{SOURCE_COMMIT}"', bootstrap)
    model = MODEL_SOURCE_CODE.split('# Cap selected integration events per source; all events estimate acceptance.')[0]
    model += "\nrun = use_practical_run(run)\nprint('Practical evaluation:', run.evaluation_root)"
    return [md(r'''# 11. Four test-statistic distributions under their own models

Generate **1,000 toys for each of four models**, at each of
$\mu_0=0$ and $\mu_0=1.4$: **8,000 fits in total**, all stat-only.
The four two-sided statistics are
$$
q_{\mu_0}=-2\log\frac{L(\mu_0)}{L(\widehat\mu)},\qquad 0\leq\widehat\mu\leq3.
$$
Each statistic is evaluated on toys generated from **its own model at
$\mu=\mu_0$**. These are separate null ensembles, not four fits of a common
physical sample. Their distributions may be similar without being identical.
There are no nuisance parameters, upper-limit truncation, MC-statistical
nuisances, or fluctuations of trained models/templates.

| Statistic | Toy generation |
|---|---|
| Analytical unbinned | Fresh continuous physical Poisson experiments, including preselection |
| Learned unbinned | Poisson-resample the existing selected $S$ bank with learned intensity weights |
| Moving learned observable, 36 bins | Independent Poisson counts from the frozen 36-bin means |
| Fixed learned ratios, $9\times9\times9$ bins | Independent Poisson counts from the nested 3D means |

The baseline remains **36 moving bins**, **$a=100$**, **no observable floor**,
and **9 bins on each fixed 3D axis**. No analytical density defines a learned
observable or either set of histogram yields.
''', 'title'), md(r'''### Run from the top after 08 and 10

Use the completed run and filtered Unity ensembles from 03/08/10. The setup
loads 08's baseline manifest, verifies/stages cached ratios on Colab local
disk, and reuses 10's fixed-grid count cache. It never trains networks or
extends the reference/template banks beyond their existing **10M selected
events per source**. Notebook 09's interpolation products are not needed.

A Colab GPU helps apply the frozen preselection to fresh analytical toy
events. Ratio networks are **not evaluated per toy**. Scalar likelihood fits
run on the CPU. Fresh analytical experiments are the requested toy data;
they are separate from the fixed 10M reference banks.

The default is 1,000 per case and hypothesis even in smoke mode. The settings
cell explicitly controls this budget. Results are saved on Drive; the final
plots can be regenerated from checkpoints without repeating completed toys.
''', 'runtime'), code(bootstrap, 'bootstrap'), code(COMMON_IMPORTS, 'imports'),
    code('''MU_VALUES = (0.0, 1.4)  # Either hypothesis may be run alone in a new OUTPUT_TAG.
N_TOYS = 1000  # Per model AND per hypothesis; no 200-toy pilot.
EXPOSURE = 1.0
SEED = 110923
MU_BOUNDS = (0.0, 3.0)
FIT_GRID_SIZE = 129
MAX_SELECTED_PER_SOURCE = 10_000_000
REFERENCE_RATIO_A = 100.0
REFERENCE_RATIO_ETA_MIN = 0.0
MC_ZERO_MAX_BIN = 0.01
MC_ZERO_MAX_TOTAL = 0.05
CHECKPOINT_EVERY = 5
PROGRESS_EVERY = 25
OUTPUT_TAG = "11_own_model_statonly_10M_36_9_v1"
run = run.with_config(reference_ratio_power=REFERENCE_RATIO_A,
    reference_ratio_eta_floor=REFERENCE_RATIO_ETA_MIN,
    mc_zero_max_bin=MC_ZERO_MAX_BIN, mc_zero_max_total=MC_ZERO_MAX_TOTAL)
print('Requested fits:', len(MU_VALUES) * 4 * N_TOYS)
''', 'settings'), md('''### Reuse the frozen model selection and 08 bank

Retain notebook 03's strict ensemble filtering and the same Unity bundle.
The shared baseline loader checks model identity, rates and normalization
constants. The historical five-case notebook 11 outputs are not reused.
''', 'model-note'), code(model, 'model-source'), code('''from poodemo.colab_reference import load_08_baseline
run, samples_08 = load_08_baseline(run, 36, max_selected=MAX_SELECTED_PER_SOURCE,
    local_cache=Path('/content/poodemo-11-ratios') if IN_COLAB else None)
RESULT_DIR = run.path('results', OUTPUT_TAG)
FIGURE_DIR = run.path('plots', OUTPUT_TAG)
print('Selected bank events per source:', samples_08['target'])
print('Checkpoint directory:', RESULT_DIR)
''', 'load-bank'), md(r'''### Define each generating model and matching likelihood once

For the learned case, let $r_j(x_i)$ be the saved learned component ratios
on the selected $S$ bank, with $r_S=1$. Set
$$
\overline r_j=\frac{1}{M}\sum_{i=1}^{M}r_j(x_i),\qquad
f_{ji}=\nu_j\frac{r_j(x_i)}{\overline r_j},\qquad
w_i(\mu)=\frac{1}{M}\sum_j c_j(\mu)f_{ji},
$$
where $c(\mu)=(\mu-\sqrt\mu,\sqrt\mu,1-\sqrt\mu,1)$ in $(S,SBI,B,NI)$ order.
Draw $N\sim\mathrm{Poisson}(\sum_i w_i)$ and then sample bank indices with
replacement with probability $w_i/\sum_iw_i$. Duplicate events enter the
likelihood as integer multiplicities. The constant bank measure cancels from
$q$, and the likelihood uses the same integrated rate
$\sum_iw_i=\sum_jc_j\nu_j$.

These normalizers are computed **once on the entire frozen S bank**, never
on a toy. They preserve the selected component rates and deliberately define
a self-consistent finite-bank learned model. They can differ from 03/08's
normalizers on the smaller original integration sample; those original
normalizers still define both histogram observables. Network artifacts are
unchanged. The table below shows both sets. This sampler approximates a
continuous learned density; its bank size, effective size, largest sampling
probability and expected number of unique events are recorded.

All combined learned sampling weights must be nonnegative. Signed coherent
components are never sampled independently. A negative learned intensity
makes that generating case unavailable; it is reported instead of clipped.
Fits enforce positivity on the **whole** learned bank, including unsampled
points, with the allowed parameter intervals reported.

For both histogram models, use the direct source MC templates and the same
bounded zeroing as 08/10: at most **0.01 expected events per bin and 0.05 in
total**. The corrected total means enter both the Poisson generator and the
full likelihood, including its rate term. A zero predicted mean with a
positive observed count is invalid. The 3D partition merges adjacent groups
of four bins from the existing 36-bin grid on every axis.

For analytical toys, use fresh continuous physical events and the original
analytical quadrature for the selected rate integrals. Those integrals retain
finite numerical error: this is the existing analytical reference, not an
exact analytic integral of the NN preselection. The count-mean pull in the
summary helps detect normalization discrepancies across toys.
''', 'models'), code('''from poodemo.own_model_toys import prepare_own_model_toys, run_own_model_toys
prepared = prepare_own_model_toys(run, samples_08, mu_values=MU_VALUES, exposure=EXPOSURE)
display(pd.DataFrame({
    'component': ['S', 'SBI', 'B', 'NI'],
    'observable_normalizer_08': samples_08['normalizers'],
    'learned_toy_bank_normalizer': prepared['configuration']['learned_bank_normalizers'],
    'selected_MC_rate': prepared['learned_rates'],
    'analytical_quadrature_rate': prepared['analytic_rates'],
}))
print('Learned-bank forbidden intervals in sqrt(mu):', prepared['learned_intervals'])
''', 'prepare'), md(r'''### Run the ensembles, keeping the global fit

For every toy, search all allowed intervals with a 129-point grid in
$\kappa=\sqrt\mu$, refine every resolved local minimum, and compare interval
boundaries, the tested point and both physical fit bounds. This keeps the
secondary minimum in the fit. For the moving observable, **$\eta=\mu_0$ stays
fixed throughout the denominator fit**; only the physical yields vary with
$\mu$. Nothing is frozen at 0.2.

Independent random streams depend on the case, hypothesis and toy ID.
Checkpoints are saved every five completed toys and on an ordinary notebook
interrupt. Rerunning this cell resumes; increasing `N_TOYS` extends the same
streams. A hard runtime loss can repeat at most five deterministic toys.
Scientific changes require a new `OUTPUT_TAG`; an input fingerprint blocks
mixing incompatible results. Failures are retained and never replaced by
new toys. Progress messages report the most recent toy's runtime.
''', 'execution-note'), code('''study = run_own_model_toys(prepared, RESULT_DIR, n_toys=N_TOYS, seed=SEED,
    mu_bounds=MU_BOUNDS, grid_size=FIT_GRID_SIZE,
    checkpoint_every=CHECKPOINT_EVERY, progress_every=PROGRESS_EVERY)
display(study['sources'])
display(study['summary'])
failures = study['results'].loc[~study['results'].valid]
print('Failed fits:', len(failures))
display(failures)
print('Upper-bound fits:', int(study['results'].at_upper.sum()))
print('Detailed per-toy rows and source diagnostics:', study['output'])
''', 'run-toys'), md(r'''### Compare the distributions

Overlay all four distributions separately at each generating hypothesis.
Histograms use common edges; empirical CDFs retain any atom at $q=0$, and
survival curves show the tails. The summary gives means with their toy
standard errors, quantiles, the fraction at zero, upper-bound frequency,
failures and count-mean pulls. Plots state the valid/attempted count for every
case and are explicitly conditional on successful fits.

At $\mu_0=0$, the physical boundary matters. At either point, branch switching
or finite-sample effects can change the shape. No asymptotic law is imposed.
With 1,000 toys the 99th-percentile tail contains only about ten events;
its precision should be assessed before drawing tail conclusions.

This study measures null distributions conditional on the frozen models and
finite MC banks. It does not establish closure on common physical data or
include training/template uncertainty. The analytical count-mean pull also
contains quadrature error. Large pulls, failed fits or upper-bound pileups
should be understood before interpreting a distributional difference.
''', 'plot-note'), code('''from poodemo.own_model_toy_plots import plot_own_model_toys
for name, figure in plot_own_model_toys(study, FIGURE_DIR):
    print(name)
    display(figure)
    plt.close(figure)
print('Saved plots:', FIGURE_DIR)
''', 'plots')]


if __name__ == '__main__':
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells = toy_cells()
    nbformat.write(notebook, NOTEBOOK)
