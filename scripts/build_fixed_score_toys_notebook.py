"""Build only notebook 13: own-model toys with the score frozen at one."""
from pathlib import Path
import re

import nbformat

from build_notebooks import BOOTSTRAP, COMMON_IMPORTS
from build_reference_notebooks import MODEL_SOURCE_CODE
from build_reference_spline_notebook import fresh_bootstrap

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = 'af50d5aaad8bc1d0b2c1eff247b5e97b1b2144a1'  # Tested fixed-score toy backend.
NAME = '13_fixed_score_toys.ipynb'


def score_toy_cells():
    md = lambda text, name: nbformat.v4.new_markdown_cell(text.strip(), id=name)
    code = lambda text, name: nbformat.v4.new_code_cell(text.strip(), id=name)
    bootstrap = fresh_bootstrap(BOOTSTRAP).replace('poodemo-09-', 'poodemo-13-').replace(
        'Notebook 09', 'Notebook 13').replace('notebook 09 source', 'notebook 13 source')
    bootstrap = re.sub(r'SOURCE_COMMIT = "[0-9a-f]+"', f'SOURCE_COMMIT = "{SOURCE_COMMIT}"', bootstrap)
    model = MODEL_SOURCE_CODE.split('# Cap selected integration events per source; all events estimate acceptance.')[0]
    model += "\nrun = use_practical_run(run)\nprint('Practical evaluation:', run.evaluation_root)"
    return [md(r'''# 13. Own-model toys for the fixed learned score

Test whether the close Asimov agreement in 12 also gives similar **sampling
distributions** and global best-fit branches. Use **10,000 stat-only toys per
case and hypothesis**, at $\mu_0=0$ and $1.4$:

| Statistic | Toy generation |
|---|---|
| Fixed learned score, 36 bins | Independent Poisson counts in the fixed bins from 12 |
| Fixed learned score, 144 bins | Independent Poisson counts; resolution control |
| Analytical unbinned | Fresh continuous physical Poisson experiments, including preselection |
| Learned unbinned | Poisson resampling of the frozen selected $S$ bank with learned weights |

Every statistic is evaluated under **its own generating model**. The four
ensembles are not fits to shared pseudo-data. The score anchor stays at
$\eta=1$ for both generating hypotheses and throughout every fit:
$$
q_{\mu_0}=-2\log\frac{L(\mu_0)}{L(\widehat\mu)},\qquad
0\leq\widehat\mu\leq3.
$$
Use the full physical $\mu$ and $\sqrt\mu$ dependence, including the competing
minimum. There are no nuisance fits, one-sided truncation, template fluctuations
or retraining. Histograms and survival plots use logarithmic vertical axes.
''', 'title'), md(r'''### Run from the top after 12

Use the same run and filtered ensembles as 04/12. The completed 04 bank
supplies the cached learned ratios; 12's 36/144-bin score counts are reused.
There is no expansion beyond **10M selected events per source**, and no
ratio-network evaluation per toy.

By default, reuse matching analytical and learned unbinned toy rows from 07,
including failed fits. The notebook checks the bank, rates, normalizers,
selection, physics, reference implementation, seed and fit settings before
reusing them. It never changes 07's files. Missing reference toys are generated
in 13; a configuration mismatch stops with an explanation.

With a completed matching 07, only **40,000 new binned fits** are needed.
Without those references, there are 80,000 fits in total. A CPU runtime is
sufficient when all reference toys are reused. A GPU helps apply the frozen
selector if fresh analytical toys are needed. Those experiments are toy data,
not additional reference-bank events.
''', 'runtime'), code(bootstrap, 'bootstrap'), code(COMMON_IMPORTS, 'imports'),
        code('''MU_VALUES = (0.0, 1.4)
N_TOYS = 10_000  # Per case and per hypothesis, including in smoke mode.
BIN_COUNTS = (36, 144)
SCORE_SCALE = 0.04
EXPOSURE = 1.0
SEED = 110923  # Same reference streams as 07.
MU_BOUNDS = (0.0, 3.0)
FIT_GRID_SIZE = 129
CHECKPOINT_EVERY = 100
PROGRESS_EVERY = 100
MAX_SELECTED_PER_SOURCE = 10_000_000
REUSE_07 = True
REFERENCE_TAG = '07_signal_own_model_statonly_10M_36_9_v1'
OUTPUT_TAG = '13_fixed_score_statonly_10M_36_144_v1'
run = run.with_config(reference_ratio_denominator='S', reference_ratio_power=2.0,
    reference_ratio_eta_floor=0.0, mc_zero_max_bin=0.01, mc_zero_max_total=0.05)
print('Total requested toy rows:', len(MU_VALUES) * (len(BIN_COUNTS)+2) * N_TOYS)
''', 'settings'), code(model, 'model-source'), code('''from poodemo.signal_reference import load_signal_baseline
run, samples = load_signal_baseline(run, 36, max_selected=MAX_SELECTED_PER_SOURCE,
    local_cache=Path('/content/poodemo-13-ratios') if IN_COLAB else None)
RESULT_DIR = run.path('results', OUTPUT_TAG)
FIGURE_DIR = run.path('plots', OUTPUT_TAG)
REFERENCE_DIR = run.path('results', REFERENCE_TAG) if REUSE_07 else None
print('Selected events per source:', samples['target'])
print('Optional 07 reference checkpoints:', REFERENCE_DIR)
print('13 checkpoints:', RESULT_DIR)
''', 'load-bank'), md(r'''### Freeze the score and construct matching generators

Use precisely notebook 12's nominal-NI learned score and centering:
$$
s_1(x)=\frac{F_S+F_{SBI}-F_B}{2(F_{SBI}+F_{NI})}
-\frac{\nu_S+\nu_{SBI}-\nu_B}{2(\nu_{SBI}+\nu_{NI})},\qquad
z_1(x)=\operatorname{sigmoid}(s_1(x)/0.04).
$$
The learned fields include selected rates and frozen observable normalizers.
Use the **same fixed component histograms at every $\mu$**, and draw
$n_b\sim\mathrm{Poisson}(\lambda_b(\mu_0))$ independently. Both generator and
likelihood use the same bounded zeroing: at most 0.01 expected events per bin
and 0.05 in total. No positive floor is introduced. Larger negative yields
make a source/fit invalid and remain in the diagnostics.

The learned unbinned reference uses the same finite $S$-bank model as 07:
ratios are normalized once over that bank, not over individual toys. Draw a
Poisson event count and sample indices with replacement according to the
combined physical intensity. Integer multiplicities enter the likelihood.
Positivity is enforced over the entire bank, including unsampled points.

The analytical reference retains 07's selected-rate quadrature. Its finite
integration error remains visible through the count-mean pull; it is not an
exact integral of the neural preselection. The references are **own-model toy
ensembles from 07**, not the physical-Asimov learned curve from 03 shown in 12.
''', 'models'), code('''from poodemo.fixed_score_toys import prepare_fixed_score_toys, run_fixed_score_toys
prepared = prepare_fixed_score_toys(run, samples, mu_values=MU_VALUES,
    bin_counts=BIN_COUNTS, score_scale=SCORE_SCALE, exposure=EXPOSURE)
display(prepared['bank']['occupancy'])
display(pd.DataFrame({
    'component': ['S', 'SBI', 'B', 'NI'],
    'observable_normalizer_12': samples['normalizers'],
    'learned_toy_bank_normalizer': prepared['configuration']['reference']['learned_bank_normalizers'],
    'selected_MC_rate': prepared['learned_rates'],
    'analytical_quadrature_rate': prepared['analytic_rates'],
}))
print('Learned-bank forbidden intervals in sqrt(mu):', prepared['learned_intervals'])
''', 'prepare'), md(r'''### Run or resume the global fits

Search all allowed intervals on a 129-point grid in $\kappa=\sqrt\mu$, refine
every resolved local minimum, and check boundaries and the tested point.
The score anchor stays at 1 even when the global best fit switches branch.

Random streams depend on case, hypothesis and toy ID. Save checkpoints every
100 new toys and on an ordinary notebook interruption. Rerun this cell to
resume; increasing `N_TOYS` extends the same streams. Failed IDs are retained,
never replaced by fresh trials. Changing the scientific settings requires a
new `OUTPUT_TAG`; an input fingerprint prevents incompatible results mixing.

At $\mu_0=0$, a zero-mean bin generates zero counts exactly. The support table
also reports whether such a bin has positive expected yield at $\mu=1$.
This explains how notebook 12 can have an infinite Asimov likelihood ratio
at zero while own-model toys **generated at zero** remain well defined.
Finite-MC support zeros and positivity restrictions can still affect the
global fit, so they remain visible without smoothing or dropping bins.
''', 'execution-note'), code('''study = run_fixed_score_toys(prepared, RESULT_DIR, reference_directory=REFERENCE_DIR,
    n_toys=N_TOYS, seed=SEED, mu_bounds=MU_BOUNDS, grid_size=FIT_GRID_SIZE,
    checkpoint_every=CHECKPOINT_EVERY, progress_every=PROGRESS_EVERY)
print('Reference reuse:', study['reference_reuse'])
display(study['sources'])
display(study['summary'])
failures = study['results'].loc[~study['results'].valid]
print('Failed fits:', len(failures))
display(failures)
print('Upper-bound fits:', int(study['results'].at_upper.sum()))
print('Detailed results:', study['output'])
''', 'run-toys'), md(r'''### Test-statistic distributions and branch switching

For each generating hypothesis, compare the log-density histogram, empirical
CDF (linear scale, including the atom at zero) and logarithmic survival curve.
The horizontal $q$ axis stays linear. Also compare the global $\widehat\mu$
histogram and CDF. The line at $\widehat\mu=0.5$ is a descriptive branch split,
not a fitting restriction: the other branch is above it at truth zero and
below it at truth 1.4.

The tables include mean $q$ with its toy standard error, quantiles, mass at
zero, branch fractions with binomial errors, boundary fits, failed fits,
yield corrections and count-mean pulls. Curves are explicitly conditional on
successful fits. No asymptotic distribution is imposed. Close agreement in
12 is motivation for this test, not a guarantee of matching toy distributions
or proof that the fixed score preserves all global information.

These distributions condition on frozen learned models and finite templates;
they do not test coverage on common physical pseudo-data or training uncertainty.
''', 'plot-note'), code('''from poodemo.fixed_score_toy_plots import plot_fixed_score_toys
for name, figure in plot_fixed_score_toys(study, FIGURE_DIR):
    print(name)
    display(figure)
    plt.close(figure)
print('Figures:', FIGURE_DIR)
''', 'plots')]


def build():
    notebook = nbformat.v4.new_notebook(cells=score_toy_cells(), metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}})
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT/'notebooks'/NAME)
    print(NAME)


if __name__ == '__main__':
    build()
