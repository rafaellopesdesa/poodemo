"""Build only notebook 12: a learned score frozen at mu=1."""
from pathlib import Path
import re

import nbformat

from build_notebooks import BOOTSTRAP, COMMON_IMPORTS
from build_reference_notebooks import MODEL_SOURCE_CODE
from build_reference_spline_notebook import fresh_bootstrap

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = '36091e8eb77c8e5fda8654ea161879b99c40ce18'  # Tested fixed-score backend.
NAME = '12_fixed_score_histograms.ipynb'


def score_cells():
    md = lambda text,name: nbformat.v4.new_markdown_cell(text.strip(),id=name)
    code = lambda text,name: nbformat.v4.new_code_cell(text.strip(),id=name)
    bootstrap = fresh_bootstrap(BOOTSTRAP).replace('poodemo-09-','poodemo-12-').replace(
        'Notebook 09','Notebook 12').replace('notebook 09 source','notebook 12 source')
    bootstrap = re.sub(r'SOURCE_COMMIT = "[0-9a-f]+"',f'SOURCE_COMMIT = "{SOURCE_COMMIT}"',bootstrap)
    model = MODEL_SOURCE_CODE.split('# Cap selected integration events per source; all events estimate acceptance.')[0]
    model += "\nrun = use_practical_run(run)\nprint('Practical evaluation:', run.evaluation_root)"
    return [md(r'''# 12. A fixed learned score: local versus global inference

Freeze the score observable at **$\mu=1$, nominal NI** and compare its binned
likelihood with the analytical and learned unbinned curves from notebook 03.
The test is whether a locally informative observable remains adequate near
zero and around the secondary minimum. Agreement away from the anchor is
measured, not assumed.

Use **36 bins**, with **144 bins** as a simple resolution control. Both
stat-only and NI-profiled curves use the same fixed event partition throughout
the entire scan. The physical likelihood keeps the exact $\mu$ and $\sqrt\mu$
coefficients; it is not replaced by a Taylor expansion.

Reuse the completed 04 **10M selected-event bank**, its filtered learned ratios,
and the matching saved unbinned references. No new events, training, interpolation
study or toys are required. This notebook replaces the retired notebook-12
diagnostic study; its old outputs remain untouched.
''','title'),md('''### Run from the top

Use the same `RUN_NAME`, `MODE`, overrides, Unity bundle and ensemble filter as
03 and the completed 04. The setup downloads a fresh pinned source checkout.
Cached ratios are verified and staged on Colab local disk; histogram filling
and the scalar nuisance fits are CPU work. A CPU Colab runtime is sufficient.
Run 04's final export cell first if its baseline manifest is not present.
''','runtime'),code(bootstrap,'bootstrap'),code(COMMON_IMPORTS,'imports'),
        md(r'''### The score convention and sigmoid

Write $F_j(x)=\nu_jp_j(x)$ for a selected component intensity and use
$$
f(x;\mu)=(\mu-\sqrt\mu)F_S(x)+\sqrt\mu F_{SBI}(x)
+(1-\sqrt\mu)F_B(x)+F_{NI}(x).
$$
The per-event intensity score at nominal NI is
$$
t_1(x)=\left.\partial_\mu\log f(x;\mu)\right|_1
=\frac{1}{2}\frac{1+F_{SBI}/F_S-F_B/F_S}{F_{SBI}/F_S+F_{NI}/F_S}
=\frac{F_S+F_{SBI}-F_B}{2(F_{SBI}+F_{NI})}.
$$
The implementation includes this factor of $1/2$ before centering and applying
the sigmoid, so the observable uses the derivative with respect to $\mu$.
When the component ratios refer to normalized PDFs, their selected rates must
be included. The full extended-likelihood score is
$-\nu'_1+\sum_i t_1(x_i)$, rather than just one event's score.

For the histogram coordinate, retain the earlier centered shape-score convention:
$$
s_1(x)=t_1(x)-\frac{\nu'_1}{\nu_1},\qquad
\nu'_1=\frac{\nu_S+\nu_{SBI}-\nu_B}{2},\qquad
\nu_1=\nu_{SBI}+\nu_{NI},\qquad
z_1(x)=\operatorname{sigmoid}\!\left(\frac{s_1(x)}{0.04}\right).
$$
The constant shift and factor of two preserve event ordering; the sigmoid scale
controls finite-bin resolution. The extended Poisson term retains the count
information. The coordinate, centering and scale never change during a fit.

Evaluate this using only the frozen learned fields
$\widehat F_j/p_S=\nu_j\widehat r_j/Z_j$, with $\widehat r_S=Z_S=1$.
The common $p_S$ cancels. No analytical density defines the observable or
the component bin yields. On the NI up/down samples, evaluate the same nominal
coordinate and then morph the integrated NI templates in the likelihood.
''','definition'),code('''BIN_COUNTS = (36, 144)  # 36 is the baseline; 144 diagnoses bin resolution.
SCORE_SCALE = 0.04      # Earlier centered shape-score convention.
MAX_SELECTED_PER_SOURCE = 10_000_000
MC_ZERO_MAX_BIN = 0.01
MC_ZERO_MAX_TOTAL = 0.05
# These settings identify the completed 04 bank; they do not define the score.
run = run.with_config(reference_ratio_denominator="S", reference_ratio_power=2.0,
    reference_ratio_eta_floor=0.0, mc_zero_max_bin=MC_ZERO_MAX_BIN,
    mc_zero_max_total=MC_ZERO_MAX_TOTAL)
''','settings'),code(model,'model-source'),code('''from poodemo.signal_reference import load_signal_baseline
from poodemo.signal_benchmarks import load_signal_benchmarks

run, samples = load_signal_baseline(run, 36, max_selected=MAX_SELECTED_PER_SOURCE,
    local_cache=Path('/content/poodemo-12-ratios') if IN_COLAB else None)
references, reference_provenance = load_signal_benchmarks(run)
print('Selected events per source:', samples['target'])
print('Unbinned references:', references.model.unique().tolist())
print(reference_provenance['asimov_note'])
''','load-bank'),md(r'''### Direct Asimov likelihood scans

Fill each fixed histogram once from all six source banks. Generate Asimov bin
counts at $(\mu_*,\alpha_{NI})=(1,0)$ from these same templates. For each tested
$\mu$, evaluate the full extended Poisson likelihood with NI fixed at zero or
profiled with its existing constraint. The known generating point attains the
nonnegative Asimov objective's global lower bound; conditional NI fits search
all resolved basins and the bounds.

Keep the established total-yield policy: zero a tiny negative total bin mean
only within 0.01 event/bin and 0.05 events in total. Larger deficits and invalid
fits remain visible. The score scan includes $\mu=0$ exactly; saved unbinned
references retain 03's grid and are not extrapolated to zero.

The learned unbinned reference uses 03's physical Asimov construction and its
original normalization, while these binned curves use same-template MC Asimov
counts. Their difference includes finite-MC and modeling effects as well as
compression. The 144-bin control keeps the same score and sample to check how
much of the discrepancy can be explained by coarse binning.
''','study-note'),code('''from poodemo.fixed_score import run_fixed_score_study, plot_fixed_score_study

study = run_fixed_score_study(run, samples, benchmarks=references,
    benchmark_provenance=reference_provenance,
    bin_counts=BIN_COUNTS, score_scale=SCORE_SCALE)
display(study['bank']['occupancy'])
display(study['summary'])
display(study['scans'].loc[~study['scans'].valid | study['scans'].alpha_boundary])
print('All score fits valid:', study['metadata']['all_fits_valid'])
print('Results:', study['output'])
''','study'),code('''FIGURE_DIR = run.path('plots', '12_fixed_score', study['metadata']['identity'][:20])
for name, figure in plot_fixed_score_study(study, FIGURE_DIR).items():
    print(name)
    display(figure)
    plt.close(figure)
print('Figures:', FIGURE_DIR)
''','plots'),md(r'''### Reading the result

Compare the neighborhood of 1 first, then the low-$\mu$ region and competing
minimum. Similar 36- and 144-bin curves with a persistent unbinned discrepancy
would indicate that simply refining this fixed coordinate is insufficient.
This is an empirical test; local sufficiency does not require global failure
in every model, and a binned score likelihood remains a valid compressed-model
likelihood away from 1 even when it loses information.

For an exact score, local information preservation applies to the parameter
directions represented by the score. A single $\mu$ score at nominal NI need
not retain all information required when NI is profiled; the full local score
vector would include its nuisance direction. Learned-score errors, finite bins,
MC statistics and nuisance morphing after binning are additional approximations.
Thus do not require perfect local profiled closure or universal global optimality.

These are Asimov comparisons, not a coverage or toy-distribution study.
For the local-score argument, see
[Brehmer et al., arXiv:1805.00020](https://arxiv.org/abs/1805.00020).
''','interpretation')]


def build():
    notebook = nbformat.v4.new_notebook(cells=score_cells(),metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}})
    nbformat.validate(notebook)
    nbformat.write(notebook,ROOT/'notebooks'/NAME)
    print(NAME)


if __name__ == '__main__':
    build()
