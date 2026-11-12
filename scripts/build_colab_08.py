"""Maintain notebook 08's Colab-only large-MC workflow; preserve user controls."""
import re
import ast
from pathlib import Path
import nbformat

SOURCE_COMMIT = 'c567cfaee8c8cf6b70ae747646c2b7c3f89e7cc6'

SETUP = '''from poodemo.colab_mc import (prepare_colab_samples, fill_colab_histograms,
                               template_diagnostics, convergence_scans)
from poodemo.pipeline import _positive_study_grid
import torch

SELECTED_EVENTS_PER_SOURCE = 10_000_000  # AFTER the frozen notebook-02 preselection.
MC_PREFIXES = [250_000, 1_000_000, SELECTED_EVENTS_PER_SOURCE]
MC_SEED = 20261111
SELECTED_CHUNK = 250_000
GENERATION_CHUNK = 250_000
INFERENCE_BATCH = 8192  # Reduce if the GPU runs out of memory.
NORMALIZATION_EVENTS = 250_000  # Original held-out S sample; fixed across MC prefixes.
BIN_COUNTS = [4, 12, 24, 36, 60]
ETA_VALUES = [0.0, 0.3, 0.6, 1.0, 1.5]
COMPARISON_ETA_MIN = 0.2
CHECK_ETAS = [0., 0.99, 0.999, 1., 1.001, 1.01]

if IN_COLAB and not torch.cuda.is_available():
    raise RuntimeError("Select a GPU runtime in Colab, then rerun from the top.")
print("Inference device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU smoke check")
print(f"Target: {SELECTED_EVENTS_PER_SOURCE:,} selected events for EACH of S, SBI, B, NI, NI-up, NI-down")
print("Generation and neural evaluation run here in Colab; no cluster jobs or retraining.")

# Preserve the original practical run for calibration and sample caching.
mc_run = run
samples = prepare_colab_samples(
    mc_run, selected_per_source=SELECTED_EVENTS_PER_SOURCE, seed=MC_SEED,
    selected_chunk=SELECTED_CHUNK, generation_chunk=GENERATION_CHUNK,
    inference_batch=INFERENCE_BATCH, normalization_events=NORMALIZATION_EVENTS,
    model_cache=Path('/content/poodemo-model-cache') if IN_COLAB else None,
)
display(samples['sample_counts'])
print('Persistent sample cache:', samples['directory'])
'''

HISTOGRAMS = '''# NN predictions are cached. Changing eta/bin settings only refills histograms.
scan_grid = _positive_study_grid(mc_run)
requested_etas = np.unique(np.r_[scan_grid, ETA_VALUES, CHECK_ETAS, 1.,
                                 COMPARISON_ETA_MIN, REFERENCE_RATIO_ETA_MIN,
                                 np.maximum(scan_grid, REFERENCE_RATIO_ETA_MIN)])
requested_bins = sorted(set(BIN_COUNTS + [N_BINS]))
templates = fill_colab_histograms(
    mc_run, samples, etas=requested_etas, bin_counts=requested_bins,
    prefixes=MC_PREFIXES,
)
# Separate results protect historical 08/09 runs and different MC studies.
run = templates.study_run(mc_run)
print('10M-study outputs:', run.evaluation_root)
print('Stored direct templates:', templates.directory / 'counts.npz')
'''

DIAGNOSTICS = '''diagnostics = template_diagnostics(run, templates, etas=scan_grid, n_bins=N_BINS)
by_partition = diagnostics.groupby(['selected_per_source', 'eta']).agg(
    minimum_expected=('minimum_expected', 'min'),
    negative_bins=('negative', 'sum'),
    nonregular_bins_at_truth=('nonregular_mu_at_truth', 'sum'),
    incomplete_NI_support_bins=('incomplete_NI_support_bins', 'max'),
).reset_index()
display(by_partition.groupby('selected_per_source').agg(
    partitions=('eta', 'size'), minimum_expected=('minimum_expected', 'min'),
    partitions_with_negative_bins=('negative_bins', lambda x: int((x > 0).sum())),
    nonregular_partitions_at_truth=('nonregular_bins_at_truth', lambda x: int((x > 0).sum())),
))
problem_bins = diagnostics.loc[diagnostics.negative | diagnostics.nonregular_mu_at_truth]
if len(problem_bins):
    print('Unsupported physical predictions remain. No yields are clipped or floored.')
    display(problem_bins.head(30))
print('Full bin diagnostics, including MC uncertainties:',
      run.path('results', 'reference_ratio_mc_bin_diagnostics.csv'))
'''

DISTRIBUTIONS = r'''import mplhep as hep
from poodemo.pipeline import observable_bin_edges
OBSERVABLE = 'reference_ratio'
PLOT_PREFIX = '08'
OBSERVABLE_LABEL = rf"$z_\eta=R_\eta^a/(1+R_\eta^a),\quad a={REFERENCE_RATIO_A:g}$"
HIST_BINS = N_BINS
edges = observable_bin_edges(HIST_BINS, OBSERVABLE)
distributions = []
for eta in ETA_VALUES:
    nu_s, nu_sbi, nu_b, nu_ni = templates.anchors(max(eta, REFERENCE_RATIO_ETA_MIN), HIST_BINS)[0]
    root_eta = np.sqrt(eta)
    coherent = (eta-root_eta)*nu_s + root_eta*nu_sbi + (1-root_eta)*nu_b
    distributions.append((eta, coherent, nu_ni))

# Each panel has its own scale. Signed components remain visible on symlog.
with plt.style.context(hep.style.ATLAS), plt.rc_context({'font.size': 14}):
    ncols = min(2, len(distributions)); nrows = (len(distributions)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 4.8*nrows), squeeze=False)
    for ax, (eta, coherent, ni) in zip(axes.flat, distributions):
        total = coherent+ni
        if np.any(coherent < 0):
            hep.histplot([ni, coherent, total], bins=edges, histtype='step',
                         color=['#888888','#0072B2','black'],
                         label=['Non-interfering background', 'Coherent SBI', 'Total'], yerr=False, ax=ax)
            ax.set_yscale('symlog', linthresh=.1)
            ax.text(.04,.79,f'{np.count_nonzero(total < 0)} negative total-yield bins', transform=ax.transAxes, fontsize=11)
        else:
            hep.histplot([ni,coherent], bins=edges, stack=True, histtype='fill',
                         color=['#B8B8B8','#56B4E9'], edgecolor='black', linewidth=.7,
                         label=['Non-interfering background', r'Coherent SBI ($\mu=\eta$)'], yerr=False, ax=ax)
            ax.set_yscale('log')
            positive=total[total>0]
            if len(positive): ax.set_ylim(max(.001,positive.min()*.3),positive.max()*30)
        ax.text(.04,.94,fr'$\eta={eta:g},\quad\alpha_{{NI}}=0$',transform=ax.transAxes,va='top')
        ax.set(xlim=(0,1),xlabel=OBSERVABLE_LABEL,ylabel='Expected events / bin')
        ax.legend(fontsize=10,loc='upper right'); ax.grid(False)
    for ax in axes.flat[len(distributions):]: ax.set_visible(False)
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(run.path('plots', f'08_reference_ratio_distributions.{ext}'),dpi=160,bbox_inches='tight')
    plt.show()
display(pd.DataFrame([dict(eta=e, coherent_SBI_yield=c.sum(), NI_yield=n.sum(),
                           total_yield=(c+n).sum(),minimum_bin_yield=(c+n).min()) for e,c,n in distributions]))
'''

COLLAPSE = '''collapse = templates.collapse_table(CHECK_ETAS, N_BINS)
display(collapse)
assert collapse.loc[collapse.eta.eq(1.), 'occupied_bins'].item() == 1
np.testing.assert_array_equal(collapse.loc[collapse.eta.eq(1.), ['z_min','z_max']].to_numpy(), [[.5,.5]])
np.testing.assert_allclose(templates.anchors(1.,N_BINS)[0].sum(axis=-1), samples['rates'][:4], rtol=1e-12)
collapse.to_csv(run.path('results','reference_ratio_collapse.csv'),index=False)
'''

CONVERGENCE = r'''mc_scans = convergence_scans(run, templates, etas=scan_grid, n_bins=N_BINS)
display(mc_scans.groupby(['selected_per_source','systematics']).agg(
    scan_points=('valid','size'), valid_points=('valid','sum')))
fig, axes = plt.subplots(1,2,figsize=(13,4.5),sharex=True)
for ax, syst in zip(axes,[False,True]):
    for size, group in mc_scans.loc[mc_scans.systematics.eq(syst)].groupby('selected_per_source'):
        group=group.sort_values('mu')
        line,=ax.plot(group.mu,group.q.where(group.valid),label=f'{size:,} selected / source')
        bad=group.loc[~group.valid]
        ax.plot(bad.mu,np.zeros(len(bad)),marker='x',ls='none',color=line.get_color(),
                transform=ax.get_xaxis_transform(),clip_on=False)
    ref=unbinned.loc[unbinned.systematics.eq(syst)].sort_values('mu')
    ax.plot(ref.mu,ref.q.where(ref.valid),'k--',label='Analytical unbinned')
    ax.set(xlabel=r'Tested $\mu_0$',ylabel=r'$q(\mu_0)$',title='NI profiled' if syst else 'NI fixed',ylim=(0,None))
    ax.legend(fontsize=8)
fig.suptitle('MC convergence: crosses on the axis mark invalid fits')
fig.tight_layout();fig.savefig(run.path('plots','08_mc_convergence.png'),dpi=160,bbox_inches='tight');plt.show()
'''


def configure_colab_08(cells):
    previous_setup = next((c.source for c in cells if c.id == 'colab-sampling-setup'), None)
    setup = SETUP
    if previous_setup:
        # Retain editable sampling/plot controls when regenerating the notebook.
        names = {'SELECTED_EVENTS_PER_SOURCE', 'MC_PREFIXES', 'MC_SEED', 'SELECTED_CHUNK',
                 'GENERATION_CHUNK', 'INFERENCE_BATCH', 'NORMALIZATION_EVENTS', 'BIN_COUNTS',
                 'ETA_VALUES', 'COMPARISON_ETA_MIN', 'CHECK_ETAS'}
        for node in ast.parse(previous_setup).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', '') in names:
                name = node.targets[0].id
                replacement = '\n'.join(previous_setup.splitlines()[node.lineno-1:node.end_lineno])
                current = next(n for n in ast.parse(setup).body if isinstance(n, ast.Assign)
                               and getattr(n.targets[0], 'id', '') == name)
                lines = setup.splitlines()
                lines[current.lineno-1:current.end_lineno] = [replacement]
                setup = '\n'.join(lines)
    cells=[c for c in cells if c.source.strip() and not c.id.startswith('colab-')]
    for c in cells:
        s=c.source
        if c.id=='cell-000':
            s='''# 8. Learned reference-ratio histograms with 10M selected events per source

Generate fresh independent simulation **in Colab**, applying notebook 02's
frozen preselection until each of S, SBI, B, NI, NI-up and NI-down has at least
10 million selected events. Use exactly 10 million per source for the primary
histograms, with 250k and 1M prefixes as MC convergence controls.
Keep the filtered trained ensembles, no floor, power $a=100$ and 36 primary bins.
Generation, GPU predictions and histogram accumulation resume from Drive.
'''
        elif c.id=='cell-001':
            s='''### Runtime and persistent run

Use **a Colab GPU runtime** and Run all from the top. Existing models and the
preselection are reused; no training or cluster submission is needed. The new
simulation is independent of the original training and integration samples.
Keep RUN_NAME, Unity model directory and filter policy as in notebook 03.
The Unity directory here is only the existing model bundle already on Drive.

Checkpoints persist on Drive; interrupted chunks are replayed. Restart the
runtime and Run all to resume. GPU model files are staged on Colab local disk;
events and predictions are processed in bounded chunks. The 60M selected events
are never concatenated in RAM. Allow roughly 2.2 GB on Drive for selected
coordinates and three float64 ratios per event, plus small manifests/templates;
local model staging uses additional disk. Rejected events are not saved.
'''
        elif c.id=='cell-002':
            s=re.sub(r'SOURCE_COMMIT = "[0-9a-f]+"',f'SOURCE_COMMIT = "{SOURCE_COMMIT}"',s)
        elif c.id=='reference-model-intro' and 'All tables and figures' in s:
            s=s[:s.index('All tables and figures')]+'''The next cells create an independent Colab simulation cache. The selected
filtered models remain unchanged; predictions on the new events are cached.
All new results use a separate study directory printed below.
'''
        elif c.id=='reference-model-source':
            marker='# Cap selected integration events per source; all events estimate acceptance.'
            if marker in s: s=s[:s.index(marker)]+'''_filtered_run = run
run = use_practical_run(run)
print("Practical evaluation:", run.evaluation_root)
'''
        elif c.id=='practical-definition':
            s=r'''### What changes with the larger sample

Fresh events determine the component **shapes**. The selected process yields
$\nu_j$ remain the estimates from the original integration samples, and the
learned-map normalizers remain fixed on the original held-out S sample.
Each prefix of $N$ fresh selected events receives weight $w_j=\nu_j/N$.
Increasing MC statistics does not change the experimental luminosity or yields.
Acceptance-rate uncertainty is not reduced by this shape-only convergence study.
The selected-target stopping count is never used as an acceptance estimate.

All coordinates use learned ratios. The only analytical inference calculation
is the unbinned benchmark; analytical PDFs are not histogram weights.
Same-template Asimov observations test each fixed MC model, not independent
closure against the true experiment. Prefixes are nested and correlated.
NI retains the existing after-binning morph, including its explicit sparse-bin
convention. Negative physical predictions and nonregular Fisher points remain
invalid; no yield floor or automatic merging is introduced.
'''
        elif c.id=='cell-004':
            s=s.replace('Rerun **08 first, then 09** with matching settings and the\nsame filtered ensemble evaluation; no generation or training is needed.',
                        'This notebook validates the new high-statistics direct templates first.\nNotebook 09 is not automatically switched to the new cache by this update.')
        elif c.id=='cell-006': s=DISTRIBUTIONS
        elif c.id=='cell-007': s=COLLAPSE
        elif c.id=='cell-011':
            s='''from poodemo.pipeline import run_binning_study
binning_run = run.with_config(bin_counts=BIN_COUNTS)
binning = run_binning_study(binning_run, observable="reference_ratio", template_bank=templates)
display(binning["scans"])
display(binning["fidelity"])
display(binning['scans'].groupby(['model','n_bins'],dropna=False).agg(
    scan_points=('valid','size'), valid_points=('valid','sum')))
'''
        elif c.id=='cell-012':
            if '.where(group["valid"])' not in s:
                s=s.replace('group[y_column], marker=', 'group[y_column].where(group["valid"]) if "valid" in group else group[y_column], marker=')
            old='        ax.plot(group[x_column], '
            # Preserve the existing helper and add visible failure markers once.
            if 'Invalid fits are shown' not in s:
                s=s.replace('    ax.axhline(1.0,', '''        # Invalid fits are shown on the x axis, never joined by a valid curve.
        if "valid" in group:
            bad = group.loc[~group["valid"]]
            ax.plot(bad[x_column], np.zeros(len(bad)), "rx", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.axhline(1.0,''')
        elif c.id=='cell-013':
            s=s.replace('SCAN_XLIM = (0.02, 1.15)','SCAN_XLIM = None').replace('SCAN_YLIM = (0, 8)','SCAN_YLIM = None')
        elif c.id=='cell-017':
            s=s.replace('same 36 bins, MC bank','same 36 bins, fresh MC bank')
        elif c.id=='cell-018':
            s=s.replace('COMPARISON_ETA_MIN = 0.2\n','').replace('COMPARISON_ETA_MIN = COMPARISON_ETA_MIN\n','')
            # Each call uses compact templates, including the floor control.
            if 'template_bank=templates' not in s:
                s=s.replace('n_bins=N_BINS)', 'n_bins=N_BINS, template_bank=templates)')
                s=s.replace('n_bins=N_BINS,\n', 'n_bins=N_BINS, template_bank=templates,\n')
        elif c.id=='cell-020':
            s='''Review the MC convergence and validity diagnostics before updating 09.
This change updates **08 only**. Notebook 09 still uses its earlier bank;
it does not yet consume this 10M Colab cache. The cached selected coordinates
and learned ratios are retained for a subsequent interpolation study.
'''
        c.source=s.strip()
    md=lambda s,i:nbformat.v4.new_markdown_cell(s.strip(),id=i)
    code=lambda s,i:nbformat.v4.new_code_cell(s.strip(),id=i)
    where=next(i for i,c in enumerate(cells) if c.id=='practical-definition')+1
    cells[where:where]=[
        md('''### Generate and evaluate fresh selected events

The target is **per source after preselection**, not a cap on existing samples.
Progress and hashes are saved after each completed chunk and each ratio task.
A new seed creates a separate independent campaign. Changing batch/chunk settings
creates a separate cache; keep them unchanged when resuming.
''','colab-sampling-note'),code(setup,'colab-sampling-setup'),
        code(HISTOGRAMS,'colab-histograms'),
        md(r'''### Physical-domain and MC precision checks

For each bin, minimize the physical prediction exactly as a quadratic in
$\sqrt{\mu}$ over the configured fit bounds, at a declared grid of NI values.
The nuisance grid is a diagnostic, not a guarantee between its points. Save MC
uncertainties and zero-yield/nonzero-derivative checks at the generating truth.
These run before likelihood fits. Increasing the MC target does not guarantee
that every rare bin becomes valid.
''','colab-diagnostic-note'),code(DIAGNOSTICS,'colab-diagnostics')]
    where=next(i for i,c in enumerate(cells) if c.id=='cell-010')+1
    cells[where:where]=[md('### MC convergence at fixed map and yields','colab-convergence-note'),code(CONVERGENCE,'colab-convergence')]
    for c in cells:
        if c.cell_type=='code': c.outputs=[];c.execution_count=None
    from refresh_mc_reference_notebooks import configure_review_notebook
    return configure_review_notebook(cells, 8)


if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'notebooks/08_reference_ratio_histograms.ipynb'
    nb=nbformat.read(path,as_version=4);nb.cells=configure_colab_08(nb.cells);nbformat.write(nb,path)
