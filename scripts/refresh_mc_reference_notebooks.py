"""Keep 08/09 on one bounded-zeroing policy and the completed Colab bank."""
import re
from pathlib import Path
import nbformat

SOURCE_COMMIT = "81153be9b15241529ac040a7447421779d80e766"

POLICY = '''# Bounded zeroing of TOTAL MC bin predictions, in expected event counts.
# Fixed before fitting; larger per-bin OR aggregate deficits remain invalid.
MC_ZERO_MAX_BIN = 0.01
MC_ZERO_MAX_TOTAL = 0.05
run = run.with_config(mc_zero_max_bin=MC_ZERO_MAX_BIN, mc_zero_max_total=MC_ZERO_MAX_TOTAL)
from poodemo.mc_yields import yield_policy
print("MC total-yield policy:", yield_policy(run.config).metadata())'''

NOTE = r'''### Tiny negative MC yields

After combining the signed components and applying the NI morph, replace a
negative **total bin mean** by zero only when every correction is at most
$0.01$ expected events and their sum at that parameter point is at most $0.05$.
Both limits are fixed before fitting and shared by 08 and 09. Larger deficits
remain invalid. Positive means, component templates and interference are unchanged;
the corrected total is used consistently in the Poisson rate and logarithm.

Save raw minima, affected bins and summed corrections. A zero prediction with a
positive observed count still has zero Poisson probability. Clipping creates
kinks: Fisher points with negative raw means or zero means with nonzero derivatives
remain invalid. This policy does not freeze $\eta$ or remove its exact $\eta=1$ collapse.
'''

DIAGNOSTICS = '''diagnostics = template_diagnostics(run, templates, etas=scan_grid, n_bins=N_BINS)
by_partition = diagnostics.groupby(['selected_per_source','eta']).agg(
    minimum_expected=('minimum_expected','min'), negative_bins=('negative','sum'),
    nonregular_bins_at_truth=('nonregular_mu_at_truth','sum'),
    zeroed_bins_at_test=('zeroed_at_test','sum'),
    zeroed_total_at_test=('test_zeroed_total','max'),
    test_zeroing_accepted=('test_zeroing_accepted','all'),
).reset_index()
display(by_partition.groupby('selected_per_source').agg(
    partitions=('eta','size'), raw_domain_minimum=('minimum_expected','min'),
    partitions_with_raw_negatives=('negative_bins',lambda x:int((x>0).sum())),
    corrected_test_points=('zeroed_bins_at_test',lambda x:int((x>0).sum())),
    max_test_correction=('zeroed_total_at_test','max'),
    rejected_test_points=('test_zeroing_accepted',lambda x:int((~x).sum())),
    nonregular_partitions_at_truth=('nonregular_bins_at_truth',lambda x:int((x>0).sum())),
))
print('Raw domain minima are retained; bounded zeroing is audited separately at each tested point.')
display(diagnostics.loc[diagnostics.negative | diagnostics.nonregular_mu_at_truth].head(30))
print('Full raw/used bin diagnostics:',run.path('results','reference_ratio_mc_bin_diagnostics.csv'))'''

POWER = r'''from poodemo.power_study import run_power_study
POWER_VALUES = [50., 100., 150.]
# Map positive probes to the actual scan so the analytical benchmark is matched.
POWER_ETA_TARGETS = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.4, 0.8, 1.2]
POWER_ETAS = np.unique(np.r_[0., [scan_grid[np.argmin(abs(scan_grid-e))] for e in POWER_ETA_TARGETS]])
power_controls = run_power_study(run, samples, powers=POWER_VALUES, etas=POWER_ETAS, n_bins=N_BINS)
display(power_controls.loc[~power_controls.systematics,
    ['eta','power','reference_end_bin_fraction','reference_numerical_saturation_fraction','q','q_analytic','valid']])
fig, axes = plt.subplots(1,2,figsize=(12,4.5))
for ax,syst in zip(axes,[False,True]):
    for power,g in power_controls.loc[power_controls.systematics.eq(syst)&power_controls.eta.gt(0)&power_controls.eta.le(.3)].groupby('power'):
        ax.plot(g.eta,g.q.where(g.valid),'.-',label=f'a={power:g}')
        bad=g.loc[~g.valid]
        ax.plot(bad.eta,np.zeros(len(bad)),'rx',transform=ax.get_xaxis_transform(),clip_on=False)
    ref=unbinned.loc[unbinned.systematics.eq(syst)&unbinned.mu.le(.3)]
    ax.plot(ref.mu,ref.q.where(ref.valid),'k--',label='Analytical unbinned')
    ax.set(xlabel=r'Tested $\mu_0$',ylabel=r'$q(\mu_0)$',title='NI profiled' if syst else 'NI fixed')
    ax.legend()
fig.suptitle('36-bin power controls: low strengths and secondary minimum')
fig.tight_layout()
for ext in ('png','pdf'): fig.savefig(run.path('plots',f'08_power_controls.{ext}'),dpi=160,bbox_inches='tight')
plt.show()
print('Power is kept fixed at',REFERENCE_RATIO_A,'; these paired controls do not automatically tune it.')'''

LOAD09 = '''from poodemo.colab_reference import load_08_baseline, ColabTemplateBank
_mc_base_run = run
run, samples_08 = load_08_baseline(
    _mc_base_run, N_BINS,
    local_cache=Path('/content/poodemo-09-ratios') if IN_COLAB else None,
)
print('Reusing completed notebook 08:',run.evaluation_root)
print('Selected events per source:',samples_08['target'])
print('No generation, neural evaluation or training is performed in 09.')'''


def configure_review_notebook(cells, number):
    cells = [c for c in cells if c.source.strip() and not c.id.startswith('mc-review-')]
    md = lambda s,i:nbformat.v4.new_markdown_cell(s.strip(),id=i)
    code = lambda s,i:nbformat.v4.new_code_cell(s.strip(),id=i)
    for c in cells:
        if c.cell_type=='code' and 'SOURCE_COMMIT = ' in c.source:
            c.source=re.sub(r'SOURCE_COMMIT = [\'\"][0-9a-f]+[\'\"]',f'SOURCE_COMMIT = "{SOURCE_COMMIT}"',c.source)
        if c.id=='cell-003':
            policy=POLICY
            for key in ('MC_ZERO_MAX_BIN','MC_ZERO_MAX_TOTAL'):
                old=re.search(rf'^{key} = (.+)$',c.source,flags=re.M)
                if old:policy=re.sub(rf'^{key} = .+$',lambda _:key+' = '+old.group(1),policy,flags=re.M)
            c.source=c.source.split('# Bounded zeroing of TOTAL MC bin predictions')[0].rstrip()+'\n\n'+policy
        if number==8:
            if c.id=='colab-diagnostics':c.source=DIAGNOSTICS
            if c.id=='practical-definition':
                c.source=c.source.replace('Negative physical predictions and nonregular Fisher points remain\ninvalid; no yield floor or automatic merging is introduced.',
                    'Tiny negative total yields use the bounded zeroing policy below. Larger\ndeficits and nonregular Fisher points remain invalid; bins are not merged.')
            if c.id=='cell-020':c.source='Review the raw/used yield diagnostics and power controls before interpreting 09.'
        else:
            if c.id=='spline-source-check':c.source=c.source.replace('!= 4','!= 5')
            if c.id=='reference-model-source':
                marker='# Cap selected integration events per source; all events estimate acceptance.'
                if marker in c.source:c.source=c.source[:c.source.index(marker)]+'''_filtered_run = run
run = use_practical_run(run)
print("Practical evaluation:",run.evaluation_root)'''
            if c.id=='spline-run':c.source='''cdf_bank = ColabTemplateBank(run, samples_08, SPLINE_BINS, SPLINE_SETTINGS.cdf_cache_size)
splines = run_reference_spline_comparison(run, n_bins=SPLINE_BINS, settings=SPLINE_SETTINGS, template_bank=cdf_bank)
display(splines["summary"])
print("Results:",splines["output"])'''
            if c.id=='floor-run':c.source='''floor_study = run_reference_floor_study(
    run, n_bins=SPLINE_BINS, template_bank=cdf_bank, settings=FLOOR_SETTINGS)
display(floor_study["summary"])
display(floor_study["scans"].loc[~floor_study["scans"].valid | floor_study["scans"].alpha_boundary])
print("Results:",floor_study["output"])'''
            if c.id=='spline-tables' and 'Zeroing at conditional' not in c.source:
                c.source+='''\nprint("Zeroing at conditional and unrestricted fit points:")
display(splines["scans"].filter(regex="^(method|eta|systematics|valid|.*zeroed.*|.*raw_minimum.*|.*rejected_negative.*)$"))
print("Raw mixture errors are retained before zeroing:")
display(splines["validation"].nlargest(10,"raw_mixture_max_abs_error"))'''
            if c.id=='cell-001':
                c.source='''### Runtime and persistent run

Run the updated **08 first**, including its final baseline-export cell, then
Run all here. Setup downloads a fresh pinned checkout; the next cell must report
study version 5. Keep the same run, filtered Unity ensemble, 36 bins, power and
zeroing limits. Notebook 09 loads the exact completed 08 sample campaign and
analytical benchmark; it never falls back to the old 250k integration bank.

Ratios are verified and staged on Colab local disk once (about 1.4 GB for the
10M-per-source bank). CDFs are built one source at a time; two anchors require
about 1 GB of coordinate arrays. There is no generation, NN evaluation or training.
Sorting and interpolation refinement are CPU work and can take appreciable time.
Historical 08 products are preserved; this study writes its own subdirectories.'''
            if c.id=='practical-definition':
                c.source='''### Shared 10M simulation baseline

The completed 08 manifest selects the same six labelled selected streams,
frozen rates, learned-map normalizers and filtered ensembles. Both direct fills
and exact empirical CDFs use these events. The analytical density is used only
for the saved unbinned benchmark. Asimov counts come from direct templates;
this is interpolation closure on a fixed MC bank, not independent MC closure.'''
            if c.id=='reference-model-intro' and 'All tables and figures' in c.source:
                c.source=c.source[:c.source.index('All tables and figures')]+'The next cells verify the selected ensemble, then load the matching completed 08 bank.\n'
        if c.id=='reference-model-source' and '# Preserve selected model provenance' not in c.source:
            c.source+='''\n# Preserve selected model provenance in the practical evaluation.
save_json(run.path("results","reference_ratio_model_source.json"),
          __import__("json").loads(_filtered_run.path("results","reference_ratio_model_source.json").read_text()))'''
    where=next(i for i,c in enumerate(cells) if c.id=='cell-003')+1
    cells.insert(where,md(NOTE,'mc-review-policy'))
    if number==8:
        cells += [md(r'''### Check the power near zero and the secondary minimum

Keep $a=100$ and 36 bins. Compare $a=50,100,150$ on the same full selected bank,
with rates and learned-map normalization fixed. This reads cached predictions
once, resumes from compact count checkpoints, and performs no neural evaluation.
Near $\eta=0$ and the secondary minimum, compare $q$, endpoint-bin occupancy and
exact numerical saturation separately. Merely approaching an endpoint does not
imply lost information. The $\eta=0$ row is distribution-only; physical fits use
positive scan strengths. These finite-MC controls do not automatically optimize
power against the analytical reference.
''','mc-review-power-note'),code(POWER,'mc-review-power'),
        code('''from poodemo.colab_reference import publish_08_baseline
baseline_manifest = publish_08_baseline(mc_run, run, samples, templates, N_BINS)
print("Completed 08 baseline for notebook 09:",baseline_manifest)''','mc-review-publish')]
    else:
        where=next(i for i,c in enumerate(cells) if c.id=='reference-model-source')+1
        cells.insert(where,code(LOAD09,'mc-review-load'))
    for c in cells:
        if c.cell_type=='code':c.outputs=[];c.execution_count=None
    return cells


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    for number,name in [(8,'08_reference_ratio_histograms'),(9,'09_reference_ratio_spline_templates')]:
        path=root/'notebooks'/f'{name}.ipynb'
        nb=nbformat.read(path,as_version=4);nb.cells=configure_review_notebook(nb.cells,number)
        nbformat.write(nb,path)
