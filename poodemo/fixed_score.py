"""Notebook 12: one learned score at mu=1, with direct simulation bin yields."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .colab_mc import _digest, _integer
from .colab_reference import ColabTemplateBank
from .data import SAMPLES, save_json
from .inference import component_coefficients
from .pipeline import _histogram_model, _positive_study_grid, observable_bin_edges
from .reference_floor import FloorStudySettings, direct_asimov_fits
from .mc_yields import yield_policy
from .toy_integration_diagnostics import _atomic_npz

VERSION = 1


def shape_score_at_one(fields, rates):
    """d_mu log p(x;mu)|_1 from learned intensity fields relative to p_S.

    Fields already include selected yields and frozen ratio normalizers.
    This is a fixed nominal-NI coordinate, independent of the fitted mu/alpha.
    The extended count term is retained separately by the Poisson likelihood.
    """
    fields, rates = np.asarray(fields, float), np.asarray(rates, float)
    if (fields.ndim != 2 or fields.shape[0] != 4 or rates.shape != (4,)
            or np.any(~np.isfinite(fields)) or np.any(fields < 0)
            or np.any(~np.isfinite(rates)) or np.any(rates <= 0)):
        raise ValueError('Expected nonnegative finite (4,N) fields and positive finite (4,) rates')
    # A common per-event scale cancels and prevents overflow in the sums.
    scale = fields.max(axis=0)
    if np.any(scale <= 0):
        raise ValueError('The nominal physical intensity must be positive')
    s, sbi, b, ni = fields/scale
    denominator = sbi+ni
    if np.any(denominator <= 0):
        raise ValueError('The nominal physical intensity must be positive')
    r = rates/rates.max()
    center = .5*(r[0]+r[1]-r[2])/(r[1]+r[3])
    return .5*(s+sbi-b)/denominator-center


def _payload_digest(counts):
    digest = hashlib.sha256()
    for bins, values in sorted(counts.items()):
        digest.update(str(bins).encode())
        digest.update(np.ascontiguousarray(values, dtype=np.int64).tobytes())
    return digest.hexdigest()


def fill_fixed_score(run, samples, *, bin_counts=(36, 144), score_scale=.04, progress=print):
    """Read cached ratios once, evaluating the same coordinate on all six sources."""
    target = _integer(samples['target'], 'selected events per source', 2)
    if target > 10_000_000:
        raise ValueError('Notebook 12 may not exceed 10M selected events per source')
    if tuple(samples['manifests']) != tuple(SAMPLES):
        raise ValueError('Expected S, SBI, B, NI, NI_up and NI_down sources in order')
    if not np.isfinite(score_scale) or score_scale <= 0:
        raise ValueError('score_scale must be finite and positive')
    bins = sorted(set(_integer(n, 'bin count', 2) for n in bin_counts))
    if not bins:
        raise ValueError('Provide at least one bin count')
    edges = {n: observable_bin_edges(n, 'score') for n in bins}
    config = dict(version=VERSION, sample_fingerprint=samples['fingerprint'],
        selected_per_source=target, score_anchor=1., score_nuisance=0.,
        coordinate='centered d_mu log intensity at mu=1; sigmoid(score/scale)',
        score_scale=float(score_scale), rates=samples['rates'].tolist(),
        normalizers=samples['normalizers'].tolist(), edges={str(n):e.tolist() for n,e in edges.items()})
    identity = _digest(config)
    directory = samples['directory']/'fixed_score'/identity[:20]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'counts.npz'
    cached = path.exists()
    if cached:
        with np.load(path, allow_pickle=False) as saved:
            counts = {n:saved[f'counts_{n}'] for n in bins}
            ranges = saved['ranges']
            if str(saved['identity']) != identity or str(saved['payload_sha256']) != _payload_digest(counts):
                raise ValueError('Fixed-score histogram cache identity/content mismatch')
        progress('Reusing completed fixed-score counts')
    else:
        counts = {n:np.zeros((6,n), dtype=np.int64) for n in bins}
        ranges = np.zeros((6,4), float)  # score min/max, z min/max
        stream = ColabTemplateBank(run, samples)
        for i,(source,state) in enumerate(samples['manifests'].items()):
            low, high, zlow, zhigh = np.inf, -np.inf, np.inf, -np.inf
            for _,fields in stream._fields(state):
                score = shape_score_at_one(fields, samples['rates'][:4])
                z = expit(score/score_scale)
                if np.any(~np.isfinite(score)) or np.any(~np.isfinite(z)):
                    raise ValueError('Nonfinite learned score')
                for n in bins:
                    counts[n][i] += np.histogram(z, edges[n])[0]
                low, high = min(low,float(score.min())), max(high,float(score.max()))
                zlow, zhigh = min(zlow,float(z.min())), max(zhigh,float(z.max()))
            ranges[i] = low,high,zlow,zhigh
            progress(f'Fixed score: {source}, {counts[bins[0]][i].sum():,}/{target:,} events')
    for n,values in counts.items():
        if values.shape != (6,n) or np.any(values < 0) or np.any(values.sum(axis=1) != target):
            raise ValueError('Incomplete fixed-score source counts')
    if not cached:
        _atomic_npz(path, identity=np.array(identity), ranges=ranges,
            payload_sha256=np.array(_payload_digest(counts)),
            **{f'counts_{n}':values for n,values in counts.items()})
    save_json(directory/'configuration.json', {**config,'identity':identity})
    occupancy = []
    for n,values in counts.items():
        for i,source in enumerate(SAMPLES):
            occupancy.append(dict(source=source,n_bins=n,selected=target,
                occupied_bins=int(np.count_nonzero(values[i])),
                endpoint_fraction=float((values[i,0]+values[i,-1])/target),
                score_min=ranges[i,0],score_max=ranges[i,1],z_min=ranges[i,2],z_max=ranges[i,3]))
    return dict(counts=counts,edges=edges,configuration=config,identity=identity,
                directory=directory,occupancy=pd.DataFrame(occupancy))


def run_fixed_score_study(run, samples, *, benchmarks, benchmark_provenance,
                          bin_counts=(36,144), score_scale=.04, progress=print):
    """Fixed partition, exact physical coefficients, nominal-NI Asimov truth at 1."""
    if float(run.config['asimov_mu']) != 1.:
        raise ValueError('Notebook 12 compares the score and Asimov experiment at mu=1')
    bank = fill_fixed_score(run,samples,bin_counts=bin_counts,score_scale=score_scale,progress=progress)
    grid = np.unique(np.r_[0.,_positive_study_grid(run)])
    settings = FloorStudySettings()  # Reuse the robust scalar NI profiler; no observable floor.
    policy = yield_policy(run.config)
    config = dict(**bank['configuration'],histogram_identity=bank['identity'],
        asimov_mu=1.,mu_grid=grid.tolist(),mu_fit_bounds=run.config['mu_fit_bounds'],
        nuisance_bounds=run.config['nuisance_bounds'],yield_policy=policy.metadata(),
        nuisance_morphing='after binning; frozen nominal score on every source')
    identity = _digest(config)
    output = run.path('results','12_fixed_score',identity[:20]);output.mkdir(parents=True,exist_ok=True)
    scans,models = [],{}
    for n,raw in bank['counts'].items():
        yields = raw*samples['rates'][:,None]/samples['target']
        anchors = yields[[[0,1,2,3],[0,1,2,5],[0,1,2,4]]]
        observed = component_coefficients(1.) @ anchors[0]
        model = _histogram_model(anchors,observed,practical=True,yield_policy=policy)
        models[n] = model
        for mu in grid:
            for result in direct_asimov_fits(run,model,float(mu),settings):
                scans.append(dict(model=f'Fixed learned score ({n} bins)',mu=float(mu),
                                  eta=1.,n_bins=n,**result))
        progress(f'Fixed-score fits complete: {n} bins, stat-only and NI profiled')
    frame = pd.DataFrame(scans)
    summary = frame.groupby(['n_bins','systematics']).agg(points=('valid','size'),
        valid=('valid','sum'),max_zeroed_total=('zeroed_total','max'),
        alpha_boundaries=('alpha_boundary','sum')).reset_index()
    summary['invalid'] = summary.points-summary.valid
    metadata = dict(configuration=config,status='complete',identity=identity,
        all_fits_valid=bool(frame.valid.all()),unbinned_benchmarks=benchmark_provenance,
        interpretation='Same-template MC Asimov histograms; 03 learned benchmark uses physical Asimov')
    frame.to_csv(output/'scans.csv',index=False)
    summary.to_csv(output/'summary.csv',index=False)
    bank['occupancy'].to_csv(output/'occupancy.csv',index=False)
    benchmarks.to_csv(output/'unbinned.csv',index=False)
    save_json(output/'metadata.json',metadata)
    return dict(scans=frame,summary=summary,bank=bank,models=models,unbinned=benchmarks.copy(),
                output=output,metadata=metadata)


def plot_fixed_score_study(study, output):
    """Three likelihood views with both unbinned references and one shape view."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    figures = {}
    scans,refs = study['scans'],study['unbinned']
    def save(fig,name):
        fig.tight_layout()
        for extension in ('png','pdf'):
            fig.savefig(output/f'12_{name}.{extension}',bbox_inches='tight',dpi=150)
        figures[name] = fig
    with plt.style.context(hep.style.ATLAS):
        regions = dict(full=(0.,float(scans.mu.max())),near_one=(.8,1.2),secondary=(0.,.3))
        for name,limits in regions.items():
            fig,axes = plt.subplots(1,2,figsize=(14,5))
            for ax,syst in zip(axes,(False,True)):
                heights = []
                for n,group in scans.loc[scans.systematics.eq(syst)].groupby('n_bins'):
                    group = group.sort_values('mu')
                    ax.plot(group.mu,group.q.where(group.valid),label=f'Fixed score: {n} bins')
                    bad = group.loc[~group.valid]
                    ax.scatter(bad.mu,np.zeros(len(bad)),marker='x',color='red',s=25)
                    heights.extend(group.loc[group.valid & group.mu.between(*limits),'q'])
                for label,group in refs.loc[refs.systematics.eq(syst)].groupby('model'):
                    group = group.sort_values('mu')
                    color,style = ('black','--') if label=='analytic unbinned' else ('#CC79A7','-.')
                    ax.plot(group.mu,group.q.where(group.valid),color=color,ls=style,label=label)
                    heights.extend(group.loc[group.valid & group.mu.between(*limits),'q'])
                finite = np.asarray(heights);finite = finite[np.isfinite(finite)]
                top = max(.1,1.1*float(finite.max())) if len(finite) else 1.
                ax.set(xlim=limits,ylim=(0.,top),xlabel=r'Tested $\mu$',ylabel=r'$q(\mu)$',
                       title='NI profiled' if syst else 'Stat-only (NI fixed)')
                ax.axhline(1.,color='.6',ls=':',lw=1)
                ax.legend(fontsize=10)
            save(fig,name)
        fig,ax = plt.subplots(figsize=(8,5))
        bins = min(study['models']);model = study['models'][bins]
        for mu in (0.,1.,1.4):
            values = model.intensity(mu,[0.])
            if np.any(values < 0) or not np.isfinite(values).all() or values.sum() <= 0:
                continue
            ax.stairs(values/values.sum(),study['bank']['edges'][bins],label=fr'$\mu={mu:g}$')
        ax.set(xlabel=r'Fixed $z_1=\operatorname{sigmoid}(s_1/0.04)$',ylabel='Probability per bin',
               title=fr'Fixed learned score at $\mu=1$; {bins} bins')
        scale = study['bank']['configuration']['score_scale']
        ax.set_xlabel(fr'Fixed $z_1=\operatorname{{sigmoid}}(s_1/{scale:g})$')
        ax.legend();save(fig,'distributions')
    return figures
