"""Resumable, bounded-memory selected simulation for notebook 08.

Fresh iid component samples supply SHAPES. Selected rates and learned-map
normalizers stay frozen on the original integration samples. Stopping after a
selected-event target must not be used to re-estimate an acceptance as N/G.
No analytical density is evaluated outside the simulator itself.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .data import PROCESSES, SAMPLES, SAMPLE_STREAMS, SPLITS, selected_split, sample_size, file_digest, save_json
from .inference import component_coefficients, _bin_indices
from .practical import PracticalRun, SimulationHistogram
from .toy_integration_diagnostics import _atomic_npz

VERSION = 1


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _count_digest(counts, completed, ranges):
    digest = hashlib.sha256()
    for value in (counts, completed, ranges):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _array(path, value):
    temporary = path.with_suffix('.npy.partial')
    with temporary.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)
    return file_digest(path)


def _read_array(directory, item):
    path = directory / item['file']
    if not path.is_file() or file_digest(path) != item['sha256']:
        raise ValueError(f'Cached simulation changed or is missing: {path}')
    return np.load(path, allow_pickle=False)


def _stage_predictor(source, target):
    """Relocate only inference files, resolving filtered relative member paths."""
    target.mkdir(parents=True, exist_ok=True)
    if (source/'ensemble.json').exists():
        payload = json.loads((source/'ensemble.json').read_text())
        members = []
        for i, member in enumerate(payload['members']):
            name = f'members/{i:04d}'
            _stage_predictor((source/member).resolve(), target/name)
            members.append(name)
        save_json(target/'ensemble.json', {**payload, 'members': members})
    else:
        for name in ('model.pt', 'calibrator.joblib'):
            src, dst = source/name, target/name
            if src.exists():
                if not dst.exists() or file_digest(src) != file_digest(dst):
                    temporary = dst.with_suffix(dst.suffix+'.partial')
                    shutil.copyfile(src, temporary)
                    os.replace(temporary, dst)
            elif dst.exists():
                dst.unlink()


def prepare_colab_samples(run, *, selected_per_source=10_000_000, seed=20261111,
                          selected_chunk=250_000, generation_chunk=250_000,
                          inference_batch=8192, normalization_events=250_000,
                          model_cache=None, progress=print):
    """Generate six independent selected streams and cache three learned ratios.

    The independent stream namespace differs from training and toy streams.
    Complete chunks, RNG states and ratio predictions persist on Drive. Reuse
    validates their hashes; interruption replays only the unfinished chunk.
    Extending the target preserves earlier chunks. Exact analysis prefixes can
    end inside a chunk; excess cached events are not included in their weights.
    """
    from .training import load_predictor
    from .toy_sources import load_frozen_selector
    selected_per_source = _integer(selected_per_source, 'selected_per_source', 2)
    selected_chunk = _integer(selected_chunk, 'selected_chunk', 2)
    generation_chunk = _integer(generation_chunk, 'generation_chunk')
    inference_batch = _integer(inference_batch, 'inference_batch')
    normalization_events = _integer(normalization_events, 'normalization_events', 2)
    seed = _integer(seed, 'seed', 0)
    owner = run
    while not hasattr(owner, 'filter_manifest') and hasattr(owner, 'base_run'):
        owner = owner.base_run
    if not hasattr(owner, 'filter_manifest'):
        raise ValueError('Reuse notebook 03 filtered ensembles before generating inference samples')
    selector, threshold, selection = load_frozen_selector(run)
    records = []
    for sample, (process, alpha) in SAMPLES.items():
        source = selected_split(run, sample, 'integration')
        a, b = SPLITS['integration']
        generated = int(b*sample_size(run, sample))-int(a*sample_size(run, sample))
        if not 2 <= len(source) <= generated:
            raise ValueError(f'Invalid original acceptance sample: {sample}')
        records.append(dict(sample=sample, selected=len(source), generated=generated,
            rate=run.model.component_yield(process, alpha_ni=alpha)*len(source)/generated,
            source_sha256=file_digest(run.path('selected', f'{sample}_integration.npy'))))
    configuration = dict(version=VERSION, seed=seed, stream_namespace=0x08C01,
        selected_chunk=selected_chunk, generation_chunk=generation_chunk,
        inference_batch=inference_batch, normalization_events=normalization_events,
        filter_identity=owner.filter_manifest['identity'], selection=selection,
        physics=run.model.to_dict(), original_samples=records,
        normalization='Frozen original selected rates; fresh samples estimate conditional shapes only')
    fingerprint = _digest(configuration)
    directory = run.path('results', 'colab_mc', fingerprint[:20])
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory/'configuration.json', {**configuration, 'fingerprint': fingerprint})
    rates = np.array([r['rate'] for r in records])
    predictors = {}
    def predictor(task):
        if task not in predictors:
            source = run.path('models', 'ratios', task)
            if model_cache is not None:
                target = Path(model_cache)/fingerprint[:20]/task
                _stage_predictor(source, target)
                source = target
            predictors[task] = load_predictor(source)
        return predictors[task]
    norm_path = directory/'normalizers.json'
    if norm_path.exists():
        normalizers = np.array(json.loads(norm_path.read_text()), dtype=float)
    else:
        points = selected_split(run, 'S', 'integration')[:normalization_events]
        normalizers = np.r_[1., [predictor(f'{p}_over_S').predict_ratio(
            points, batch_size=inference_batch).mean(dtype=np.float64) for p in PROCESSES[1:]]]
        save_json(norm_path, normalizers.tolist())
    if normalizers.shape != (4,) or not np.isfinite(normalizers).all() or np.any(normalizers <= 0):
        raise ValueError('Invalid frozen learned-map normalization')
    configuration['normalizers'] = normalizers.tolist()
    manifests = {}
    for si, (sample, (process, alpha)) in enumerate(SAMPLES.items()):
        manifest_path = directory/f'{sample}.json'
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x08C01, SAMPLE_STREAMS[sample]]))
        if manifest_path.exists():
            state = json.loads(manifest_path.read_text())
            if state['fingerprint'] != fingerprint:
                raise ValueError('Simulation checkpoint configuration mismatch')
            rng.bit_generator.state = state['rng_state']
        else:
            state = dict(fingerprint=fingerprint, chunks=[], generated=0, selected=0,
                         rng_state=rng.bit_generator.state)
        if state['selected'] != len(state['chunks'])*selected_chunk:
            raise ValueError('Malformed selected simulation checkpoint')
        while state['selected'] < selected_per_source:
            points = np.empty((selected_chunk, 3), dtype=np.float32)
            filled, generated = 0, 0
            acceptance = records[si]['selected']/records[si]['generated']
            while filled < selected_chunk:
                draw = min(generation_chunk, max(1, int(np.ceil((selected_chunk-filled)/acceptance))))
                raw = np.asarray(run.model.sample_component(process, draw, rng, alpha_ni=alpha), dtype=np.float32)
                accepted = raw[selector.predict_proba(raw, batch_size=inference_batch)[:, 0] >= threshold]
                take = min(len(accepted), selected_chunk-filled)
                points[filled:filled+take] = accepted[:take]
                filled += take
                generated += draw
                if generated >= max(1000*selected_chunk, 100*selected_chunk/acceptance):
                    raise RuntimeError(f'Preselection accepts too few {sample} events; inspect selector')
            path = directory/f'{sample}_{len(state["chunks"]):05d}_events.npy'
            item = dict(file=path.name, sha256=_array(path, points), count=selected_chunk, ratios={})
            state['chunks'].append(item)
            state.update(selected=state['selected']+selected_chunk, generated=state['generated']+generated,
                         rng_state=rng.bit_generator.state)
            save_json(manifest_path, state)
            progress(f'{sample}: {state["selected"]:,}/{selected_per_source:,} selected events cached')
        for ci, chunk in enumerate(state['chunks'][:int(np.ceil(selected_per_source/selected_chunk))]):
            points = _read_array(directory, chunk)
            if points.shape != (selected_chunk, 3) or not np.isfinite(points).all():
                raise ValueError('Malformed cached selected events')
            for process_ratio in PROCESSES[1:]:
                if process_ratio in chunk['ratios']:
                    ratio = _read_array(directory, chunk['ratios'][process_ratio])
                else:
                    ratio = predictor(f'{process_ratio}_over_S').predict_ratio(points, batch_size=inference_batch)
                    if ratio.shape != (selected_chunk,) or not np.isfinite(ratio).all() or np.any(ratio <= 0):
                        raise ValueError('Invalid learned ratio on fresh simulation; inspect ensembles')
                    path = directory/f'{sample}_{ci:05d}_{process_ratio}_ratio.npy'
                    chunk['ratios'][process_ratio] = dict(file=path.name, sha256=_array(path, ratio))
                    save_json(manifest_path, state)
                    progress(f'{sample}: chunk {ci+1}, {process_ratio}/S evaluated')
                if ratio.shape != (selected_chunk,) or not np.isfinite(ratio).all() or np.any(ratio <= 0):
                    raise ValueError('Malformed cached learned ratios')
        manifests[sample] = state
    summary = pd.DataFrame([dict(sample=s, selected_cached=m['selected'], selected_used=selected_per_source,
        generated=m['generated'], frozen_selected_yield=rates[i], event_weight=rates[i]/selected_per_source)
        for i,(s,m) in enumerate(manifests.items())])
    summary.to_csv(directory/'sample_counts.csv', index=False)
    progress(f'Colab sample cache: {directory}')
    return dict(directory=directory, fingerprint=fingerprint, rates=rates, normalizers=normalizers,
                manifests=manifests, target=selected_per_source, sample_counts=summary)


class ColabStudyRun(PracticalRun):
    def __init__(self, base_run, root, target):
        self.base_run = base_run
        self.root, self.model = base_run.root, base_run.model
        self.config = {**base_run.config, 'simulation_per_sample': target,
                       'template_source': 'generated_colab_mc_v1'}
        self.evaluation_root = Path(root)
        for name in ('results', 'plots', 'workspaces'):
            (self.evaluation_root/name).mkdir(parents=True, exist_ok=True)


class HistogramBank:
    """Compact direct counts for frozen eta/edge specifications and MC prefixes."""
    def __init__(self, samples, specifications, prefixes, counts, ranges, directory, yield_policy=None):
        self.samples, self.specifications, self.prefixes = samples, specifications, prefixes
        self.counts, self.ranges, self.directory = counts, ranges, directory
        self.lookup = {(s['eta'], s['n_bins']): i for i,s in enumerate(specifications)}
        self.target = max(prefixes)
        self.yield_policy = yield_policy

    def raw_counts(self, eta, n_bins, size=None):
        key = (float(eta), int(n_bins))
        if key not in self.lookup:
            raise ValueError(f'Partition {key} not cached; rerun the histogram-bank cell with this specification')
        return self.counts[self.prefixes.index(self.target if size is None else size), self.lookup[key], :, :n_bins]

    def anchors(self, eta, n_bins, size=None):
        size = self.target if size is None else size
        yields = self.raw_counts(eta, n_bins, size)*self.samples['rates'][:, None]/size
        nominal = yields[:4]
        down, up = nominal.copy(), nominal.copy()
        down[3], up[3] = yields[5], yields[4]
        return np.stack([nominal, down, up])

    def model(self, eta, n_bins, *, truth=1., systematics=True, size=None):
        a = self.anchors(eta, n_bins, size)
        variations = a[1:2] if systematics else np.empty((0,4,n_bins))
        up = a[2:3] if systematics else np.empty((0,4,n_bins))
        return SimulationHistogram(a[0], variations, up, np.ones(n_bins), component_coefficients(truth)@a[0],
                                   yield_policy=self.yield_policy)

    def study_run(self, run):
        from .mc_yields import yield_policy
        policy = yield_policy(run.config)
        root = self.directory / ('inference_' + _digest(policy.metadata())[:12])
        return ColabStudyRun(run, root, self.target)

    def collapse_table(self, etas, n_bins):
        from .pipeline import observable_bin_edges
        edges = observable_bin_edges(n_bins, 'reference_ratio')
        central = np.searchsorted(edges,.5,side='right')-1
        rows=[]
        for eta in etas:
            i=self.lookup[float(eta),n_bins]
            rows.append(dict(eta=eta, n_bins=n_bins, z_min=self.ranges[i,0], z_max=self.ranges[i,1],
                occupied_bins=np.count_nonzero(self.raw_counts(eta,n_bins).sum(axis=0)),
                central_low=edges[central], central_high=edges[central+1]))
        return pd.DataFrame(rows)


def fill_colab_histograms(run, samples, *, etas, bin_counts=(4,12,24,36,60),
                          prefixes=(250_000,1_000_000,10_000_000), progress=print):
    """Stream cached predictions once; checkpoint integer counts after each chunk."""
    from .pipeline import observable_bin_edges
    from .toy_study import reference_observable
    prefixes=sorted(set(_integer(n,'prefix',2) for n in prefixes))
    if not prefixes or max(prefixes)>samples['target']:
        raise ValueError('Prefixes must not exceed the prepared selected-event target')
    bin_counts=sorted(set(_integer(n,'n_bins',2) for n in bin_counts))
    etas=np.unique(np.asarray(etas,dtype=float))
    if not len(etas) or not np.isfinite(etas).all() or np.any(etas<0):
        raise ValueError('etas must be finite and nonnegative')
    power=float(run.config['reference_ratio_power'])
    specs=[dict(eta=float(eta),n_bins=n,edges=observable_bin_edges(n,'reference_ratio').tolist())
           for eta in etas for n in bin_counts]
    metadata=dict(version=VERSION,sample_fingerprint=samples['fingerprint'],prefixes=prefixes,
                  specifications=specs,power=power,rates=samples['rates'].tolist(),
                  normalizers=samples['normalizers'].tolist())
    fingerprint=_digest(metadata)
    directory=samples['directory']/'histograms'/fingerprint[:20]
    directory.mkdir(parents=True,exist_ok=True)
    save_json(directory/'configuration.json',{**metadata,'fingerprint':fingerprint})
    checkpoint=directory/'counts.npz'
    shape=(len(prefixes),len(specs),6,max(bin_counts))
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as saved:
            if str(saved['fingerprint'])!=fingerprint:
                raise ValueError('Histogram checkpoint fingerprint mismatch')
            counts=saved['counts']; completed=saved['completed']; ranges=saved['ranges']
            if str(saved['payload_sha256']) != _count_digest(counts, completed, ranges):
                raise ValueError('Histogram checkpoint payload changed')
        if counts.shape!=shape or completed.shape!=(6,) or ranges.shape!=(len(specs),2):
            raise ValueError('Malformed histogram checkpoint')
    else:
        counts=np.zeros(shape,dtype=np.int64); completed=np.zeros(6,dtype=np.int64)
        ranges=np.tile([np.inf,-np.inf],(len(specs),1))
    for si,(sample,manifest) in enumerate(samples['manifests'].items()):
        start=0
        for ci,chunk in enumerate(manifest['chunks']):
            if start>=max(prefixes): break
            length=min(chunk['count'],max(prefixes)-start)
            if ci<completed[si]:
                start+=chunk['count']; continue
            ratio=np.vstack([np.ones(length)]+[_read_array(samples['directory'],chunk['ratios'][p])[:length]
                                              for p in PROCESSES[1:]])
            fields=samples['rates'][:4,None]*ratio/samples['normalizers'][:,None]
            for ei,eta in enumerate(etas):
                z=reference_observable(fields,samples['rates'][:4],eta,power,0.)
                for bi,n in enumerate(bin_counts):
                    i=ei*len(bin_counts)+bi
                    indices=_bin_indices(z,np.asarray(specs[i]['edges']))
                    ranges[i]=[min(ranges[i,0],z.min()),max(ranges[i,1],z.max())]
                    for pi,size in enumerate(prefixes):
                        take=min(length,max(0,size-start))
                        if take: counts[pi,i,si,:n]+=np.bincount(indices[:take],minlength=n)
            completed[si]=ci+1
            _atomic_npz(checkpoint,counts=counts,completed=completed,ranges=ranges,fingerprint=np.array(fingerprint),
                        payload_sha256=np.array(_count_digest(counts, completed, ranges)))
            progress(f'Histogram cache: {sample} {min(start+length,max(prefixes)):,}/{max(prefixes):,}')
            start+=chunk['count']
    for pi,size in enumerate(prefixes):
        if not np.all(counts[pi].sum(axis=-1)==size):
            raise ValueError('Histogram totals do not equal the selected-event prefix')
    from .mc_yields import yield_policy
    return HistogramBank(samples,specs,prefixes,counts,ranges,directory,yield_policy(run.config))


def template_diagnostics(run, bank, *, etas, n_bins=36):
    """Exact quadratic minimum in sqrt(mu), at a declared NI nuisance grid.

    The NI grid is a diagnostic, not a proof between nuisance grid points.
    Fixed selected sample sizes use multinomial marginal MC variances; these
    uncertainties do not include the original acceptance-rate uncertainty.
    """
    low,high=np.sqrt(np.asarray(run.config['mu_fit_bounds'],float))
    alphas=np.unique(np.r_[np.linspace(*run.config['nuisance_bounds'],13),-1.,0.,1.])
    alphas=alphas[(alphas>=run.config['nuisance_bounds'][0])&(alphas<=run.config['nuisance_bounds'][1])]
    rows=[]
    for size in bank.prefixes:
        for eta in etas:
            model=bank.model(eta,n_bins,truth=run.config['asimov_mu'],size=size)
            minimum=np.full(n_bins,np.inf); worst_alpha=np.zeros(n_bins); worst_mu=np.zeros(n_bins)
            for alpha in alphas:
                s,sbi,b,ni=model.components([alpha]);interference=sbi-s-b
                vertex=np.clip(np.divide(-interference,2*s,out=np.full(n_bins,low),where=s>0),low,high)
                ks=np.stack([np.full(n_bins,low),np.full(n_bins,high),vertex])
                values=s*ks**2+interference*ks+b+ni
                which=np.argmin(values,axis=0);values=values[which,np.arange(n_bins)]
                update=values<minimum
                minimum[update]=values[update];worst_alpha[update]=alpha
                worst_mu[update]=ks[which,np.arange(n_bins)][update]**2
            nominal=model.nominal;truth=float(run.config['asimov_mu'])
            truth_means=component_coefficients(truth)@nominal
            inv=.5/np.sqrt(truth)
            derivative=np.array([1-inv,inv,-inv,0.])@nominal
            tested=component_coefficients(eta)@nominal
            used, audit = model.yield_policy.apply(tested)
            p=bank.raw_counts(eta,n_bins,size)[:4]/size
            variance=np.sum(component_coefficients(eta)[:,None]**2*bank.samples['rates'][:4,None]**2*p*(1-p)/(size-1),axis=0)
            for b in range(n_bins):
                rows.append(dict(selected_per_source=size,eta=float(eta),bin=b,
                    minimum_expected=minimum[b],mu_at_minimum=worst_mu[b],alpha_at_minimum=worst_alpha[b],
                    negative=bool(minimum[b]<0),truth_mean=truth_means[b],
                    nonregular_mu_at_truth=bool(truth_means[b]==0 and derivative[b]!=0),
                    expected_at_test=tested[b],mc_sigma_at_test=np.sqrt(variance[b]),
                    expected_used_at_test=used[b],zeroed_at_test=bool(used[b]!=tested[b]),
                    test_zeroing_accepted=audit['zeroing_accepted'],test_negative_total=audit['negative_total'],
                    test_zeroed_total=audit['zeroed_total'],
                    incomplete_NI_support_bins=model.incomplete_support_bins))
    frame=pd.DataFrame(rows)
    frame.to_csv(run.path('results','reference_ratio_mc_bin_diagnostics.csv'),index=False)
    save_json(run.path('results','reference_ratio_mc_diagnostic_grid.json'),
              dict(mu_bounds=run.config['mu_fit_bounds'],alpha_grid=alphas.tolist(),
                   yield_policy=model.yield_policy.metadata(),
                   note='Raw quadratic mu minimum at sampled NI values; separately report bounded total-yield zeroing'))
    return frame


def convergence_scans(run, bank, *, etas, n_bins=36):
    from .practical import fit_direct_asimov
    rows=[]
    for size in bank.prefixes:
        for eta in etas:
            model=bank.model(eta,n_bins,truth=run.config['asimov_mu'],size=size)
            for systematics in (False,True):
                fit=fit_direct_asimov(run,model,float(eta),systematics)
                rows.append(dict(selected_per_source=size,mu=float(eta),n_bins=n_bins,
                                 systematics=systematics,**fit))
    frame=pd.DataFrame(rows)
    frame.to_csv(run.path('results','reference_ratio_mc_convergence.csv'),index=False)
    return frame
