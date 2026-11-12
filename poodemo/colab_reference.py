"""Reuse notebook 08's selected streams for bounded-memory empirical CDFs."""
from collections import OrderedDict
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.special import logit

from .colab_mc import ColabStudyRun
from .data import PROCESSES, SAMPLES, file_digest, save_json
from .inference import component_coefficients
from .mc_yields import yield_policy
from .pipeline import observable_bin_edges, _histogram_model
from .practical import LearnedMap
from .toy_study import reference_observable


def publish_08_baseline(base_run, study_run, samples, templates, n_bins):
    """Publish only after 08's analysis cells finish; raw samples stay reusable."""
    root = base_run.path('results')
    metadata = dict(version=1, sample_fingerprint=samples['fingerprint'],
        sample_directory=str(samples['directory'].relative_to(root)),
        study_directory=str(study_run.evaluation_root.relative_to(root)),
        histogram_directory=str(templates.directory.relative_to(root)),
        selected_per_source=int(samples['target']), n_bins=int(n_bins),
        power=float(study_run.config['reference_ratio_power']),
        eta_floor=float(study_run.config['reference_ratio_eta_floor']),
        yield_policy=yield_policy(study_run.config).metadata())
    save_json(root/'colab_08_baseline.json', metadata)
    provenance = root/'reference_ratio_model_source.json'
    if provenance.exists():
        save_json(study_run.path('results','reference_ratio_model_source.json'), json.loads(provenance.read_text()))
    return root/'colab_08_baseline.json'


def load_08_baseline(base_run, n_bins, *, local_cache=None, progress=print):
    """Read the matching completed 08 campaign, with no generation or NN calls."""
    if isinstance(base_run, ColabStudyRun):
        base_run = base_run.base_run
    root = base_run.path('results')
    marker = root/'colab_08_baseline.json'
    if not marker.exists():
        raise FileNotFoundError('Rerun the updated notebook 08 through its final cell first')
    saved = json.loads(marker.read_text())
    expected = dict(n_bins=int(n_bins), power=float(base_run.config['reference_ratio_power']),
                    eta_floor=float(base_run.config['reference_ratio_eta_floor']),
                    yield_policy=yield_policy(base_run.config).metadata())
    if any(saved.get(k) != v for k,v in expected.items()):
        raise ValueError('08/09 bin count, power, eta floor or yield-zeroing policy differ; rerun matching 08')
    def contained(name):
        path = (root/saved[name]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('08 manifest path escapes this evaluation')
        return path
    directory = contained('sample_directory')
    configuration = json.loads((directory/'configuration.json').read_text())
    owner = base_run
    while not hasattr(owner, 'filter_manifest') and hasattr(owner, 'base_run'):
        owner = owner.base_run
    if configuration['filter_identity'] != owner.filter_manifest['identity']:
        raise ValueError('The completed 08 bank uses a different filtered ensemble')
    if configuration['fingerprint'] != saved['sample_fingerprint']:
        raise ValueError('08 sample fingerprint mismatch')
    if configuration['physics'] != base_run.model.to_dict():
        raise ValueError('08/09 simulator configuration differs')
    manifests = {s: json.loads((directory/f'{s}.json').read_text()) for s in SAMPLES}
    target = saved['selected_per_source']
    for state in manifests.values():
        if state['fingerprint'] != saved['sample_fingerprint'] or state['selected'] < target:
            raise ValueError('Incomplete or inconsistent 08 sample manifest')
    samples = dict(directory=directory, fingerprint=saved['sample_fingerprint'], target=target,
        rates=np.array([r['rate'] for r in configuration['original_samples']]),
        normalizers=np.array(json.loads((directory/'normalizers.json').read_text())), manifests=manifests)
    histogram_config = json.loads((contained('histogram_directory')/'configuration.json').read_text())
    if (histogram_config['sample_fingerprint'] != samples['fingerprint']
            or not np.array_equal(histogram_config['rates'], samples['rates'])
            or not np.array_equal(histogram_config['normalizers'], samples['normalizers'])
            or max(histogram_config['prefixes']) != target):
        raise ValueError('08 rates, normalizers or selected target changed after histogramming')
    # Stage ratios once, not events. CDF construction repeatedly reads local
    # arrays; 60M selected events never become a dense multi-component matrix.
    target_dir = Path(local_cache)/samples['fingerprint'][:20] if local_cache else directory
    target_dir.mkdir(parents=True, exist_ok=True)
    for sample, state in manifests.items():
        count = 0
        for chunk in state['chunks']:
            if count >= target: break
            for process in PROCESSES[1:]:
                item = chunk['ratios'][process]
                if Path(item['file']).name != item['file']:
                    raise ValueError('Ratio manifest must contain plain file names')
                source, dest = directory/item['file'], target_dir/item['file']
                if not dest.exists() or file_digest(dest) != item['sha256']:
                    if file_digest(source) != item['sha256']:
                        raise ValueError('08 ratio cache changed: '+str(source))
                    if source.resolve() != dest.resolve():
                        temporary = dest.with_suffix('.partial')
                        shutil.copyfile(source, temporary); temporary.replace(dest)
                    if file_digest(dest) != item['sha256']:
                        raise ValueError('Staged ratio cache hash mismatch')
            count += chunk['count']
        progress(f'08 ratios verified/staged: {sample}, {target:,} events')
    samples['ratio_directory'] = target_dir
    return ColabStudyRun(base_run, contained('study_directory'), target), samples


class ColabTemplateBank:
    """Exact per-source ECDFs; uniform source weights require no prefix arrays."""
    def __init__(self, run, samples, n_bins=36, cdf_cache_size=2):
        self.run, self.samples = run, samples
        self.edges = observable_bin_edges(n_bins, 'reference_ratio')
        self.floor = float(run.config['reference_ratio_eta_floor'])
        self.power = float(run.config['reference_ratio_power'])
        self.yield_policy = yield_policy(run.config)
        self.cache_size = cdf_cache_size
        self.histograms, self.cdf_cache = {}, OrderedDict()
        self.central = np.searchsorted(self.edges, .5, side='right')-1
        self.quad = dict(practical=True, fingerprint=samples['fingerprint'])
        r = samples['rates']
        self.totals = r[[[0,1,2,3],[0,1,2,5],[0,1,2,4]]]
        self.map = LearnedMap(None, r[:4], samples['normalizers'])

    def _fields(self, state):
        start = 0
        directory = self.samples.get('ratio_directory', self.samples['directory'])
        for chunk in state['chunks']:
            take = min(chunk['count'], self.samples['target']-start)
            if take <= 0: break
            ratios = np.vstack([np.ones(take)] + [
                np.load(directory/chunk['ratios'][p]['file'], mmap_mode='r')[:take]
                for p in PROCESSES[1:]])
            yield start, self.samples['rates'][:4,None]*ratios/self.samples['normalizers'][:,None]
            start += take

    def _anchors(self, source_counts):
        yields = np.asarray(source_counts)*self.samples['rates'][:,None]/self.samples['target']
        return yields[[[0,1,2,3],[0,1,2,5],[0,1,2,4]]]

    def model(self, anchors, counts):
        return _histogram_model(anchors, counts, practical=True, yield_policy=self.yield_policy)

    def exact(self, eta):
        eta = max(float(eta), self.floor)
        if eta not in self.histograms:
            raw = np.zeros((6, len(self.edges)-1), dtype=np.int64)
            if eta == 1.:
                raw[:,self.central] = self.samples['target']
            else:
                for si,state in enumerate(self.samples['manifests'].values()):
                    for _,fields in self._fields(state):
                        z = reference_observable(fields,self.samples['rates'][:4],eta,self.power,0.)
                        raw[si] += np.histogram(z,self.edges)[0]
            self.histograms[eta] = self._anchors(raw)
        return self.histograms[eta]

    def counts(self, eta):
        return component_coefficients(self.run.config['asimov_mu']) @ self.exact(eta)[0]

    def _cdf(self, anchor):
        anchor = max(float(anchor), self.floor)
        if anchor in self.cdf_cache:
            self.cdf_cache.move_to_end(anchor)
            return self.cdf_cache[anchor]
        while len(self.cdf_cache) >= self.cache_size:
            self.cdf_cache.popitem(last=False)
        coordinates = []
        for state in self.samples['manifests'].values():
            u = np.empty(self.samples['target'])
            for start,fields in self._fields(state):
                u[start:start+fields.shape[1]] = self.map.u(fields, anchor)
            u.sort()
            coordinates.append(u)
        self.cdf_cache[anchor] = coordinates
        return coordinates

    def cdf_u_bins(self, anchor, eta):
        if not np.isfinite(eta) or eta == 1.:
            raise ValueError('Use exact collapse for eta=1')
        boundaries = logit(self.edges)/(self.power*(eta-1.))
        reverse = eta < 1.
        if reverse: boundaries = boundaries[::-1]
        raw = np.array([np.diff(np.searchsorted(u, boundaries, side='right' if reverse else 'left'))
                        for u in self._cdf(anchor)])
        return self._anchors(raw[:,::-1] if reverse else raw)

    def with_binning(self, run, n_bins):
        other = ColabTemplateBank(run, self.samples, n_bins, self.cache_size)
        other.cdf_cache = self.cdf_cache
        return other
