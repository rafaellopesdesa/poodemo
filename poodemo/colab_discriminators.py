"""Fixed discriminator histograms on the completed 08 bank; never generate MC.

All moments describe finite selected-sample SHAPE precision conditional on the
frozen yields and learned map. They are diagnostics, not likelihood nuisances.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from . import pipeline
from .colab_mc import HistogramBank, _count_digest, _digest, _integer, _read_array, fill_colab_histograms
from .data import PROCESSES, SAMPLES, save_json
from .diagnostics import local_asimov_information, local_width
from .discriminator_study import AXES, DEFAULT_POWERS, FIXED_LABELS, REFERENCE_LABEL, _information
from .discriminator_unbinned import LEARNED_LABEL, learned_reference
from .inference import component_coefficients
from .mc_yields import yield_policy
from .practical import SimulationHistogram, fit_direct_asimov
from .toy_integration_diagnostics import _atomic_npz

MAX_SELECTED = 10_000_000
VERSION = 1


def cached_analytic_quadrature(run):
    """Require the completed reference; never fall back to preselection calls."""
    names = ('x', 'weights', 'nominal', 'down', 'up', 'truth', 'stratum', 'half')
    if not all(run.path('quadrature', name+'.npy').is_file() for name in names):
        raise FileNotFoundError('Notebook 10 requires the completed analytical quadrature from 03/08')
    return pipeline.prepare_quadrature(run)  # Validates the saved selection identity.


def completed_histograms(run, samples):
    """Read and validate 08's compact direct counts without refilling anything."""
    root = samples['histogram_directory']
    meta = json.loads((root/'configuration.json').read_text())
    if _digest({k: v for k, v in meta.items() if k != 'fingerprint'}) != meta['fingerprint']:
        raise ValueError('Completed 08 histogram metadata changed')
    if (meta['sample_fingerprint'] != samples['fingerprint']
            or meta['power'] != run.config['reference_ratio_power']
            or not np.array_equal(meta['rates'], samples['rates'])
            or not np.array_equal(meta['normalizers'], samples['normalizers'])):
        raise ValueError('Completed 08 histogram configuration does not match this study')
    with np.load(root/'counts.npz', allow_pickle=False) as z:
        counts, completed, ranges = z['counts'], z['completed'], z['ranges']
        if (str(z['fingerprint']) != meta['fingerprint']
                or str(z['payload_sha256']) != _count_digest(counts, completed, ranges)):
            raise ValueError('Completed 08 histogram checkpoint changed')
    if max(meta['prefixes']) != samples['target']:
        raise ValueError('08 histogram target differs from the completed sample target')
    for i, size in enumerate(meta['prefixes']):
        if not np.all(counts[i].sum(axis=-1) == size):
            raise ValueError('Incomplete 08 histogram counts')
    return HistogramBank(samples, meta['specifications'], meta['prefixes'], counts,
                         ranges, root, yield_policy(run.config))


def fixed_counts(samples, n_bins=36, powers=None, progress=print):
    """Stream cached ratios into a resumable six-source 3D count array.

    The 1D and 2D histograms are exact projections of the same fixed 3D cells.
    Only one selected chunk is in memory. No events or predictors are loaded.
    """
    target = _integer(samples['target'], 'selected target', 2)
    if target > MAX_SELECTED:
        raise ValueError('Notebook 10 is limited to 10M selected events per source')
    n_bins = _integer(n_bins, 'n_bins', 2)
    powers = dict(DEFAULT_POWERS if powers is None else powers)
    if set(powers) != set(AXES) or any(not np.isfinite(powers[k]) or powers[k] <= 0 for k in AXES):
        raise ValueError('Provide positive finite NI, SBI and B powers')
    if tuple(samples['manifests']) != tuple(SAMPLES):
        raise ValueError('Unexpected source order in the completed sample manifest')
    edges = pipeline.observable_bin_edges(n_bins, 'reference_ratio')
    meta = dict(version=VERSION, sample_fingerprint=samples['fingerprint'], target=target,
                powers=powers, edges=edges.tolist(), rates=samples['rates'].tolist(),
                normalizers=samples['normalizers'].tolist())
    fingerprint = _digest(meta)
    root = samples['directory']/'fixed_discriminators'/fingerprint[:20]
    root.mkdir(parents=True, exist_ok=True)
    save_json(root/'configuration.json', meta)
    path = root/'counts.npz'
    shape = (6, n_bins, n_bins, n_bins)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            counts, completed, saturated = z['counts'], z['completed'], z['saturated']
            if (str(z['fingerprint']) != fingerprint
                    or str(z['payload_sha256']) != _count_digest(counts, completed, saturated)):
                raise ValueError('Fixed discriminator checkpoint changed')
        if counts.shape != shape or completed.shape != (6,) or saturated.shape != (6, 3):
            raise ValueError('Malformed fixed discriminator checkpoint')
    else:
        counts = np.zeros(shape, dtype=np.int64)
        completed = np.zeros(6, dtype=np.int64)
        saturated = np.zeros((6, 3), dtype=np.int64)
    directory = samples.get('ratio_directory', samples['directory'])
    for si, (source, state) in enumerate(samples['manifests'].items()):
        start = 0
        for ci, chunk in enumerate(state['chunks']):
            take = min(chunk['count'], target-start)
            if take <= 0:
                break
            if ci < completed[si]:
                start += take
                continue
            indices = []
            for ai, name in enumerate(AXES):
                j = PROCESSES.index(name)
                ratio = _read_array(directory, chunk['ratios'][name])[:take]
                if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0):
                    raise ValueError('Cached learned ratios must be positive and finite')
                z = expit(powers[name]*(np.log(ratio)-np.log(samples['normalizers'][j])
                                       + np.log(samples['normalizers'][0])))
                indices.append(np.minimum(np.searchsorted(edges, z, side='right')-1, n_bins-1))
                saturated[si, ai] += np.count_nonzero((z == 0) | (z == 1))
            flat = np.ravel_multi_index(indices, (n_bins,)*3)
            counts[si] += np.bincount(flat, minlength=n_bins**3).reshape((n_bins,)*3)
            completed[si] = ci+1
            start += take
            _atomic_npz(path, counts=counts, completed=completed, saturated=saturated,
                        fingerprint=np.array(fingerprint),
                        payload_sha256=np.array(_count_digest(counts, completed, saturated)))
            progress(f'Fixed histograms: {source} {start:,}/{target:,} cached events')
    if not np.all(counts.sum(axis=(1, 2, 3)) == target):
        raise ValueError('The existing sample bank is incomplete; no events will be generated')
    return dict(counts=counts, saturated=saturated, edges=edges, powers=powers,
                fingerprint=fingerprint, directory=root)


def source_moments(counts, rates, target):
    """Unbiased multinomial marginal variance, conditional on each fixed rate.

    The negative covariance between distinct cells is not plotted. An empty
    sampled cell does not establish a zero true probability or zero uncertainty.
    """
    p = np.asarray(counts, float)/target
    rates = np.asarray(rates, float).reshape((-1,)+(1,)*(p.ndim-1))
    return rates*p, rates**2*p*(1-p)/(target-1)


def precision_table(counts, rates, target, mu=None, source=None):
    """Per-cell RAW yield and uncertainty; never divide by clipped predictions."""
    counts = np.asarray(counts).reshape(6, -1)
    means, variances = source_moments(counts, rates, target)
    if source is not None:
        i = list(SAMPLES).index(source)
        expected, variance, support = means[i], variances[i], counts[i]
    else:
        coefficients = component_coefficients(mu)
        expected = coefficients @ means[:4]
        variance = coefficients**2 @ variances[:4]
        support = counts[:4][coefficients != 0].sum(axis=0)
    sigma = np.sqrt(variance)
    relative = np.divide(sigma, expected, out=np.full_like(sigma, np.nan), where=expected > 0)
    status = np.where(expected > 0, 'positive', np.where(support == 0, 'no_sampled_support', 'nonpositive'))
    result = pd.DataFrame(dict(bin=np.arange(len(expected)), raw_yield=expected,
                               mc_sigma=sigma, relative_mc_sigma=relative, status=status))
    for si, name in enumerate(SAMPLES):
        result['n_'+name] = counts[si]
    return result


def _histogram(counts, samples, truth, policy, systematics=True):
    y = counts.reshape(6, -1)*samples['rates'][:, None]/samples['target']
    nominal = y[:4]
    down, up = nominal.copy(), nominal.copy()
    down[3], up[3] = y[5], y[4]
    variations = down[None] if systematics else np.empty((0, 4, y.shape[1]))
    highs = up[None] if systematics else np.empty((0, 4, y.shape[1]))
    return SimulationHistogram(nominal, variations, highs, np.ones(y.shape[1]),
                               component_coefficients(truth)@nominal, yield_policy=policy)


def _occupancy(counts, samples, truth, label, eta=np.nan):
    flat = counts.reshape(6, -1)
    p = precision_table(flat, samples['rates'], samples['target'], mu=truth)
    support = flat[3:6]
    positive = p.raw_yield > 0
    return dict(model=label, eta=eta, total_cells=len(p), occupied_cells=int(positive.sum()),
                empty_cells=int((~positive).sum()), expected_events=float(p.raw_yield.sum()),
                incomplete_NI_support_bins=int(((support.min(axis=0) == 0)&(support.max(axis=0) > 0)).sum()),
                cells_above_10pct=int((p.relative_mc_sigma > .1).sum()),
                cells_above_30pct=int((p.relative_mc_sigma > .3).sum()),
                expected_fraction_above_10pct=float(p.loc[p.relative_mc_sigma > .1, 'raw_yield'].sum()/p.raw_yield.sum()))


def run_colab_discriminator_study(run, samples, *, n_bins=36, powers=None,
                                 recap_bins=(36, 72, 108, 144),
                                 output_tag='10_colab_10M_precision', progress=print):
    """Fixed direct fills versus saved direct 08 fills, on exactly the same bank."""
    if not output_tag or Path(output_tag).name != output_tag or output_tag in ('.', '..'):
        raise ValueError('output_tag must be a directory name')
    if run.config['reference_ratio_eta_floor'] != 0:
        raise ValueError('This 10M comparison uses the no-floor baseline')
    fixed = fixed_counts(samples, n_bins, powers, progress)
    moving = completed_histograms(run, samples)
    edges, counts = fixed['edges'], fixed['counts']
    truth = float(run.config['asimov_mu'])
    policy = yield_policy(run.config)
    reference = pd.read_csv(run.path('results', 'reference_ratio_mc_convergence.csv'))
    reference = reference[(reference.selected_per_source == samples['target']) & (reference.n_bins == n_bins)]
    grid = np.sort(reference.mu.unique())
    if not len(grid) or reference.duplicated(['mu', 'systematics']).any():
        raise ValueError('Missing or duplicated completed 08 scans')
    analytic = pd.read_csv(run.path('results', 'unbinned_scans.csv'))
    analytic = analytic[analytic.mu.isin(grid)].copy()
    for frame in (reference, analytic):
        for syst in (False, True):
            rows = frame[frame.systematics == syst]
            if len(rows) != len(grid) or not rows.valid.all() or not np.isfinite(rows.q).all():
                raise ValueError('Incomplete/invalid completed 08 reference scans')
    learned, learned_widths, learned_provenance = learned_reference(run, grid)
    quad = cached_analytic_quadrature(run)  # Only the analytical UNBINNED comparator.
    scans, information, occupancy, checks = [analytic, learned], [], [], []
    baselines = {}
    dimensions = {d: counts.sum(axis=tuple(range(d+1, 4))) if d < 3 else counts for d in (1, 2, 3)}
    for syst in (False, True):
        analytic_parent = pipeline._template_likelihood(quad, systematics=syst)
        baseline = local_width(local_asimov_information(analytic_parent, truth)['matrix'], systematics=syst)
        baselines[syst] = baseline
        information.append(_information(analytic_parent, label='analytic unbinned', truth=truth,
                                        systematics=syst, baseline=baseline))
        information.append(dict(model=LEARNED_LABEL, eta=np.nan, systematics=syst, information_mu=truth,
            information_fraction=learned_widths[syst]['information_mu_effective']/baseline['information_mu_effective'],
            information_note='physical-Asimov objective curvature; not coverage', **learned_widths[syst]))
        for d, label in enumerate(FIXED_LABELS, 1):
            raw = dimensions[d]
            model = _histogram(raw, samples, truth, policy, syst)
            information.append(_information(model, label=label, truth=truth, systematics=syst, baseline=baseline))
            rows = [dict(mu=float(mu), model=label, systematics=syst, dimension=d,
                         n_bins=n_bins, total_cells=n_bins**d, **fit_direct_asimov(run, model, mu, syst)) for mu in grid]
            scans.append(pd.DataFrame(rows))
            if not syst:
                occupancy.append(_occupancy(raw, samples, truth, label))
            progress(f'{label}: {sum(r["valid"] for r in rows)}/{len(grid)} valid; NI profiled={syst}')
        rows = []
        for mu in grid:
            eta = min((e for e, n in moving.lookup if n == n_bins), key=lambda e: abs(e-mu))
            if abs(eta-mu) > 1e-12:
                raise ValueError('08 does not contain the requested moving partition')
            model = moving.model(eta, n_bins, truth=truth, systematics=syst)
            fit = fit_direct_asimov(run, model, mu, syst)
            old = reference[(reference.systematics == syst) & (reference.mu == mu)].iloc[0]
            delta = fit['q']-old.q
            checks.append(dict(mu=mu, systematics=syst, q_08=old.q, q_10=fit['q'], delta_q=delta,
                               passed=bool(fit['valid'] and abs(delta) <= 1e-7)))
            rows.append(dict(mu=mu, model=REFERENCE_LABEL, systematics=syst, eta=mu,
                             n_bins=n_bins, total_cells=n_bins, **fit))
            information.append(_information(model, label=REFERENCE_LABEL, truth=truth, systematics=syst,
                                            baseline=baseline, eta=mu))
            if not syst:
                occupancy.append(_occupancy(moving.raw_counts(eta, n_bins), samples, truth, REFERENCE_LABEL, mu))
        scans.append(pd.DataFrame(rows))
        progress(f'Moving reference: direct 08/10 comparison complete; NI profiled={syst}')
    marginals, summaries, joints = [], [], {}
    weights = component_coefficients(truth)*samples['rates'][:4]/samples['target']
    for ai, name in enumerate(AXES):
        raw = counts.sum(axis=tuple(j for j in (1, 2, 3) if j != ai+1))
        y = raw[:4]*samples['rates'][:4, None]/samples['target']
        expected = weights @ raw[:4]
        summaries.append(dict(axis=name, power=fixed['powers'][name],
                              endpoint_event_fraction=expected[[0, -1]].sum()/expected.sum(),
                              saturated_event_fraction=(weights @ fixed['saturated'][:4, ai])/expected.sum()))
        for j, process in enumerate(PROCESSES):
            for b in range(n_bins):
                marginals.append(dict(axis=name, process=process, bin=b, low=edges[b], high=edges[b+1],
                                      expected=y[j, b], probability=y[j, b]/y[j].sum()))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        raw = counts.sum(axis=next(j+1 for j in range(3) if j not in (a, b)))
        joints[AXES[a]+'_'+AXES[b]] = np.tensordot(weights, raw[:4], axes=1)
    config = dict(version=VERSION, n_bins=n_bins, powers=fixed['powers'], systematics=[False, True],
                  reference_ratio_power=run.config['reference_ratio_power'], reference_ratio_eta_floor=0.,
                  bin_edges=edges.tolist(), asimov_mu=truth, scan_grid=grid.tolist(), morph_order='after',
                  selected_per_source=samples['target'], max_selected_per_source=MAX_SELECTED,
                  sample_fingerprint=samples['fingerprint'], fixed_fingerprint=fixed['fingerprint'],
                  rates=samples['rates'].tolist(), normalizers=samples['normalizers'].tolist(),
                  yield_policy=policy.metadata(), likelihood_source='simulation', ratio_source='learned',
                  asimov_source='same MC templates', uncertainty='fixed-N multinomial marginal; frozen rates/map',
                  no_generation=True, adaptive_binning=False, learned_unbinned=learned_provenance)
    result = dict(scans=pd.concat(scans, ignore_index=True), information=pd.DataFrame(information),
                  occupancy=pd.DataFrame(occupancy), axis_summary=pd.DataFrame(summaries),
                  axis_histograms=pd.DataFrame(marginals), joint_histograms=joints,
                  baseline_checks=pd.DataFrame(checks), config=config,
                  fixed_counts=dimensions, moving_bank=moving, samples=samples)
    output = run.path('results', output_tag)
    output.mkdir(parents=True, exist_ok=True)
    result['output'] = output
    save_json(output/'config.json', config)
    for name in ('scans', 'information', 'occupancy', 'axis_summary', 'axis_histograms', 'baseline_checks'):
        result[name].to_csv(output/(name+'.csv'), index=False)
    _atomic_npz(output/'source_counts.npz', counts=counts, rates=samples['rates'], target=np.array(samples['target']), edges=edges)
    if not result['baseline_checks'].passed.all():
        raise ValueError('Moving 10 histogram does not reproduce 08; inspect baseline_checks.csv')
    # Refill ONLY histograms from existing cached ratios, with an explicit
    # <=10M prefix. No generation/evaluation entry point is reachable here.
    recap_bins = sorted(set(_integer(n, 'recap bins', 2) for n in recap_bins) | {n_bins})
    config['recap_bin_counts'] = recap_bins
    etas = [min((e for e, n in moving.lookup if n == n_bins), key=lambda e: abs(e-mu)) for mu in grid]
    recap = fill_colab_histograms(run, samples, etas=etas, bin_counts=recap_bins,
                                  prefixes=[samples['target']], progress=progress)
    recap_scans, recap_information, recap_occupancy = [analytic, learned], [], []
    for syst in (False, True):
        recap_information.extend(r for r in information if r['systematics'] == syst and r['model'] in ('analytic unbinned', LEARNED_LABEL))
        for bins in recap_bins:
            label = f'moving reference ratio ({bins} bins)'
            for mu, eta in zip(grid, etas):
                if bins == n_bins:
                    if not np.array_equal(recap.raw_counts(eta, bins), moving.raw_counts(eta, bins)):
                        raise ValueError('The recap does not reproduce the completed 08 counts')
                model = recap.model(eta, bins, truth=truth, systematics=syst)
                fit = fit_direct_asimov(run, model, mu, syst)
                recap_scans.append(pd.DataFrame([dict(mu=mu, eta=mu, model=label, systematics=syst, n_bins=bins, **fit)]))
                recap_information.append(_information(model, label=label, truth=truth, systematics=syst,
                                                       baseline=baselines[syst], eta=mu))
                if not syst:
                    recap_occupancy.append(_occupancy(recap.raw_counts(eta, bins), samples, truth, label, mu))
            progress(f'Recap {bins} bins complete; NI profiled={syst}')
    result.update(resolution_scans=pd.concat(recap_scans, ignore_index=True),
                  resolution_information=pd.DataFrame(recap_information),
                  resolution_occupancy=pd.DataFrame(recap_occupancy), resolution_bank=recap)
    for name in ('resolution_scans', 'resolution_information', 'resolution_occupancy'):
        result[name].to_csv(output/(name+'.csv'), index=False)
    save_json(output/'config.json', config)
    return result
