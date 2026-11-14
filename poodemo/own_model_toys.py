"""Four own-model, stat-only ensembles for notebook 11.

The learned source is an explicitly normalized finite S-bank model. Histogram
sources condition on frozen component templates, including bounded zeroing.
The analytical reference uses fresh continuous events and numerical selected
rate integrals, whose integration error remains a separate limitation.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .colab_mc import _digest, _integer, _read_array
from .data import save_json, file_digest
from .mc_yields import YieldPolicy, yield_policy
from .physics import PhysicsModel
from .toy_likelihood import StatOnlyLikelihood, _fields, _polynomial, _negative_intervals, _merge_intervals
from .toy_sources import QuadraturePoissonSource
from .toy_study import _array_digest, _atomic_csv, _failure, _fit, _rng

VERSION = 1
CASES = {
    'analytic': 'Analytical unbinned',
    'learned': 'Learned unbinned (S bank)',
    'moving36': 'Moving learned observable: 36 bins',
    'fixed9': 'Fixed learned ratios: 9×9×9 bins',
}


def _policy_intervals(polynomial, policy):
    """Exact forbidden intervals for both correction budgets in kappa.

    Between roots of individual bins, the total required correction is a
    quadratic. Sweep its changes so the global fit also searches narrow valid
    regions and zeroing boundaries, instead of masking a coarse grid.
    """
    shifted = polynomial.copy()
    shifted[2] += policy.max_bin
    forbidden = list(_negative_intervals(shifted))
    events = {}
    for index, row in enumerate(polynomial.T):
        for lo, hi in _negative_intervals(row[:, None]):
            events.setdefault(lo, []).append((index, True))
            if np.isfinite(hi):
                events.setdefault(hi, []).append((index, False))
    active = set()
    edges = sorted(events)
    for i, lo in enumerate(edges):
        for index, present in events[lo]:
            if present:
                active.add(index)
            else:
                active.remove(index)
        if not active:
            continue
        hi = edges[i + 1] if i + 1 < len(edges) else np.inf
        total = polynomial[:, sorted(active)].sum(axis=1)
        total[2] += policy.max_total
        for start, end in _negative_intervals(total[:, None]):
            if max(start, lo) < min(end, hi):
                forbidden.append((max(start, lo), min(end, hi)))
    return _merge_intervals(forbidden)


class PoissonTemplate:
    """One fixed partition and its corrected mean function, shared by toys."""
    def __init__(self, fields, *, policy=YieldPolicy(.01, .05), exposure=1.):
        self.fields = _fields(fields, 'component bin means').copy()
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError('exposure must be finite and positive')
        self.polynomial = _polynomial(self.fields) * exposure
        self.policy = policy
        self.negative_intervals = _policy_intervals(self.polynomial, policy)

    def means(self, mu):
        if not np.isfinite(mu) or mu < 0:
            raise ValueError('mu must be finite and nonnegative')
        k = np.sqrt(mu)
        a, b, c = self.polynomial
        corrected, report = self.policy.apply((a*k+b)*k+c)
        return corrected, report

    def likelihood(self, counts):
        return BinnedToyLikelihood(self, counts)


class BinnedToyLikelihood(StatOnlyLikelihood):
    """Independent Poissons; correction applies to rate AND shape terms."""
    def __init__(self, template, counts):
        self.template = template
        self.counts = np.asarray(counts, dtype=float)
        if (self.counts.shape != (template.fields.shape[1],)
                or np.any(~np.isfinite(self.counts)) or np.any(self.counts < 0)):
            raise ValueError('Counts must be finite, nonnegative and match the partition')
        self.positive = self.counts > 0
        self.negative_intervals = template.negative_intervals
        self._reference_cache = {}

    def expected_count(self, mu):
        means, report = self.template.means(mu)
        return float(means.sum()) if report['zeroing_accepted'] else np.nan

    def nll(self, mu, *, reference_mu=None):
        if not np.isfinite(mu) or mu < 0:
            return np.inf
        means, report = self.template.means(mu)
        if not report['zeroing_accepted'] or np.any(means[self.positive] <= 0):
            return np.inf
        if reference_mu is None:
            return float(2*(means.sum()-np.sum(self.counts[self.positive]*np.log(means[self.positive]))))
        if reference_mu not in self._reference_cache:
            reference, check = self.template.means(reference_mu)
            if not check['zeroing_accepted'] or np.any(reference[self.positive] <= 0):
                raise ValueError('Invalid reference hypothesis')
            self._reference_cache[reference_mu] = reference
        reference = self._reference_cache[reference_mu]
        delta = means-reference
        relative = delta[self.positive]/reference[self.positive]
        # log1p is accurate near the tested point; logs avoid rounding to -1
        # when a positive candidate mean is extremely small.
        close = np.abs(relative) < .5
        logs = np.empty_like(relative)
        logs[close] = np.log1p(relative[close])
        logs[~close] = np.log(means[self.positive][~close])-np.log(reference[self.positive][~close])
        return float(2*(delta.sum()-np.sum(self.counts[self.positive]*logs)))


def load_signal_fields(samples):
    """Reuse <=10M cached S-bank ratios; no event generation or NN calls.

    The mean of each learned ratio on THIS bank is fixed once. Thus its
    empirical component integral equals the selected physical component rate,
    exactly the rate used by the generator and every likelihood evaluation.
    """
    target = _integer(samples['target'], 'selected target', 2)
    if target > 10_000_000:
        raise ValueError('Notebook 11 reference banks may not exceed 10M per source')
    fields = np.ones((4, target), dtype=np.float64)
    start = 0
    directory = samples.get('ratio_directory', samples['directory'])
    for chunk in samples['manifests']['S']['chunks']:
        take = min(chunk['count'], target-start)
        if take <= 0:
            break
        for j, name in enumerate(('SBI', 'B', 'NI'), start=1):
            values = _read_array(directory, chunk['ratios'][name])[:take]
            if len(values) != take or np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError('Invalid cached signal-bank ratio')
            fields[j, start:start+take] = values
        start += take
    if start != target:
        raise ValueError('Incomplete S bank; no additional reference events will be generated')
    normalizers = fields.mean(axis=1)
    rates = np.asarray(samples['rates'][:4], dtype=float)
    if np.any(~np.isfinite(rates)) or np.any(rates <= 0):
        raise ValueError('Selected rates must be positive and finite')
    fields *= (rates/normalizers)[:, None]
    return fields, normalizers


def prepare_own_model_toys(run, samples, *, mu_values=(0., 1.4), exposure=1., progress=print):
    """Prepare the four frozen models, reusing 08/10 and the 03 reference."""
    from .colab_discriminators import cached_analytic_quadrature, fixed_counts
    from .colab_mc import fill_colab_histograms
    from .discriminator_convergence import merge_cells
    from .toy_sources import load_frozen_selector, sample_simulator_poisson

    mus = tuple(float(mu) for mu in mu_values)
    if (not mus or len(set(mus)) != len(mus) or any(not np.isfinite(mu) or mu < 0 or mu == 1 for mu in mus)):
        raise ValueError('Use unique nonnegative hypotheses other than eta=1 (collapsed partition)')
    if run.config.get('reference_ratio_power') != 100. or run.config.get('reference_ratio_eta_floor', 0.) != 0.:
        raise ValueError('Notebook 11 requires a=100 and no eta floor')
    _integer(samples['target'], 'selected target', 2)
    if samples['target'] > 10_000_000:
        raise ValueError('Notebook 11 reference banks may not exceed 10M per source')
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError('exposure must be finite and positive')
    policy = yield_policy(run.config)
    # Direct fills include eta=0 exactly; 09 interpolation is unnecessary.
    moving = fill_colab_histograms(run, samples, etas=mus, bin_counts=[36],
                                   prefixes=[samples['target']], progress=progress)
    fine = fixed_counts(samples, n_bins=36, progress=progress)
    merged, edges = merge_cells(fine['counts'], fine['edges'], 9)
    fixed_fields = merged[:4].reshape(4, -1)*samples['rates'][:4, None]/samples['target']
    templates = {'fixed9': PoissonTemplate(fixed_fields, policy=policy, exposure=exposure)}
    templates['moving36'] = {mu: PoissonTemplate(moving.anchors(mu, 36)[0], policy=policy,
                                               exposure=exposure) for mu in mus}
    progress('Preparing the learned finite S-bank model from cached ratios')
    fields, normalizers = load_signal_fields(samples)
    rates = np.asarray(samples['rates'][:4], dtype=float)
    guard = StatOnlyLikelihood(np.empty((4, 0)), rates, exposure=exposure, validity_fields=fields)
    quad = cached_analytic_quadrature(run)
    analytic_rates = np.asarray(quad['nominal']) @ quad['weights']
    selector, threshold, selection = load_frozen_selector(run)

    def analytic_draw(mu, rng):
        points = sample_simulator_poisson(run, mu, rng, exposure=exposure,
                                         selector=selector, threshold=threshold)
        return StatOnlyLikelihood(run.model.component_densities(points).T, analytic_rates,
                                  exposure=exposure), len(points)

    configuration = dict(version=VERSION, experiment='four own-model stat-only null ensembles',
        mu_values=sorted(mus), exposure=float(exposure), sample_fingerprint=samples['fingerprint'],
        selected_per_source=int(samples['target']), rates=rates.tolist(),
        observable_normalizers=samples['normalizers'].tolist(), learned_bank_normalizers=normalizers.tolist(),
        learned_fields_sha256=_array_digest(fields), fixed_counts_sha256=_array_digest(merged),
        fixed_edges=edges.tolist(), fixed_powers=fine['powers'],
        moving_fields_sha256={str(mu): _array_digest(templates['moving36'][mu].fields) for mu in sorted(mus)},
        moving_bins=36, reference_power=100., eta_floor=0., yield_policy=policy.metadata(),
        analytic_rates=analytic_rates.tolist(), selection=selection, physics=run.model.to_dict(),
        backend_sha256={name: file_digest(Path(__file__).with_name(name+'.py'))
                        for name in ('own_model_toys', 'toy_likelihood', 'toy_sources', 'mc_yields', 'physics')},
        analytic_normalization='Original selected analytical quadrature: finite numerical integration error',
        learned_normalization='Each ratio normalized once on the frozen S bank; preserves selected component rates')
    del quad
    return dict(configuration=configuration, templates=templates, learned_fields=fields,
                learned_rates=rates, learned_intervals=guard.negative_intervals,
                analytic_rates=analytic_rates, analytic_draw=analytic_draw, exposure=float(exposure),
                mu_values=mus)


def _source(prepared, case, mu):
    exposure = prepared['exposure']
    if case == 'analytic':
        expected = exposure*float(PhysicsModel.coefficients(mu) @ prepared['analytic_rates'])
        return lambda rng: prepared['analytic_draw'](mu, rng), dict(expected_count=expected,
            source='fresh continuous physical Poisson process; numerical rate integral')
    if case == 'learned':
        fields = prepared['learned_fields']
        source = QuadraturePoissonSource({'weights': np.full(fields.shape[1], 1./fields.shape[1])},
                                         fields, mu, exposure=exposure)
        expected = exposure*float(PhysicsModel.coefficients(mu) @ prepared['learned_rates'])
        if not np.isclose(source.expected_count, expected, rtol=1e-12, atol=1e-9):
            raise ValueError('Learned generator and likelihood rates disagree')
        def draw(rng):
            indices, counts = source.draw_counts(rng)
            return StatOnlyLikelihood(fields[:, indices], prepared['learned_rates'], counts,
                exposure=exposure, validity_intervals=prepared['learned_intervals']), int(counts.sum())
        return draw, {**source.diagnostics, 'forbidden_kappa_intervals': prepared['learned_intervals'].tolist()}
    template = (prepared['templates'][case][mu] if case == 'moving36' else prepared['templates'][case])
    means, report = template.means(mu)
    if not report['zeroing_accepted']:
        raise ValueError(f'Invalid binned generator: {report}')
    def draw(rng):
        counts = rng.poisson(means)
        return template.likelihood(counts), int(counts.sum())
    return draw, dict(source='independent Poisson bin counts', expected_count=float(means.sum()),
                     occupied_bins=int(np.count_nonzero(means)), **report)


def summarize_own_model_toys(results):
    rows = []
    for (mu, case), group in results.groupby(['mu_test', 'case'], sort=False):
        good = group.loc[group.valid & np.isfinite(group.q)]
        counts = group.n_observed.dropna()
        expected = float(group.expected_count.iloc[0])
        se_count = np.sqrt(expected/len(counts)) if len(counts) and expected > 0 else np.nan
        row = dict(mu_test=mu, case=case, label=CASES[case], attempted=len(group), valid=len(good),
            failed=len(group)-len(good), mean_q=good.q.mean(), se_mean_q=good.q.sem(),
            fraction_q_zero=(good.q <= 1e-10).mean(), fraction_at_upper=good.at_upper.mean(),
            expected_count=expected, mean_count=counts.mean(),
            count_mean_pull=(counts.mean()-expected)/se_count,
            median_seconds=group.seconds.median())
        for level in (.5, .68, .9, .95, .99):
            row[f'q{int(100*level)}'] = good.q.quantile(level)
        rows.append(row)
    return pd.DataFrame(rows)


def run_own_model_toys(prepared, output, *, n_toys=1000, seed=110923,
                       mu_bounds=(0., 3.), grid_size=129, checkpoint_every=5,
                       progress_every=25, progress=print):
    """Run/resume 1000 per case and hypothesis; retain all failures, no redraw.

    A separate CSV is atomically saved every five toys and on interruption.
    Seeds depend only on case, hypothesis and toy ID. Increasing N_TOYS extends
    the ensemble; no template, training or finite-bank uncertainty is varied.
    """
    n_toys = _integer(n_toys, 'n_toys')
    grid_size = _integer(grid_size, 'grid_size', 5)
    checkpoint_every = _integer(checkpoint_every, 'checkpoint_every')
    progress_every = _integer(progress_every, 'progress_every')
    seed = _integer(seed, 'seed', 0)
    bounds = np.asarray(mu_bounds, dtype=float)
    mus = prepared['mu_values']
    if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] != 0 or bounds[1] <= 0
            or any(mu < bounds[0] or mu > bounds[1] for mu in mus)):
        raise ValueError('Fit bounds must start at zero and contain all tested hypotheses')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(**prepared['configuration'], seed=seed, mu_bounds=bounds.tolist(), grid_size=grid_size)
    fingerprint = _digest(config)
    path = output/'configuration.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('fingerprint') != fingerprint:
            raise ValueError('Toy configuration/model changed; use a new OUTPUT_TAG')
    elif list(output.glob('toys_*.csv')):
        raise ValueError('Toy results have no configuration fingerprint')
    save_json(path, dict(configuration=config, fingerprint=fingerprint, requested_toys=n_toys))
    results, audits = [], []
    for mu in mus:
        for stream, case in enumerate(CASES):
            path = output/f'toys_{case}_mu_{float(mu).hex()}.csv'
            records = pd.read_csv(path).to_dict('records') if path.exists() else []
            ids = [int(row['toy_id']) for row in records]
            if (len(set(ids)) != len(ids) or any(i < 0 for i in ids)
                    or any(row['case'] != case or row['mu_test'] != mu for row in records)):
                raise ValueError('Duplicate or incompatible cached toy IDs')
            complete = set(ids)
            error = None
            try:
                draw, audit = _source(prepared, case, mu)
                audit.update(valid=True)
            except (ValueError, FloatingPointError) as exc:
                error = f'{type(exc).__name__}: {exc}'
                audit = dict(valid=False, error=error, expected_count=None)
            audits.append(dict(mu_test=mu, case=case, **audit))
            save_json(output/'source_diagnostics.json', audits)
            progress(f'{CASES[case]}, mu={mu:g}: {sum(i < n_toys for i in complete)}/{n_toys} cached')
            dirty = 0
            try:
                for toy_id in range(n_toys):
                    if toy_id in complete:
                        continue
                    start, observed = time.perf_counter(), np.nan
                    if error:
                        fit = _failure('Generator unavailable: '+error)
                    else:
                        try:
                            likelihood, observed = draw(_rng(seed, mu, toy_id, stream))
                            fit = _fit(likelihood, mu, bounds, grid_size)
                            if case in ('moving36', 'fixed9') and fit['valid']:
                                _, check = likelihood.template.means(fit['mu_hat'])
                                fit.update(fit_zeroed_bins=check['zeroed_bins'], fit_zeroed_total=check['zeroed_total'])
                        except (ValueError, FloatingPointError, RuntimeError) as exc:
                            fit = _failure(f'{type(exc).__name__}: {exc}')
                    records.append(dict(mu_test=mu, case=case, toy_id=toy_id, n_observed=observed,
                        expected_count=(audit['expected_count'] if audit['expected_count'] is not None else np.nan), seconds=time.perf_counter()-start, **fit))
                    dirty += 1
                    if dirty >= checkpoint_every:
                        _atomic_csv(pd.DataFrame(records), path)
                        dirty = 0
                    if (toy_id+1) % progress_every == 0 or toy_id+1 == n_toys:
                        progress(f'{case}, mu={mu:g}: {toy_id+1}/{n_toys}; last toy {records[-1]["seconds"]:.2f}s')
            finally:
                if dirty:
                    _atomic_csv(pd.DataFrame(records), path)
            results.extend(row for row in records if row['toy_id'] < n_toys)
            # Release the per-hypothesis CDF before building the next source.
            draw = None
    frame = pd.DataFrame(results)
    summary = summarize_own_model_toys(frame)
    _atomic_csv(frame, output/'results.csv')
    _atomic_csv(summary, output/'summary.csv')
    return dict(results=frame, summary=summary, sources=pd.DataFrame(audits), output=output,
                fingerprint=fingerprint)
