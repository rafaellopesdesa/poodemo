"""Own-model stat-only toys for the fixed learned score from notebook 12."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .colab_mc import _digest, _integer
from .data import file_digest, save_json
from .fixed_score import fill_fixed_score
from .mc_yields import yield_policy
from .own_model_toys import PoissonTemplate, load_signal_fields, _source, summarize_own_model_toys
from .toy_likelihood import StatOnlyLikelihood
from .toy_study import _array_digest, _atomic_csv, _failure, _fit, _rng

VERSION = 1
REFERENCE_BACKENDS = ('own_model_toys', 'toy_likelihood', 'toy_sources', 'mc_yields', 'physics')


def prepare_fixed_score_toys(run, samples, *, mu_values=(0., 1.4), bin_counts=(36, 144),
                             score_scale=.04, exposure=1., progress=print):
    """Reuse 12's fixed partition; prepare the same unbinned models as 07/11."""
    from .colab_discriminators import cached_analytic_quadrature
    from .toy_sources import load_frozen_selector, sample_simulator_poisson

    mus = tuple(float(mu) for mu in mu_values)
    if not mus or len(set(mus)) != len(mus) or any(not np.isfinite(mu) or mu < 0 for mu in mus):
        raise ValueError('Use unique finite nonnegative generating hypotheses')
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError('exposure must be finite and positive')
    bank = fill_fixed_score(run, samples, bin_counts=bin_counts, score_scale=score_scale, progress=progress)
    policy = yield_policy(run.config)
    templates = {f'score{n}': PoissonTemplate(raw[:4]*samples['rates'][:4, None]/samples['target'],
        policy=policy, exposure=exposure) for n, raw in bank['counts'].items()}
    progress('Preparing the frozen learned S-bank reference; no ratio-network evaluation')
    fields, normalizers = load_signal_fields(samples)
    rates = np.asarray(samples['rates'][:4], float)
    guard = StatOnlyLikelihood(np.empty((4, 0)), rates, exposure=exposure, validity_fields=fields)
    quad = cached_analytic_quadrature(run)
    analytic_rates = np.asarray(quad['nominal']) @ quad['weights']
    selector, threshold, selection = load_frozen_selector(run)

    def analytic_draw(mu, rng):
        points = sample_simulator_poisson(run, mu, rng, exposure=exposure,
                                         selector=selector, threshold=threshold)
        return StatOnlyLikelihood(run.model.component_densities(points).T, analytic_rates,
                                  exposure=exposure), len(points)

    reference = dict(exposure=float(exposure), sample_fingerprint=samples['fingerprint'],
        selected_per_source=int(samples['target']), rates=rates.tolist(),
        observable_normalizers=samples['normalizers'].tolist(), learned_bank_normalizers=normalizers.tolist(),
        learned_fields_sha256=_array_digest(fields), analytic_rates=analytic_rates.tolist(),
        selection=selection, physics=run.model.to_dict())
    labels = {case: f'Fixed score: {case.removeprefix("score")} bins' for case in templates}
    labels.update(analytic='Analytical unbinned', learned='Learned unbinned (S bank)')
    streams = {case: 130000+int(case.removeprefix('score')) for case in templates}
    streams.update(analytic=0, learned=1)  # Same reference streams as 07/11.
    backends = (*REFERENCE_BACKENDS, 'fixed_score', 'fixed_score_toys', 'toy_study')
    hashes = {name: file_digest(Path(__file__).with_name(name+'.py')) for name in backends}
    configuration = dict(version=VERSION, experiment='fixed-score own-model stat-only toys',
        reference=reference, reference_backend_sha256={name: hashes[name] for name in REFERENCE_BACKENDS},
        mu_values=sorted(mus), histogram_identity=bank['identity'], score_anchor=1.,
        score_scale=float(score_scale), bin_counts=sorted(bank['counts']),
        fields_sha256={case: _array_digest(template.fields) for case, template in templates.items()},
        streams=streams, yield_policy=policy.metadata(), backend_sha256=hashes,
        analytic_normalization='Original analytical quadrature; finite selected-rate integration error',
        learned_normalization='Ratios normalized once on the frozen S bank, never on a toy')
    return dict(configuration=configuration, templates=templates, labels=labels, streams=streams,
        learned_fields=fields, learned_rates=rates, learned_intervals=guard.negative_intervals,
        analytic_rates=analytic_rates, analytic_draw=analytic_draw, exposure=float(exposure),
        mu_values=mus, bank=bank)


def _checked_rows(frame, case, mu):
    required = {'case', 'mu_test', 'toy_id', 'valid', 'q', 'mu_hat', 'n_observed',
                'expected_count', 'seconds', 'at_upper', 'at_lower'}
    if not required.issubset(frame.columns):
        raise ValueError('Incomplete toy checkpoint columns')
    ids = frame.toy_id.to_numpy()
    if (not np.isfinite(ids).all() or np.any(ids < 0) or np.any(ids != np.floor(ids))
            or frame.toy_id.duplicated().any() or not frame.case.eq(case).all()
            or not frame.mu_test.eq(mu).all() or not frame.valid.isin([True, False]).all()):
        raise ValueError('Duplicate or incompatible cached toy IDs/flags')
    good = frame.loc[frame.valid.astype(bool)]
    if not np.isfinite(good[['q', 'mu_hat', 'n_observed']]).all().all() or (good.q < 0).any():
        raise ValueError('Invalid values in a successful cached fit')
    return frame.to_dict('records')


def _read_references(prepared, directory, *, seed, bounds, grid_size, n_toys):
    """Read only 07's matching unbinned rows; never write to its directory."""
    if directory is None:
        return {}, {}, dict(status='disabled')
    directory = Path(directory)
    path = directory/'configuration.json'
    if not path.exists():
        if list(directory.glob('toys_*.csv')):
            raise ValueError('07 toy checkpoints have no configuration fingerprint')
        return {}, {}, dict(status='absent', directory=str(directory))
    saved = json.loads(path.read_text())
    config = saved['configuration']
    if saved['fingerprint'] != _digest(config):
        raise ValueError('07 configuration fingerprint mismatch')
    expected = dict(**prepared['configuration']['reference'], seed=seed,
                    mu_bounds=bounds.tolist(), grid_size=grid_size)
    mismatch = [key for key, value in expected.items() if config.get(key) != value]
    mismatch += ['backend:'+name for name, digest in prepared['configuration']['reference_backend_sha256'].items()
                 if config.get('backend_sha256', {}).get(name) != digest]
    if config.get('version') != 1:
        mismatch.append('version')
    if mismatch:
        raise ValueError('07 references differ in '+', '.join(mismatch)+
                         '; select the matching 07 directory or set REUSE_07=False to generate references in 13')
    rows, audits, files = {}, {}, {}
    audit_path = directory/'source_diagnostics.json'
    saved_audits = json.loads(audit_path.read_text()) if audit_path.exists() else []
    for mu in prepared['mu_values']:
        for case in ('analytic', 'learned'):
            file = directory/f'toys_{case}_mu_{float(mu).hex()}.csv'
            if not file.exists():
                continue
            records = _checked_rows(pd.read_csv(file), case, mu)
            rows[mu, case] = [{**r, 'origin': '07'} for r in records if r['toy_id'] < n_toys]
            files[file.name] = file_digest(file)
            match = [a for a in saved_audits if a['case'] == case and a['mu_test'] == mu]
            if len(match) == 1:
                audits[mu, case] = match[0]
    return rows, audits, dict(status='validated', directory=str(directory),
        fingerprint=saved['fingerprint'], files=files, available_rows=sum(map(len, rows.values())))


def score_toy_summary(frame, labels):
    rows = []
    for (mu, case), group in frame.groupby(['mu_test', 'case'], sort=False):
        row = summarize_own_model_toys(group.assign(case='moving36')).iloc[0].to_dict()
        good = group.loc[group.valid & np.isfinite(group.q)]
        high = good.mu_hat > .5
        other = high if mu == 0. else ~high if mu == 1.4 else None
        fraction = other.mean() if other is not None else np.nan
        row.update(case=case, label=labels[case], reused_from_07=int(group.origin.eq('07').sum()),
            fraction_other_branch=fraction,
            se_fraction_other_branch=np.sqrt(fraction*(1-fraction)/len(good)) if len(good) else np.nan,
            mean_mu_hat=good.mu_hat.mean(), max_fit_zeroed_total=good.fit_zeroed_total.max(),
            fits_with_zeroing=int((good.fit_zeroed_total > 0).sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def run_fixed_score_toys(prepared, output, *, reference_directory=None, n_toys=10_000,
                         seed=110923, mu_bounds=(0., 3.), grid_size=129,
                         checkpoint_every=100, progress_every=100, progress=print):
    """Independent null ensembles; resume/extend deterministic IDs, retain failures."""
    n_toys = _integer(n_toys, 'n_toys')
    seed = _integer(seed, 'seed', 0)
    grid_size = _integer(grid_size, 'grid_size', 5)
    checkpoint_every = _integer(checkpoint_every, 'checkpoint_every')
    progress_every = _integer(progress_every, 'progress_every')
    bounds = np.asarray(mu_bounds, float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] != 0 or bounds[1] <= 0
            or any(not bounds[0] <= mu <= bounds[1] for mu in prepared['mu_values'])):
        raise ValueError('Fit bounds must start at zero and contain all tested hypotheses')
    output = Path(output)
    if reference_directory is not None and output.resolve() == Path(reference_directory).resolve():
        raise ValueError('Notebook 13 must use a separate output directory from 07')
    config = dict(**prepared['configuration'], seed=seed, mu_bounds=bounds.tolist(), grid_size=grid_size)
    fingerprint = _digest(config)
    path = output/'configuration.json'
    if path.exists():
        if json.loads(path.read_text()).get('fingerprint') != fingerprint:
            raise ValueError('Score toy settings/models changed; use a new OUTPUT_TAG')
    elif list(output.glob('toys_*.csv')):
        raise ValueError('Score toy results have no configuration fingerprint')
    refs, ref_audits, provenance = _read_references(prepared, reference_directory,
        seed=seed, bounds=bounds, grid_size=grid_size, n_toys=n_toys)
    output.mkdir(parents=True, exist_ok=True)
    save_json(path, dict(configuration=config, fingerprint=fingerprint, requested_toys=n_toys))
    save_json(output/'reference_reuse.json', provenance)
    results, audits = [], []
    for mu in prepared['mu_values']:
        for case, label in prepared['labels'].items():
            path = output/f'toys_{case}_mu_{float(mu).hex()}.csv'
            records = _checked_rows(pd.read_csv(path), case, mu) if path.exists() else []
            complete = {int(row['toy_id']) for row in records}
            additions = [row for row in refs.get((mu, case), []) if row['toy_id'] not in complete]
            records.extend(additions)
            complete.update(int(row['toy_id']) for row in additions)
            if additions:
                _atomic_csv(pd.DataFrame(records), path)
            error, draw = None, None
            audit = ref_audits.get((mu, case))
            needs_draw = any(i not in complete for i in range(n_toys))
            if needs_draw or audit is None:
                try:
                    draw, audit = _source(prepared, case, mu)
                    audit.update(valid=True)
                except (ValueError, FloatingPointError) as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    audit = dict(valid=False, error=error, expected_count=None)
            if case in prepared['templates']:
                template = prepared['templates'][case]
                means, report = template.means(mu)
                at_one, _ = template.means(1.)
                support = (means == 0) & (at_one > 0)
                audit.update(n_bins=template.fields.shape[1], score_anchor=1.,
                    zero_mean_bins=int(np.count_nonzero(means == 0)),
                    zero_mean_bins_with_mu1_support=int(support.sum()),
                    mu1_yield_in_zero_mean_bins=float(at_one[support].sum()),
                    forbidden_kappa_intervals=template.negative_intervals.tolist())
            audits.append(dict(audit, mu_test=mu, case=case))
            save_json(output/'source_diagnostics.json', audits)
            progress(f'{label}, mu={mu:g}: {sum(i < n_toys for i in complete)}/{n_toys} cached/reused')
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
                            likelihood, observed = draw(_rng(seed, mu, toy_id, prepared['streams'][case]))
                            fit = _fit(likelihood, mu, bounds, grid_size)
                            if case in prepared['templates'] and fit['valid']:
                                _, check = likelihood.template.means(fit['mu_hat'])
                                fit.update(fit_zeroed_bins=check['zeroed_bins'], fit_zeroed_total=check['zeroed_total'])
                        except (ValueError, FloatingPointError, RuntimeError) as exc:
                            fit = _failure(f'{type(exc).__name__}: {exc}')
                    records.append(dict(mu_test=mu, case=case, toy_id=toy_id, origin='13', n_observed=observed,
                        expected_count=audit['expected_count'] if audit['expected_count'] is not None else np.nan,
                        seconds=time.perf_counter()-start, **fit))
                    dirty += 1
                    if dirty >= checkpoint_every:
                        _atomic_csv(pd.DataFrame(records), path); dirty = 0
                    if (toy_id+1) % progress_every == 0 or toy_id+1 == n_toys:
                        progress(f'{case}, mu={mu:g}: {toy_id+1}/{n_toys}; last toy {records[-1]["seconds"]:.2f}s')
            finally:
                if dirty:
                    _atomic_csv(pd.DataFrame(records), path)
            results.extend(row for row in records if row['toy_id'] < n_toys)
            draw = None
    frame = pd.DataFrame(results)
    if 'fit_zeroed_total' not in frame:
        frame['fit_zeroed_total'] = np.nan
    summary = score_toy_summary(frame, prepared['labels'])
    _atomic_csv(frame, output/'results.csv')
    _atomic_csv(summary, output/'summary.csv')
    return dict(results=frame, summary=summary, sources=pd.DataFrame(audits), output=output,
                labels=prepared['labels'], fingerprint=fingerprint, reference_reuse=provenance,
                mu_bounds=bounds.tolist())
