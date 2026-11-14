"""Moving-observable resolution toys, extending the immutable four-case runner.

The original module and its dependencies keep their published content hashes,
so the completed 1000-toy ensembles can be extended without migration. Only
additional binnings get new templates, streams, and independent checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .colab_mc import _digest, _integer, fill_colab_histograms
from .data import file_digest, save_json
from .mc_yields import yield_policy
from .own_model_toys import CASES, PoissonTemplate, summarize_own_model_toys
from .pipeline import observable_bin_edges
from .toy_study import _array_digest, _atomic_csv, _failure, _fit, _rng

BIN_COUNTS = (36, 72, 108, 144)
# Preserve streams 0..3 for the four original cases. Stable if order/subset changes.
STREAMS = {72: 4, 108: 5, 144: 6}


def prepare_moving_resolution(run, samples, baseline, *, bin_counts=BIN_COUNTS, progress=print):
    """Direct fills from existing ratios; the baseline 36-bin templates are reused."""
    bins = tuple(sorted({_integer(n, 'bin count', 2) for n in bin_counts}))
    if 36 not in bins or not set(bins).issubset(BIN_COUNTS):
        raise ValueError('Use a subset of 36,72,108,144 that includes the 36-bin baseline')
    config = baseline['configuration']
    if (samples['target'] > 10_000_000 or config['selected_per_source'] != samples['target']
            or config['sample_fingerprint'] != samples['fingerprint']
            or config['reference_power'] != run.config['reference_ratio_power']
            or config['eta_floor'] != run.config['reference_ratio_eta_floor']
            or config['yield_policy'] != yield_policy(run.config).metadata()
            or not np.array_equal(config['rates'], samples['rates'][:4])
            or not np.array_equal(config['observable_normalizers'], samples['normalizers'])):
        raise ValueError('Resolution study must use the same <=10M baseline, map, rates and yield policy')
    templates = {36: baseline['templates']['moving36']}
    extras = [n for n in bins if n != 36]
    if extras:
        bank = fill_colab_histograms(run, samples, etas=baseline['mu_values'],
            bin_counts=extras, prefixes=[samples['target']], progress=progress)
        for n in extras:
            templates[n] = {mu: PoissonTemplate(bank.anchors(mu, n)[0],
                policy=yield_policy(run.config), exposure=baseline['exposure']) for mu in baseline['mu_values']}
    return dict(baseline=baseline, bin_counts=bins, templates=templates)


def resolution_summary(frame):
    """Keep the same q summaries, plus explicit branch and zeroing diagnostics."""
    rows = []
    for (mu, case), group in frame.groupby(['mu_test', 'case'], sort=False):
        # Statistics are identical; proxy the label only for the legacy helper.
        row = summarize_own_model_toys(group.assign(case='moving36')).iloc[0].to_dict()
        n = int(case.removeprefix('moving')) if case.startswith('moving') else None
        row.update(case=case, label=f'Moving learned observable: {n} bins' if n else CASES[case], n_bins=n)
        good = group.loc[group.valid & np.isfinite(group.q)]
        high = good.mu_hat > .5
        other = high if mu == 0 else ~high if mu == 1.4 else None
        fraction = other.mean() if other is not None else np.nan
        row.update(fraction_other_branch=fraction,
            se_fraction_other_branch=np.sqrt(fraction*(1-fraction)/len(good)) if len(good) else np.nan,
            mean_mu_hat=good.mu_hat.mean(), max_fit_zeroed_total=good.fit_zeroed_total.max(),
            fits_with_zeroing=int((good.fit_zeroed_total > 0).sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def _run_binning(prepared, n_bins, output, *, n_toys, seed, bounds, grid_size,
                 checkpoint_every, progress_every, progress):
    baseline = prepared['baseline']
    mus = baseline['mu_values']
    templates = prepared['templates'][n_bins]
    case = f'moving{n_bins}'
    config = dict(version=1, case=case, stream=STREAMS[n_bins],
        baseline_model_fingerprint=_digest(baseline['configuration']),
        fields_sha256={str(mu): _array_digest(templates[mu].fields) for mu in sorted(mus)},
        edges=observable_bin_edges(n_bins, 'reference_ratio').tolist(),
        backend_sha256=file_digest(Path(__file__)), seed=seed, mu_bounds=bounds.tolist(), grid_size=grid_size)
    fingerprint = _digest(config)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output/'configuration.json'
    if config_path.exists():
        if json.loads(config_path.read_text()).get('fingerprint') != fingerprint:
            raise ValueError(f'{case} settings/templates changed; use a new output directory')
    elif list(output.glob('toys_*.csv')):
        raise ValueError(f'{case} results have no configuration fingerprint')
    save_json(config_path, dict(configuration=config, fingerprint=fingerprint, requested_toys=n_toys))
    results, audits = [], []
    for mu in mus:
        template = templates[mu]
        means, report = template.means(mu)
        error = None if report['zeroing_accepted'] else f'Invalid binned generator: {report}'
        expected = float(means.sum()) if error is None else None
        audits.append(dict(case=case, mu_test=mu, n_bins=n_bins, valid=error is None,
            error=error, expected_count=expected, occupied_bins=int(np.count_nonzero(means > 0)), **report))
        save_json(output/'source_diagnostics.json', audits)
        path = output/f'toys_{case}_mu_{float(mu).hex()}.csv'
        records = pd.read_csv(path).to_dict('records') if path.exists() else []
        ids = [row['toy_id'] for row in records]
        if (len(set(ids)) != len(ids) or any(not np.isfinite(i) or i < 0 or int(i) != i for i in ids)
                or any(row['case'] != case or row['mu_test'] != mu for row in records)):
            raise ValueError(f'Duplicate or incompatible cached toy IDs in {case}')
        complete = set(ids)
        progress(f'{case}, mu={mu:g}: {sum(i < n_toys for i in complete)}/{n_toys} cached')
        dirty = 0
        try:
            for toy_id in range(n_toys):
                if toy_id in complete:
                    continue
                start, observed = time.perf_counter(), np.nan
                if error:
                    fit = _failure(error)
                else:
                    try:
                        counts = _rng(seed, mu, toy_id, STREAMS[n_bins]).poisson(means)
                        observed = int(counts.sum())
                        fit = _fit(template.likelihood(counts), mu, bounds, grid_size)
                        if fit['valid']:
                            _, check = template.means(fit['mu_hat'])
                            fit.update(fit_zeroed_bins=check['zeroed_bins'], fit_zeroed_total=check['zeroed_total'])
                    except (ValueError, FloatingPointError, RuntimeError) as exc:
                        fit = _failure(f'{type(exc).__name__}: {exc}')
                records.append(dict(mu_test=mu, case=case, toy_id=toy_id, n_observed=observed,
                    expected_count=expected if expected is not None else np.nan,
                    seconds=time.perf_counter()-start, **fit))
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
    return pd.DataFrame(results), pd.DataFrame(audits)


def run_moving_resolution(prepared, baseline_study, output, *, n_toys=10_000, seed=110923,
                          mu_bounds=(0., 3.), grid_size=129, checkpoint_every=100,
                          progress_every=100, progress=print):
    """Extend three additional cases and combine with existing baseline toys.

    Completed four-case rows are checked and reused, never redrawn. Resolution
    cases have independent Poisson streams and per-binning fingerprints, so
    extending N_TOYS or changing the requested binning subset preserves work.
    """
    n_toys = _integer(n_toys, 'n_toys')
    seed = _integer(seed, 'seed', 0)
    grid_size = _integer(grid_size, 'grid_size', 5)
    checkpoint_every = _integer(checkpoint_every, 'checkpoint_every')
    progress_every = _integer(progress_every, 'progress_every')
    bounds = np.asarray(mu_bounds, float)
    baseline = prepared['baseline']
    if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] != 0 or bounds[1] <= 0
            or any(not bounds[0] <= mu <= bounds[1] for mu in baseline['mu_values'])):
        raise ValueError('Fit bounds must start at zero and contain all hypotheses')
    expected = _digest(dict(**baseline['configuration'], seed=seed, mu_bounds=bounds.tolist(), grid_size=grid_size))
    if baseline_study['fingerprint'] != expected:
        raise ValueError('Baseline toy settings differ from the resolution study')
    frame = baseline_study['results']
    if frame.duplicated(['mu_test', 'case', 'toy_id']).any():
        raise ValueError('Duplicate baseline toy IDs')
    pieces = []
    for mu in baseline['mu_values']:
        for case in CASES:
            group = frame.loc[(frame.mu_test == mu) & (frame.case == case) & (frame.toy_id < n_toys)]
            if set(group.toy_id) != set(range(n_toys)):
                raise ValueError('Extend the four baseline ensembles to N_TOYS before the resolution study')
            pieces.append(group)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audits = [baseline_study['sources']]
    for n in prepared['bin_counts']:
        if n == 36:
            continue
        result, audit = _run_binning(prepared, n, output/f'bins_{n}', n_toys=n_toys, seed=seed,
            bounds=bounds, grid_size=grid_size, checkpoint_every=checkpoint_every,
            progress_every=progress_every, progress=progress)
        pieces.append(result)
        audits.append(audit)
    results = pd.concat(pieces, ignore_index=True)
    sources = pd.concat(audits, ignore_index=True)
    summary = resolution_summary(results)
    _atomic_csv(results, output/'results.csv')
    _atomic_csv(summary, output/'summary.csv')
    _atomic_csv(sources, output/'source_diagnostics.csv')
    return dict(results=results, summary=summary, sources=sources, output=output, bin_counts=prepared['bin_counts'])
