"""Carry notebook 04's two unbinned benchmarks into 05/06 comparisons.

These curves are references for interpretation, never interpolation targets or
an acceptance test. Keep the analytical-only shared CSV unchanged.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import file_digest, save_json
from .discriminator_unbinned import LEARNED_LABEL
from .pipeline import _positive_study_grid, _saved_unbinned


def _matched(frame, grid, label, truth):
    required = {'model', 'mu', 'systematics', 'q', 'valid', 'asimov_mu'}
    if not required.issubset(frame.columns):
        raise ValueError('The saved unbinned benchmark is incomplete')
    if not pd.api.types.is_bool_dtype(frame.valid) or not pd.api.types.is_bool_dtype(frame.systematics):
        raise ValueError('Benchmark validity/nuisance flags must be boolean')
    selected = frame.loc[frame.model.eq(label) & np.isclose(frame.asimov_mu, truth, rtol=0, atol=1e-12)]
    rows = []
    for syst in (False, True):
        for mu in grid:
            match = selected.loc[selected.systematics.eq(syst) & np.isclose(selected.mu, mu, rtol=0, atol=1e-12)]
            if len(match) != 1:
                raise ValueError(f'Missing or duplicated {label} benchmark at mu={mu:g}, systematics={syst}')
            record = match.iloc[0].to_dict()
            if record['valid'] and not np.isfinite(record['q']):
                raise ValueError('A valid benchmark point must have finite q')
            record['mu'] = float(mu)
            rows.append(record)
    return pd.DataFrame(rows)


def load_signal_benchmarks(run):
    """Verify and reuse 04's snapshot, with no NN/objective evaluation."""
    owner = run
    while not hasattr(owner, 'filter_manifest') and hasattr(owner, 'base_run'):
        owner = owner.base_run
    if not hasattr(owner, 'filter_manifest'):
        raise ValueError('Select the completed notebook-03 filtered ensemble first')
    snapshot = run.path('results', 'signal_learned_unbinned_reference.csv')
    source = run.path('results', 'signal_learned_unbinned_reference.json')
    if not snapshot.is_file() or not source.is_file():
        raise FileNotFoundError('Complete notebook 04 with both unbinned benchmarks before 05/06')
    provenance = json.loads(source.read_text())
    if provenance.get('filter_identity') != owner.filter_manifest['identity'] or provenance.get('model') != LEARNED_LABEL:
        raise ValueError('Notebook 04 benchmark uses a different model or ensemble')
    scan = owner.path('results','unbinned_scans.csv')
    fits = owner.path('results','unbinned_fits.csv')
    if file_digest(scan) != provenance['scan_sha256'] or file_digest(fits) != provenance['fit_sha256']:
        raise ValueError('Notebook 03 results changed after 04; refresh the 04 benchmark before continuing')
    for artifact in provenance['workspace_artifacts']:
        name = artifact['file']
        if Path(name).name != name or file_digest(owner.path('workspaces','learned',name)) != artifact['sha256']:
            raise ValueError('Notebook 03 workspace arrays changed after 04')
    grid, truth = _positive_study_grid(run), float(run.config['asimov_mu'])
    learned = _matched(pd.read_csv(snapshot, float_precision='round_trip'), grid, LEARNED_LABEL, truth)
    current = _matched(pd.read_csv(scan), grid, LEARNED_LABEL, truth)
    if (not np.array_equal(learned.valid, current.valid)
            or not np.allclose(learned.q, current.q, rtol=1e-12, atol=1e-12, equal_nan=True)):
        raise ValueError('The copied 04 benchmark no longer matches its notebook-03 source')
    analytical = _matched(_saved_unbinned(run), grid, 'analytic unbinned', truth)
    metadata = dict(learned_source=provenance, snapshot_sha256=file_digest(snapshot),
        analytical_scan_sha256=file_digest(run.path('results','unbinned_scans.csv')),
        role='Display and diagnostic comparisons only; interpolation closure is against direct MC fills',
        asimov_note='03 learned curve: physical Asimov; binned curves: same-template MC Asimov')
    return pd.concat([analytical, learned], ignore_index=True), metadata


def add_signal_benchmarks(study, benchmarks, provenance):
    """Attach/save both curves without changing any fit or validation result."""
    study['unbinned'] = benchmarks.copy()
    study['metadata']['unbinned_benchmarks'] = provenance
    scans = study['scans']
    scans = scans.loc[scans.on_scan] if 'on_scan' in scans else scans
    comparisons = []
    for label, reference in benchmarks.groupby('model'):
        columns = reference[['mu','systematics','q','valid']].rename(
            columns={'mu':'mu_reference','q':'q_reference','valid':'reference_valid'})
        tolerance = 32*np.finfo(float).eps*max(1., float(reference.mu.abs().max()))
        joined = pd.merge_asof(scans.sort_values('mu'), columns.sort_values('mu_reference'),
            left_on='mu', right_on='mu_reference', by='systematics', direction='nearest', tolerance=tolerance)
        joined['reference_model'] = label
        joined['delta_q_to_reference'] = (joined.q-joined.q_reference).where(
            joined.valid & joined.reference_valid.eq(True))
        comparisons.append(joined)
    study['benchmark_comparison'] = pd.concat(comparisons,ignore_index=True)
    study['unbinned'].to_csv(study['output']/'unbinned.csv',index=False)
    study['benchmark_comparison'].to_csv(study['output']/'benchmark_comparison.csv',index=False)
    save_json(study['output']/'metadata.json',study['metadata'])
    return study
