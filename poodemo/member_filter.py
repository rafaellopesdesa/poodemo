"""Screen complete ratio members on frozen integration nodes before inference.

Selection never uses observed data, fitted parameters, or analytic target ratios.
The derived manifests reference the original checkpoints without modifying them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .data import Run, file_digest, save_json, selected_split
from .pipeline import RATIO_TASKS, prepare_quadrature


@dataclass(frozen=True)
class MemberFilterPolicy:
    # These are configurable numerical-quality cuts, not confidence intervals.
    max_integral_factor: float = 5.0
    max_point_mass: float = 0.01
    min_effective_fraction: float = 0.001
    peer_factor: float = 100.0
    catastrophic_peer_factor: float = 1.e6
    peer_mad_scale: float = 8.0
    peer_outlier_mass: float = 0.001
    min_retained_fraction: float = 0.5
    batch_size: int = 8192

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.max_integral_factor <= 1 or self.peer_factor <= 1:
            raise ValueError('Integral and peer factors must exceed one')
        if self.catastrophic_peer_factor < self.peer_factor:
            raise ValueError('Catastrophic peer factor must be at least peer_factor')
        for key in ('max_point_mass', 'min_effective_fraction', 'peer_outlier_mass',
                    'min_retained_fraction'):
            if getattr(self, key) > 1:
                raise ValueError(f'{key} cannot exceed one')
        if self.min_retained_fraction < .5:
            raise ValueError('Retain at least a majority of the original ensemble')
        if type(self.batch_size) is not int:
            raise ValueError('batch_size must be an integer')


class FilteredEvaluationRun(Run):
    """Route every downstream ratio read to the same retained member lists."""
    def __init__(self, base_run, evaluation_root, manifest):
        super().__init__(base_run.root, base_run.config, base_run.model)
        self.base_run = base_run
        self.evaluation_root = Path(evaluation_root)
        self.filter_manifest = manifest
        for folder in ('results', 'plots', 'workspaces'):
            (self.evaluation_root / folder).mkdir(parents=True, exist_ok=True)

    def path(self, *parts):
        if parts[:2] == ('models', 'ratios'):
            return self.evaluation_root.joinpath(*parts)
        if parts[:2] in (('results', 'preselection.json'), ('results', 'density_validation')):
            return self.base_run.path(*parts)
        if parts and parts[0] in ('results', 'plots', 'workspaces'):
            return self.evaluation_root.joinpath(*parts)
        return self.base_run.path(*parts)


def unfiltered_run(run):
    while isinstance(run, FilteredEvaluationRun):
        run = run.base_run
    return run


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _members(directory):
    path = directory / 'ensemble.json'
    payload = json.loads(path.read_text()) if path.exists() else {'members': ['.']}
    paths = [(directory / member).resolve() for member in payload['members']]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError(f'Empty or duplicate ensemble membership: {directory}')
    return paths


def _mass_metrics(log_ratio, log_mass):
    weighted = log_mass + log_ratio
    log_integral = float(logsumexp(weighted))
    fractions = np.exp(weighted - log_integral)
    return dict(log_integral=log_integral,
                max_point_mass=float(fractions.max()),
                effective_fraction=float(1. / (len(fractions)*np.dot(fractions, fractions))))


def _quality_reasons(metrics, policy):
    reasons = []
    if abs(metrics['log_integral']) > math.log(policy.max_integral_factor):
        reasons.append('integration/calibration normalization disagreement')
    if metrics['max_point_mass'] > policy.max_point_mass:
        reasons.append('integration mass concentrated on one point')
    if metrics['effective_fraction'] < policy.min_effective_fraction:
        reasons.append('insufficient effective integration points')
    return reasons


def _screen(logs, log_calibration_means, log_mass, policy):
    """Pure selection core; logs may be a disk-backed members-by-nodes matrix."""
    count, n = logs.shape
    records = []
    eligible = []
    for i in range(count):
        row = dict(member_index=i, kept=False, reasons=[], log_calibration_mean=None,
                   log_integral=None, max_point_mass=None, effective_fraction=None,
                   peer_outlier_mass=None, max_peer_log_deviation=None)
        if not np.isfinite(logs[i]).all() or not np.isfinite(log_calibration_means[i]):
            row['reasons'].append('unusable prediction or calibration mean')
        elif np.max(np.abs(logs[i])) > 700:
            row['reasons'].append('log ratio exceeds inference numeric range')
        else:
            row['log_calibration_mean'] = float(log_calibration_means[i])
            if abs(log_calibration_means[i]) > math.log(policy.max_integral_factor):
                row['reasons'].append('gross calibration mean normalization error')
            row.update(_mass_metrics(logs[i]-log_calibration_means[i], log_mass))
            row['reasons'].extend(_quality_reasons(row, policy))
            if not row['reasons']:
                eligible.append(i)
        records.append(row)

    # Compare shape differences after independent per-member centering. Do not
    # change the deployed averaging convention: it is still the arithmetic mean
    # of the original retained ratios, with ONE ensemble mean normalization.
    if len(eligible) >= 3:
        member_bad_mass = np.zeros(len(eligible))
        peer_bad_log_mass = np.full(len(eligible), -np.inf)
        peer_log_integral = -np.inf
        extreme = np.zeros(len(eligible), dtype=bool)
        max_deviation = np.zeros(len(eligible))
        for start in range(0, n, policy.batch_size):
            stop = min(n, start+policy.batch_size)
            values = np.asarray(logs[eligible, start:stop])-log_calibration_means[eligible, None]
            median = np.median(values, axis=0)
            mad = 1.4826*np.median(np.abs(values-median), axis=0)
            deviation = np.abs(values-median)
            bad = deviation > np.maximum(math.log(policy.peer_factor), policy.peer_mad_scale*mad)
            # Even negligible probability mass can invalidate the polynomial
            # morph at a node. Orders-of-magnitude extrapolation against a
            # stable peer consensus is therefore also checked pointwise.
            extreme |= np.any(deviation > np.maximum(math.log(policy.catastrophic_peer_factor),
                                                     policy.peer_mad_scale*mad), axis=1)
            max_deviation = np.maximum(max_deviation, deviation.max(axis=1))
            mass = log_mass[start:stop]
            z = np.array([records[i]['log_integral'] for i in eligible])
            member_bad_mass += np.sum(np.where(bad, np.exp(values+mass-z[:, None]), 0.), axis=1)
            peer_log_integral = np.logaddexp(peer_log_integral, logsumexp(median+mass))
            peer_bad_log_mass = np.logaddexp(peer_bad_log_mass,
                logsumexp(np.where(bad, median+mass, -np.inf), axis=1))
        peer_bad_mass = np.exp(peer_bad_log_mass-peer_log_integral)
        for k, i in enumerate(eligible):
            fraction = float(max(member_bad_mass[k], peer_bad_mass[k]))
            records[i]['peer_outlier_mass'] = fraction
            records[i]['max_peer_log_deviation'] = float(max_deviation[k])
            if fraction > policy.peer_outlier_mass:
                records[i]['reasons'].append('large peer disagreement on material integration mass')
            if extreme[k]:
                records[i]['reasons'].append('extreme pointwise peer disagreement')
    kept = [i for i in eligible if not records[i]['reasons']]
    for i in kept:
        records[i]['kept'] = True
    minimum = min(count, max(2, count//2+1, math.ceil(count*policy.min_retained_fraction)))
    if len(kept) < minimum:
        return records, kept, None, f'Only {len(kept)}/{count} members pass; require {minimum}'
    log_z = float(logsumexp(log_calibration_means[kept])-math.log(len(kept)))
    ensemble = np.full(n, -np.inf)
    for i in kept:
        np.logaddexp(ensemble, logs[i], out=ensemble)
    ensemble -= math.log(len(kept))+log_z
    metrics = _mass_metrics(ensemble, log_mass)
    reasons = _quality_reasons(metrics, policy)
    if np.max(np.abs(ensemble)) > 700:
        reasons.append('retained ensemble exceeds inference numeric range')
    return records, kept, (ensemble, log_z, metrics), '; '.join(reasons) or None


def _morph_check(down, up):
    """Sample the frozen interpolation, including the tiny steps that failed.

    This is a preflight grid, not a proof of positivity between grid points.
    The likelihood's pointwise domain checks remain authoritative during fits.
    """
    from .closure_validation import _PreparedMorph
    morph = _PreparedMorph(np.exp(down), np.exp(up))
    grid = np.unique(np.r_[np.linspace(-1., 1., 81),
                           -np.logspace(-12, -2, 21), np.logspace(-12, -2, 21)])
    for alpha in grid:
        values = morph(float(alpha))
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            return {'passed': False, 'alpha': float(alpha),
                    'invalid_points': int(np.count_nonzero(~np.isfinite(values) | (values <= 0)))}
    return {'passed': True, 'grid': grid.tolist()}


def _inventory(run, policy):
    tasks, files = {}, {}
    for num, den in RATIO_TASKS:
        task = f'{num}_over_{den}'
        directory = run.path('models', 'ratios', task).resolve()
        members = _members(directory)
        task_files = [directory/'ensemble.json'] if (directory/'ensemble.json').exists() else []
        for member in members:
            task_files.append(member/'model.pt')
            if (member/'calibrator.joblib').exists():
                task_files.append(member/'calibrator.joblib')
        files[task] = {str(p): file_digest(p) for p in task_files}
        tasks[task] = members
    inputs = {
        'schema': 1, 'policy': asdict(policy), 'models': files,
        'quadrature': {name: file_digest(run.path('quadrature', name+'.npy'))
                       for name in ('x', 'weights', 'nominal', 'down', 'up')},
        'calibration': {den: file_digest(run.path('selected', den+'_calibration.npy'))
                        for den in sorted({den for _, den in RATIO_TASKS})},
        'code': {name: file_digest(Path(__file__).parent/name)
                 for name in ('member_filter.py', 'training.py', 'closure_validation.py', 'inference.py')},
    }
    return tasks, inputs


def _write_reports(root, records, summary, manifest):
    output = root/'results/ensemble_filter'
    save_json(output/'manifest.json', manifest)
    pd.DataFrame([{**r, 'reasons': '; '.join(r['reasons'])} for r in records]).to_csv(output/'members.csv', index=False)
    pd.DataFrame(summary).to_csv(output/'summary.csv', index=False)


def prepare_ratio_members(run):
    """Drive training entry point, without evaluating an unscreened ensemble."""
    from .pipeline import _training_config
    from .training import fit_density_ratio
    run = unfiltered_run(run)
    for number, (num, den) in enumerate(RATIO_TASKS):
        directory = run.path('models', 'ratios', f'{num}_over_{den}')
        if ((directory/'ensemble.json').exists() or (directory/'model.pt').exists()):
            if all((p/'model.pt').is_file() for p in _members(directory)):
                continue
        print(f'Training/reusing members for {num}/{den} before screening', flush=True)
        config = replace(_training_config(run), seed=run.config['seed']+103*number)
        predictor = fit_density_ratio(
            selected_split(run, num, 'train'), selected_split(run, den, 'train'),
            selected_split(run, num, 'validation'), selected_split(run, den, 'validation'),
            directory, config, numerator_label=num, denominator_label=den)
        del predictor


def filter_ratio_ensembles(run, *, policy=None, device='auto'):
    """Return a cached, immutable selection view for ALL five ratio tasks.

    Requires the original Drive S/NI calibration splits, never diagnostic or
    observed events. The first pass evaluates every member on both banks;
    subsequent calls reuse results iff inputs, policy, and implementation match.
    """
    from .training import Predictor
    policy = MemberFilterPolicy() if policy is None else policy
    if not isinstance(policy, MemberFilterPolicy):
        raise TypeError('policy must be MemberFilterPolicy')
    run = unfiltered_run(run)
    q = prepare_quadrature(run)
    tasks, inputs = _inventory(run, policy)
    inputs['device'] = device
    identity = _digest(inputs)
    base = getattr(run, 'evaluation_root', run.root/'evaluations')
    root = base/'filtered'/identity[:16]
    complete = root/'results/ensemble_filter/complete.json'
    if complete.exists():
        try:
            cached = json.loads(complete.read_text())
        except ValueError:
            cached = {}
        if cached.get('identity') == identity and cached.get('artifacts') and all(
                (root/p).is_file() and file_digest(root/p) == sha
                for p, sha in cached['artifacts'].items()):
            manifest = json.loads((root/'results/ensemble_filter/manifest.json').read_text())
            print('Reusing ensemble member filter:', root, flush=True)
            return FilteredEvaluationRun(run, root, manifest)
        complete.unlink()
    root.mkdir(parents=True, exist_ok=True)
    print('Ensemble member filter reports:', root/'results/ensemble_filter', flush=True)
    x, weights, nominal = q['x'], np.asarray(q['weights']), np.asarray(q['nominal'])
    if (len(x) < 2 or weights.shape != (len(x),) or nominal.shape != (4, len(x))
            or np.any(weights <= 0) or not np.isfinite(weights).all()
            or np.any(nominal <= 0) or not np.isfinite(nominal).all()):
        raise ValueError('Member filtering needs positive finite quadrature weights and components')
    log_s_mass = np.log(weights)+np.log(nominal[0])
    log_s_mass -= logsumexp(log_s_mass)
    log_ni_mass = None
    all_records, summary, variations = [], [], {}
    manifest = dict(identity=identity, inputs=inputs, status='running', tasks={})
    for num, den in RATIO_TASKS:
        task = f'{num}_over_{den}'
        members = tasks[task]
        log_mass = log_s_mass if den == 'S' else log_ni_mass
        if log_mass is None:
            raise RuntimeError('Filter the nominal NI ensemble before NI systematic ratios')
        calibration = selected_split(run, den, 'calibration')
        if len(calibration) < 2:
            raise ValueError(f'Need at least two reserved {den} calibration events')
        log_means = np.full(len(members), np.nan)
        errors = {}
        calibration_stats = {}
        # The working matrix stays in runtime scratch, never in Drive/GitHub.
        with tempfile.TemporaryDirectory(prefix='poodemo-member-filter-') as temp:
            logs = np.lib.format.open_memmap(Path(temp)/'logs.npy', mode='w+', dtype='float64',
                                             shape=(len(members), len(x)))
            logs[:] = np.nan
            for i, member_path in enumerate(members):
                print(f'Screening {task}: member {i+1}/{len(members)} ({member_path.name})', flush=True)
                member = Predictor.load(member_path, device=device)
                try:
                    if member.kind != 'ratio' or tuple(member.labels) != (den, num):
                        raise ValueError(f'Wrong ratio labels in {member_path}')
                    values = member.predict_log_ratio(x, batch_size=policy.batch_size)
                    if values.shape != (len(x),):
                        raise ValueError(f'Wrong prediction shape in {member_path}')
                    logs[i] = values
                    calibration_values = member.predict_log_ratio(calibration, batch_size=policy.batch_size)
                    if calibration_values.shape != (len(calibration),):
                        raise ValueError(f'Wrong calibration prediction shape in {member_path}')
                    if not np.isfinite(calibration_values).all() or np.max(np.abs(calibration_values)) > 700:
                        raise FloatingPointError('Invalid calibration predictions')
                    log_means[i] = logsumexp(calibration_values)-math.log(len(calibration))
                    calibration_metrics = _mass_metrics(calibration_values,
                        np.full(len(calibration), -math.log(len(calibration))))
                    calibration_stats[i] = {
                        'calibration_log_mean': calibration_metrics['log_integral'],
                        'calibration_max_point_mass': calibration_metrics['max_point_mass'],
                        'calibration_effective_fraction': calibration_metrics['effective_fraction'],
                    }
                    if (calibration_metrics['max_point_mass'] > policy.max_point_mass
                            or calibration_metrics['effective_fraction'] < policy.min_effective_fraction):
                        log_means[i] = np.nan
                        raise FloatingPointError('Calibration mass concentrated on too few events')
                except FloatingPointError as error:
                    # Missing/corrupt checkpoints and interface errors propagate;
                    # only numeric prediction failures are eligible for rejection.
                    errors[i] = str(error)
                finally:
                    del member
            records, kept, selected, failure = _screen(logs, log_means, log_mass, policy)
            del logs
        for r in records:
            r.update(task=task, member=str(members[r['member_index']]))
            r.update(calibration_stats.get(r['member_index'], {}))
            if r['member_index'] in errors:
                r['reasons'].append(errors[r['member_index']])
        all_records.extend(records)
        row = dict(task=task, original_members=len(members), retained_members=len(kept),
                   rejected_members=len(members)-len(kept), status='failed' if failure else 'passed')
        if selected is not None:
            ensemble, log_z, metrics = selected
            row.update(metrics, log_normalization=log_z)
        summary.append(row)
        manifest['tasks'][task] = dict(kept=[str(members[i]) for i in kept],
            rejected=[r['member'] for r in records if not r['kept']])
        if failure:
            manifest.update(status='failed', error=f'{task}: {failure}')
            _write_reports(root, all_records, summary, manifest)
            raise RuntimeError(f"Ensemble screening stopped: {task}: {failure}. Inspect {root/'results/ensemble_filter'}")
        output = root/'models/ratios'/task
        save_json(output/'ensemble.json', {
            'members': [os.path.relpath(members[i], output) for i in kept],
            'labels': [den, num], 'streaming': True, 'log_normalization': log_z,
            'filter_identity': identity,
        })
        save_json(output/'normalization.json', {
            'log_normalization': log_z, 'n_calibration': len(calibration),
            'members': len(kept), 'filter_identity': identity,
            'note': 'Retained arithmetic mean; scalar normalization on reserved Drive denominator calibration split.',
        })
        if task == 'NI_over_S':
            log_ni_mass = log_s_mass+ensemble
            log_ni_mass -= logsumexp(log_ni_mass)
        elif den == 'NI':
            variations[num] = ensemble-metrics['log_integral']
        print(f'{task}: retained {len(kept)}/{len(members)} members', flush=True)
        _write_reports(root, all_records, summary, manifest)
    morph = _morph_check(variations['NI_down'], variations['NI_up'])
    manifest.update(status='passed' if morph['passed'] else 'failed', nuisance_morph=morph)
    _write_reports(root, all_records, summary, manifest)
    if not morph['passed']:
        raise RuntimeError(f"Retained NI ensembles still have an invalid nuisance morph: {morph}. Inspect {root/'results/ensemble_filter'}")
    artifacts = {str(p.relative_to(root)): file_digest(p)
                 for directory in (root/'models/ratios', root/'results/ensemble_filter')
                 for p in directory.rglob('*') if p.is_file() and p != complete}
    save_json(complete, {'identity': identity, 'artifacts': artifacts})
    return FilteredEvaluationRun(run, root, manifest)
