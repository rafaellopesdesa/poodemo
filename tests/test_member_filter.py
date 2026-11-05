"""Tail failures must remove whole members, with a reproducible inference view."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from poodemo import member_filter, training
from poodemo.data import Run, file_digest, save_json
from poodemo.member_filter import MemberFilterPolicy, _screen, filter_ratio_ensembles
from poodemo.pipeline import RATIO_TASKS


def screen(values, means=None, policy=None):
    values = np.asarray(values, float)
    if means is None:
        means = np.zeros(len(values))
    return _screen(values, np.asarray(means), np.full(values.shape[1], -np.log(values.shape[1])),
                   policy or MemberFilterPolicy())


def test_single_catastrophic_tail_is_removed_before_arithmetic_average():
    values = np.zeros((8, 1000))
    values[-1, 173] = np.log(2e34)  # The failure observed in notebook 03.
    records, kept, result, error = screen(values)
    assert error is None and kept == list(range(7))
    assert 'integration mass concentrated on one point' in records[-1]['reasons']
    assert records[-1]['max_point_mass'] > .999999
    np.testing.assert_allclose(result[0], 0., atol=1e-15)


def test_nonfinite_and_gross_uniform_miscalibration_are_rejected():
    values = np.zeros((8, 1000))
    values[-1, 3] = np.nan
    values[-2] = 10.
    means = np.zeros(8); means[-2] = 10.
    records, kept, _, error = screen(values, means)
    assert error is None and kept == list(range(6))
    assert 'gross calibration mean normalization error' in records[-2]['reasons']
    assert 'unusable prediction or calibration mean' in records[-1]['reasons']


def test_material_peer_shape_outlier_with_healthy_integral_is_removed():
    values = np.zeros((8, 1000))
    # A broad, very deep hole with compensating normalization elsewhere. No
    # single-point spike or failed integral: this exercises the peer criterion.
    values[-1, :100] = -12.
    values[-1, 100:] = np.log((1000-100*np.exp(-12))/900)
    records, kept, _, error = screen(values)
    assert error is None and kept == list(range(7))
    assert records[-1]['peer_outlier_mass'] == pytest.approx(.1)
    assert records[-1]['reasons'] == ['large peer disagreement on material integration mass']


def test_shared_real_shape_is_not_rejected_for_large_ratios_alone():
    values = np.full((8, 1000), np.log(.5))
    values[:, :100] = np.log(5.5)  # Integral one, shared tail, ESS ~308.
    records, kept, result, error = screen(values)
    assert error is None and kept == list(range(8))
    assert all(r['kept'] for r in records)
    np.testing.assert_allclose(result[0], values[0])


def test_negligible_peer_disagreement_does_not_trigger_pruning():
    values = np.zeros((8, 10000))
    values[-1, 0] = -8.
    _, kept, _, error = screen(values)
    assert error is None and kept == list(range(8))


def test_widespread_failure_stops_instead_of_selecting_a_tiny_ensemble():
    values = np.zeros((8, 1000));values[:5, 3] = 80.
    records, kept, result, error = screen(values)
    assert kept == [5, 6, 7] and result is None
    assert 'require 5' in error
    # Two-member ensembles cannot silently degrade to a singleton.
    _, _, _, error = screen(values[[0, 7]])
    assert error is not None


def test_retained_normalization_is_mean_of_ratios_not_mean_of_logits():
    values = np.repeat(np.log([.8, 1., 1.4, 1.2])[:, None], 1000, axis=1)
    _, kept, result, error = screen(values, np.log([.8, 1., 1.4, 1.2]))
    assert error is None and len(kept) == 4
    assert result[1] == pytest.approx(np.log(1.1))
    np.testing.assert_allclose(result[0], 0., atol=1e-15)


def test_nuisance_preflight_rejects_the_recorded_scale_of_distorted_anchors():
    lo = np.full(1000, 1.5e-25); hi = np.full(1000, .012)
    lo[0] = 3.487e9;hi[0] = 3.412e9
    assert not member_filter._morph_check(np.log(lo), np.log(hi))['passed']
    assert member_filter._morph_check(np.zeros(1000), np.zeros(1000))['passed']


@pytest.mark.parametrize('change', [dict(peer_factor=1.), dict(max_point_mass=2.),
                                     dict(batch_size=1.2), dict(min_retained_fraction=.1)])
def test_invalid_policy_rejected(change):
    with pytest.raises(ValueError):
        MemberFilterPolicy(**change)


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    root = tmp_path/'run'
    root.mkdir()
    run = Run(root, {}, None)
    n = 1000
    x = np.zeros((n, 3));x[:, 0] = np.arange(n)
    q = dict(x=x, weights=np.full(n, 1/n), nominal=np.ones((4, n)),
             down=np.ones((1, 4, n)), up=np.ones((1, 4, n)))
    (root/'quadrature').mkdir()
    for name, value in q.items():
        np.save(root/'quadrature'/f'{name}.npy', value)
    (root/'selected').mkdir()
    for den in ('S', 'NI'):
        # Different, reserved inputs: these never contain the quadrature spike.
        np.save(root/'selected'/f'{den}_calibration.npy', np.full((1000, 3), -1.))
    for num, den in RATIO_TASKS:
        directory = run.path('models', 'ratios', f'{num}_over_{den}')
        for i in range(5):
            member = directory/f'member_{i:03d}'
            save_json(member/'training.json', {'member': i})
            (member/'model.pt').write_bytes(f'{num}/{den}/{i}'.encode())
        save_json(directory/'ensemble.json', {'members': [f'member_{i:03d}' for i in range(5)],
            'labels': [den, num], 'streaming': True, 'log_normalization': 99.})
    calls = []
    class FakePredictor:
        kind = 'ratio'
        def __init__(self, path):
            self.path = Path(path)
            num, den = self.path.parent.name.split('_over_')
            self.labels = (den, num)
            self.index = int(self.path.name.split('_')[-1])
        def predict_log_ratio(self, points, batch_size=0):
            calls.append((self.path.parent.name, self.index, len(points)))
            result = np.full(len(points), np.log(1.+.02*self.index))
            if self.index == 4:
                result[points[:, 0] == 173] = 80.
            return result
        predict_raw_log_ratio = predict_log_ratio
    monkeypatch.setattr(training.Predictor, 'load', lambda path, device='auto': FakePredictor(path))
    monkeypatch.setattr(member_filter, 'prepare_quadrature', lambda run: q)
    return run, q, calls


def test_all_five_tasks_reuse_only_retained_checkpoints_everywhere(saved_run):
    run, q, calls = saved_run
    originals = {str(p): file_digest(p) for p in run.path('models').rglob('*') if p.is_file()}
    filtered = filter_ratio_ensembles(run)
    assert len(calls) == 50  # One quadrature + one calibration prediction per member.
    assert filtered.path('raw', 'S.npy') == run.path('raw', 'S.npy')
    assert filtered.path('quadrature', 'x.npy') == run.path('quadrature', 'x.npy')
    assert filtered.path('results', 'density_validation', 'bank') == run.path('results', 'density_validation', 'bank')
    assert filtered.path('results', 'preselection.json') == run.path('results', 'preselection.json')
    for num, den in RATIO_TASKS:
        predictor = training.load_predictor(filtered.path('models', 'ratios', f'{num}_over_{den}'))
        assert len(predictor.member_paths) == 4
        assert all(p.name != 'member_004' for p in predictor.member_paths)
        # The stale original scalar 99 is replaced with the retained mean 1.03.
        assert predictor.log_normalization == pytest.approx(np.log(1.03))
        np.testing.assert_allclose(predictor.predict_ratio(q['x']), 1., atol=1e-14)
        assert training.training_metadata(predictor)['ensemble_size'] == 4
    assert originals == {str(p): file_digest(p) for p in run.path('models').rglob('*') if p.is_file()}
    before = len(calls)
    again = filter_ratio_ensembles(filtered)
    assert again.evaluation_root == filtered.evaluation_root and len(calls) == before
    report = json.loads(again.path('results', 'ensemble_filter', 'manifest.json').read_text())
    assert report['status'] == 'passed'
    assert all(len(t['rejected']) == 1 for t in report['tasks'].values())


def test_cache_tracks_models_banks_policy_and_generated_manifests(saved_run):
    run, q, calls = saved_run
    first = filter_ratio_ensembles(run)
    before = len(calls)
    # A damaged derived file must not be accepted as a completed audit.
    first.path('models', 'ratios', 'SBI_over_S', 'ensemble.json').write_text('{}')
    same = filter_ratio_ensembles(run)
    assert same.evaluation_root == first.evaluation_root and len(calls) > before
    model = run.path('models', 'ratios', 'SBI_over_S', 'member_000', 'model.pt')
    model.write_bytes(b'changed')
    changed = filter_ratio_ensembles(run)
    assert changed.evaluation_root != first.evaluation_root
    np.save(run.path('selected', 'S_calibration.npy'), np.full((1000, 3), -2.))
    bank_changed = filter_ratio_ensembles(run)
    assert bank_changed.evaluation_root != changed.evaluation_root
    policy_changed = filter_ratio_ensembles(run, policy=replace(MemberFilterPolicy(), peer_factor=200.))
    assert policy_changed.evaluation_root != bank_changed.evaluation_root


def test_failed_audit_leaves_report_but_no_active_completion(saved_run, monkeypatch):
    run, _, _ = saved_run
    monkeypatch.setattr(member_filter, '_morph_check', lambda *args: {'passed': False, 'alpha': .1})
    with pytest.raises(RuntimeError, match='invalid nuisance morph'):
        filter_ratio_ensembles(run)
    assert not list(run.root.rglob('complete.json'))
    reports = list(run.root.rglob('manifest.json'))
    assert len(reports) == 1 and json.loads(reports[0].read_text())['status'] == 'failed'


def test_drive_members_are_reused_without_unscreened_predictions(saved_run, monkeypatch):
    run, _, calls = saved_run
    def unexpected(*args, **kwargs):
        raise AssertionError('Existing members must not be trained or evaluated before screening')
    monkeypatch.setattr(training, 'fit_density_ratio', unexpected)
    member_filter.prepare_ratio_members(run)
    assert calls == []


def test_missing_drive_members_are_trained_without_ensemble_diagnostics(tmp_path, monkeypatch):
    from poodemo import pipeline
    run = Run(tmp_path, {'seed': 7}, None)
    seen = []
    config = training.TrainingConfig.smoke(ensemble_size=3)
    monkeypatch.setattr(pipeline, '_training_config', lambda run: config)
    monkeypatch.setattr(member_filter, 'selected_split', lambda run, sample, split: (sample, split))
    def train(nt, dt, nv, dv, directory, cfg, *, numerator_label, denominator_label):
        assert nt == (numerator_label, 'train') and nv == (numerator_label, 'validation')
        assert dt == (denominator_label, 'train') and dv == (denominator_label, 'validation')
        seen.append((numerator_label, denominator_label, cfg.seed))
        return object()
    monkeypatch.setattr(training, 'fit_density_ratio', train)
    member_filter.prepare_ratio_members(run)
    assert seen == [(num, den, 7+103*i) for i, (num, den) in enumerate(RATIO_TASKS)]


def test_systematic_screen_uses_retained_learned_ni_measure(saved_run, monkeypatch):
    run, q, _ = saved_run
    # Make the analytic NI measure different from the learned, retained NI.
    # The unfiltered NI is different again because of the catastrophic spike.
    q['nominal'][3] = np.linspace(.2, 3., len(q['x']))
    np.save(run.path('quadrature', 'nominal.npy'), q['nominal'])
    seen = []
    original = member_filter._screen
    def capture(logs, means, log_mass, policy):
        seen.append(np.exp(log_mass))
        return original(logs, means, log_mass, policy)
    monkeypatch.setattr(member_filter, '_screen', capture)
    filter_ratio_ensembles(run)
    assert len(seen) == 5
    for mass in seen[-2:]:
        np.testing.assert_allclose(mass, np.full(1000, .001), rtol=1e-13)
        assert not np.allclose(mass, q['nominal'][3]/q['nominal'][3].sum())


def test_model_load_errors_are_not_silently_treated_as_bad_training(saved_run, monkeypatch):
    run, _, _ = saved_run
    def corrupt(*args, **kwargs):
        raise OSError('incomplete transfer')
    monkeypatch.setattr(training.Predictor, 'load', corrupt)
    with pytest.raises(OSError, match='incomplete transfer'):
        filter_ratio_ensembles(run)
    assert not list(run.root.rglob('complete.json'))


def test_extreme_peer_extrapolation_is_rejected_even_at_negligible_mass_node():
    values = np.zeros((8, 1000)); values[-1, 0] = np.log(1e8)
    mass = np.full(1000, 1/999);mass[0] = 1e-20;mass /= mass.sum()
    records, kept, _, error = _screen(values, np.zeros(8), np.log(mass), MemberFilterPolicy())
    assert error is None and kept == list(range(7))
    assert records[-1]['peer_outlier_mass'] < .001
    assert records[-1]['reasons'] == ['extreme pointwise peer disagreement']
