"""Interpolation conservation, contraction, withheld closure and notebook routing."""
from dataclasses import replace
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import nbformat
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from test_ratio_observable import selected_quadrature
from poodemo import pipeline
from poodemo.data import Run, save_json
from poodemo.inference import component_coefficients
from poodemo.reference_splines import (
    METHODS, SplineStudySettings, TemplateBank, CumulativeTemplates,
    adapt_templates, heldout_eta_grid, initial_eta_grid, make_interpolator,
    metric_pass, rescaled_log_ratio, run_reference_spline_comparison, template_metrics)

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / 'notebooks/09_reference_ratio_spline_templates.ipynb'


@pytest.fixture
def bank(selected_quadrature):
    run, quad = selected_quadrature
    run.config.update(reference_ratio_power=100., reference_ratio_eta_floor=.2,
                      mu_min=.02, mu_max=1.4, asimov_mu=1.)
    return TemplateBank(run, quad, pipeline.observable_bin_edges(36, 'reference_ratio'))


def test_stable_coordinate_and_exact_score_limit(bank):
    for eta in (.2, .7, .974074074, .991111111, 1.008, 1.3):
        u = rescaled_log_ratio(bank.quad, eta)
        actual = pipeline.observable_values(bank.run, bank.quad, eta, 'reference_ratio')
        np.testing.assert_allclose(expit(100 * (eta-1) * u), actual, atol=8e-14)
    score = pipeline._score(bank.run, bank.quad, 1.)
    np.testing.assert_allclose(rescaled_log_ratio(bank.quad, 1.), score, atol=1e-16)
    for eta in (np.nextafter(1., 0.), np.nextafter(1., 2.), 1.-1e-10, 1.+1e-10):
        np.testing.assert_allclose(rescaled_log_ratio(bank.quad, eta), score, atol=5e-10)


@pytest.mark.parametrize('method', METHODS)
def test_totals_exact_plateau_collapse_and_no_extrapolation(bank, method):
    model = make_interpolator(bank, [0., .2, .9, 1., 1.1, 1.4], method)
    plateau = model(np.array([0., .1, .2]))
    np.testing.assert_allclose(plateau, np.broadcast_to(bank.exact(.2), plateau.shape), atol=1e-10)
    np.testing.assert_allclose(model(1.), bank.exact(1.), rtol=1e-12, atol=0.)
    np.testing.assert_array_equal(model(1.) == 0., bank.exact(1.) == 0.)
    for eta in (.23, .974, .991, 1.009, 1.03, 1.4):
        value = model(eta)
        assert np.all(value >= 0)
        np.testing.assert_allclose(value.sum(axis=-1), bank.totals, rtol=1e-12)
    assert len(bank.cdf_cache) <= 2
    with pytest.raises(ValueError, match='outside'):
        model(-.1)
    with pytest.raises(ValueError, match='outside'):
        model(1.41)


def test_empirical_cdf_closes_at_anchors_without_direct_fill_fallback(bank, monkeypatch):
    anchors = np.array([0., .2, .37, .666171875, .7596875, .974074, .991436,
                        1., 1.008545, 1.0273, 1.15625, 1.4])
    reference = {eta: bank.exact(eta).copy() for eta in anchors}
    model = make_interpolator(bank, anchors, 'ecdf_u')
    original = bank.exact
    def special_only(eta):
        assert eta <= bank.floor or eta == 1., 'Empirical anchors must use the CDF, not direct-fill fallback'
        return original(eta)
    monkeypatch.setattr(bank, 'exact', special_only)
    for eta in anchors:
        np.testing.assert_allclose(model(eta), reference[eta], rtol=2e-11, atol=1e-9)
        assert metric_pass(template_metrics(reference[eta], model(eta), [.001, .5, 1., 2.]),
                           SplineStudySettings())


@pytest.mark.parametrize('anchor,eta', [(.72, .73), (1., .974), (1., 1.027), (1.25, 1.3)])
def test_empirical_cdf_uses_query_edges_and_anchor_coordinates(bank, anchor, eta):
    # Independent event histogram of the transported anchor coordinates. This
    # must hold off-anchor too, unlike a fallback to the query's direct fills.
    z = expit(bank.power * (eta - 1) * rescaled_log_ratio(bank.quad, anchor))
    expected = np.array([np.histogram(z, bank.edges, weights=w)[0]
                         for w in bank.weighted.reshape(12, -1)]).reshape(3, 4, -1)
    actual = bank.cdf_u_bins(anchor, eta)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-9)
    np.testing.assert_array_equal(actual[expected == 0.], 0.)


@pytest.mark.parametrize('eta', [.7, 1.3])
def test_empirical_cdf_keeps_ties_on_correct_edges_and_saturated_endpoints(monkeypatch, eta):
    from poodemo import reference_splines
    nominal = np.arange(1., 25.).reshape(4, 6)
    down, up = nominal.copy(), nominal.copy()
    down[-1] *= .5
    up[-1] *= 1.5
    quad = dict(nominal=nominal, down=down[None], up=up[None], weights=np.arange(1., 7.),
                truth=nominal[1]+nominal[3], x=np.zeros((6, 2)))
    run = SimpleNamespace(config={'reference_ratio_power': 100., 'reference_ratio_eta_floor': .2})
    bank = TemplateBank(run, quad, pipeline.observable_bin_edges(36, 'reference_ratio'))
    z = np.array([0., bank.edges[10], bank.edges[10], .5, bank.edges[25], 1.])
    u = logit(z)/(bank.power*(eta-1.))
    u[0], u[-1] = -np.sign(eta-1.)*1e4, np.sign(eta-1.)*1e4
    monkeypatch.setattr(reference_splines, 'rescaled_log_ratio', lambda quad, anchor: u)
    expected = np.array([np.histogram(z, bank.edges, weights=w)[0]
                         for w in bank.weighted.reshape(12, -1)]).reshape(3, 4, -1)
    np.testing.assert_array_equal(bank.cdf_u_bins(eta, eta), expected)
    np.testing.assert_array_equal(expected.sum(axis=-1), bank.totals)


def test_empirical_eta_interpolation_is_convex_and_cache_eviction_is_reproducible(bank):
    model = make_interpolator(bank, [0., .2, .7, 1., 1.4], 'ecdf_u')
    eta = .8
    fraction = (eta-.7)/(1.-.7)
    low = bank.cdf_u_bins(.7, eta)
    high = bank.cdf_u_bins(1., eta)
    np.testing.assert_allclose(model(eta), (1.-fraction)*low+fraction*high, rtol=1e-12)
    before = bank.cdf_u_bins(.7, .73)
    bank.cdf_u_bins(.8, .73)
    bank.cdf_u_bins(.9, .73)
    assert .7 not in bank.cdf_cache
    after = bank.cdf_u_bins(.7, .73)
    np.testing.assert_array_equal(before, after)
    assert len(bank.cdf_cache) == 2


def test_point_mass_cdf_and_shared_nuisance_support():
    nominal = np.tile(np.array([2., 3., 4., 5.])[:, None], (1, 4))
    quad = dict(nominal=nominal, down=nominal[None], up=nominal[None], weights=np.ones(4),
                truth=nominal[1]+nominal[3], x=np.zeros((4, 2)))
    run = SimpleNamespace(config={'reference_ratio_power': 100., 'reference_ratio_eta_floor': .2})
    bank = TemplateBank(run, quad, pipeline.observable_bin_edges(36, 'reference_ratio'))
    model = CumulativeTemplates(bank, [0., .2, .9, 1., 1.1, 1.4])
    for eta in (.3, .99, 1., 1.01):
        assert np.count_nonzero(model(eta).sum(axis=(0, 1))) == 1
        np.testing.assert_allclose(model(eta), bank.exact(eta))


class SmoothBank:
    floor = .2
    totals = np.tile([10., 20., 20., 40.], (3, 1))
    def cdf_u_bins(self, anchor, eta):
        # Controlled moving template for testing adaptive eta refinement.
        return self.exact(anchor)
    def exact(self, eta):
        fraction = .5 + .35*np.tanh((max(eta, self.floor)-.977)/.012)
        totals = np.tile([10., 20., 20., 40.], (3, 1))
        return totals[..., None] * np.array([fraction, 1-fraction])


def test_adaptation_resolves_transition_and_leaves_independent_checks():
    settings = SplineStudySettings(max_anchors=301, max_rounds=8,
                                   max_bin_pull=.025, max_deviance=.002)
    grid = np.array([0., .2, .85, 1., 1.2])
    model, history = adapt_templates(SmoothBank(), grid, 'ecdf_u', settings, [.02, 1., 2.], lambda _: None)
    assert len(model.etas) > len(grid)
    assert history.iloc[-1].stop_reason == 'tolerances_met'
    checks = heldout_eta_grid(model, .2, 25)
    assert all(np.min(abs(model.etas-e)) > 1e-12 for e in checks)
    assert max(template_metrics(SmoothBank().exact(e), model(e), [.02, 1., 2.])['max_deviance']
               for e in checks) < .004
    assert len(model.etas[(model.etas > .94) & (model.etas < 1.01)]) > 5


def test_refinement_limits_are_not_silently_accepted():
    settings = SplineStudySettings(max_anchors=5, max_rounds=2, max_bin_pull=1e-8, max_deviance=1e-10)
    model, history = adapt_templates(SmoothBank(), [0., .2, .85, 1., 1.2], 'ecdf_u',
                                    settings, [1.], lambda _: None)
    assert history.iloc[-1].stop_reason == 'anchor_limit'
    assert history.iloc[-1].n_failed > 0


def test_empty_bins_and_small_yields_are_not_hidden_by_total_normalization():
    direct = SmoothBank().exact(.5)
    direct[..., 0] = 0.
    prediction = direct.copy()
    prediction[..., 0] = 1.
    metrics = template_metrics(direct, prediction, [1.])
    assert not metric_pass(metrics, SplineStudySettings())
    prediction[..., 1] = 0.
    assert template_metrics(direct, prediction, [1.])['max_deviance'] == np.inf
    assert np.all(direct[..., 0] == 0.)  # No diagnostic floor mutates input.


def test_grid_has_true_reference_focus_and_no_ulp_intervals(bank):
    grid = initial_eta_grid(bank.run, SplineStudySettings())
    assert all(e in grid for e in [0., .2, 1., 1.4])
    assert np.min(np.diff(grid)) > 1e-8
    assert len(grid[(grid >= .9) & (grid <= 1.1)]) >= 81


def _small_study(tmp_path, monkeypatch, selected_quadrature):
    original, quad = selected_quadrature
    config = dict(reference_ratio_power=100., reference_ratio_eta_floor=.2,
                  mu_min=.95, mu_max=1.05, mu_points=3, asimov_mu=1.,
                  mu_fit_bounds=[.001, 2.], nuisance_bounds=[-3., 3.], bin_counts=[4])
    run = Run(tmp_path/'evaluation', config, original.model)
    run.path('results').mkdir(parents=True)
    save_json(run.path('results', 'reference_ratio_binning_config.json'), {
        **pipeline.observable_configuration(run, 'reference_ratio'),
        'bin_counts': [4], 'bin_edges': {'4': pipeline.observable_bin_edges(4, 'reference_ratio').tolist()}})
    pd.DataFrame([dict(n_bins=4, information_fraction=.99,
                      **pipeline.observable_configuration(run, 'reference_ratio'))]).to_csv(
                          run.path('results', 'reference_ratio_binning_fisher.csv'), index=False)
    pd.DataFrame([dict(mu=1., systematics=False, model='analytic unbinned', q=0.)]).to_csv(
        run.path('results', 'unbinned_scans.csv'), index=False)
    sentinel = run.path('results', 'reference_ratio_spline_templates.npz')
    sentinel.write_bytes(b'legacy untouched')
    monkeypatch.setattr(pipeline, 'prepare_quadrature', lambda _: quad)
    settings = SplineStudySettings(initial_anchors=3, focus_anchors=3, max_anchors=10,
        max_rounds=1, likelihood_refine_points=2, validation_intervals=2, validation_fit_points=1,
        surface_points=1, surface_mu_points=5, surface_alpha_points=5)
    result = run_reference_spline_comparison(run, n_bins=4, settings=settings, progress=lambda _: None)
    assert sentinel.read_bytes() == b'legacy untouched'
    return run, result


def test_complete_workflow_with_real_fits_and_output_roundtrip(tmp_path, monkeypatch, selected_quadrature):
    run, study = _small_study(tmp_path, monkeypatch, selected_quadrature)
    assert set(study['models']) == {'ecdf_u'}
    assert set(study['scans'].method) == {'direct', 'ecdf_u'}
    for row in study['summary'].itertuples():
        validation = study['validation'].query('method == @row.method')
        heldout = validation.query("role == 'heldout'")
        anchors = validation.query("role == 'anchor_coordinate_check'")
        assert row.heldout_checks == len(heldout)
        assert row.heldout_failures == (~heldout.passed).sum()
        assert row.anchor_checks == len(anchors) == row.n_anchors
        assert row.anchor_failures == (~anchors.passed).sum()
        assert row.fit_checks == len(study['comparison'].query('method == @row.method'))
    empirical = study['summary'].query("method == 'ecdf_u'").iloc[0]
    assert empirical.anchor_failures == 0
    assert 'template_failures' not in study['summary']
    assert study['comparison'].q_direct.notna().any()
    # A failed optimizer must be reported, never turned into passing closure.
    invalid = ~study['comparison'].valid | ~study['comparison'].valid_direct
    assert not study['comparison'].loc[invalid, 'q_passed'].any()
    assert all(n > 0 for n in study['summary'].off_anchor_fits)
    for name in ('summary', 'scans', 'comparison', 'fixed_checks', 'surfaces', 'bin_errors', 'refinement', 'likelihood_refinement'):
        saved = pd.read_csv(study['output']/f'{name}.csv')
        assert len(saved) == len(study[name])
    metadata = json.loads((study['output']/'metadata.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata['schema_version'] == 3
    assert set(metadata['cdf_recipes']) == set(METHODS)
    assert len(metadata['quadrature_fingerprint']) == 64
    for method in METHODS:
        with np.load(study['output']/f'{method}_anchors.npz') as archive:
            np.testing.assert_array_equal(archive['etas'], study['models'][method].etas)
    for eta in study['surfaces'].eta.unique():
        direct = study['surfaces'].query('method == "direct" and eta == @eta and mu == 1 and alpha_NI == 0')
        assert direct.nll.item() < 1e-15
    import matplotlib
    matplotlib.use('Agg')
    from poodemo.reference_spline_plots import plot_reference_spline_comparison
    figures = plot_reference_spline_comparison(run, study)
    assert len(figures) >= 5
    assert len(list(run.path('plots', '09_interpolation').glob('*.png'))) == len(figures)
    import matplotlib.pyplot as plt
    plt.close('all')


def test_notebook_builder_is_repeatable_and_preserves_controls(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    builder = runpy.run_path(str(REPO/'scripts/build_reference_notebooks.py'))
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    controls = next(c for c in notebook.cells if c.id == 'spline-controls')
    controls.source = controls.source.replace('max_rounds=12', 'max_rounds=3')
    source = next(c for c in notebook.cells if c.id == 'reference-model-source')
    source.source = source.source.replace('RATIO_SOURCE = "unity"', 'RATIO_SOURCE = "drive"')
    notebook.cells = builder['configure_reference_cells'](notebook.cells, 9)
    first = [c.source for c in notebook.cells]
    notebook.cells = builder['configure_reference_cells'](notebook.cells, 9)
    assert first == [c.source for c in notebook.cells]
    assert any('max_rounds=3' in c for c in first)
    assert any('RATIO_SOURCE = "drive"' in c for c in first)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert 'ROOT / "results"' not in cell.source and 'ROOT / "plots"' not in cell.source
            assert 'replace(run,' not in cell.source
            compile(cell.source, '<notebook09>', 'exec')


def test_notebook_runs_study_with_selected_evaluation(monkeypatch, tmp_path):
    # Existing model-selection tests exercise bundle/filter construction. Here
    # execute 09's actual controls/run cells and verify the selected object is
    # passed through unchanged, including its result routing.
    from poodemo import reference_splines
    from poodemo.member_filter import FilteredEvaluationRun
    from poodemo.ensemble_bundle import BundleEvaluationRun
    base = Run(tmp_path/'base', {}, None)
    bundle = BundleEvaluationRun(base, tmp_path/'bundle', {'campaign_id': 'campaign', 'name': 'unity100'})
    selected = FilteredEvaluationRun(bundle, tmp_path/'evaluation', {'identity': 'filter'})
    seen = []
    def study(run, *, n_bins, settings):
        assert run is selected and n_bins == 36
        seen.append(run.path('results'))
        return dict(summary=pd.DataFrame(), output=run.path('results', 'reference_ratio_interpolation'))
    monkeypatch.setattr(reference_splines, 'run_reference_spline_comparison', study)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    ids = [c.id for c in notebook.cells]
    assert ids.index('reference-model-source') < ids.index('spline-run')
    ns = dict(run=selected, N_BINS=36, display=lambda _: None)
    for ident in ('spline-controls', 'spline-run'):
        exec(next(c.source for c in notebook.cells if c.id == ident), ns)
    assert seen == [selected.path('results')]


def test_rescaled_cdf_respects_local_rate_only_plateau(bank):
    model = CumulativeTemplates(bank, [0., .2, .9, 1., 1.1, 1.4])
    for eta in (1.-1e-8, 1.+1e-8):
        value = model(eta)
        assert np.count_nonzero(value.sum(axis=(0, 1))) == 1
        np.testing.assert_allclose(value, bank.exact(eta), rtol=1e-12, atol=1e-10)


def test_fit_certificate_requires_the_actual_nonnegative_bound(bank):
    from poodemo.inference import FitResult
    from poodemo.reference_splines import _fit
    run = bank.run.with_config(mu_fit_bounds=[.001, 2.], nuisance_bounds=[-3., 3.])
    class FailedLineSearch:
        value = 0.
        def fit(self, **kwargs):
            return FitResult(1., np.zeros(1), self.value, False, 'ABNORMAL line search', 3)
        def nll(self, mu, alpha):
            return self.value
    model = FailedLineSearch()
    result = _fit(run, model, 1., True)
    assert result['valid'] and result['null_certified'] and result['best_certified']
    assert not result['null_optimizer_success'] and result['q'] == 0.
    model.value = .1
    result = _fit(run, model, 1., True)
    assert not result['valid'] and not result['null_certified'] and np.isnan(result['q'])


def test_likelihood_refinement_requests_neighbor_anchors_and_caches_direct_fits(monkeypatch):
    from poodemo import reference_splines as module
    bank = SmoothBank()
    bank.run = SimpleNamespace()
    bank.counts = lambda eta: np.ones(2)
    settings = SplineStudySettings(max_anchors=10, max_rounds=1,
                                  max_bin_pull=1e6, max_deviance=1e6)
    # Template tolerances deliberately cannot trigger refinement. A controlled
    # fit discrepancy alone must request knots, but not insert the fit point.
    calls = []
    def fake_fit(run, model, eta, syst):
        calls.append((eta, syst))
        return dict(q=0. if len(calls) <= 2 else .1, valid=True, message='test')
    monkeypatch.setattr(module, '_fit', fake_fit)
    model, history = adapt_templates(bank, [0., .2, .85, 1., 1.2], 'ecdf_u',
        settings, [1.], lambda _: None, likelihood_points=[.925])
    assert len(calls) == 6  # 2 direct once + 2 approximate on each of two rounds.
    assert len(model.etas) == 6 and .925 not in model.etas
    assert (history.n_failed == 0).all() and (history.q_failed == 2).all()
    assert history.iloc[-1].stop_reason == 'round_limit'
    reserved = heldout_eta_grid(model, .2, 3)
    assert all(not np.isclose(e, .925, atol=1e-13, rtol=0.) for e in reserved)
    assert set(model.refinement_checks.eta) == {.925}
