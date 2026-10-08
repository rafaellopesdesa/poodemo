"""Pairwise compression, exact collapse, and separate workflow products."""
import json
import numpy as np
import pandas as pd
import pytest

from test_ratio_observable import selected_quadrature, _categorical_quadrature
from poodemo import pipeline
from poodemo.data import Run
from poodemo.diagnostics import local_asimov_information, run_estimator_study
from poodemo.inference import component_coefficients


@pytest.mark.parametrize("power", [1., 10., 100.])
def test_stretched_ratio_keeps_order_and_expands_about_half(selected_quadrature, power):
    run, quad = selected_quadrature
    original = pipeline.observable_values(run, quad, .6, 'reference_ratio')
    run.config['reference_ratio_power'] = power
    actual = pipeline.observable_values(run, quad, .6, 'reference_ratio')
    from scipy.special import expit, logit
    np.testing.assert_allclose(actual, expit(power * logit(original)), rtol=1e-10, atol=1e-14)
    assert np.all(np.diff(actual[np.argsort(original)]) >= 0)
    assert np.all(np.abs(actual - .5) >= np.abs(original - .5) - 1e-14)
    assert np.all(np.isfinite(actual)) and np.all((actual >= 0) & (actual <= 1))
    np.testing.assert_array_equal(pipeline.observable_values(run, quad, 1., 'reference_ratio'), .5)
    # The stretch belongs only to the pairwise observable.
    ratio = pipeline.observable_values(run, quad, .6, 'ratio')
    run.config['reference_ratio_power'] = 1.
    np.testing.assert_array_equal(ratio, pipeline.observable_values(run, quad, .6, 'ratio'))


@pytest.mark.parametrize("power", [0., -1., np.nan, np.inf, "invalid"])
def test_stretch_requires_a_positive_finite_power(selected_quadrature, power):
    run, quad = selected_quadrature
    run.config['reference_ratio_power'] = power
    with pytest.raises(ValueError, match="reference_ratio_power"):
        pipeline.observable_values(run, quad, 1., 'reference_ratio')


@pytest.mark.parametrize("observable", ["ratio", "reference_ratio"])
def test_ratio_plots_allow_zero_eta(selected_quadrature, observable):
    run, quad = selected_quadrature
    values = pipeline.observable_values(run, quad, 0., observable)
    assert np.all(np.isfinite(values)) and np.all((values >= 0) & (values <= 1))
    with pytest.raises(ValueError, match="strictly positive"):
        pipeline.observable_values(run, quad, 0., "score")


@pytest.mark.parametrize("n_bins", [1, 2, 3, 4, 20, 31, 60, 100, 1024])
def test_reference_edges_keep_half_and_roundoff_in_one_bin(n_bins):
    edges = pipeline.observable_bin_edges(n_bins, 'reference_ratio')
    assert len(edges) == n_bins + 1 and edges[0] == 0 and edges[-1] == 1
    assert np.all(np.diff(edges) > 0) and not np.any(edges == .5)
    center = np.searchsorted(edges, .5, side='right') - 1
    assert edges[center] < .5 < edges[center + 1]
    if n_bins % 2 == 0:
        np.testing.assert_allclose(np.diff(edges)[[0, -1]], [.5 / n_bins, 1.5 / n_bins])
        np.testing.assert_allclose(np.diff(edges)[1:-1], 1. / n_bins)
        if n_bins >= 4:
            np.testing.assert_allclose((edges[center] + edges[center + 1]) / 2, .5)
            np.testing.assert_allclose(edges[center + 1] - edges[center], 1. / n_bins)
    points = np.array([np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.)])
    counts = np.histogram(points, edges)[0]
    assert counts[center] == 3 and np.count_nonzero(counts) == 1
    for observable in ['score', 'ratio']:
        np.testing.assert_array_equal(pipeline.observable_bin_edges(n_bins, observable),
                                      np.linspace(0., 1., n_bins + 1))


def test_fisher_study_rejects_zero_scan_before_loading_data(tmp_path, monkeypatch):
    run = Run(tmp_path, dict(mu_min=0., mu_max=1.5, mu_points=3, asimov_mu=1.), None)
    def no_data(run):
        pytest.fail('Invalid Fisher grid must be rejected before loading data')
    monkeypatch.setattr(pipeline, 'prepare_quadrature', no_data)
    with pytest.raises(ValueError, match='strictly positive'):
        pipeline.run_binning_study(run, observable='reference_ratio')


def test_normalized_physical_ratio_and_exact_collapse(selected_quadrature):
    run, quad = selected_quadrature
    reference = component_coefficients(1.) @ quad['nominal']
    for eta in (.5, .73, 1., 1.3):
        target = component_coefficients(eta) @ quad['nominal']
        ratio = (target / (target @ quad['weights'])) / (reference / (reference @ quad['weights']))
        z = pipeline.observable_values(run, quad, eta, 'reference_ratio')
        np.testing.assert_allclose(z, ratio / (1 + ratio), rtol=3e-14)
        # The common S reference must cancel, including selected normalization.
        r_eta = pipeline.observable_values(run, quad, eta, 'ratio')
        r_one = pipeline.observable_values(run, quad, 1., 'ratio')
        quotient = (r_eta / (1-r_eta)) / (r_one / (1-r_one))
        np.testing.assert_allclose(z, quotient / (1+quotient), rtol=3e-13)
    np.testing.assert_array_equal(pipeline.observable_values(run, quad, 1., 'reference_ratio'), .5)
    parent = pipeline._template_likelihood(quad, systematics=False)
    hist = parent.binned(np.full(len(reference), .5), pipeline.observable_bin_edges(20, 'reference_ratio'), eta=1.)
    assert np.count_nonzero(hist.nominal.sum(axis=0)) == 1
    information = local_asimov_information(hist, 1.)
    np.testing.assert_allclose(information['shape_matrix'], 0., atol=1e-7)
    np.testing.assert_allclose(information['matrix'], information['rate_matrix'])


def test_resolved_ratio_preserves_pairwise_likelihood_not_fisher(tmp_path):
    quad = _categorical_quadrature()
    run = Run(tmp_path, {}, None)
    parent = pipeline._template_likelihood(quad, systematics=False)
    for eta in (.5, .8, 1.3):
        z = pipeline.observable_values(run, quad, eta, 'reference_ratio')
        unique = np.unique(z)
        edges = np.r_[0., (unique[:-1]+unique[1:])/2, 1.]
        frozen = parent.binned(z, edges, eta=eta)
        np.testing.assert_allclose(frozen.nll(eta)-frozen.nll(1.),
                                   parent.nll(eta)-parent.nll(1.), rtol=1e-10, atol=1e-13)


def test_reference_workflows_include_collapse_and_isolate_artifacts(tmp_path, monkeypatch):
    run = Run(tmp_path, dict(asimov_mu=1., mu_min=.5, mu_max=1.5, mu_points=5,
                            mu_fit_bounds=[.2, 2.], nuisance_bounds=[-2., 2.],
                            bin_counts=[4, 20], score_scale=.15, reference_ratio_power=10., spline_anchors=5,
                            spline_focus_anchors=0, spline_focus_halfwidth=.1), None)
    run.path('results').mkdir()
    for name in ['binning_scans.csv', 'ratio_binning_scans.csv']:
        run.path('results', name).write_text('preserve me')
    monkeypatch.setattr(pipeline, 'prepare_quadrature', lambda run: _categorical_quadrature())
    monkeypatch.setattr(pipeline, '_saved_unbinned', lambda *a, **k: pd.DataFrame())
    bins = pipeline.run_binning_study(run, observable='reference_ratio')
    spline = pipeline.run_spline_study(run, n_bins=20, observable='reference_ratio')
    estimators = run_estimator_study(run, observable='reference_ratio')
    for frame in [bins['scans'], spline['scans']]:
        assert frame.valid.all()
        assert np.isfinite(frame.q).all()
        np.testing.assert_allclose(frame.loc[frame.mu == 1., 'q'], 0., atol=1e-8)
    collapse = bins['fidelity'].query('eta == 1.')
    assert (collapse.occupied_bins == 1).all()
    np.testing.assert_allclose(collapse.shape_information_fraction, 0., atol=1e-12)
    assert 1. in spline['spline'].etas
    assert estimators.fit_valid.all() and estimators.width_valid.all()
    for name in ['binning_scans.csv', 'ratio_binning_scans.csv']:
        assert run.path('results', name).read_text() == 'preserve me'
    assert run.path('results', 'reference_ratio_estimator_eta.csv').exists()
    settings = pipeline.observable_configuration(run, 'reference_ratio')
    for frame in [bins['scans'], bins['fidelity'], spline['scans'], estimators]:
        for key, value in settings.items():
            assert (frame[key] == value).all()
    with np.load(run.path('results', 'reference_ratio_spline_templates.npz')) as archive:
        np.testing.assert_array_equal(archive['edges'], pipeline.observable_bin_edges(20, 'reference_ratio'))
        assert archive['reference_ratio_power'] == 10.
    saved = json.loads(run.path('results', 'reference_ratio_spline_choice.json').read_text())
    assert saved['reference_ratio_power'] == 10.
    assert .5 not in saved['bin_edges']

    # An updated notebook 09 cannot silently reuse the original/another-a study.
    run.config['reference_ratio_power'] = 100.
    with pytest.raises(ValueError, match='Rerun notebook 08'):
        pipeline.run_spline_study(run, n_bins=20, observable='reference_ratio')
    run.config['reference_ratio_power'] = 10.
    config_path = run.path('results', 'reference_ratio_binning_config.json')
    original = config_path.read_text()
    config_path.unlink()
    with pytest.raises(ValueError, match='Rerun notebook 08'):
        pipeline.run_spline_study(run, n_bins=20, observable='reference_ratio')
    incompatible = json.loads(original)
    incompatible['binning_policy'] = 'uniform_v1'
    config_path.write_text(json.dumps(incompatible))
    with pytest.raises(ValueError, match='Rerun notebook 08'):
        pipeline.run_spline_study(run, n_bins=20, observable='reference_ratio')
