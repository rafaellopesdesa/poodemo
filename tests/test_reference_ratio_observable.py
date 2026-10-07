"""Pairwise compression, exact collapse, and separate workflow products."""
import numpy as np
import pandas as pd
import pytest

from test_ratio_observable import selected_quadrature, _categorical_quadrature
from poodemo import pipeline
from poodemo.data import Run
from poodemo.diagnostics import local_asimov_information, run_estimator_study
from poodemo.inference import component_coefficients


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
    hist = parent.binned(np.full(len(reference), .5), np.linspace(0, 1, 21), eta=1.)
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
                            bin_counts=[4, 20], score_scale=.15, spline_anchors=8,
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
