"""Statistical checks for independent fixed-compression integration."""
from types import SimpleNamespace

import numpy as np
import pytest

from poodemo.physics import PhysicsModel
from poodemo.toy_integration_diagnostics import (
    _blank, _accumulate, _combine, _moments,
    diagnostic_bin_edges, run_integration_diagnostics,
)


class AnalyticWorkspace:
    def __init__(self, model, factor=1.):
        self.model, self.factor = model, factor
        self.calls = 0

    def evaluate(self, x):
        self.calls += 1
        nominal = self.model.component_densities(x).T * self.factor
        return nominal, nominal, nominal


def _context(tmp_path, n=1500):
    model = PhysicsModel(lambda_s=4., lambda_b=15., lambda_ni=35., exposure=1.)
    rng = np.random.default_rng(77219)
    x = np.concatenate([model.sample_component(p, n, rng) for p in ("S", "B", "NI")])
    fields = model.component_densities(x).T
    weights = 1. / (n * sum(model.component_pdf(x, p) for p in ("S", "B", "NI")))
    quad = dict(x=x, nominal=fields, weights=weights, stratum=np.repeat(np.arange(3), n))
    rates = fields @ weights
    # A deliberate learned rescaling must survive independent integration;
    # the diagnostic must never renormalize it away.
    factor = 1.07
    return dict(run=SimpleNamespace(model=model), quad=quad, rates=rates,
                learned_rates=factor*rates, workspace=AnalyticWorkspace(model, factor),
                selector=None, threshold=None, original_n_per_source=n,
                configuration=dict(mu_values=[0., 1.4], exposure_multiplier=2.,
                                   reference_power=100., n_bins=12),
                fingerprint="test-model-bank-frozen", output_dir=tmp_path)


def test_nested_refinement_preserves_bins_and_half():
    base, fine = diagnostic_bin_edges(12), diagnostic_bin_edges(36)
    np.testing.assert_array_equal(fine[::3], base)
    assert not np.any(fine == .5)
    for invalid in (24, 35, 36.5):
        with pytest.raises(ValueError):
            diagnostic_bin_edges(invalid)


def test_fixed_generated_covariance_includes_rejected_zeros():
    # Two selected weighted draws and three rejected draws in a fixed n=5
    # stratum. Cross-component correlation must survive the reduction.
    draws = np.array([[2., 1.], [6., 5.], [0., 0.], [0., 0.], [0., 0.]])
    mean, covariance = _moments(draws[:2].sum(axis=0), draws[:2].T @ draws[:2], len(draws))
    np.testing.assert_allclose(mean, draws.mean(axis=0))
    np.testing.assert_allclose(covariance, np.cov(draws, rowvar=False, ddof=1)/len(draws))
    assert covariance[0, 1] > 0


def test_score_variance_includes_cross_bin_covariance():
    # The first two events occupy one bin and the third occupies the other.
    # Fixed-size sampling creates a negative cross-bin covariance.
    fields = np.array([[1., 2., 4.], [4., 7., 3.], [1., 5., 8.], [3., 1., 4.]])
    rates = fields.sum(axis=1)
    spec = dict(mu=0., edges=np.array([0., .5, 1.]), coefficients=np.array([0., 0., 1., 1.]),
                score_weights=np.array([-.2, .6]))
    state = _blank([spec])
    _accumulate(state, fields, fields, np.array([1., 2., 3.]), [spec], rates, 1., 2.)
    result = _combine([state], 5, [spec])
    scalar = np.r_[2.*(spec["coefficients"] @ fields)*np.array([1., 2., 3.])*np.array([-.2, -.2, .6]), 0., 0.]
    assert result["score_means"][0] == pytest.approx(scalar.mean())
    assert result["score_variances"][0] == pytest.approx(scalar.var(ddof=1)/5)
    # Same-bin covariance retains cancellation in the physical sum.
    projected = np.einsum("i,ijb,j->b", spec["coefficients"], result["covariance"][0], spec["coefficients"])
    physical = np.array([[4., 0.], [12., 0.], [0., 36.], [0., 0.], [0., 0.]])
    np.testing.assert_allclose(projected, physical.var(axis=0, ddof=1)/5)
    independent_bins_variance = 4.*np.sum(projected*spec["score_weights"]**2)
    assert not np.isclose(independent_bins_variance, result["score_variances"][0])


def test_independent_rates_templates_and_resume(tmp_path):
    context = _context(tmp_path)
    settings = dict(generated_per_source=2000, multiples=(1, 3), replicas=2,
                    n_bins=(12, 36), seed=82721, chunk_size=701)
    original_rates = context["rates"].copy()
    study = run_integration_diagnostics(context, **settings)
    assert len(study["templates"]) == 4
    np.testing.assert_array_equal(context["rates"], original_rates)
    np.testing.assert_allclose(study["learned_rates"], study["rates"]*1.07, rtol=1e-14)
    for (bins, mu), template in study["templates"].items():
        np.testing.assert_allclose(template.sum(axis=1), study["rates"], rtol=2e-14)
        assert np.all(template >= 0)
    # Nested finer yields sum exactly to the coarser integral, because the
    # same proposal events populate both.
    for mu in (0., 1.4):
        np.testing.assert_allclose(study["templates"][36, mu].reshape(4, 12, 3).sum(axis=2),
                                   study["templates"][12, mu], rtol=1e-13)
    for row in study["rate_closure"].itertuples():
        expected = 2.*context["run"].model.total_yield(row.mu_test)
        if row.model == "learned":
            expected *= 1.07
        assert abs(row.estimate-expected) < 5.*row.standard_error
    assert np.isfinite(study["bin_closure"].baseline_se_fixed_partition).all()
    calls = context["workspace"].calls
    repeat = run_integration_diagnostics(context, **settings)
    assert context["workspace"].calls == calls
    np.testing.assert_array_equal(study["rates"], repeat["rates"])
    for name in ("bin_closure", "rate_closure", "convergence", "score_closure"):
        assert study[name].equals(repeat[name])
    context["fingerprint"] = "changed-input"
    with pytest.raises(ValueError, match="changed"):
        run_integration_diagnostics(context, **settings)


def test_selected_rate_variance_counts_rejected_draws(tmp_path):
    from scipy.special import ndtr
    context = _context(tmp_path)
    class Selector:
        def predict_proba(self, x):
            return (x[:, :1] >= .4).astype(float)
    context["selector"], context["threshold"] = Selector(), .5
    # The deliberately inconsistent old full-space rates are frozen. The
    # independent proposal result must discover the acceptance loss.
    study = run_integration_diagnostics(context, generated_per_source=6000,
                                        multiples=(1,), replicas=1, n_bins=(12,), chunk_size=1200)
    model = context["run"].model
    expected = 0.
    for process in ("B", "NI"):
        covariance = np.asarray(getattr(model, f"cov_{process.lower()}"))
        probability = ndtr((model.mean(process)[0]-.4)/np.sqrt(covariance[0, 0]))
        expected += model.component_yield(process)*probability
    row = study["rate_closure"].query("model == 'analytic' and mu_test == 0").iloc[0]
    assert abs(row.estimate-2.*expected) < 5.*row.standard_error
    assert row.selected < 3*6000
    assert row.ratio < .8
