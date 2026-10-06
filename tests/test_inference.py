"""Numerical/statistical invariants of the shared likelihood implementation."""

import numpy as np
import pytest
from decimal import Decimal, localcontext
from scipy.special import expit

from poodemo.inference import (
    TemplateLikelihood, YieldFractionSpline, component_coefficients,
    exp_poly_factor, histogram_components, poisson_asimov_nll,
)


def small_model():
    x = np.linspace(-2, 2, 301)
    w = np.full(len(x), x[1] - x[0])
    w[[0, -1]] /= 2
    signal = 3 * np.exp(-x**2 / 0.7)

    def fields(alpha_ni=0):
        background = 8 * np.exp(-(x - 0.3)**2 / 1.4)
        interference = 0.35 * np.sqrt(signal * background) * np.sin(x)
        ni = 4 * np.exp(-(x - 0.2 * alpha_ni)**2 / 3)
        return np.stack([signal, signal + background + interference, background, ni])

    nominal = fields()
    down = np.stack([fields(-1)])
    up = np.stack([fields(1)])
    truth = nominal[1] + nominal[3]
    return x, TemplateLikelihood(nominal, down, up, w, truth)


def test_physical_coefficients():
    np.testing.assert_array_equal(component_coefficients(0), [0, 0, 1, 1])
    np.testing.assert_array_equal(component_coefficients(1), [0, 1, 0, 1])
    # The four-template representation must recover B + sqrt(mu)*I + mu*S + NI.
    components = np.array([[3.0, 4], [12.0, 14], [8.0, 9], [2.0, 2]])
    interference = components[1] - components[0] - components[2]
    mu = 2.25
    expected = mu * components[0] + np.sqrt(mu) * interference + components[2] + components[3]
    np.testing.assert_allclose(component_coefficients(mu) @ components, expected)
    with pytest.raises(ValueError):
        component_coefficients(-0.1)


def test_exp_poly_anchor_values_and_c2_matching():
    down, up = np.array([0.72, 1.13, 0.9]), np.array([1.25, 0.91, 1.4])
    np.testing.assert_allclose(exp_poly_factor(-1, down, up), down)
    np.testing.assert_allclose(exp_poly_factor(0, down, up), 1)
    np.testing.assert_allclose(exp_poly_factor(1, down, up), up)
    np.testing.assert_allclose(exp_poly_factor(1.7, down, up), up**1.7)
    np.testing.assert_allclose(exp_poly_factor(-1.7, down, up), down**1.7)
    h = 2e-4
    for endpoint, value, derivative in [(-1, down, -down * np.log(down)),
                                        (1, up, up * np.log(up))]:
        first = (exp_poly_factor(endpoint + h, down, up) - exp_poly_factor(endpoint - h, down, up)) / (2 * h)
        second = (exp_poly_factor(endpoint + h, down, up) - 2 * value
                  + exp_poly_factor(endpoint - h, down, up)) / h**2
        np.testing.assert_allclose(first, derivative, atol=2e-7)
        np.testing.assert_allclose(second, value * np.log(value)**2, atol=2e-4)


def test_common_quadrature_and_nuisance_anchors():
    _, model = small_model()
    assert model.n_nuisance == 1
    assert abs(model.nll(1, [0])) < 1e-12
    assert model.nll(1.2, [0]) > 0
    for k in range(model.n_nuisance):
        alpha = np.zeros(model.n_nuisance)
        alpha[k] = -1
        np.testing.assert_allclose(model.components(alpha), model.down[k], rtol=2e-12)
        alpha[k] = 1
        np.testing.assert_allclose(model.components(alpha), model.up[k], rtol=2e-12)
    alpha = np.array([0.45])
    morphed_rates = model.rates.copy()
    for k, value in enumerate(alpha):
        morphed_rates *= exp_poly_factor(value, model.rates_down[k] / model.rates,
                                        model.rates_up[k] / model.rates)
    np.testing.assert_allclose(model.components(alpha) @ model.quadrature_weights,
                               morphed_rates, rtol=2e-12)
    # This demonstration has only an NI nuisance: S, SBI, and B stay fixed.
    np.testing.assert_allclose(model.components(alpha)[:3], model.nominal[:3], rtol=2e-12)


def test_poisson_kl_handles_empty_bins_and_physical_domain():
    assert poisson_asimov_nll([0, 2], [0, 2]) == 0
    assert poisson_asimov_nll([1, 2], [0, 2]) == 1
    assert np.isinf(poisson_asimov_nll([0, 2], [1, 2]))
    assert np.isinf(poisson_asimov_nll([-1, 2], [0, 2]))


def test_histogram_refills_truth_and_morph_order_does_not_commute():
    x, model = small_model()
    scores = (x + 2) / 4
    edges = np.linspace(0, 1, 9)
    before = model.binned(scores, edges, morph_order="before", eta=1.0)
    after = model.binned(scores, edges, morph_order="after", eta=1.0)
    manual_counts = np.histogram(scores, bins=edges, weights=model.asimov_weights)[0]
    np.testing.assert_allclose(before.data_counts, manual_counts)
    np.testing.assert_allclose(after.data_counts, manual_counts)
    np.testing.assert_allclose(before.components([0]), after.components([0]))
    alpha = [-0.45]
    integrated = histogram_components(scores, model.components(alpha), model.quadrature_weights, edges)
    np.testing.assert_allclose(before.components(alpha), integrated)
    assert np.max(np.abs(before.components(alpha) - after.components(alpha))) > 1e-6
    # Both preserve the same component rates; the difference is bin shape.
    np.testing.assert_allclose(before.components(alpha).sum(axis=1),
                               after.components(alpha).sum(axis=1), rtol=2e-12)


def test_fit_recovers_model_asimov_truth():
    _, model = small_model()
    fit = model.fit(initial_mu=0.65, initial_alpha=[-0.1], mu_starts=[0.5, 1.4])
    assert fit.success, fit.message
    np.testing.assert_allclose(fit.parameters, [1, 0], atol=2e-4)
    assert abs(fit.nll) < 1e-7
    statistic = model.test_statistic(1)
    assert statistic["t"] < 1e-7
    fixed_nuisance = model.fit(initial_mu=0.65, systematics=False)
    assert fixed_nuisance.success
    np.testing.assert_array_equal(fixed_nuisance.alpha, [0])
    assert abs(fixed_nuisance.mu - 1) < 2e-4


def test_unit_ni_gaussian_constraint_survives_histograms_and_spline_templates():
    x, base = small_model()
    # Neutral NI anchors isolate the auxiliary likelihood from morphing.
    # A unit Gaussian exp(-alpha_NI**2/2) adds alpha_NI**2 to -2 log L.
    neutral = TemplateLikelihood(base.nominal, base.nominal[None], base.nominal[None],
                                 base.quadrature_weights, base.truth_intensity)
    edges = np.linspace(0, 1, 9)
    eta = 0.8
    scores = expit(eta * x)
    direct_before = neutral.binned(scores, edges, morph_order="before", eta=eta)
    direct_after = neutral.binned(scores, edges, morph_order="after", eta=eta)

    anchors = np.array([0.4, 1.0, 1.7])
    templates = []
    for anchor in anchors:
        nominal = histogram_components(expit(anchor * x), neutral.nominal,
                                       neutral.quadrature_weights, edges)
        templates.append(np.stack([nominal, nominal, nominal]))
    spline = YieldFractionSpline(anchors, np.stack(templates))
    nominal, down, up = spline(eta)
    # Data are refilled at this held-out eta, never interpolated by the spline.
    counts = np.histogram(scores, edges, weights=neutral.asimov_weights)[0]
    splined = TemplateLikelihood(nominal, down[None], up[None], np.ones(len(counts)), counts)

    for model in (neutral, direct_before, direct_after, splined):
        np.testing.assert_array_equal(model.aux_center, [0])
        np.testing.assert_array_equal(model.aux_sigma, [1])
        baseline = model.nll(1, [0])
        for alpha_ni in [-2.0, -0.4, 0.6, 1.5]:
            difference = model.nll(1, [alpha_ni]) - baseline
            assert difference == pytest.approx(alpha_ni**2, abs=2e-12)
            assert np.exp(-difference / 2) == pytest.approx(np.exp(-alpha_ni**2 / 2), rel=2e-12)
        fitted = model.fit(mu_fixed=1, initial_alpha=[0.8])
        assert fitted.success, fitted.message
        assert abs(fitted.alpha[0]) < 1e-6


def test_fraction_spline_preserves_rates_and_rejects_extrapolation():
    etas = np.array([0.4, 0.8, 1.3, 2.0])
    total = np.array([3.0, 7.0])
    raw = np.stack([np.array([[1 + eta, 2, 3 - eta, 0],
                             [2, 1 + eta, 4 - eta, 0]]) for eta in etas])
    fractions = raw / raw.sum(axis=-1, keepdims=True)
    yields = fractions * total[None, :, None]
    spline = YieldFractionSpline(etas, yields)
    np.testing.assert_allclose(spline(etas), yields, rtol=1e-12, atol=1e-13)
    heldout = spline(np.array([0.5, 1.0, 1.7]))
    assert np.all(heldout >= 0)
    np.testing.assert_allclose(heldout.sum(axis=-1), np.broadcast_to(total, (3, 2)))
    np.testing.assert_array_equal(heldout[..., -1], 0)
    with pytest.raises(ValueError, match="extrapolation"):
        spline(2.1)
    with pytest.raises(ValueError, match="totals"):
        YieldFractionSpline(etas, yields * np.arange(1, 5)[:, None, None])


def test_fraction_spline_handles_zero_anchor_bins_without_log_floor():
    etas = np.array([0.2, 0.8, 1.4, 2.0])
    fractions = np.array([[0.0, 0.8, 0.2, 0.0],
                          [0.2, 0.0, 0.8, 0.0],
                          [0.5, 0.2, 0.3, 0.0],
                          [0.0, 0.4, 0.6, 0.0]])
    total = 120.0
    spline = YieldFractionSpline(etas, total * fractions)
    restored = spline(etas)
    np.testing.assert_allclose(restored, total * fractions, rtol=2e-15, atol=2e-14)
    np.testing.assert_array_equal(restored[fractions == 0], 0)
    intermediate = spline(np.linspace(etas[0], etas[-1], 151))
    assert np.all(intermediate >= 0)
    np.testing.assert_array_equal(intermediate[:, -1], 0)
    np.testing.assert_allclose(intermediate.sum(axis=-1), total, rtol=2e-15)
    # An empty endpoint transitioning to a populated bin has a finite,
    # positive interpolation; no arbitrary positive anchor floor enters.
    assert spline(0.5)[0] > 0


def sparse_spline_templates():
    # A moving bin becomes empty at eta=.535. The nuisance templates have
    # common support, but different slopes as they approach the empty bin.
    etas = np.array([.1, .39, .535, 3.])
    fractions = np.array([[.2, .8, 0], [.5, .5, 0],
                          [0, 1, 0], [.4, .6, 0]])
    nominal = fractions[:, None, :] * np.array([3., 11., 8., 12.])[None, :, None]
    down, up = nominal.copy(), nominal.copy()
    lower, upper = fractions.copy(), fractions.copy()
    lower[:2, :2] = [[.1, .9], [.3, .7]]
    upper[:2, :2] = [[.4, .6], [.7, .3]]
    down[:, 3] = 10 * lower
    up[:, 3] = 15 * upper
    templates = np.stack([nominal, down, up], axis=1)
    return etas, templates, YieldFractionSpline(etas, templates)


def test_sparse_spline_preserves_exact_support_at_float_equivalent_anchors():
    etas, templates, spline = sparse_spline_templates()
    # The independent 41-point fit grid and 61-point spline grid used in
    # production represent this same anchor one floating-point step apart.
    assert np.linspace(.1, 3, 41)[6] == np.nextafter(etas[2], -np.inf)
    queries = np.array([[etas[0], np.nextafter(etas[0], np.inf), etas[1]],
                        [np.nextafter(etas[2], -np.inf), etas[2],
                         np.nextafter(etas[2], np.inf)],
                        [np.nextafter(etas[-1], -np.inf), etas[-1], etas[-1]]])
    indices = np.argmin(np.abs(queries[..., None] - etas), axis=-1)
    expected = templates[indices]
    restored = spline(queries)
    np.testing.assert_array_equal(restored == 0, expected == 0)
    np.testing.assert_allclose(restored, expected, rtol=2e-15, atol=0)
    for eta, index in zip(queries.flat, indices.flat):
        scalar = spline(eta)
        np.testing.assert_array_equal(scalar == 0, templates[index] == 0)
        np.testing.assert_allclose(scalar, templates[index], rtol=2e-15, atol=0)
    # Genuine interior values still interpolate, and snapping is not allowed
    # to admit even a one-ULP extrapolation beyond the supported domain.
    assert np.all(spline(.53)[..., 0] > 0)
    for outside in (np.nextafter(etas[0], -np.inf), np.nextafter(etas[-1], np.inf)):
        with pytest.raises(ValueError, match="extrapolation"):
            spline(outside)


@pytest.mark.parametrize("query", [.535, np.nextafter(.535, -np.inf),
                                  np.nextafter(.535, np.inf)])
def test_sparse_spline_anchor_support_allows_nominal_and_profile_fits(query):
    _, templates, spline = sparse_spline_templates()
    nominal, down, up = spline(query)
    # Refill data from the exact anchor; do not make a self-Asimov data set
    # from interpolated templates, which could hide a support mismatch.
    truth = templates[2, 0, 1] + templates[2, 0, 3]
    model = TemplateLikelihood(nominal, down[None], up[None], np.ones(3), truth)
    assert model.nll(1, [0]) == pytest.approx(0., abs=1e-12)
    assert np.all(np.isfinite([model.nll(1, [alpha]) for alpha in (-1., .4, 1.)]))
    for systematics in (False, True):
        result = model.test_statistic(.535, systematics=systematics)
        assert result["null_fit"].success
        assert result["best_fit"].success
        assert np.isfinite(result["t"]) and result["t"] > 0
        assert result["best_fit"].mu == pytest.approx(1., abs=2e-4)


def test_histogram_rejects_unaccounted_tail_events():
    x, model = small_model()
    with pytest.raises(ValueError, match="cover every score"):
        model.binned((x + 2) / 4, [0.1, 0.5, 0.9])


def test_asimov_kl_is_stable_close_to_large_expected_counts():
    observed = np.array([1e12, 3e9, 8e6])
    expected = observed * (1 + np.array([2e-8, -3e-8, 1e-7]))
    with localcontext() as context:
        context.prec = 70
        exact = Decimal(0)
        for x, y in zip(expected, observed):
            xd, yd = Decimal.from_float(float(x)), Decimal.from_float(float(y))
            exact += xd - yd + yd * (yd / xd).ln()
    actual = poisson_asimov_nll(expected, observed)
    assert actual > 0
    np.testing.assert_allclose(actual, float(exact), rtol=2e-14)


def test_scalar_fit_checks_multiple_minimum_brackets(monkeypatch):
    _, model = small_model()
    # The initial point is near a local minimum; the other basin is lower.
    monkeypatch.setattr(model, "nll", lambda mu, alpha=None:
                        ((mu - .65) * (mu - 2.85))**2 + .1 * (mu - 2.85)**2)
    result = model.fit(systematics=False, initial_mu=.5, mu_starts=[.5], mu_bounds=(0, 4))
    assert result.success, result.message
    assert abs(result.mu - 2.85) < 1e-7
    assert result.nll < 1e-14


def test_sparse_frozen_histogram_scalar_asimov_fit_regression():
    _, base = small_model()
    scale = 1e6
    model = TemplateLikelihood(scale * base.nominal, scale * base.down, scale * base.up,
                               base.quadrature_weights, scale * base.truth_intensity)
    eta = 2.275
    s, sbi, b, _ = model.nominal
    derivative = s + (sbi - s - b) / (2 * np.sqrt(eta))
    intensity = component_coefficients(eta) @ model.nominal
    score = derivative / intensity - (derivative @ model.quadrature_weights) / (intensity @ model.quadrature_weights)
    histogram = model.binned(expit(score), np.linspace(0, 1, 101), eta=eta)
    assert np.count_nonzero(histogram.data_counts) < 100
    result = histogram.test_statistic(eta, systematics=False,
                                     mu_starts=[.05, .3, 1, 2.5, 3.8])
    assert result["best_fit"].success
    assert abs(result["best_fit"].mu - 1) < 1e-7
    assert result["best_fit"].nll < 1e-10
    assert result["t"] > 0
