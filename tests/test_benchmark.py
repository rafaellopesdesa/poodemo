"""Analytic benchmark checks, independent of neural-network training quality.

The small common quadrature uses the ideal balanced S/B/NI classifier for
selection. Production still calibrates its trained selector on held-out data.
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from poodemo.inference import TemplateLikelihood, component_coefficients
from poodemo.physics import PhysicsModel


@pytest.fixture(scope="module")
def oracle_selected_benchmark():
    physics = PhysicsModel()
    rng = np.random.default_rng(20261006)
    x = np.concatenate([physics.sample_component(process, 8192, rng)
                        for process in ("S", "B", "NI")])
    pdfs = np.stack([physics.component_pdf(x, process)
                     for process in ("S", "B", "NI")])
    weights = 1 / (len(x) * pdfs.mean(axis=0))
    nominal = physics.component_densities(x).T
    classifier = pdfs[0] / pdfs.sum(axis=0)

    # Retain the largest weighted acceptance meeting the physical S/NI=.1
    # target. This deterministic oracle is a model-design regression only.
    order = np.argsort(classifier)[::-1]
    signal = np.cumsum(weights[order] * nominal[0, order])
    ni = np.cumsum(weights[order] * nominal[3, order])
    accepted = np.flatnonzero(signal / ni >= .1)
    assert len(accepted)
    threshold = classifier[order[accepted[-1]]]
    keep = classifier >= threshold
    x, weights, nominal = x[keep], weights[keep], nominal[:, keep]
    down = physics.component_densities(x, alpha_ni=-1).T[None]
    up = physics.component_densities(x, alpha_ni=1).T[None]
    truth = nominal[1] + nominal[3]
    return TemplateLikelihood(nominal, down, up, weights, truth)


def test_unbinned_interference_retains_a_lifted_secondary_minimum(oracle_selected_benchmark):
    model = oracle_selected_benchmark
    assert model.nll(1, [0]) == pytest.approx(0, abs=1e-12)
    fixed = lambda mu: model.nll(mu, [0])
    secondary = minimize_scalar(fixed, bounds=(.03, .25), method="bounded")
    barrier = minimize_scalar(lambda mu: -fixed(mu), bounds=(.25, .6), method="bounded")
    assert secondary.success and barrier.success
    assert .08 < secondary.x < .17
    assert 3 < secondary.fun < 6
    assert .3 < barrier.x < .45
    assert -barrier.fun > secondary.fun + 2
    assert fixed(.02) > secondary.fun + 5
    assert fixed(1.3) > secondary.fun + 1

    # Equal total rates do not make the full point-process model degenerate:
    # distinct Gaussian shapes leave a strictly positive likelihood penalty.
    rates = model.nominal @ model.quadrature_weights
    interference = rates[1] - rates[0] - rates[2]
    partner = (-interference / rates[0] - 1)**2
    np.testing.assert_allclose(component_coefficients(partner) @ rates,
                               component_coefficients(1) @ rates, rtol=2e-14)
    assert model.nll(partner, [0]) > .05


def test_ni_profiling_visibly_broadens_the_benchmark(oracle_selected_benchmark):
    model = oracle_selected_benchmark
    # Profiling cannot increase the statistic. Its size is not constant along
    # the scan: the nearby secondary basin shifts more than the barrier does.
    for mu in (1.05, .38, .12):
        fixed = model.nll(mu, [0])
        fitted = model.fit(mu_fixed=mu)
        assert fitted.success, fitted.message
        assert 0 <= fitted.nll <= fixed + 1e-9
        assert abs(fitted.alpha[0]) < 1

    def profile(mu):
        fitted = model.fit(mu_fixed=mu)
        assert fitted.success, fitted.message
        return fitted.nll

    secondary = minimize_scalar(profile, bounds=(.01, .2), method="bounded")
    fixed_secondary = minimize_scalar(lambda mu: model.nll(mu, [0]),
                                      bounds=(.03, .25), method="bounded")
    assert secondary.success
    assert .04 < secondary.x < .11
    assert 1 < secondary.fun < 4
    assert secondary.x < fixed_secondary.x
    assert secondary.fun < fixed_secondary.fun - .75
    assert profile(.36) > secondary.fun + 3

    # The local curvature also broadens near the generating hypothesis,
    # independently of the movement of the distant second minimum.
    step = .005
    fixed_curvature = (model.nll(1 - step, [0]) + model.nll(1 + step, [0])) / step**2
    profile_curvature = (profile(1 - step) + profile(1 + step)) / step**2
    width_ratio = np.sqrt(fixed_curvature / profile_curvature)
    assert 1.03 < width_ratio < 1.25

    nominal_ni = model.rates[3]
    assert model.rates_down[0, 3] < .8 * nominal_ni
    assert model.rates_up[0, 3] > 1.2 * nominal_ni


def test_signal_is_distinct_while_coherent_sbi_stays_close_to_background():
    physics = PhysicsModel()
    mean_delta = physics.mean("S") - physics.mean("B")
    mahalanobis = np.sqrt(mean_delta @ np.linalg.solve(physics.cov_s, mean_delta))
    assert .3 < mahalanobis < .6
    width_ratio = np.sqrt(np.diag(physics.cov_b) / np.diag(physics.cov_s))
    assert np.all((width_ratio > 1.08) & (width_ratio < 1.25))

    # Full-space normalized shapes, so this distinction cannot be caused
    # solely by different yields or by an unusually selective cut.
    rng = np.random.default_rng(8114)
    x = np.concatenate([physics.sample_component(process, 8192, rng)
                        for process in ("S", "B")])
    signal, background, sbi = [physics.component_pdf(x, process)
                               for process in ("S", "B", "SBI")]
    proposal = (signal + background) / 2
    tv_signal_background = np.mean(np.abs(signal - background) / proposal) / 2
    tv_sbi_background = np.mean(np.abs(sbi - background) / proposal) / 2
    assert .12 < tv_signal_background < .23
    assert tv_sbi_background < .02
    assert tv_signal_background > 8 * tv_sbi_background
