"""Analytic benchmark checks, independent of neural-network training quality.

The small common quadrature uses the ideal balanced S/B/NI classifier for
selection. Production still calibrates its trained selector on held-out data.
"""

import numpy as np
import pytest

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
    candidates = np.linspace(5.5, 10, 31)
    values = np.array([model.nll(mu, [0]) for mu in candidates])
    index = np.argmin(values)
    assert 0 < index < len(candidates) - 1
    assert 6 < candidates[index] < 9
    assert .2 < values[index] < 5
    assert model.nll(3.5, [0]) > values[index] + 2
    assert model.nll(10, [0]) > values[index] + 2

    # Equal total rates do not make the full point-process model degenerate:
    # distinct Gaussian shapes leave a strictly positive likelihood penalty.
    rates = model.nominal @ model.quadrature_weights
    interference = rates[1] - rates[0] - rates[2]
    partner = (-interference / rates[0] - 1)**2
    np.testing.assert_allclose(component_coefficients(partner) @ rates,
                               component_coefficients(1) @ rates, rtol=2e-14)
    assert model.nll(partner, [0]) > .2


def test_ni_profiling_visibly_broadens_the_benchmark(oracle_selected_benchmark):
    model = oracle_selected_benchmark
    # Test the local shoulder, the interference barrier, and the second basin
    # using the same rate/normalized-shape exp-poly model as the notebooks.
    for mu in (1.3, 3.5, 7.2):
        fixed = model.nll(mu, [0])
        fitted = model.fit(mu_fixed=mu)
        assert fitted.success, fitted.message
        assert 0 <= fitted.nll < .85 * fixed
        assert abs(fitted.alpha[0]) < 1

    nominal_ni = model.rates[3]
    assert model.rates_down[0, 3] < .8 * nominal_ni
    assert model.rates_up[0, 3] > 1.2 * nominal_ni
