"""Checks of physical marginal templates used in the notebook-1 figures."""
from dataclasses import replace

import numpy as np
import pytest

from poodemo.physics import COMPONENTS, PhysicsModel
from poodemo.plotting import coherent_marginal_yields, marginal_component_yields


def test_marginal_counts_preserve_full_yields_and_positive_stacks():
    model = PhysicsModel()
    edges = np.r_[-np.inf, np.linspace(-8, 9, 200), np.inf]
    for coordinate in range(3):
        components = marginal_component_yields(model, coordinate, edges)
        np.testing.assert_allclose(components.sum(axis=1),
                                   [model.component_yield(p) for p in COMPONENTS], rtol=2e-14)
        for mu in (0., .5, 1., 2.):
            coherent, ni = coherent_marginal_yields(model, coordinate, edges, mu)
            assert np.all(coherent >= 0) and np.all(ni >= 0)
            np.testing.assert_allclose((coherent + ni).sum(), model.total_yield(mu), rtol=2e-14)
            if mu == 0:
                np.testing.assert_allclose(coherent, components[2], rtol=1e-14)
            if mu == 1:
                np.testing.assert_allclose(coherent, components[1], rtol=1e-14)


def test_marginal_bins_match_generated_sbi_for_distinct_correlated_shapes():
    # Independent stochastic check of the overlap Gaussian's mean/covariance.
    model = replace(PhysicsModel(), mean_b=(-.4, .2, 1.1), cov_b=(
        (1.2, .2, -.1), (.2, .9, .1), (-.1, .1, 1.4)), phase_cos=-.8)
    points = model.sample_component("SBI", 160_000, 918)
    edges = np.r_[-np.inf, np.linspace(-2, 4, 7), np.inf]
    for coordinate in range(3):
        probabilities = marginal_component_yields(model, coordinate, edges)[1] / model.component_yield("SBI")
        observed = np.histogram(points[:, coordinate], edges)[0] / len(points)
        sigma = np.sqrt(probabilities * (1 - probabilities) / len(points))
        assert np.all(np.abs(observed - probabilities) < 6 * sigma + 2 / len(points))


def test_coherent_bins_stay_nonnegative_at_exact_amplitude_cancellation():
    base = PhysicsModel()
    model = replace(base, mean_b=base.mean_s, cov_b=base.cov_s, phase_cos=-1.)
    edges = np.linspace(-9, 12, 300)
    for coordinate in range(3):
        coherent, _ = coherent_marginal_yields(
            model, coordinate, edges, model.lambda_b / model.lambda_s)
        assert np.all(coherent >= 0)
        assert coherent.sum() < 1e-7 * model.component_yield("B")


@pytest.mark.parametrize("mu", [-1., np.nan, np.inf])
def test_coherent_bins_reject_unphysical_mu(mu):
    with pytest.raises(ValueError):
        coherent_marginal_yields(PhysicsModel(), 0, [-1, 0, 1], mu)
