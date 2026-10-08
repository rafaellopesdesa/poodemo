"""Physical yield checks for plots made after a frozen preselection."""
from dataclasses import replace

import numpy as np

from poodemo.physics import PhysicsModel
from poodemo.plotting import (
    coherent_marginal_yields,
    selected_marginal_component_yields,
)


def _selected_quadrature(model):
    """A small independent common quadrature with a nontrivial fixed cut."""
    rng = np.random.default_rng(7251)
    points = rng.uniform(-4., 5., size=(12_000, 3))
    accepted = (points[:, 0] + .3 * points[:, 1] > .5) & (points[:, 2] > -1.5)
    selected = points[accepted]
    return {
        "x": selected,
        "weights": np.full(len(selected), 9.**3 / len(points)),
        "nominal": model.component_densities(selected).T,
    }


def test_selected_marginals_preserve_accepted_yields_and_visible_tail_losses():
    model = PhysicsModel()
    quadrature = _selected_quadrature(model)
    rates = quadrature["nominal"] @ quadrature["weights"]
    full_edges = np.r_[-np.inf, np.linspace(-4., 5., 30), np.inf]
    finite_edges = np.linspace(.8, 1.4, 7)
    for coordinate in range(3):
        full = selected_marginal_component_yields(quadrature, coordinate, full_edges)
        np.testing.assert_allclose(full.sum(axis=1), rates, rtol=2e-13)

        visible = selected_marginal_component_yields(quadrature, coordinate, finite_edges)
        x = quadrature["x"][:, coordinate]
        in_range = (x >= finite_edges[0]) & (x <= finite_edges[-1])
        expected_visible = quadrature["nominal"][:, in_range] @ quadrature["weights"][in_range]
        np.testing.assert_allclose(visible.sum(axis=1), expected_visible, rtol=2e-13)
        # The selected shape is normalized over all accepted events. A cropped
        # view must not redistribute the omitted probability into visible bins.
        probability_visible = visible.sum(axis=1) / rates
        assert np.all((probability_visible > 0.) & (probability_visible < .9))


def test_selected_coherent_anchors_are_b_and_sbi_with_unchanged_ni():
    model = PhysicsModel()
    quadrature = _selected_quadrature(model)
    edges = np.r_[-np.inf, np.linspace(-3., 4., 24), np.inf]
    for coordinate in range(3):
        components = selected_marginal_component_yields(quadrature, coordinate, edges)
        for mu, index in ((0., 2), (1., 1)):
            coherent, ni = coherent_marginal_yields(
                model, coordinate, edges, mu, quadrature=quadrature)
            np.testing.assert_allclose(coherent, components[index], rtol=2e-13)
            np.testing.assert_array_equal(ni, components[3])
        for mu in (.5, 2.):
            _, ni = coherent_marginal_yields(
                model, coordinate, edges, mu, quadrature=quadrature)
            np.testing.assert_array_equal(ni, components[3])


def test_selected_stacks_match_positive_amplitudes_under_destructive_interference():
    model = replace(PhysicsModel(), phase_cos=-.99)
    quadrature = _selected_quadrature(model)
    edges = np.r_[-np.inf, np.linspace(-3., 4., 24), np.inf]
    # The direct complex-amplitude expression checks the signed S/SBI/B basis
    # independently, including mu values for which some basis coefficients are negative.
    for mu in (0., .5, 1., 2.):
        amplitude = (np.sqrt(mu) * model.amplitude(quadrature["x"], "S")
                     + model.amplitude(quadrature["x"], "B"))
        event_weights = quadrature["weights"] * np.abs(amplitude)**2
        for coordinate in range(3):
            coherent, ni = coherent_marginal_yields(
                model, coordinate, edges, mu, quadrature=quadrature)
            expected = np.histogram(quadrature["x"][:, coordinate], edges,
                                    weights=event_weights)[0]
            np.testing.assert_allclose(coherent, expected, rtol=1e-10, atol=2e-11)
            assert np.all(coherent >= 0.) and np.all(ni >= 0.)


def test_selected_coherent_stack_remains_nonnegative_at_amplitude_cancellation():
    base = PhysicsModel()
    model = replace(base, mean_b=base.mean_s, cov_b=base.cov_s, phase_cos=-1.)
    quadrature = _selected_quadrature(model)
    edges = np.r_[-np.inf, np.linspace(-3., 4., 24), np.inf]
    selected_b_yield = quadrature["nominal"][2] @ quadrature["weights"]
    for coordinate in range(3):
        coherent, _ = coherent_marginal_yields(
            model, coordinate, edges, model.lambda_b / model.lambda_s,
            quadrature=quadrature)
        assert np.all(coherent >= 0.)
        assert coherent.sum() < 1e-10 * selected_b_yield
