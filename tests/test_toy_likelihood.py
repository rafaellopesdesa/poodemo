"""Stat-only toy likelihoods use event data and independent rate integrals."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from poodemo.toy_likelihood import StatOnlyLikelihood


def count_model(count, *, exposure=1.0):
    # lambda(mu) = 3 mu + 7, with no interference.
    return StatOnlyLikelihood.from_binned(np.array([[3.], [8.], [5.], [2.]]),
                                          np.array([count]), exposure=exposure)


def poisson_q(expected, observed, best):
    return 2 * (expected - best + (observed * np.log(best / expected) if observed else 0))


@pytest.mark.parametrize("count,mu_hat", [(0, 0), (5, 0), (10, 1), (20, 2)])
def test_poisson_count_fits_including_both_physical_boundaries(count, mu_hat):
    model = count_model(count)
    result = model.test_statistic(1.4)
    assert result["valid"], result
    assert result["mu_hat"] == pytest.approx(mu_hat, abs=2e-7)
    assert result["q"] == pytest.approx(poisson_q(11.2, count, 3 * mu_hat + 7), abs=2e-10)
    assert result["at_lower"] == (mu_hat == 0)
    assert result["at_upper"] == (mu_hat == 2)
    result_zero = model.test_statistic(0)
    assert result_zero["valid"]
    assert result_zero["q"] == pytest.approx(poisson_q(7, count, 3 * mu_hat + 7), abs=2e-10)


def test_exposure_scales_model_not_observations():
    model = count_model(20, exposure=2)
    result = model.test_statistic(1.4)
    assert model.expected_count(1.4) == pytest.approx(22.4)
    assert result["mu_hat"] == pytest.approx(1, abs=2e-7)
    assert result["q"] == pytest.approx(poisson_q(22.4, 20, 20), abs=2e-10)


def test_interference_two_minima_and_dense_grid_agreement():
    # One bin has two equally good solutions kappa=0.5 and 1.5. The
    # second bin breaks the degeneracy, requiring both minima to be searched.
    fields = np.array([[40., 1.], [0., 1.9], [40., 1.], [10., .1]])
    model = StatOnlyLikelihood.from_binned(fields, [20, 3])
    result = model.test_statistic(0, mu_bounds=(0, 4))
    grid = np.linspace(0, 2, 4001)
    values = np.array([model.nll(k * k, reference_mu=0) for k in grid])
    local = np.flatnonzero((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])) + 1
    assert len(local) == 2
    candidates = [minimize_scalar(lambda k: model.nll(k * k, reference_mu=0),
                                 bounds=(grid[i - 1], grid[i + 1]), method="bounded",
                                 options={"xatol": 1e-12}) for i in local]
    best = min(candidates, key=lambda fit: fit.fun)
    assert result["valid"], result
    assert result["mu_hat"] == pytest.approx(best.x**2, abs=2e-7)
    assert result["q"] == pytest.approx(-best.fun, abs=1e-10)
    assert result["mu_hat"] > 1


def test_empty_unbinned_toy_minimizes_rate_not_event_shape():
    # nu(kappa)=4 kappa^2-8 kappa+5 has its minimum at kappa=1.
    model = StatOnlyLikelihood(np.empty((4, 0)), [4, 0, 4, 1])
    result = model.test_statistic(0)
    assert result["valid"]
    assert result["mu_hat"] == pytest.approx(1, abs=1e-8)
    assert result["q"] == pytest.approx(8, abs=1e-12)


def test_weighted_unique_bank_points_equal_repeated_events():
    fields = np.array([[.7, .1, .2], [.9, .2, .4], [.3, .2, .5], [.2, .4, .1]])
    rates = np.array([5., 8., 7., 2.])
    counts = np.array([2, 0, 5])
    repeated = StatOnlyLikelihood(np.repeat(fields, counts, axis=1), rates)
    weighted = StatOnlyLikelihood(fields, rates, counts)
    for mu in (0, .2, 1, 1.4, 2):
        assert repeated.nll(mu, reference_mu=1) == pytest.approx(weighted.nll(mu, reference_mu=1), abs=1e-13)
    a, b = repeated.test_statistic(1.4), weighted.test_statistic(1.4)
    assert a["q"] == pytest.approx(b["q"], abs=1e-12)
    assert a["mu_hat"] == pytest.approx(b["mu_hat"], abs=2e-7)


def test_binned_zero_means_obey_zero_log_zero():
    fields = np.array([[3., 0.], [8., 0.], [5., 0.], [2., 0.]])
    empty = StatOnlyLikelihood.from_binned(fields, [10, 0])
    ordinary = count_model(10)
    assert empty.test_statistic(1.4)["q"] == pytest.approx(ordinary.test_statistic(1.4)["q"])
    impossible = StatOnlyLikelihood.from_binned(fields, [10, 1])
    result = impossible.test_statistic(1.4)
    assert not result["valid"]
    assert np.isnan(result["q"])
    assert np.isinf(impossible.nll(1.4))
    # A pure signal bin is allowed to be empty at mu=0.
    pure_signal = StatOnlyLikelihood.from_binned(np.array([[1.], [1.], [0.], [0.]]), [0])
    assert pure_signal.test_statistic(0)["q"] == 0
    assert pure_signal.test_statistic(0)["at_lower"]


def test_invalid_negative_predictions_rejected_even_in_empty_bins():
    # lambda=kappa^2-3 kappa+2 is negative for 1<kappa<2.
    fields = np.array([[1.], [0.], [2.], [0.]])
    model = StatOnlyLikelihood.from_binned(fields, [0])
    assert np.isinf(model.nll(2.25))
    result = model.test_statistic(2.25, mu_bounds=(0, 9))
    assert not result["valid"] and np.isnan(result["q"])
    fit = model.fit(mu_bounds=(0, 9))
    assert fit["valid"]
    assert fit["mu_hat"] == pytest.approx(1)


def test_cached_independent_bank_positivity_includes_unobserved_locations():
    # Actual observed fields are benign, but the independent bank forbids
    # kappa between one and two. The optimum must honor this additional test.
    bank = np.array([[1.], [0.], [2.], [0.]])
    guard = StatOnlyLikelihood(np.empty((4, 0)), [3., 8., 5., 2.], validity_fields=bank)
    direct = StatOnlyLikelihood(np.array([[3.], [8.], [5.], [2.]]),
                               [3., 8., 5., 2.], [15], validity_fields=bank)
    cached = StatOnlyLikelihood(np.array([[3.], [8.], [5.], [2.]]),
                               [3., 8., 5., 2.], [15], validity_intervals=guard.negative_intervals)
    np.testing.assert_array_equal(direct.negative_intervals, cached.negative_intervals)
    assert np.isinf(cached.nll(2.25))
    assert cached.fit(mu_bounds=(0, 9))["mu_hat"] == pytest.approx(4)


def test_stable_small_likelihood_differences_at_large_event_counts():
    # Subtracting absolute NLLs of order 1e15 would erase a 1e-6 statistic.
    count = 1e13
    fields = np.array([[count / 2], [count], [count / 2], [0.]])
    model = StatOnlyLikelihood.from_binned(fields, [count])
    displacement = 2e-9
    actual = model.nll(1 + displacement, reference_mu=1)
    expected = count * (displacement / 2)**2
    assert actual > 0
    assert actual == pytest.approx(expected, rel=2e-5)


@pytest.mark.parametrize("kwargs", [
    {"fields": np.ones((3, 4)), "rates": np.ones(4)},
    {"fields": np.ones((4, 4)), "rates": [-1, 1, 1, 1]},
    {"fields": np.ones((4, 4)), "rates": np.ones(4), "observation_weights": [1, 2]},
    {"fields": np.ones((4, 4)), "rates": np.ones(4), "exposure": 0},
    {"fields": np.ones((4, 4)), "rates": np.ones(4), "validity_intervals": [[2, 1]]},
])
def test_bad_input_validation(kwargs):
    with pytest.raises(ValueError):
        StatOnlyLikelihood(**kwargs)
