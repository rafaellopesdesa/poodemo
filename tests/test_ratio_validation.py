"""Checks that independent density diagnostics do not manufacture closure."""
import numpy as np
import pytest

from poodemo.ratio_validation import calibration_table, reweighting_table


def _two_bin_sample(low_count, high_count):
    return np.repeat(np.array([[.2, .2, .2], [1.2, 1.2, 1.2]]),
                     [low_count, high_count], axis=0)


def test_reweighting_recovers_target_without_hiding_global_or_tail_mass():
    target = _two_bin_sample(400, 600)
    reference = _two_bin_sample(500, 500)
    weights = np.repeat([.8, 1.2], [500, 500])
    full = reweighting_table(target, reference, weights, [0, 1, 2])
    np.testing.assert_allclose(full.reweighted_over_target, 1.)
    biased = reweighting_table(target, reference, 1.3 * weights, [0, 1, 2])
    np.testing.assert_allclose(biased.reweighted_over_target, 1.3)
    np.testing.assert_allclose(biased.denominator_mean_ratio, 1.3)
    cropped = reweighting_table(target, reference, 1.3 * weights, [0, 1])
    np.testing.assert_allclose(cropped.numerator_plot_mass, .4)
    np.testing.assert_allclose(cropped.reweighted_plot_mass, 1.3 * .4)
    np.testing.assert_allclose(cropped.reweighted_over_target, 1.3)


def test_histogram_errors_match_independent_fixed_sample_moments():
    target = _two_bin_sample(10, 30)
    reference = _two_bin_sample(20, 30)
    weights = np.linspace(.2, 2., len(reference))
    row = reweighting_table(target, reference, weights, [0, 1, 2]).iloc[0]
    weighted_indicator = weights * (reference[:, 0] < 1)
    target_indicator = (target[:, 0] < 1).astype(float)
    variance_target = target_indicator.var(ddof=1) / len(target)
    variance_weighted = weighted_indicator.var(ddof=1) / len(reference)
    np.testing.assert_allclose(row.reweighted_se ** 2, variance_weighted)
    np.testing.assert_allclose(row.numerator_se ** 2, variance_target)
    expected_ratio_variance = (variance_weighted / target_indicator.mean() ** 2
                              + weighted_indicator.mean() ** 2 * variance_target
                              / target_indicator.mean() ** 4)
    np.testing.assert_allclose(row.ratio_se ** 2, expected_ratio_variance)


def test_sparse_ratio_bins_are_unavailable_not_zero_closure():
    table = reweighting_table(_two_bin_sample(50, 0), _two_bin_sample(20, 30),
                             np.ones(50), [0, 1, 2])
    sparse = table.loc[table.bin_low == 1]
    assert not sparse.ratio_valid.any()
    assert sparse.reweighted_over_target.isna().all()
    assert sparse.ratio_se.isna().all()
    assert sparse.status.str.startswith("unavailable").all()


def test_calibration_preserves_balanced_prior_with_unequal_class_sizes():
    # p_num=(.4,.6), p_den=(.5,.5), hence balanced probabilities r/(1+r).
    probabilities = [.8 / 1.8, 1.2 / 2.2]
    target_scores = np.repeat(probabilities, [800, 1200])
    reference_scores = np.repeat(probabilities, [500, 500])
    table = calibration_table(target_scores, reference_scores, [0., .5, 1.])
    np.testing.assert_allclose(table.empirical, probabilities)
    np.testing.assert_allclose(table.predicted, probabilities)
    np.testing.assert_allclose(table.residual, 0., atol=1e-14)
    np.testing.assert_allclose(table.balanced_bin_probability.sum(), 1.)


def test_calibration_residual_variance_includes_predicted_score_covariance():
    rng = np.random.default_rng(67)
    target = rng.beta(3, 2, 1300)
    reference = rng.beta(2, 3, 1700)
    edges = np.linspace(0, 1, 5)
    table = calibration_table(target, reference, edges)
    for row in table.itertuples():
        mask_target = (target >= row.bin_low) & (target <= row.bin_high)
        mask_reference = (reference >= row.bin_low) & (reference <= row.bin_high)
        zn = mask_target * (1 - target - row.residual)
        zd = mask_reference * (-reference - row.residual)
        total = mask_target.mean() + mask_reference.mean()
        variance = (zn.var(ddof=1) / len(zn) + zd.var(ddof=1) / len(zd)) / total ** 2
        np.testing.assert_allclose(row.residual_se ** 2, variance, rtol=2e-13)


def test_calibration_bad_scores_remain_bad_and_empty_bins_unavailable():
    # An uninformative balanced classifier must output .5, not .75.
    table = calibration_table(np.full(100, .75), np.full(500, .75), [0., .5, 1.])
    assert not table.iloc[0].calibration_valid
    assert np.isnan(table.iloc[0].residual_se)
    assert table.iloc[1].residual == pytest.approx(-.25)
    assert table.iloc[1].predicted == pytest.approx(.75)


@pytest.mark.parametrize("weights", [[1, -1], [1, np.inf], [1]])
def test_invalid_ratio_weights_raise(weights):
    x = _two_bin_sample(1, 1)
    with pytest.raises(ValueError, match="Ratios"):
        reweighting_table(x, x, weights, [0, 1, 2])
