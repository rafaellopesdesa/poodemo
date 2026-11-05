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


def test_narrow_scores_produce_fifteen_populated_calibration_points():
    scores = np.linspace(.495, .505, 15000)
    table = calibration_table(scores, np.repeat(scores, 2))
    assert len(table) == 15 and table.calibration_valid.all()
    assert table.requested_bins.eq(15).all()
    assert table.numerator_count.sum() == len(scores)
    assert table.denominator_count.sum() == 2*len(scores)
    np.testing.assert_allclose(table.balanced_bin_probability, 1/15, atol=1/15000)
    np.testing.assert_allclose(table.empirical, .5)
    np.testing.assert_allclose(table.residual, .5-table.predicted)
    assert table.score_min.iloc[0] == .495 and table.score_max.iloc[0] == .505


def test_quantiles_weight_classes_equally_despite_different_sample_sizes():
    from poodemo.ratio_validation import calibration_bin_edges
    num = np.linspace(.45, .5, 2100)
    den = np.linspace(.5, .55, 21000)
    edges = calibration_bin_edges(num, den)
    # Half of the balanced mixture is in each side of .5, not 1/11 vs 10/11.
    assert np.count_nonzero(edges[1:-1] < .5) == 7
    assert np.count_nonzero(edges[1:-1] > .5) == 7
    table = calibration_table(num, den)
    assert table.numerator_count.sum() == len(num)
    assert table.denominator_count.sum() == len(den)
    np.testing.assert_allclose(table.balanced_bin_probability, 1/15, atol=1/2100)
    assert not table.calibration_valid.all()  # No fabricated precision in disjoint support.


def test_tied_scores_reduce_the_bin_count_without_splitting_identical_predictions():
    table = calibration_table(np.full(100, .5), np.full(300, .5))
    assert len(table) == 1 and table.calibration_valid.all()
    assert table.requested_bins.iloc[0] == 15
    assert table.predicted.iloc[0] == .5 and table.residual.iloc[0] == 0.
    table = calibration_table(np.repeat([.49, .51], 200), np.repeat([.49, .51], 300))
    assert len(table) == 2 and table.calibration_valid.all()
    assert table.numerator_count.sum() == 400


def test_extreme_scores_are_included_in_quantile_bins():
    scores = np.r_[0., np.linspace(.499, .501, 10000), 1.]
    table = calibration_table(scores, scores)
    assert len(table) == 15
    assert table.numerator_count.sum() == len(scores)
    assert table.denominator_count.sum() == len(scores)
    assert table.score_min.iloc[0] == 0. and table.score_max.iloc[0] == 1.


def test_calibration_axes_zoom_about_half_and_keep_requested_residual_range():
    import matplotlib.pyplot as plt
    from poodemo.ratio_validation import plot_calibration
    scores = np.linspace(.49, .51, 15000)
    table = calibration_table(scores, scores)
    table['stage'] = 'normalized'
    fig = plot_calibration(table, 'NI_up', 'NI')
    try:
        ax, residual = fig.axes
        low, high = ax.get_xlim()
        assert low + high == pytest.approx(1.)
        assert .45 < low <= .49 < .51 <= high < .55
        np.testing.assert_allclose(ax.get_ylim(), (low, high))
        np.testing.assert_allclose(residual.get_ylim(), (-.01, .01))
        assert '15/15 points' in ax.get_legend().get_texts()[0].get_text()
        line = ax.containers[0].lines[0]
        assert len(line.get_xdata()) == 15
    finally:
        plt.close(fig)


def test_calibration_zoom_does_not_hide_bad_calibration():
    import matplotlib.pyplot as plt
    from poodemo.ratio_validation import plot_calibration
    table = calibration_table(np.full(1000, .75), np.full(1000, .75))
    table['stage'] = 'raw'
    fig = plot_calibration(table, 'B', 'S')
    try:
        ax = fig.axes[0]
        assert ax.get_xlim()[0] < .5 and ax.get_xlim()[1] > .75
        assert ax.get_ylim()[0] < .5 and ax.get_ylim()[1] > .75
        assert '1/15 points' in ax.get_legend().get_texts()[0].get_text()
    finally:
        plt.close(fig)


@pytest.mark.parametrize('bins', [True, 1, 15.5])
def test_invalid_adaptive_calibration_bin_count(bins):
    with pytest.raises(ValueError, match='score_bins'):
        calibration_table(np.linspace(.4, .6, 100), np.linspace(.4, .6, 100), bins=bins)


def test_task_uses_fifteen_points_for_both_stages_without_changing_ratios(tmp_path, monkeypatch):
    from poodemo import ratio_validation

    class Predictor:
        def predict_raw_log_ratio(self, x):
            return x[:, 0]+.004

        def predict_log_ratio(self, x):
            return x[:, 0]

    monkeypatch.setattr(ratio_validation, 'plot_reweighting', lambda *args, **kwargs: None)
    monkeypatch.setattr(ratio_validation, 'plot_calibration', lambda *args, **kwargs: None)
    sample = np.repeat(np.linspace(-.04, .04, 15000)[:, None], 3, axis=1)
    result = ratio_validation.plot_ratio_task(
        'NI_up', 'NI', {'NI_up': sample, 'NI': sample}, Predictor(), tmp_path)
    cal = result['calibration']
    assert cal.groupby('stage').calibration_valid.sum().to_dict() == {'raw': 15, 'normalized': 15}
    for stage, offset in [('raw', .004), ('normalized', 0.)]:
        rw = result['reweighting'].query('stage == @stage')
        np.testing.assert_allclose(rw.denominator_mean_ratio, np.exp(sample[:, 0]+offset).mean())
