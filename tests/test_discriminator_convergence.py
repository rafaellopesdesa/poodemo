"""Nested event partitions and finite-MC contrast diagnostics in notebook 10."""
from types import SimpleNamespace

import numpy as np
import pytest

from poodemo import discriminator_convergence as module
from poodemo.pipeline import observable_bin_edges
from poodemo.inference import component_coefficients, poisson_asimov_nll
from poodemo.colab_discriminators import source_moments


def test_nested_36_18_9_counts_match_direct_event_histograms():
    rng = np.random.default_rng(20261114)
    fine_edges = observable_bin_edges(36, 'reference_ratio')
    points = [np.r_[rng.uniform(size=(400, 3)), np.column_stack([fine_edges]*3)] for _ in range(6)]
    counts = np.array([np.histogramdd(x, bins=[fine_edges]*3)[0] for x in points], dtype=np.int64)
    for bins in [9, 18, 36]:
        merged, edges = module.merge_cells(counts, fine_edges, bins)
        for actual, x in zip(merged, points):
            np.testing.assert_array_equal(actual, np.histogramdd(x, bins=[edges]*3)[0])
        np.testing.assert_array_equal(merged.sum(axis=(1, 2, 3)), counts.sum(axis=(1, 2, 3)))
    intermediate, middle_edges = module.merge_cells(counts, fine_edges, 18)
    coarse, edges = module.merge_cells(intermediate, middle_edges, 9)
    direct, direct_edges = module.merge_cells(counts, fine_edges, 9)
    np.testing.assert_array_equal(coarse, direct)
    np.testing.assert_array_equal(edges, direct_edges)
    assert not np.array_equal(edges, observable_bin_edges(9, 'reference_ratio'))
    with pytest.raises(ValueError, match='divide'):
        module.merge_cells(counts, fine_edges, 10)


def test_exact_merging_preserves_rates_and_reduces_fixed_ni_deviance():
    rng = np.random.default_rng(8)
    counts = rng.integers(10, 100, size=(6, 8, 8, 8))
    # Separate source rates chosen to guarantee positive physical means here.
    rates = np.array([1., 10., 8., 100., 110., 90.])
    target = counts.sum(axis=(1, 2, 3))
    fields = counts*rates[:, None, None, None]/target[:, None, None, None]
    values = []
    for n in [2, 4, 8]:
        merged, _ = module.merge_cells(fields, observable_bin_edges(8, 'reference_ratio'), n)
        np.testing.assert_allclose(merged.sum(axis=(1, 2, 3)), rates, rtol=1e-14)
        m = merged.reshape(6, -1)[:4]
        x, y = component_coefficients(.2) @ m, component_coefficients(1.) @ m
        values.append(2*poisson_asimov_nll(x, y))
    assert values[0] < values[1] < values[2]


def test_contrast_bias_matches_numerical_poisson_hessian():
    counts = np.array([[300, 700], [450, 550], [600, 400], [250, 750], [260, 740], [240, 760]])
    rates = np.array([10., 90., 80., 300., 330., 270.])
    mu, truth, target = .12, 1., 1000
    result = module.contrast_table(counts, rates, target, mu, truth)
    means, variances = source_moments(counts, rates, target)
    c, c0 = component_coefficients(mu), component_coefficients(truth)
    np.testing.assert_allclose(result.mc_sigma_delta**2, (c-c0)**2 @ variances[:4])
    np.testing.assert_allclose(result.raw_q_contribution.sum(),
        2*poisson_asimov_nll(c @ means[:4], c0 @ means[:4]), rtol=1e-12)
    for b in range(2):
        m = means[:4, b]
        objective = lambda p:2*poisson_asimov_nll(c @ p, c0 @ p)
        hessian_diagonal = []
        for j in range(4):
            shift = np.eye(4)[j]*.01
            hessian_diagonal.append((objective(m+shift)-2*objective(m)+objective(m-shift))/.01**2)
        numerical = .5*np.dot(hessian_diagonal, variances[:4, b])
        assert result.mc_bias_second_order.iloc[b] == pytest.approx(numerical, rel=2e-5)
    at_truth = module.contrast_table(counts, rates, target, 1.)
    assert (at_truth.mc_sigma_delta == 0).all()
    assert at_truth.relative_mc_sigma_delta.isna().all()
    # Large NI uncertainty cancels from the nominal-NI hypothesis contrast.
    bigger = rates.copy(); bigger[3] *= 10
    np.testing.assert_allclose(module.contrast_table(counts, bigger, target, mu).mc_sigma_delta,
                               result.mc_sigma_delta)


def test_nonpositive_and_empty_contrasts_are_not_reported_as_precise():
    counts = np.array([[3, 0, 0], [0, 0, 0], [0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    result = module.contrast_table(counts, np.ones(6), 10, .25)
    assert list(result.status) == ['nonpositive_prediction', 'no_sampled_support', 'support_boundary']
    assert result.mc_bias_second_order.isna().all()
    assert np.isnan(result.raw_q_contribution.iloc[0])
    assert np.isnan(result.relative_mc_sigma_delta.iloc[1])


def test_support_audit_distinguishes_permitted_zeroing_from_finite_likelihood():
    from poodemo.mc_yields import YieldPolicy
    from poodemo.colab_discriminators import _histogram
    # A tiny positive SBI Asimov count is overwhelmed by a negative signal
    # coefficient at the test point. Zeroing is permitted but q is infinite.
    counts = np.zeros((6, 2))
    counts[:, 1] = 1000
    counts[0] = [10, 990]
    counts[1] = [1, 999]
    samples = dict(rates=np.ones(6)*.1, target=1000)
    model = _histogram(counts, samples, 1., YieldPolicy(.01, .05), False)
    corrected, report = model.yield_policy.apply(model.raw_intensity(.25))
    assert report['zeroing_accepted'] and corrected[0] == 0
    assert np.isinf(model.nll(.25))
    audit = module.support_audit(model, .25, [])
    assert audit['support_mismatch_bins'] == 1
    assert audit['support_mismatch_asimov'] == pytest.approx(.0001)


@pytest.mark.parametrize('bins,sizes,message', [([9, 12, 36], [10], 'nested'),
    ([9, 18], [10], 'nested'), ([9, 18, 36], [5, 11], '10M'),
    ([9, 18, 36], [5], 'completed')])
def test_invalid_design_rejected_before_reading_ratios(bins, sizes, message, monkeypatch):
    study = dict(samples=dict(target=10), config=dict(n_bins=36))
    monkeypatch.setattr(module, 'fixed_counts', lambda *a:pytest.fail('Invalid design read ratios'))
    with pytest.raises(ValueError, match=message):
        module.run_convergence_study(SimpleNamespace(), study, bin_counts=bins, prefixes=sizes)
