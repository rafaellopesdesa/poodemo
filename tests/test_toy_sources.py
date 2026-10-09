"""Check toy laws, not just sampler implementation details."""
from dataclasses import replace
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr

from poodemo.data import Run, file_digest
from poodemo.physics import PhysicsModel
from poodemo.toy_sources import (
    QuadraturePoissonSource, load_frozen_selector, sample_simulator_poisson,
)


def _small_model(phase=-.6):
    return PhysicsModel(lambda_s=5., lambda_b=15., lambda_ni=35.,
                        exposure=1., phase_cos=phase)


def _interference_gaussian(model):
    ps, pb = np.linalg.inv(model.cov_s), np.linalg.inv(model.cov_b)
    covariance = np.linalg.inv(.5*(ps+pb))
    mean = covariance @ (.5*(ps @ model.mean("S") + pb @ model.mean("B")))
    return mean, covariance


@pytest.mark.parametrize("phase,mu", [(-.7, 0.), (-.7, .3), (-.7, 1.4), (.8, 1.4)])
def test_exact_simulator_count_and_coordinate_moments(phase, mu):
    model, rng, repetitions = _small_model(phase), np.random.default_rng(627), 1200
    toys = [sample_simulator_poisson(model, mu, rng, chunk_size=31) for _ in range(repetitions)]
    counts = np.array([len(x) for x in toys])
    rate = model.total_yield(mu)
    assert abs(counts.mean()-rate) < 5*np.sqrt(rate/repetitions)
    assert abs(counts.var(ddof=1)-rate) < 6*rate*np.sqrt(2/(repetitions-1))
    # E(sum of coordinates) and Var(sum of coordinates) are known integrals
    # for a Poisson point process, including the signed interference measure.
    int_mean, int_cov = _interference_gaussian(model)
    int_yield = (2*phase*np.sqrt(mu*model.component_yield("S")*model.component_yield("B"))
                 * model.gaussian_overlap())
    mean = int_yield*int_mean
    variance = int_yield*(np.diag(int_cov)+int_mean**2)
    for process, coefficient in (("S", mu), ("B", 1.), ("NI", 1.)):
        component_mean = model.mean(process)
        component_cov = np.asarray(getattr(model, f"cov_{process.lower()}"))
        coefficient *= model.component_yield(process)
        mean += coefficient*component_mean
        variance += coefficient*(np.diag(component_cov)+component_mean**2)
    observed = np.array([x.sum(axis=0) for x in toys]).mean(axis=0)
    assert np.all(np.abs(observed-mean) < 6*np.sqrt(variance/repetitions))


class HalfSpaceSelector:
    def predict_proba(self, x):
        accepted = (x[:, 0] >= .9).astype(float)
        return np.column_stack((accepted, 1.-accepted))


def test_frozen_selection_and_additional_exposure_preserve_poisson_law():
    model, rng, mu, exposure = _small_model(), np.random.default_rng(639), .2, 2.3
    count = []
    for _ in range(1200):
        toy = sample_simulator_poisson(model, mu, rng, exposure=exposure,
                                        selector=HalfSpaceSelector(), threshold=.5)
        assert np.all(toy[:, 0] >= .9)
        count.append(len(toy))
    rate = 0.
    for process, coefficient in (("S", mu), ("B", 1.), ("NI", 1.)):
        sigma = np.sqrt(getattr(model, f"cov_{process.lower()}")[0][0])
        rate += coefficient*model.component_yield(process)*ndtr((model.mean(process)[0]-.9)/sigma)
    mean, covariance = _interference_gaussian(model)
    rate += (2*model.phase_cos*np.sqrt(mu*model.component_yield("S")*model.component_yield("B"))
             * model.gaussian_overlap()*ndtr((mean[0]-.9)/np.sqrt(covariance[0, 0])))
    rate *= exposure
    assert abs(np.mean(count)-rate) < 5*np.sqrt(rate/len(count))
    assert abs(np.var(count, ddof=1)-rate) < 6*rate*np.sqrt(2/(len(count)-1))


def test_zero_event_experiment_and_invalid_parameters():
    model = replace(_small_model(), lambda_b=0., lambda_ni=0.)
    assert sample_simulator_poisson(model, 0., 728).shape == (0, 3)
    with pytest.raises(ValueError, match="nonnegative"):
        sample_simulator_poisson(model, -.1)
    with pytest.raises(ValueError, match="positive"):
        sample_simulator_poisson(model, 1., exposure=0.)
    with pytest.raises(ValueError, match="chunk_size"):
        sample_simulator_poisson(model, 1., chunk_size=0)
    with pytest.raises(ValueError, match="threshold"):
        sample_simulator_poisson(model, 1., selector=HalfSpaceSelector())


def test_bank_toys_have_independent_poisson_bin_counts_and_report_finite_support():
    fields = np.array([[2., 4., 1.], [8., 11., 6.], [4., 5., 3.], [1., 3., 2.]])
    weights = np.array([.2, .4, .1])
    source = QuadraturePoissonSource(dict(weights=weights), fields, 1., exposure=1.7)
    means = 1.7*weights*(fields[1]+fields[3])
    rng = np.random.default_rng(761)
    draws = np.array([np.bincount(source.draw_indices(rng), minlength=3) for _ in range(6000)])
    assert source.expected_count == pytest.approx(means.sum())
    np.testing.assert_allclose(draws.mean(axis=0), means, atol=.10)
    np.testing.assert_allclose(np.cov(draws.T), np.diag(means), atol=.40)
    assert source.diagnostics["continuous_generator"] is False
    assert source.diagnostics["effective_bank_size"] == pytest.approx(means.sum()**2/np.sum(means**2))
    assert source.diagnostics["expected_unique_nodes"] == pytest.approx(np.sum(1-np.exp(-means)))
    indices, counts = source.draw_counts(790)
    full = source.draw_indices(790)
    np.testing.assert_array_equal(np.repeat(indices, counts), np.sort(full))


def test_bank_source_rejects_invalid_learned_intensity_without_clipping():
    fields = np.array([[1000., 1.], [1., 1.], [1., 1.], [1., 1.]])
    with pytest.raises(ValueError, match="physical intensity"):
        QuadraturePoissonSource(dict(weights=np.ones(2)), fields, .25)
    with pytest.raises(ValueError, match="nonnegative"):
        QuadraturePoissonSource(dict(weights=np.array([-1., 1.])), fields, 1.)
    zero = QuadraturePoissonSource(dict(weights=np.ones(2)), np.zeros((4, 2)), 0.)
    assert len(zero.draw_indices(98)) == 0
    assert zero.expected_count == 0.


def test_selector_fingerprint_checked_before_loading(tmp_path, monkeypatch):
    run = Run(tmp_path, {}, _small_model())
    model = run.path("models", "preselection", "model.pt")
    model.parent.mkdir(parents=True)
    model.write_bytes(b"original checkpoint")
    selection = run.path("selected", "selection_state.json")
    selection.parent.mkdir()
    threshold = .123
    fingerprint = hashlib.sha256((file_digest(model)+repr(threshold)).encode()).hexdigest()
    selection.write_text(json.dumps(dict(threshold=threshold, selection_fingerprint=fingerprint)))
    sentinel = object()
    monkeypatch.setitem(sys.modules, "poodemo.training", SimpleNamespace(load_predictor=lambda path: sentinel))
    selector, saved_threshold, metadata = load_frozen_selector(run)
    assert selector is sentinel and saved_threshold == threshold
    assert metadata["selector_sha256"] == file_digest(model)
    model.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="changed"):
        load_frozen_selector(run)
