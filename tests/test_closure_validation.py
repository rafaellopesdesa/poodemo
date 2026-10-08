"""Non-tautological checks for the independent notebook03 validation."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from poodemo.closure_validation import (
    FrozenWorkspaceFields, IndependentValidationLikelihood, _PreparedMorph,
    expectation_convergence, prepare_validation_bank, scan_independent_likelihood,
)
from poodemo.data import file_digest
from poodemo.inference import TemplateLikelihood, exp_poly_factor
from poodemo.physics import PhysicsModel


def discrete_model():
    nominal = 20*np.array([[4., 1., .5], [10., 3., 1.], [7., 2., 1.], [3., 5., 2.]])
    weights = np.array([.2, .3, .5])
    down, up = nominal.copy(), nominal.copy()
    down[3] *= np.array([.7, 1.1, 1.2])
    up[3] *= np.array([1.3, .9, .8])
    return (nominal, down, up), weights


def test_raw_expectation_bias_survives_fresh_bank_and_error_is_correct():
    values = np.tile([.5, 1., 1.5], 100)
    result = expectation_convergence(values, [30, 300])
    assert result.iloc[-1].expectation == pytest.approx(1.)
    assert result.iloc[-1].standard_error == pytest.approx(values.std(ddof=1)/np.sqrt(len(values)))
    assert result.iloc[-1].effective_sample_size == pytest.approx(values.sum()**2/np.sum(values**2))
    biased = expectation_convergence(values*1.3, [300])
    assert biased.iloc[-1].expectation == pytest.approx(1.3)
    assert biased.iloc[-1].residual == pytest.approx(.3)


def test_prepared_morph_matches_existing_interpolator():
    rng = np.random.default_rng(213)
    lo, hi = np.exp(rng.normal(size=(2, 70)))
    prepared = _PreparedMorph(lo, hi)
    for alpha in (-2.5, -1., -.2, 0., .15, .7, 1., 2.5):
        np.testing.assert_allclose(prepared(alpha), exp_poly_factor(alpha, lo, hi), atol=1e-14)


def test_external_likelihood_matches_existing_model_when_integrals_are_exact():
    fields, weights = discrete_model()
    nominal, down, up = fields
    truth = nominal[1]+nominal[3]
    old = TemplateLikelihood(nominal, down[None], up[None], weights, truth)
    separate = IndependentValidationLikelihood(fields, weights, fields, weights*truth,
                                               reference_intensity=truth)
    for mu, alpha in ((.2, -.7), (.7, .6), (1., 0.), (1.4, 1.5)):
        assert separate.nll(mu, [alpha]) == pytest.approx(old.nll(mu, [alpha]), abs=1e-11)


def test_oracle_mle_closes_but_biased_density_does_not():
    fields, weights = discrete_model()
    nominal = fields[0]
    truth = nominal[1]+nominal[3]
    oracle = IndependentValidationLikelihood(fields, weights, fields, weights*truth)
    fit = oracle.fit(systematics=False, mu_bounds=(.001, 2.))
    assert fit.success and abs(fit.mu-1.) < 1e-6
    biased = nominal.copy()
    biased[1] *= np.array([1.2, .7, .5])
    biased[1] *= (nominal[1]@weights)/(biased[1]@weights)
    wrong_fields = (biased, fields[1].copy(), fields[2].copy())
    wrong_fields[1][:3] = biased[:3]
    wrong_fields[2][:3] = biased[:3]
    wrong = IndependentValidationLikelihood(wrong_fields, weights, wrong_fields, weights*truth,
                                            reference_intensity=truth)
    wrong_fit = wrong.fit(systematics=False, mu_bounds=(.001, 2.))
    assert wrong_fit.success and abs(wrong_fit.mu-1.) > .2


def test_independent_observation_mass_is_not_renormalized_to_force_stationarity():
    fields, weights = discrete_model()
    truth = fields[0][1]+fields[0][3]
    independent_weights = weights*truth*np.array([1.2, .8, 1.1])
    likelihood = IndependentValidationLikelihood(fields, weights, fields, independent_weights)
    fit = likelihood.fit(systematics=False, mu_bounds=(.001, 2.))
    assert fit.success and abs(fit.mu-1.) > .02


def test_negative_intensity_on_integration_nodes_is_invalid_even_if_data_positive():
    fields, _ = discrete_model()
    q = fields[0].copy()
    q[0, 0] = 10000.
    q_fields = (q, q.copy(), q.copy())
    likelihood = IndependentValidationLikelihood(q_fields, np.array([1e-4, .4, .6]),
                                                fields, np.ones(3))
    assert np.isinf(likelihood.nll(.1, [0.]))
    assert np.isfinite(likelihood.nll(1., [0.]))


def test_scan_refines_the_mle_and_distinguishes_joint_from_profiled_truth():
    fields, weights = discrete_model()
    truth = fields[0][1]+fields[0][3]
    likelihood = IndependentValidationLikelihood(fields, weights, fields, weights*truth)
    scans, fits = scan_independent_likelihood(likelihood, [.001, .5, 1., 1.5, 2.], profile=True)
    assert fits.valid.all()
    assert len(scans.loc[~scans.systematics]) > 5
    for syst in (False, True):
        fit = fits.loc[fits.systematics == syst].iloc[0]
        at_one = scans.loc[(scans.systematics == syst) & (scans.mu == 1.)].iloc[0]
        assert fit.q_mu1 == pytest.approx(at_one.q)
        assert fit.q_joint_at_truth >= fit.q_mu1-1e-8
        assert fit.mu_hat == pytest.approx(1., abs=1e-5)


def test_frozen_workspace_normalization_is_unchanged_by_new_evaluation_points():
    model = PhysicsModel()
    rng = np.random.default_rng(59)
    x = np.concatenate([model.sample_component(p, 1000, rng) for p in ("S", "B", "NI")])
    proposal = sum(model.component_pdf(x, p) for p in ("S", "B", "NI"))/3
    weights = 1/(len(x)*proposal)
    nominal = model.component_densities(x).T
    down = model.component_densities(x, alpha_ni=-1.).T
    up = model.component_densities(x, alpha_ni=1.).T
    quad = dict(x=x, weights=weights, nominal=nominal, down=down[None], up=up[None])
    rates = nominal@weights
    class Oracle:
        def __init__(self, numerator, denominator, alpha=0.):
            self.num, self.den, self.alpha = numerator, denominator, alpha
        def predict_ratio(self, points):
            # Any constant normalization cancels on the frozen workspace q.
            return model.component_density(points, self.num, self.alpha)/model.component_density(points, self.den)
    predictors = {f"{p}_over_S": Oracle(p, "S") for p in ("SBI", "B", "NI")}
    predictors.update({"NI_down_over_NI": Oracle("NI", "NI", -1.),
                       "NI_up_over_NI": Oracle("NI", "NI", 1.)})
    frozen = FrozenWorkspaceFields(SimpleNamespace(model=model), quad, predictors)
    for actual, expected in zip(frozen.learned_quadrature, (nominal, down, up)):
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
    before = dict(frozen.normalizers)
    outside = model.sample_component("NI", 77, rng)
    evaluated = frozen.evaluate(outside)
    assert frozen.normalizers == before
    np.testing.assert_allclose(evaluated[0], model.component_densities(outside).T, rtol=1e-12)
    np.testing.assert_allclose(frozen.rates, rates)


def test_fresh_bank_reuses_only_its_own_frozen_selector_cache(tmp_path, monkeypatch):
    model = PhysicsModel()
    (tmp_path/"models"/"preselection").mkdir(parents=True)
    checkpoint = tmp_path/"models"/"preselection"/"model.pt"
    checkpoint.write_bytes(b"fixed selector")
    (tmp_path/"selected").mkdir()
    threshold = .5
    fingerprint = hashlib.sha256((file_digest(checkpoint)+repr(threshold)).encode()).hexdigest()
    (tmp_path/"selected"/"selection_state.json").write_text(json.dumps({
        "threshold": threshold, "selection_fingerprint": fingerprint}))
    run = SimpleNamespace(model=model, config={"seed": 230, "generation_chunk": 300},
                          path=lambda *parts: tmp_path.joinpath(*parts))
    class Selector:
        def predict_proba(self, x):
            p = (x[:, 0] > 0).astype(float)
            return np.c_[p, 1-p]
    monkeypatch.setattr("poodemo.training.load_predictor", lambda path: Selector())
    bank = prepare_validation_bank(run, n_per_sample=500)
    for sample, points in bank["samples"].items():
        assert np.all(points[:, 0] > 0)
        assert bank["generated_counts"][sample] == 500
    cached = prepare_validation_bank(run, n_per_sample=500)
    np.testing.assert_array_equal(cached["samples"]["S"], bank["samples"]["S"])
    checkpoint.write_bytes(b"changed selector")
    with pytest.raises(ValueError, match="selector checkpoint changed"):
        prepare_validation_bank(run, n_per_sample=500)
