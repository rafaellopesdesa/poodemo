"""Check estimator widths against actual Asimov likelihood curvature."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar

from poodemo.diagnostics import (
    _unbinned_baselines, local_asimov_information, local_width, run_estimator_study,
)
from poodemo.inference import TemplateLikelihood


def toy_model():
    signal = np.array([2., 5., 8., 3.])
    background = np.array([15., 10., 12., 20.])
    ni = np.array([20., 13., 8., 17.])
    # Genuine negative interference, with a positive coherent component.
    sbi = signal + background - .7 * np.sqrt(signal * background)
    nominal = np.stack([signal, sbi, background, ni])
    down, up = nominal[None].copy(), nominal[None].copy()
    displacement = np.array([-.12, .08, .18, -.05])
    down[0, 3] *= np.exp(-displacement)
    up[0, 3] *= np.exp(displacement)
    return TemplateLikelihood(nominal, down, up, np.ones(4), nominal[1] + ni)


def test_fisher_equals_numerical_asimov_hessian_with_gaussian_constraint():
    model = toy_model()
    information = local_asimov_information(model, 1.)
    h = 2e-4
    origin = np.array([1., 0.])
    def nll(theta):
        return model.nll(theta[0], theta[1:])
    hessian = np.empty((2, 2))
    for i in range(2):
        ei = np.eye(2)[i] * h
        hessian[i, i] = (nll(origin + ei) - 2*nll(origin) + nll(origin - ei)) / h**2
    e0, e1 = np.eye(2) * h
    hessian[0, 1] = hessian[1, 0] = (
        nll(origin + e0 + e1) - nll(origin + e0 - e1)
        - nll(origin - e0 + e1) + nll(origin - e0 - e1)
    ) / (4*h**2)
    np.testing.assert_allclose(information["matrix"], hessian / 2, rtol=3e-7, atol=2e-8)
    np.testing.assert_array_equal(information["auxiliary_matrix"], [[0, 0], [0, 1]])


def test_profiled_width_matches_directly_profiled_curvature():
    model = toy_model()
    information = local_asimov_information(model, 1.)
    width = local_width(information["matrix"], systematics=True)
    h = 1e-3
    def profile(mu):
        result = minimize_scalar(lambda alpha: model.nll(mu, [alpha]),
                                 bounds=(-1, 1), method="bounded",
                                 options={"xatol": 1e-13})
        assert result.success
        return result.fun
    curvature_information = (profile(1+h) + profile(1-h) - 2*profile(1)) / (2*h*h)
    np.testing.assert_allclose(width["information_mu_effective"], curvature_information, rtol=2e-6)
    np.testing.assert_allclose(width["sigma_mu"], 1/np.sqrt(curvature_information), rtol=2e-6)
    assert width["sigma_mu"] >= local_width(information["matrix"])["sigma_mu"]


def test_rate_shape_decomposition_and_information_loss_from_binning():
    model = toy_model()
    full = local_asimov_information(model, 1.)
    np.testing.assert_allclose(full["data_matrix"], full["rate_matrix"] + full["shape_matrix"], atol=1e-14)
    coarse = model.binned([.1, .2, .8, .9], [0, .5, 1])
    binned = local_asimov_information(coarse, 1.)
    np.testing.assert_allclose(binned["rate_matrix"][0, 0], full["rate_matrix"][0, 0], rtol=1e-13)
    assert 0 <= binned["shape_matrix"][0, 0] <= full["shape_matrix"][0, 0]
    assert local_width(binned["matrix"])["sigma_mu"] >= local_width(full["matrix"])["sigma_mu"]


def test_unidentified_width_is_not_reported_as_zero_or_regularized_finite():
    for matrix, profiled in (([[0.]], False), ([[1., 1.], [1., 1.]], True)):
        result = local_width(matrix, systematics=profiled)
        assert not result["width_valid"]
        assert np.isinf(result["sigma_mu"])
    # Small but resolved information is allowed to produce a large width.
    assert local_width([[1e-20]])["sigma_mu"] == 1e10


def test_estimator_study_holds_generating_truth_fixed_and_saves_schema(tmp_path):
    model = toy_model()
    quad = dict(nominal=model.nominal, down=model.down, up=model.up,
                weights=model.quadrature_weights, truth=model.truth_intensity)
    config = dict(asimov_mu=1., mu_min=.2, mu_max=3., mu_points=3,
                  mu_fit_bounds=[.01, 5.], nuisance_bounds=[-3., 3.],
                  bin_counts=[8, 64], score_scale=1.)
    run = SimpleNamespace(config=config, path=lambda *parts: tmp_path.joinpath(*parts))
    with patch("poodemo.pipeline.prepare_quadrature", return_value=quad):
        result = run_estimator_study(run)
    assert (tmp_path / "results" / "estimator_eta.csv").exists()
    assert set(result.systematics) == {False, True}
    assert set(result.model) == {"analytic unbinned", "direct score histogram"}
    assert set(result.truth) == {1.}
    assert set(result.information_mu) == {1.}
    assert result.valid.all(), result.loc[~result.valid, ["fit_message", "width_message"]]
    np.testing.assert_allclose(result.mu_hat, 1., atol=2e-5)
    assert np.all(result.sigma_mu > 0)
    for _, baseline in result[result.model == "analytic unbinned"].groupby("systematics"):
        assert baseline.mu_hat.nunique() == 1
        assert baseline.sigma_mu.nunique() == 1


def saved_baseline_fixture(tmp_path):
    model = toy_model()
    run = SimpleNamespace(config=dict(asimov_mu=1., mu_fit_bounds=[.01, 5.],
                                      nuisance_bounds=[-3., 3.]),
                          path=lambda *parts: tmp_path.joinpath(*parts))
    path = run.path("results", "unbinned_fits.csv")
    path.parent.mkdir()
    records = pd.DataFrame([dict(model="analytic unbinned", systematics=syst,
                                 mu_hat=1., alpha_NI_hat=0., nll=model.nll(1., [0.]), valid=True)
                            for syst in (False, True)])
    records.to_csv(path, index=False)
    return model, run, path, records


def test_saved_toolkit_baselines_skip_optimization_but_recheck_likelihood(tmp_path):
    model, run, _, _ = saved_baseline_fixture(tmp_path)
    information = local_asimov_information(model, 1.)
    with patch.object(model, "fit", side_effect=AssertionError("Cached baselines must not refit")), \
            patch.object(model, "nll", wraps=model.nll) as evaluate:
        result = _unbinned_baselines(run, model, information)
    assert evaluate.call_count == 3  # generating truth once, each saved fit once
    assert set(result) == {False, True}
    for record in result.values():
        assert record["valid"]
        assert record["mu_hat"] == 1.
        assert record["fit_source"] == "Saved notebook 3 toolkit fit"


@pytest.mark.parametrize("failure", ["invalid", "missing_mode", "wrong_nll"])
def test_bad_saved_baselines_are_not_silently_refitted(tmp_path, failure):
    model, run, path, records = saved_baseline_fixture(tmp_path)
    if failure == "invalid":
        records.loc[1, "valid"] = False
    elif failure == "missing_mode":
        records = records.iloc[:1]
    else:
        records.loc[1, "nll"] += .1
    records.to_csv(path, index=False)
    with patch.object(model, "fit", side_effect=AssertionError("Bad saved baselines must not refit")):
        with pytest.raises(ValueError, match="rerun notebook 3"):
            _unbinned_baselines(run, model, local_asimov_information(model, 1.))
