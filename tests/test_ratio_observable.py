"""Physical ratio compression, frozen bins, and isolated study artifacts."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from poodemo import pipeline
from poodemo.data import Run
from poodemo.diagnostics import local_asimov_information, run_estimator_study
from poodemo.inference import TemplateLikelihood, component_coefficients
from poodemo.physics import PhysicsModel


@pytest.fixture
def selected_quadrature(tmp_path):
    """A fixed selection whose S and physical acceptances are different."""
    model = PhysicsModel()
    rng = np.random.default_rng(730)
    x = np.concatenate([model.sample_component(p, 600, rng) for p in ("S", "B", "NI")])
    proposal = sum(model.component_pdf(x, p) for p in ("S", "B", "NI")) / 3
    selected = x[:, 0] + x[:, 1] > .5
    weights = (1 / (len(x) * proposal))[selected]
    x = x[selected]
    nominal = model.component_densities(x).T
    quad = dict(x=x, weights=weights, nominal=nominal,
                down=model.component_densities(x, alpha_ni=-1.).T[None],
                up=model.component_densities(x, alpha_ni=1.).T[None],
                truth=model.intensity(x, 1.))
    return Run(tmp_path, {"score_scale": .005}, model), quad


def test_ratio_uses_selected_normalized_densities_and_is_exposure_invariant(selected_quadrature):
    run, quad = selected_quadrature
    signal = quad["nominal"][0]
    for eta in (.3, .73, 1.3):
        physical = run.model.intensity(quad["x"], eta)
        ratio = (physical / (physical @ quad["weights"])) / (signal / (signal @ quad["weights"]))
        actual = pipeline.observable_values(run, quad, eta, observable="ratio")
        np.testing.assert_allclose(actual, ratio / (1 + ratio), rtol=2e-14)
        assert np.all(np.isfinite(actual)) and np.all((actual >= 0) & (actual <= 1))
        assert np.ptp(actual) > .1

        # Full-space rates omit the selection acceptance and are not the
        # densities that define this selected experiment's observable.
        full_ratio = (physical / run.model.total_yield(eta)) / (signal / run.model.component_yield("S"))
        assert np.max(np.abs(actual - full_ratio / (1 + full_ratio))) > .01

        scaled = {**quad, **{key: 17. * quad[key] for key in ("nominal", "down", "up", "truth")}}
        np.testing.assert_allclose(pipeline.observable_values(run, scaled, eta, "ratio"), actual, rtol=2e-14)

    previous = pipeline.observable_values(run, quad, .73, "ratio")
    run.config["score_scale"] = 123.
    np.testing.assert_array_equal(pipeline.observable_values(run, quad, .73, "ratio"), previous)


def test_default_score_observable_is_unchanged(selected_quadrature):
    run, quad = selected_quadrature
    expected = expit(pipeline._score(run, quad, .73) / run.config["score_scale"])
    np.testing.assert_array_equal(pipeline.observable_values(run, quad, .73), expected)
    np.testing.assert_array_equal(pipeline.observable_values(run, quad, .73, "score"), expected)
    assert pipeline.observable_metadata()["prefix"] == ""
    assert pipeline.observable_metadata("ratio")["prefix"] == "ratio_"


@pytest.mark.parametrize("eta", [-1., np.nan, np.inf])
def test_ratio_rejects_unphysical_plot_hypotheses(selected_quadrature, eta):
    run, quad = selected_quadrature
    with pytest.raises(ValueError, match="eta"):
        pipeline.observable_values(run, quad, eta, "ratio")


def test_physical_mu_changes_expected_yields_inside_the_same_frozen_ratio_bins(selected_quadrature):
    run, quad = selected_quadrature
    eta = .73
    z = pipeline.observable_values(run, quad, eta, "ratio")
    edges = np.linspace(0., 1., 33)
    parent = TemplateLikelihood(quad["nominal"], quad["down"], quad["up"],
                                quad["weights"], quad["truth"])
    frozen = parent.binned(z, edges, eta=eta)
    before_templates, before_data = frozen.nominal.copy(), frozen.data_counts.copy()
    for mu in (.25, .8, 1.4):
        expected = np.histogram(z, edges, weights=quad["weights"] * run.model.intensity(quad["x"], mu))[0]
        np.testing.assert_allclose(frozen.intensity(mu, [0.]), expected, rtol=2e-12, atol=1e-8)
        np.testing.assert_array_equal(frozen.scores, z)
        assert frozen.eta == eta
    np.testing.assert_array_equal(frozen.nominal, before_templates)
    np.testing.assert_array_equal(frozen.data_counts, before_data)


def _categorical_quadrature():
    # Positive interference model for all mu >= 0. At eta=1 the first two
    # categories have equal p_eta/p_S but different mu derivatives.
    nominal = np.array([[1., 1., 1.], [2., 2., 3.],
                        [1., 3., 2.], [1., 1., 1.]])
    down, up = nominal[None].copy(), nominal[None].copy()
    displacement = np.array([-.1, .1, .05])
    down[0, 3] *= np.exp(-displacement)
    up[0, 3] *= np.exp(displacement)
    return dict(x=np.arange(3.)[:, None], weights=np.ones(3), nominal=nominal,
                down=down, up=up, truth=component_coefficients(1.) @ nominal)


def test_arbitrarily_fine_ratio_bins_need_not_preserve_local_fisher_information(tmp_path):
    quad = _categorical_quadrature()
    run = Run(tmp_path, {}, None)
    parent = pipeline._template_likelihood(quad, systematics=False)
    z = pipeline.observable_values(run, quad, 1., "ratio")
    assert z[0] == z[1] and z[0] != z[2]
    full_info = local_asimov_information(parent, 1.)["matrix"][0, 0]
    np.testing.assert_allclose(full_info, 7 / 12)
    for n_bins in (128, 4096):
        frozen = parent.binned(z, np.linspace(0, 1, n_bins + 1), eta=1.)
        compressed_info = local_asimov_information(frozen, 1.)["matrix"][0, 0]
        np.testing.assert_allclose(compressed_info, 5 / 12)
        np.testing.assert_allclose(full_info - compressed_info, 1 / 6)


def test_ratio_studies_fit_frozen_models_and_preserve_all_score_outputs(tmp_path, monkeypatch):
    """Run real SciPy likelihood fits; stub only expensive data preparation."""
    run = Run(tmp_path, dict(asimov_mu=1., mu_min=.5, mu_max=1.5, mu_points=3,
                            mu_fit_bounds=[.2, 2.], nuisance_bounds=[-2., 2.],
                            bin_counts=[4, 8], score_scale=.15, spline_anchors=3,
                            spline_focus_anchors=0, spline_focus_halfwidth=.1), None)
    run.path("results").mkdir()
    quad = _categorical_quadrature()
    monkeypatch.setattr(pipeline, "prepare_quadrature", lambda current_run: quad)
    monkeypatch.setattr(pipeline, "_saved_unbinned", lambda *args, **kwargs: pd.DataFrame())

    pipeline.run_binning_study(run)
    pipeline.run_spline_study(run, n_bins=8)
    run_estimator_study(run)
    score_files = {path.name: path.read_bytes() for path in run.path("results").iterdir()}
    assert "binning_scans.csv" in score_files and "estimator_eta.csv" in score_files

    binning = pipeline.run_binning_study(run, observable="ratio")
    splines = pipeline.run_spline_study(run, n_bins=8, observable="ratio")
    estimators = run_estimator_study(run, observable="ratio")
    for name, original in score_files.items():
        assert run.path("results", name).read_bytes() == original, name
        assert run.path("results", "ratio_" + name).exists(), name
    assert {p.name for p in run.path("results").iterdir()} == set(score_files) | {
        "ratio_" + name for name in score_files}

    assert set(binning["scans"].model) == {"direct ratio histogram", "fixed ratio histogram"}
    assert set(splines["scans"].model) == {"direct ratio histogram", "spline ratio histogram"}
    assert set(estimators.model) == {"direct ratio histogram", "analytic unbinned"}
    for scans in (binning["scans"], splines["scans"], estimators):
        assert scans.valid.all()
        np.testing.assert_allclose(scans.mu_hat, 1., atol=1e-5)
    assert (estimators.truth == 1.).all() and (estimators.information_mu == 1.).all()
    assert set(estimators.systematics) == {False, True}
    assert (estimators.loc[estimators.model == "direct ratio histogram", "n_bins"] == 8).all()
    assert (binning["fidelity"].information_fraction < .99).any()

    # Physical scans keep eta fixed at truth while scaling component bins.
    z = pipeline.observable_values(run, quad, 1., "ratio")
    edges = np.linspace(0, 1, 9)
    for mu, rows in splines["physical_yields"].groupby("mu"):
        expected = np.histogram(z, edges, weights=quad["weights"] *
                                (component_coefficients(mu) @ quad["nominal"]))[0]
        assert (rows.eta == 1.).all()
        np.testing.assert_allclose(rows.sort_values("bin").yield_value, expected, atol=1e-14)


@pytest.mark.parametrize("study", [pipeline.run_binning_study, pipeline.run_spline_study, run_estimator_study])
def test_invalid_observable_is_rejected_before_preparing_data(tmp_path, monkeypatch, study):
    def unexpected_prepare(run):
        pytest.fail("Invalid observable must be rejected before loading quadrature")

    monkeypatch.setattr(pipeline, "prepare_quadrature", unexpected_prepare)
    with pytest.raises(ValueError, match="observable"):
        study(Run(tmp_path, {}, None), observable="invalid")
