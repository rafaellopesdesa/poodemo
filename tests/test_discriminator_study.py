"""Nested fixed discriminators, selected normalization, and scan provenance."""
import json

import numpy as np
import pytest
from scipy.special import expit

from poodemo import discriminator_study as study_module
from poodemo import pipeline
from poodemo.data import Run
from poodemo.diagnostics import local_asimov_information
from test_ratio_observable import _categorical_quadrature, selected_quadrature


def test_process_ratios_are_selected_normalized_and_exposure_invariant(selected_quadrature):
    _, quad = selected_quadrature
    powers = {"NI": .5, "SBI": 2., "B": 3.}
    values = study_module.process_ratio_values(quad, powers)
    nominal, weights = quad["nominal"], quad["weights"]
    rates = nominal @ weights
    for name, index in (("NI", 3), ("SBI", 1), ("B", 2)):
        ratio = (nominal[index]/rates[index])/(nominal[0]/rates[0])
        np.testing.assert_allclose(values[name], expit(powers[name]*np.log(ratio)), atol=1e-15)
        assert np.all((values[name] >= 0.) & (values[name] <= 1.))
    scaled = {**quad, "nominal": nominal*np.array([.1, 10., 20., 100.])[:, None]}
    for name, actual in study_module.process_ratio_values(scaled, powers).items():
        np.testing.assert_allclose(actual, values[name], atol=2e-15)


@pytest.mark.parametrize("powers", [{"NI": 1.}, {"NI": 0., "SBI": 2., "B": 2.},
                                    {"NI": 1., "SBI": np.nan, "B": 2.},
                                    {"NI": 1., "SBI": 2., "B": "bad"}])
def test_invalid_discriminator_powers_are_rejected(powers):
    with pytest.raises(ValueError, match="power"):
        study_module.process_ratio_values(_categorical_quadrature(), powers)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_nd_histograms_match_numpy_and_preserve_the_same_assignment(selected_quadrature, dimension):
    _, quad = selected_quadrature
    parent = pipeline._template_likelihood(quad, systematics=True)
    axes = study_module.process_ratio_values(quad)
    coordinates = np.column_stack([axes[name] for name in study_module.AXES[:dimension]])
    # Exercise the closed endpoints as well as interior edge assignments.
    coordinates[0] = 0.
    coordinates[1] = 1.
    coordinates[2] = .5
    edges = pipeline.observable_bin_edges(12, "reference_ratio")
    histogram = study_module.multidimensional_histogram(parent, coordinates, edges)
    assert histogram.nominal.shape == (4, 12**dimension)
    original = histogram._index.copy()
    for mu in (.02, .6, 1., 1.4):
        expected = np.histogramdd(coordinates, bins=[edges]*dimension,
                                  weights=quad["weights"]*parent.intensity(mu))[0].ravel()
        np.testing.assert_allclose(histogram.intensity(mu), expected, atol=1e-9, rtol=2e-12)
        np.testing.assert_array_equal(histogram._index, original)
    np.testing.assert_allclose(histogram.rates, parent.rates, rtol=1e-13)
    np.testing.assert_allclose(histogram.rates_up, parent.rates_up, rtol=1e-13)
    np.testing.assert_allclose(histogram.rates_down, parent.rates_down, rtol=1e-13)
    assert histogram.nll(1.) == pytest.approx(0., abs=1e-8)


def test_nested_discriminators_cannot_lose_statistical_information(selected_quadrature):
    _, quad = selected_quadrature
    parent = pipeline._template_likelihood(quad, systematics=False)
    axes = study_module.process_ratio_values(quad)
    edges = pipeline.observable_bin_edges(12, "reference_ratio")
    information = []
    divergences = []
    for dimension in (1, 2, 3):
        coordinates = np.column_stack([axes[name] for name in study_module.AXES[:dimension]])
        histogram = study_module.multidimensional_histogram(parent, coordinates, edges)
        information.append(local_asimov_information(histogram, 1.)["matrix"][0, 0])
        divergences.append(histogram.nll(.3))
    information.append(local_asimov_information(parent, 1.)["matrix"][0, 0])
    divergences.append(parent.nll(.3))
    assert np.all(np.diff(information) >= -1e-9)
    assert np.all(np.diff(divergences) >= -1e-9)


def test_run_isolated_artifacts_and_unchanged_compression_settings(tmp_path, monkeypatch):
    quad = _categorical_quadrature()
    run = Run(tmp_path, dict(asimov_mu=1., mu_min=.2, mu_max=1.4, mu_points=3,
                            mu_fit_bounds=[.001, 2.], nuisance_bounds=[-2., 2.],
                            reference_ratio_power=5.), None)
    run.path("results").mkdir()
    sentinel = run.path("results", "reference_ratio_spline_templates.npz")
    sentinel.write_bytes(b"preserve notebook 09")
    monkeypatch.setattr(pipeline, "prepare_quadrature", lambda run: quad)
    def no_learned_fields(*args):
        pytest.fail("Analytical compression must not load any trained network")
    monkeypatch.setattr(pipeline, "_learned_fields", no_learned_fields)
    result = study_module.run_discriminator_study(run, scan_grid=[.3, 1., 1.4], n_bins=12)
    assert result["scans"].valid.all()
    assert set(result["scans"].model) == set(study_module.FIXED_LABELS) | {
        "analytic unbinned", study_module.REFERENCE_LABEL}
    assert set(result["scans"].systematics) == {False, True}
    np.testing.assert_allclose(result["scans"].loc[result["scans"].mu == 1., "q"], 0., atol=1e-8)
    occupancy = result["occupancy"].set_index("model")
    for label, expected in zip(study_module.FIXED_LABELS, (12, 144, 1728)):
        assert occupancy.loc[label, "total_cells"] == expected
    assert run.config["reference_ratio_power"] == 5.
    assert sentinel.read_bytes() == b"preserve notebook 09"
    saved = json.loads(run.path("results", "10_discriminator_study", "config.json").read_text())
    assert saved["reference_ratio_power"] == 100.
    assert .5 not in saved["bin_edges"]
    for name in ("scans", "information", "occupancy", "axis_summary", "axis_histograms"):
        assert run.path("results", "10_discriminator_study", name + ".csv").exists()
    assert set(result["joint_histograms"]) == {"NI_SBI", "NI_B", "SBI_B"}


def test_learned_axes_keep_analytical_templates_and_asimov(tmp_path, monkeypatch):
    quad = _categorical_quadrature()
    run = Run(tmp_path, dict(asimov_mu=1., mu_min=.3, mu_max=1.4, mu_points=2,
                            mu_fit_bounds=[.001, 2.], nuisance_bounds=[-2., 2.]), None)
    learned = quad["nominal"][:, ::-1].copy()
    called = []
    monkeypatch.setattr(pipeline, "prepare_quadrature", lambda run: quad)
    def fields(run, bank):
        called.append(True)
        return learned, learned[None], learned[None]
    monkeypatch.setattr(pipeline, "_learned_fields", fields)
    result = study_module.run_discriminator_study(run, n_bins=4, ratio_source="learned", systematics=(False,))
    assert called == [True]
    assert result["config"]["likelihood_source"] == "analytic"
    assert result["config"]["asimov_source"] == "analytic"
    np.testing.assert_allclose(result["occupancy"].expected_events, quad["truth"] @ quad["weights"])
