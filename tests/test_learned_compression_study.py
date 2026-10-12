"""Independent physical yields for frozen learned compression maps."""
import json

import numpy as np
import pytest

from poodemo import learned_compression_study as study
from poodemo import pipeline
from poodemo.data import Run
from poodemo.inference import component_coefficients, histogram_components
from poodemo.toy_study import reference_observable
from test_ratio_observable import _categorical_quadrature


def _bank():
    quad = _categorical_quadrature()
    return {**quad, "learned_nominal": quad["nominal"][:, ::-1].copy(),
            "stratum": np.array([0, 1, 2]),
            "metadata": {"sources": ["S", "B", "NI"],
                         "generated_counts": {"S": 2, "B": 2, "NI": 2}}}


def _run(tmp_path):
    return Run(tmp_path, dict(asimov_mu=1., mu_min=0., mu_max=1.4, mu_points=3,
                             mu_fit_bounds=[0., 2.], nuisance_bounds=[-2., 2.]), None)


def _banks():
    a, b = _bank(), _bank()
    b["weights"] = b["weights"] * np.array([1.1, .9, 1.05])
    return dict(template=a, validation=b, rates=a["nominal"] @ a["weights"], metadata={"fingerprint": "test"})


def test_map_changes_coordinates_only_and_keeps_original_normalizers():
    banks = _banks()
    a, b = banks["template"], banks["validation"]
    za, da = study._map(a, banks["rates"], 0., 10., .2, "learned", "template")
    zb, db = study._map(b, banks["rates"], .1, 10., .2, "learned", "validation")
    np.testing.assert_array_equal(za, zb)
    expected = reference_observable(a["learned_nominal"], banks["rates"], .2, 10.)
    np.testing.assert_array_equal(za, expected)
    assert da["valid"] and db["valid"]
    assert da["eta_effective"] == .2
    # The learned values are a different map, but integration weights are physical.
    assert not np.array_equal(za, reference_observable(a["nominal"], banks["rates"], .2, 10.))


def test_fixed_stratum_mc_errors_include_rejected_events():
    bank = _bank()
    # One retained event and one rejected zero-contribution event per stratum.
    counts, errors, ess = study._bin_moments(bank, np.array([.1, .1, .9]), np.array([0., .5, 1.]))
    contributions = bank["weights"] * bank["truth"]
    np.testing.assert_allclose(counts, [contributions[:2].sum(), contributions[2]])
    np.testing.assert_allclose(errors**2, [np.square(contributions[:2]).sum(), contributions[2]**2])
    np.testing.assert_allclose(ess, [contributions[:2].sum()**2/np.square(contributions[:2]).sum(), 1.])


def test_study_uses_physical_yields_and_independent_data_without_forcing_closure(tmp_path, monkeypatch):
    run, banks = _run(tmp_path), _banks()
    def no_saved_results(*args, **kwargs):
        pytest.fail("The analytical baseline must be recomputed on the fresh banks")
    monkeypatch.setattr(pipeline, "_saved_unbinned", no_saved_results)
    result = study.run_learned_compression_study(run, banks, n_bins=36, reference_power=10.,
                                                scan_grid=[0., .2, 1., 1.4],
                                                distribution_etas=[0., .1, .2, 1.])
    assert result["scans"].valid.all()
    assert set(result["scans"].model) == {study.ANALYTIC_MODEL, *study.MAP_MODELS.values()}
    assert set(result["scans"].comparison) == {"matched", "independent"}
    matched = result["scans"].query("comparison == 'matched' and mu == 1")
    np.testing.assert_allclose(matched.q, 0., atol=1e-9)
    independent = result["scans"].query("comparison == 'independent' and mu == 1")
    assert independent.q.max() > 1e-5
    assert np.max(np.abs(independent.mu_hat - 1.)) > .01
    a = banks["template"]
    edges = np.asarray(result["metadata"]["bin_edges"])
    z = reference_observable(a["learned_nominal"], banks["rates"], 0., 10., reference_eta_floor=.2)
    expected = histogram_components(z, a["nominal"], a["weights"], edges)
    table = result["component_templates"].query("model == 'binned learned map' and eta == 0")
    for index, process in enumerate(("S", "SBI", "B", "NI")):
        np.testing.assert_allclose(table.query("process == @process").expected_yield, expected[index])
    # Eta=0 and 0.1 share the map, while physical coefficient vectors differ.
    plateau = result["component_templates"].query("model == 'binned learned map'")
    np.testing.assert_array_equal(plateau.query("eta == 0").expected_yield,
                                  plateau.query("eta == .1").expected_yield)
    assert not np.allclose(component_coefficients(0.) @ expected, component_coefficients(.1) @ expected)
    assert .5 not in edges
    assert (result["bin_closure"].query("eta == 1").template > 0).sum() == 2
    information = result["information"].query("systematics == False")
    assert information.information_fraction.max() <= 1. + 1e-10
    output = run.path("results", "13_learned_compression_36_floor02")
    saved = json.loads((output / "metadata.json").read_text())
    assert saved["bank_metadata"]["fingerprint"] == "test"
    assert saved["reference_ratio_eta_floor"] == .2
    for name in ("scans", "information", "bin_closure", "component_templates", "map_diagnostics"):
        assert (output / (name + ".csv")).exists()


def test_fresh_analytic_unbinned_scan_matches_original_matched_likelihood(tmp_path):
    run, bank = _run(tmp_path), _bank()
    for systematic in (False, True):
        rows, _ = study._fresh_unbinned_scans(run, bank, bank, [0., .6, 1., 1.4], systematic, "matched")
        parent = pipeline._template_likelihood(bank, systematics=systematic)
        for row in rows:
            expected = pipeline._fit_test(run, parent, row["mu"], systematic)
            assert row["valid"]
            assert row["q"] == pytest.approx(expected["q"], rel=2e-6, abs=1e-8)


def test_invalid_learned_interference_is_reported_without_losing_analytic_control(tmp_path):
    banks = _banks()
    # At eta=.2 the S coefficient is negative; a bad learned S can make the
    # combined density negative despite positive component classifier ratios.
    banks["validation"]["learned_nominal"][0, 0] = 1000.
    result = study.run_learned_compression_study(_run(tmp_path), banks, n_bins=4,
                                                reference_power=10., scan_grid=[0., 1.],
                                                systematics=(False,), distribution_etas=[])
    scans = result["scans"]
    assert scans.loc[scans.model != "binned learned map", "valid"].all()
    bad = scans.query("model == 'binned learned map' and comparison == 'independent' and mu == 0")
    assert not bad.valid.any() and bad.q.isna().all()
    good = scans.query("model == 'binned learned map' and comparison == 'matched' and mu == 0")
    assert good.valid.all()
    report = result["map_diagnostics"].query("model == 'binned learned map' and bank == 'validation' and eta == 0")
    assert report.invalid_fraction.iloc[0] == pytest.approx(1/3)
    assert report.invalid_expected_fraction.iloc[0] > 0


def test_positive_validation_bin_with_no_template_support_is_explicitly_invalid(tmp_path):
    nominal = np.array([[1., 0.], [2., 0.], [1., 0.], [1., 0.]])
    result, model = study._fit_bins(_run(tmp_path), nominal, nominal[None], nominal[None],
                                    np.array([3., .1]), 0., False)
    assert not result["valid"] and np.isnan(result["q"])
    assert model is None
    assert "zero template support" in result["fit_message"]


@pytest.mark.parametrize("floor", [-.1, 1., np.nan])
def test_invalid_floor_rejected(tmp_path, floor):
    with pytest.raises(ValueError):
        study.run_learned_compression_study(_run(tmp_path), _banks(), reference_eta_floor=floor)
