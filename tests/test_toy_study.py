"""Direct fixed-eta histograms, paired sources and resumable five-case toys."""
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from poodemo.data import Run
from poodemo.inference import component_coefficients
from poodemo.toy_study import (CASES, _direct_templates, reference_observable,
                               run_toy_study, summarize_toys)


class DiscretePhysics:
    """Two cells with distinct S:B mixtures and positive interference model."""
    fields = np.asarray([[2., 1.], [4., 5.], [3., 4.], [2., 1.]])

    def to_dict(self):
        return {"model": "two-cell toy-study regression"}

    def component_densities(self, x):
        return self.fields[:, np.asarray(x[:, 0], int)].T


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    from poodemo import toy_sources, toy_study
    model = DiscretePhysics()
    run = Run(tmp_path, {"seed": 57}, model)
    run.path("results").mkdir()
    run.path("models").mkdir()
    quad = dict(x=np.asarray([[0., 0., 0.], [1., 0., 0.]]), weights=np.ones(2), nominal=model.fields.copy())
    quad["down"], quad["up"] = quad["nominal"][None].copy(), quad["nominal"][None].copy()
    monkeypatch.setattr(toy_study, "prepare_quadrature", lambda run: quad)
    monkeypatch.setattr(toy_sources, "load_frozen_selector", lambda run: (object(), .5, {"selector": "frozen"}))
    frozen = SimpleNamespace(learned_quadrature=(model.fields.copy(), None, None),
                             evaluate=lambda x: (model.component_densities(x).T, None, None))
    monkeypatch.setattr(toy_sources, "make_frozen_workspace", lambda run, quad: frozen)
    calls = []

    def simulate(run, mu, rng, **kwargs):
        from poodemo.inference import component_coefficients
        counts = rng.poisson(kwargs["exposure"] * (component_coefficients(mu) @ model.fields))
        x = np.repeat(quad["x"], counts, axis=0)
        calls.append((mu, len(x)))
        return x

    monkeypatch.setattr(toy_sources, "sample_simulator_poisson", simulate)
    return run, quad, calls


def test_reference_coordinates_use_frozen_rates_and_exact_collapse():
    fields = DiscretePhysics.fields
    rates = fields.sum(axis=1)
    z = reference_observable(fields, rates, .3, 100.)
    replicated = fields[:, [0, 0, 1, 0]]
    np.testing.assert_array_equal(reference_observable(replicated, rates, .3, 100.), z[[0, 0, 1, 0]])
    np.testing.assert_array_equal(reference_observable(replicated, rates, 1., 100.), .5)


def test_both_hypotheses_use_direct_fixed_eta_templates_without_notebook09(workspace):
    run, quad, _ = workspace
    assert not run.path("results", "reference_ratio_spline_templates.npz").exists()
    templates, edges, yields, metadata = _direct_templates(quad, (0., 1.4), 12, 100.)
    assert metadata["template_method"] == "direct_quadrature"
    assert set(yields.columns) == {"mu_test", "eta", "process", "bin", "expected_yield", "template_source"}
    for eta in (0., 1.4):
        z = reference_observable(quad["nominal"], quad["nominal"].sum(axis=1), eta, 100.)
        expected = np.asarray([np.histogram(z, edges, weights=quad["weights"] * fields)[0]
                               for fields in quad["nominal"]])
        np.testing.assert_array_equal(templates[eta], expected)
        np.testing.assert_array_equal(yields.loc[yields.eta == eta].expected_yield, expected.ravel())
        # Physical mu varies during the fit, but the observable stays at eta.
        for fitted_mu in (0., .1, .5, 1.3, 2.):
            expected_counts = np.histogram(z, edges, weights=quad["weights"] *
                                          (component_coefficients(fitted_mu) @ quad["nominal"]))[0]
            np.testing.assert_allclose(component_coefficients(fitted_mu) @ templates[eta], expected_counts)


def test_existing_or_malformed_notebook09_files_are_ignored(workspace):
    run, _, calls = workspace
    original = run_toy_study(run, n_toys=2, grid_size=9)
    path = run.path("results", "reference_ratio_spline_templates.npz")
    # An unrelated (and subsequently malformed) old spline must not affect
    # either the likelihood or its cache fingerprint.
    np.savez(path, etas=np.asarray([.02, 2.]), wrong_templates=np.zeros(3))
    existing = run_toy_study(run, n_toys=2, grid_size=9)
    path.write_bytes(b"not an npz file")
    malformed = run_toy_study(run, n_toys=2, grid_size=9)
    assert len(calls) == 4
    for actual in (existing, malformed):
        pd.testing.assert_frame_equal(original["results"], actual["results"], check_dtype=False)
        pd.testing.assert_frame_equal(original["bin_yields"], actual["bin_yields"])
        assert original["metadata"]["fingerprint"] == actual["metadata"]["fingerprint"]
    assert run.path("results", "toy_study_direct_bin_yields.csv").exists()


def test_binned_toy_means_and_fit_templates_use_the_same_direct_yields(workspace, monkeypatch):
    import poodemo.toy_study as module
    run, quad, _ = workspace
    expected, _, _, _ = _direct_templates(quad, (0., 1.4), 12, 100.)
    original_rng = module._rng
    original_binned = module.StatOnlyLikelihood.from_binned
    means, fitted_fields = [], []

    def record_rng(seed, mu, toy_id, stream):
        rng = original_rng(seed, mu, toy_id, stream)
        if stream != 2:
            return rng
        def poisson(values):
            means.append((mu, np.asarray(values).copy()))
            return rng.poisson(values)
        return SimpleNamespace(poisson=poisson)

    def record_binned(fields, counts, *, exposure):
        fitted_fields.append(np.asarray(fields).copy())
        return original_binned(fields, counts, exposure=exposure)

    monkeypatch.setattr(module, "_rng", record_rng)
    monkeypatch.setattr(module.StatOnlyLikelihood, "from_binned", record_binned)
    run_toy_study(run, n_toys=2, grid_size=9, exposure=3.)
    for mu, values in means:
        np.testing.assert_allclose(values, 3. * (component_coefficients(mu) @ expected[mu]))
    assert len(means) == 4 and len(fitted_fields) == 8
    for fields, mu in zip(fitted_fields, [0.] * 4 + [1.4] * 4):
        np.testing.assert_array_equal(fields, expected[mu])


def test_old_interpolated_template_cache_is_rejected(workspace):
    run, _, _ = workspace
    study = run_toy_study(run, n_toys=1, grid_size=9)
    path = run.path("results", "toy_study_direct_config.json")
    metadata = study["metadata"]
    metadata["configuration"]["version"] = 1
    metadata["configuration"].pop("template_method")
    metadata["configuration"]["spline_sha256"] = "old template hash"
    metadata["fingerprint"] = hashlib.sha256(json.dumps(
        metadata["configuration"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="differ"):
        run_toy_study(run, n_toys=1, grid_size=9)


def test_five_cases_share_simulator_and_resume_deterministically(workspace):
    run, quad, calls = workspace
    first = run_toy_study(run, n_toys=4, grid_size=9)
    assert len(calls) == 8  # one simulator draw per hypothesis/toy, not per fit
    assert set(first["results"].case) == set(CASES)
    assert len(first["results"]) == 2 * 4 * 5
    assert first["results"].valid.all()
    simulator = first["results"].loc[first["results"].source_stream == 0]
    assert (simulator.groupby(["mu_test", "toy_id"]).n_observed.nunique() == 1).all()
    # Both likelihoods are exactly the same here and see the identical toy.
    paired = simulator.pivot(index=["mu_test", "toy_id"], columns="case", values="q")
    np.testing.assert_allclose(paired.analytic_simulator, paired.learned_simulator)
    assert (first["results"].eta == first["results"].mu_test).all()
    assert first["metadata"]["source_diagnostics"]["0.0"]["continuous_generator"] is False
    same = run_toy_study(run, n_toys=4, grid_size=9)
    assert len(calls) == 8
    pd.testing.assert_frame_equal(first["results"], same["results"], check_dtype=False, atol=1e-12)
    extended = run_toy_study(run, n_toys=6, grid_size=9)
    assert len(calls) == 12
    # Fresh run with a different output name reproduces every toy exactly.
    fresh = run_toy_study(run, n_toys=6, grid_size=9, output_tag="fresh")
    pd.testing.assert_frame_equal(extended["results"], fresh["results"], check_dtype=False, atol=1e-12)
    with pytest.raises(ValueError, match="differ"):
        run_toy_study(run, n_toys=6, grid_size=9, exposure=2.)
    quad["nominal"][3, 0] += .1
    with pytest.raises(ValueError, match="differ"):
        run_toy_study(run, n_toys=6, grid_size=9)


def test_fit_failures_stay_in_rows_and_denominators(workspace, monkeypatch):
    import poodemo.toy_study as module
    run, _, _ = workspace
    original = module.StatOnlyLikelihood.test_statistic

    def failed_at_zero(self, mu, **kwargs):
        if mu == 0.:
            raise FloatingPointError("test fit failure")
        return original(self, mu, **kwargs)

    monkeypatch.setattr(module.StatOnlyLikelihood, "test_statistic", failed_at_zero)
    result = run_toy_study(run, n_toys=4, grid_size=9)
    assert len(result["results"]) == 40
    zero = result["summary"].loc[result["summary"].mu_test == 0.]
    assert (zero.n_failed == 4).all()
    assert (zero.failure_fraction == 1.).all()
    assert result["results"].loc[result["results"].mu_test == 0., "message"].str.contains("test fit failure").all()
    assert result["coverage"].loc[result["coverage"].mu_test == 0., "n_evaluation_failed"].eq(2).all()


def test_coverage_uses_disjoint_calibration_and_evaluation_ids():
    rows = []
    for case in CASES:
        for toy_id in range(6):
            # Even toys have q=1, odd toys q=100: in-sample coverage would hide this.
            rows.append(dict(mu_test=0., case=case, toy_id=toy_id, q=1. if toy_id % 2 == 0 else 100.,
                             valid=True, mu_hat=0., at_lower=True, at_upper=False))
    summary, coverage = summarize_toys(pd.DataFrame(rows))
    assert (coverage.n_calibration == 3).all()
    assert (coverage.n_evaluation == 3).all()
    assert (coverage.critical_q == 1.).all()
    assert (coverage.coverage == 0.).all()
    assert (summary.lower_boundary_fraction == 1.).all()


def test_invalid_learned_toy_source_does_not_discard_other_comparisons(workspace, monkeypatch):
    from poodemo import toy_sources
    run, _, _ = workspace
    original = toy_sources.QuadraturePoissonSource

    def invalid_source(quad, fields, mu, **kwargs):
        if mu == 1.4:
            raise ValueError("negative learned intensity")
        return original(quad, fields, mu, **kwargs)

    monkeypatch.setattr(toy_sources, "QuadraturePoissonSource", invalid_source)
    study = run_toy_study(run, n_toys=2, grid_size=9)
    invalid = study["results"].loc[(study["results"].mu_test == 1.4)
                                   & (study["results"].case == "learned_model")]
    assert len(invalid) == 2 and not invalid.valid.any()
    assert invalid.message.str.contains("negative learned intensity").all()
    assert study["results"].valid.sum() == 18
    assert study["metadata"]["source_diagnostics"]["1.4"]["valid"] is False
