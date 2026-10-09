"""Five-case pairing, fixed observable, boundary templates and resumable toys."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from poodemo.data import Run
from poodemo.inference import histogram_components
from poodemo.pipeline import observable_bin_edges
from poodemo.toy_study import (CASES, _load_templates, reference_observable,
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
    etas = np.asarray([.02, .6, 1., 1.4, 2.])
    edges = observable_bin_edges(12, "reference_ratio")
    rates = quad["nominal"].sum(axis=1)
    values = []
    for eta in etas:
        z = reference_observable(quad["nominal"], rates, eta, 100.)
        nominal = histogram_components(z, quad["nominal"], quad["weights"], edges)
        values.append(np.repeat(nominal[None], 3, axis=0))
    values = np.asarray(values)
    np.savez(run.path("results", "reference_ratio_spline_templates.npz"),
             etas=etas, edges=edges, bin_yields=values, totals=values[0].sum(axis=-1),
             reference_ratio_power=100., binning_policy="reference_halfstep_interior_v1")
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


def test_zero_uses_exact_template_without_extrapolation(workspace):
    run, quad, _ = workspace
    templates, edges, closure, metadata = _load_templates(run, quad, (0., 1.4), 12, 100.)
    z = reference_observable(quad["nominal"], quad["nominal"].sum(axis=1), 0., 100.)
    expected = histogram_components(z, quad["nominal"], quad["weights"], edges)
    np.testing.assert_array_equal(templates[0.], expected)
    assert "no extrapolation" in metadata["template_sources"]["0.0"]
    assert not np.any(closure.fraction_difference)
    with pytest.raises(ValueError, match="extrapolation"):
        _load_templates(run, quad, (2.1,), 12, 100.)
    with pytest.raises(ValueError, match="match"):
        _load_templates(run, quad, (0.,), 12, 20.)


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
    with pytest.raises(ValueError, match="totals differ"):
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
