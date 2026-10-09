"""Paired reconstruction, one-factor refits, selection and cache safety."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from poodemo.data import Run
from poodemo.inference import component_coefficients, histogram_components
from poodemo.toy_integration_diagnostics import diagnostic_bin_edges
from poodemo.toy_likelihood import StatOnlyLikelihood
from poodemo.toy_refit_diagnostics import (SIMULATOR_CASES, plot_refit_diagnostics,
                                          run_refit_diagnostics, select_refit_toys)
from poodemo.toy_study import reference_observable, run_toy_study


class CellPhysics:
    fields = np.array([[2., 1., .3], [4., 5., .7], [3., 4., .5], [2., 1., 3.]])

    def to_dict(self):
        return {"model": "three-cell independent refit test"}

    def component_densities(self, x):
        return self.fields[:, x[:, 0].astype(int)].T


@pytest.fixture
def refit_context(tmp_path, monkeypatch):
    from poodemo import toy_sources, toy_study
    model = CellPhysics()
    run = Run(tmp_path, {"seed": 57}, model)
    run.path("results").mkdir()
    run.path("models").mkdir()
    quad = dict(x=np.column_stack((np.arange(3), np.zeros((3, 2)))),
                weights=np.ones(3), nominal=model.fields.copy())
    quad["down"], quad["up"] = quad["nominal"][None].copy(), quad["nominal"][None].copy()
    frozen = SimpleNamespace(learned_quadrature=(model.fields.copy(), None, None),
                             evaluate=lambda x: (model.component_densities(x).T, None, None))
    monkeypatch.setattr(toy_study, "prepare_quadrature", lambda run: quad)
    monkeypatch.setattr(toy_sources, "load_frozen_selector", lambda run: (object(), .5, {"frozen": True}))
    monkeypatch.setattr(toy_sources, "make_frozen_workspace", lambda run, quad: frozen)
    calls = []

    def simulate(run, mu, rng, **kwargs):
        counts = rng.poisson(kwargs["exposure"] * (component_coefficients(mu) @ model.fields))
        calls.append((mu, counts.copy()))
        return np.repeat(quad["x"], counts, axis=0)

    monkeypatch.setattr(toy_sources, "sample_simulator_poisson", simulate)
    source = run_toy_study(run, n_toys=3, grid_size=9, progress_every=10)
    calls.clear()
    rates = model.fields.sum(axis=1)
    guard = StatOnlyLikelihood(np.empty((4, 0)), rates, validity_fields=model.fields)
    context = dict(run=run, source=source, configuration=source["metadata"]["configuration"],
                   rates=rates, learned_rates=rates.copy(), quad=quad, workspace=frozen,
                   selector=object(), threshold=.5, learned_guard=guard, fingerprint="test-original-context",
                   output_dir=tmp_path / "diagnostics")
    return context, calls


def _integration(context, scale=1.):
    templates = {}
    edges = {n: diagnostic_bin_edges(n) for n in (12, 36)}
    for mu in (0., 1.4):
        z = reference_observable(context["quad"]["nominal"], context["rates"], mu, 100.)
        for n, bins in edges.items():
            templates[n, mu] = scale * histogram_components(
                z, context["quad"]["nominal"], context["quad"]["weights"], bins)
    return dict(templates=templates, rates=context["rates"] * scale, edges=edges,
                metadata={"fingerprint": "independent-test-bank", "source_fingerprint": context["fingerprint"]})


def test_all_upper_bound_cases_and_independent_random_controls():
    rows = []
    for mu in (0., 1.4):
        for toy_id in range(30):
            for ci, case in enumerate(SIMULATOR_CASES):
                rows.append(dict(mu_test=mu, toy_id=toy_id, case=case, q=toy_id * .1 + ci,
                                 mu_hat=2. if toy_id < 5 and ci == 2 else .2,
                                 at_upper=toy_id < 5 and ci == 2))
    source = {"results": pd.DataFrame(rows)}
    selected = select_refit_toys(source, n_random=8)
    unflagged_source = {"results": source["results"].assign(at_upper=False)}
    independent = select_refit_toys(unflagged_source, n_random=8)
    for mu, group in selected.groupby("mu_test"):
        assert set(range(5)) <= set(group.toy_id)
        assert group.is_random.sum() == 8
        assert set(group.loc[group.is_random, "toy_id"]) == set(independent.loc[independent.mu_test == mu, "toy_id"])
        assert group.loc[group.toy_id < 5, "branch_disagreement"].all()
        assert (group.n_population == 30).all()
    pd.testing.assert_frame_equal(selected, select_refit_toys(source, n_random=8))


def test_reproduces_original_experiments_and_one_factor_settings(refit_context):
    context, calls = refit_context
    selection = select_refit_toys(context["source"], n_random=3)
    study = run_refit_diagnostics(context, selection, grid_size=17, n_scans=1,
                                  integration=_integration(context))
    assert len(calls) == 6
    original = study["comparisons"].query("variant == 'original'")
    assert original.reproduction_pass.all()
    assert original.count_reproduced.all()
    assert len(original) == 18
    assert study["refits"].valid.all()
    original = study["refits"].query("variant == 'original'")
    dense = study["refits"].query("variant == 'dense'")
    wide = study["refits"].query("variant == 'wide'")
    assert (original.mu_upper == 2.).all() and (original.grid_size == 9).all()
    assert (dense.mu_upper == 2.).all() and (dense.grid_size == 17).all()
    assert (wide.mu_upper == 3.).all() and (wide.grid_size == 17).all()
    assert not study["scans"].empty
    for _, group in study["scans"].groupby(["mu_test", "toy_id", "case", "variant"]):
        assert group.loc[group.mu == group.mu_test.iloc[0], "relative_nll"].item() == 0.
    assert set(study["summary"].summary_population) == {"random controls only"}
    assert (study["summary"].n_random == 3).all()
    resumed = run_refit_diagnostics(context, selection, grid_size=17, n_scans=1,
                                    integration=_integration(context))
    assert len(calls) == 6
    pd.testing.assert_frame_equal(study["refits"], resumed["refits"], check_dtype=False, atol=1e-14)
    pd.testing.assert_frame_equal(study["scans"], resumed["scans"], check_dtype=False, atol=1e-14)


def test_nested_bin_refinement_and_independent_rates_are_frozen(refit_context):
    context, _ = refit_context
    original_rates = context["rates"].copy()
    independent = _integration(context, scale=1.15)
    for mu in (0., 1.4):
        np.testing.assert_allclose(independent["templates"][36, mu].reshape(4, 12, 3).sum(axis=2),
                                   independent["templates"][12, mu])
    selection = select_refit_toys(context["source"], n_random=3)
    study = run_refit_diagnostics(context, selection, grid_size=17, n_scans=0, integration=independent)
    np.testing.assert_array_equal(context["rates"], original_rates)
    np.testing.assert_array_equal(context["learned_rates"], original_rates)
    assert set(study["refits"].variant) == {"original", "dense", "wide", "bins36", "integrated12", "integrated36", "analytic_integrated"}
    # At eta=1.4 every physical cell lands in a separate fine bin. Updating
    # the same rate scale must therefore agree for analytical and binned
    # descriptions. At eta=0 saturation merges cells, so no such identity is
    # assumed there.
    compared = study["refits"].pivot(index=["mu_test", "toy_id"], columns=["case", "variant"], values="q")
    np.testing.assert_allclose(compared.loc[1.4, ("analytic_simulator", "analytic_integrated")],
                               compared.loc[1.4, ("binned_simulator", "integrated36")], atol=1e-6)
    assert np.any(np.abs(compared["analytic_simulator", "analytic_integrated"]
                         - compared["analytic_simulator", "wide"]) > .01)


def test_checkpoint_rejects_changed_scientific_inputs(refit_context):
    context, _ = refit_context
    selection = select_refit_toys(context["source"], n_random=1, max_flagged=0)
    run_refit_diagnostics(context, selection, grid_size=17, n_scans=0)
    with pytest.raises(ValueError, match="different inputs"):
        run_refit_diagnostics(context, selection, grid_size=19, n_scans=0)
    with pytest.raises(ValueError, match="different inputs"):
        run_refit_diagnostics(context, selection, grid_size=17, n_scans=0, integration=_integration(context))


def test_learned_failures_are_retained_without_discarding_other_fits(refit_context):
    context, _ = refit_context
    def broken_evaluate(x):
        raise ValueError("synthetic failed learned model")
    context["workspace"].evaluate = broken_evaluate
    selection = select_refit_toys(context["source"], n_random=1, max_flagged=0)
    study = run_refit_diagnostics(context, selection, grid_size=17, n_scans=0)
    bad = study["refits"].query("case == 'learned_simulator'")
    assert not bad.valid.any()
    assert bad.q.isna().all()
    assert bad.message.str.contains("synthetic failed learned model").all()
    assert study["refits"].query("case != 'learned_simulator'").valid.all()
    assert (study["summary"].query("case == 'learned_simulator'").n_failed == 1).all()
    resumed = run_refit_diagnostics(context, selection, grid_size=17, n_scans=0)
    pd.testing.assert_frame_equal(study["refits"], resumed["refits"], check_dtype=False)


def test_refit_figures_render_with_negative_relative_likelihoods(refit_context, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    context, _ = refit_context
    selection = select_refit_toys(context["source"], n_random=1, max_flagged=0)
    study = run_refit_diagnostics(context, selection, grid_size=17, n_scans=1,
                                  integration=_integration(context))
    figures = plot_refit_diagnostics(study, tmp_path / "figures")
    assert set(figures) == {"fit_stability", "binning_integration", "representative_scans", "random_boundary_fractions"}
    for name, fig in figures.items():
        assert (tmp_path / "figures" / f"12_{name}.pdf").stat().st_size > 1000
        plt.close(fig)
