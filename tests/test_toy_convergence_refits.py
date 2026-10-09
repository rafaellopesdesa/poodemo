"""Convergence bank matching, shared events, paired errors and safe resumption."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from poodemo.data import Run
from poodemo.inference import component_coefficients, histogram_components
from poodemo.toy_convergence_refits import (load_convergence_selection, plot_convergence_refits,
                                           run_convergence_refits, _tables)
from poodemo.toy_integration_diagnostics import diagnostic_bin_edges
from poodemo.toy_study import _rng, reference_observable


class CellPhysics:
    fields = np.array([[2., 1., .3], [4., 5., .7], [3., 4., .5], [2., 1., 3.]])

    def component_densities(self, x):
        return self.fields[:, x[:, 0].astype(int)].T


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    from poodemo import toy_sources
    model = CellPhysics()
    run = Run(tmp_path, {"seed": 57}, model)
    x = np.column_stack((np.arange(3), np.zeros((3, 2))))
    quad = dict(x=x, weights=np.ones(3), nominal=model.fields.copy())
    rates = model.fields.sum(axis=1)
    config = dict(mu_bounds=[0., 2.], grid_size=9, n_bins=12,
                  reference_power=100., seed=120913, exposure_multiplier=1.)
    rows, selected, calls = [], [], []

    def events(mu, rng):
        return np.repeat(x, rng.poisson(component_coefficients(mu) @ model.fields), axis=0)

    for mu in (0., 1.4):
        for toy_id in range(3):
            sample = events(mu, _rng(config["seed"], mu, toy_id, 0))
            rows.append(dict(mu_test=mu, toy_id=toy_id, case="analytic_simulator", n_observed=len(sample)))
            selected.append(dict(mu_test=mu, toy_id=toy_id, is_random=True, is_flagged=False,
                                 branch_split=1., selection_reason="random"))

    def simulate(run, mu, rng, **kwargs):
        result = events(mu, rng)
        calls.append(result.copy())
        return result

    def no_networks(*args, **kwargs):
        raise AssertionError("Convergence refits must not evaluate any network")

    monkeypatch.setattr(toy_sources, "sample_simulator_poisson", simulate)
    context = dict(run=run, quad=quad, rates=rates, configuration=config,
                   source={"results": pd.DataFrame(rows)}, selector=object(), threshold=.5,
                   workspace=SimpleNamespace(evaluate=no_networks), fingerprint="frozen-context",
                   original_n_per_source=100, output_dir=tmp_path / "diagnostics")
    banks = {}
    edges = {n: diagnostic_bin_edges(n) for n in (12, 36)}
    for replica in (0, 1):
        for size in (100, 200):
            scale = 1. + (1 + replica) * 10. / size
            templates = {}
            for mu in (0., 1.4):
                z = reference_observable(quad["nominal"], rates, mu, config["reference_power"])
                for n, bins in edges.items():
                    templates[n, mu] = scale * histogram_components(z, quad["nominal"], quad["weights"], bins)
            banks[replica, size] = dict(templates=templates, rates=scale * rates, edges=edges,
                                       metadata=dict(fingerprint=f"bank-{replica}-{size}", source_fingerprint=context["fingerprint"],
                                                     replica=replica, generated_per_source=size))
    return context, pd.DataFrame(selected), {"banks": banks, "metadata": {}}, calls


def test_same_events_no_nn_and_matched_rate_changes(experiment):
    context, selection, banks, calls = experiment
    original_rates = context["rates"].copy()
    study = run_convergence_refits(context, selection, banks, grid_size=17)
    assert len(calls) == len(selection)
    assert len(study["refits"]) == 90
    assert study["refits"].valid.all()
    assert study["refits"].count_reproduced.all()
    assert (study["refits"].groupby(["mu_test", "toy_id"]).event_sha256.nunique() == 1).all()
    np.testing.assert_array_equal(context["rates"], original_rates)
    # Three physical cells occupy distinct fine bins at eta=1.4. With a
    # common bank normalization their likelihood ratios must agree exactly.
    fine = study["paired"].query("mu_test == 1.4 and n_bins == 36")
    np.testing.assert_allclose(fine.q, fine.analytic_q, atol=1e-6)
    analytic = study["refits"].query("model == 'analytic'")
    assert (analytic.groupby(["mu_test", "toy_id"]).q.max()
            - analytic.groupby(["mu_test", "toy_id"]).q.min()).max() > .01
    assert study["summary"].n_random.eq(3).all()
    assert study["summary"].n_failed.eq(0).all()
    assert np.isfinite(study["summary"].mean_q_se).all()
    assert {"prefix", "replica", "original"} == set(study["convergence"].comparison_type)
    assert np.isfinite(study["convergence"].paired_se).all()


def test_resume_and_reject_changed_actual_bank_values(experiment):
    context, selection, banks, calls = experiment
    study = run_convergence_refits(context, selection, banks, grid_size=17)
    again = run_convergence_refits(context, selection, banks, grid_size=17)
    assert len(calls) == len(selection)
    pd.testing.assert_frame_equal(study["refits"], again["refits"], check_dtype=False)
    altered = copy.deepcopy(banks)
    bank = altered["banks"][0, 100]
    bank["rates"] *= 1.01
    bank["templates"] = {key: value * 1.01 for key, value in bank["templates"].items()}
    # A claimed metadata fingerprint cannot disguise changed numerical inputs.
    with pytest.raises(ValueError, match="different inputs"):
        run_convergence_refits(context, selection, altered, grid_size=17)
    with pytest.raises(ValueError, match="different inputs"):
        run_convergence_refits(context, selection, banks, grid_size=19)
    assert not (context["output_dir"] / "refit_results.csv").exists()


def test_bad_bank_matching_and_frozen_edges_are_rejected(experiment):
    context, selection, banks, _ = experiment
    altered = copy.deepcopy(banks)
    altered["banks"][0, 100]["rates"][0] *= 1.1
    with pytest.raises(ValueError, match="same bank"):
        run_convergence_refits(context, selection, altered, grid_size=17)
    altered = copy.deepcopy(banks)
    altered["banks"][0, 100]["edges"][12][1] += 1e-5
    with pytest.raises(ValueError, match="frozen bin edges"):
        run_convergence_refits(context, selection, altered, grid_size=17)


def test_partial_toy_resume_rejects_changed_coordinates(experiment, monkeypatch):
    from poodemo import toy_sources
    context, selection, banks, _ = experiment
    study = run_convergence_refits(context, selection, banks, grid_size=17)
    checkpoint = context["output_dir"] / "convergence/bank_convergence/refits/results.csv"
    frame = pd.read_csv(checkpoint)
    # Leave a partial toy with its original event digest, then change event
    # coordinates without changing the number of observed events.
    frame.drop(index=0).to_csv(checkpoint, index=False)
    original = toy_sources.sample_simulator_poisson

    def changed_coordinates(*args, **kwargs):
        x = original(*args, **kwargs)
        x[:, 1] += .25
        return x

    monkeypatch.setattr(toy_sources, "sample_simulator_poisson", changed_coordinates)
    with pytest.raises(ValueError, match="event coordinates differ"):
        run_convergence_refits(context, selection, banks, grid_size=17)
    assert len(pd.read_csv(checkpoint)) == len(study["refits"]) - 1


def test_reconstruction_failure_remains_counted_and_resumes(experiment, monkeypatch):
    from poodemo import toy_sources
    context, selection, banks, calls = experiment
    original = toy_sources.sample_simulator_poisson

    def wrong_count(*args, **kwargs):
        return original(*args, **kwargs)[:0]

    monkeypatch.setattr(toy_sources, "sample_simulator_poisson", wrong_count)
    study = run_convergence_refits(context, selection, banks, grid_size=17)
    assert not study["refits"].valid.any()
    assert not study["refits"].count_reproduced.any()
    assert study["refits"].message.str.contains("does not reproduce").all()
    assert study["summary"].n_failed.eq(3).all()
    assert study["convergence"].n_failed.eq(3).all()
    assert study["paired"].delta_q.isna().all()
    again = run_convergence_refits(context, selection, banks, grid_size=17)
    assert len(calls) == len(selection)
    pd.testing.assert_frame_equal(study["refits"], again["refits"], check_dtype=False)


def test_paired_error_uses_differences_and_excludes_flag_only(experiment):
    context, selection, banks, _ = experiment
    study = run_convergence_refits(context, selection, banks, grid_size=17)
    frame = study["refits"].copy()
    # Strongly correlated q values: uncertainty in their difference is tiny
    # compared with either marginal error and must be computed on paired IDs.
    frame["q"] = 100 * (frame.toy_id + 1.)
    frame.loc[frame.model == "binned12", "q"] += frame.loc[frame.model == "binned12", "toy_id"]
    frame.loc[frame.model == "binned36", "q"] += 2 * frame.loc[frame.model == "binned36", "toy_id"]
    extra = frame.loc[frame.toy_id == 0].assign(toy_id=99, is_random=False, is_flagged=True, q=1e9)
    _, summary, _ = _tables(pd.concat([frame, extra], ignore_index=True))
    rows = summary.query("model == 'binned12'")
    np.testing.assert_allclose(rows.mean_delta_q, 1.)
    np.testing.assert_allclose(rows.mean_delta_q_se, 1 / np.sqrt(3))
    assert rows.n_random.eq(3).all()
    # Drop one analytical fit: its binned partner must be excluded as well.
    frame.loc[(frame.model == "analytic") & (frame.toy_id == 2), "valid"] = False
    _, summary, changes = _tables(frame)
    rows = summary.query("model == 'binned12'")
    assert rows.n_paired_valid.eq(2).all()
    assert rows.n_paired_failed.eq(1).all()
    np.testing.assert_allclose(rows.mean_delta_q, .5)
    np.testing.assert_allclose(rows.mean_delta_q_se, .5)
    assert changes.query("metric == 'gap'").n_failed.eq(1).all()


def test_load_saved_selection_checks_original_file_bytes(experiment):
    context, selection, _, _ = experiment
    output = context["output_dir"]
    output.mkdir()
    path = output / "refit_selection.csv"
    selection.to_csv(path, index=False)
    identity = dict(context_fingerprint=context["fingerprint"],
                    selection_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output / "refit_config.json").write_text(json.dumps({"configuration": identity}))
    pd.testing.assert_frame_equal(load_convergence_selection(context), selection)
    path.write_text(path.read_text().replace("random", "altered"))
    with pytest.raises(ValueError, match="fingerprint"):
        load_convergence_selection(context)


def test_convergence_figures_render(experiment, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    context, selection, banks, _ = experiment
    study = run_convergence_refits(context, selection, banks, grid_size=17)
    figures = plot_convergence_refits(study, tmp_path / "plots")
    assert len(figures) == 5
    for name, fig in figures.items():
        assert (tmp_path / "plots" / f"12_{name}.pdf").stat().st_size > 1000
        plt.close(fig)
