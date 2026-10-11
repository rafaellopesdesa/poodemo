"""Frozen per-toy reference, free denominator, and authenticated reuse."""
import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

from test_toy_convergence_refits import experiment
from poodemo.toy_convergence_refits import run_convergence_refits
from poodemo.toy_integration_diagnostics import diagnostic_bin_edges
from poodemo.toy_likelihood import StatOnlyLikelihood
from poodemo.toy_mle_reference import (mle_reference_observable, _bank_templates,
                                       _fit_branches, run_mle_reference_diagnostics)
from poodemo.toy_study import reference_observable


def test_general_mapping_fixed_reference_equal_reference_and_scaling():
    fields = np.array([[2., 1., .3], [4., 5., .7], [3., 4., .5], [2., 1., 3.]])
    rates = fields.sum(axis=1)
    for eta in (0., .3, 1., 1.4):
        np.testing.assert_array_equal(mle_reference_observable(fields, rates, eta, 1., 100.),
                                      reference_observable(fields, rates, eta, 100.))
        np.testing.assert_array_equal(mle_reference_observable(fields, rates, eta, eta, 100.), np.full(3, .5))
    for anchor in (0., .23, 1.7):
        result = mle_reference_observable(fields, rates, 1.4, anchor, 100.)
        np.testing.assert_allclose(mle_reference_observable(fields * np.array([.2, 17., 5.]), rates, 1.4, anchor, 100.),
                                   result, atol=2e-15)
        assert np.isfinite(result).all() and (result >= 0).all() and (result <= 1).all()
    close = mle_reference_observable(fields, rates, .23 + 1e-13, .23, 100.)
    assert np.isfinite(close).all()
    np.testing.assert_allclose(close, .5, atol=1e-10)


def test_equal_reference_templates_are_exact_yield_only_and_can_have_positive_q():
    # Five events in A favor kappa=0 through shape, but the total count alone
    # favors a larger intensity. Equal-reference compression discards that shape.
    fields = np.array([[4., 1.], [2., 4.], [2., 2.], [0., 0.]])
    rates = fields.sum(axis=1)
    full = StatOnlyLikelihood(fields, rates, [5., 0.])
    afit = _fit_branches(full, 0., (0., 3.), 1., 65)
    assert afit["valid"] and afit["mu_hat"] == 0.
    edges = {n: diagnostic_bin_edges(n) for n in (12, 36)}
    bank = dict(fields=fields * 100, rates=rates, normalization_divisor=100,
                metadata=dict(generated_per_source=100))
    templates, occupancy = _bank_templates(bank, rates, 0., 0., 100., edges)
    for n in edges:
        np.testing.assert_array_equal(templates[n].sum(axis=1), rates)
        assert np.count_nonzero(templates[n].sum(axis=0)) == 1
        assert np.count_nonzero(occupancy[n]) == 1
        counts = np.histogram(np.full(5, .5), edges[n])[0]
        binned = StatOnlyLikelihood.from_binned(templates[n], counts)
        fit = _fit_branches(binned, 0., (0., 3.), 1., 65, extra_mu=(0.,))
        yield_fit = _fit_branches(StatOnlyLikelihood.from_binned(rates[:, None], [5.]), 0., (0., 3.), 1., 65)
        assert fit["valid"] and fit["q"] > .1
        np.testing.assert_allclose(fit["q"], yield_fit["q"], atol=1e-12)
        assert binned.nll(0., reference_mu=1.) - binned.nll(afit["mu_hat"], reference_mu=1.) == 0.


@pytest.fixture
def completed_reference(experiment, monkeypatch):
    import poodemo.toy_likelihood_decomposition as decomposition
    context, selection, convergence, calls = experiment
    source = run_convergence_refits(context, selection, convergence, grid_size=17)
    calls.clear()
    monkeypatch.setattr(decomposition, "load_completed_convergence_banks", lambda *args, **kwargs: convergence)
    banks = {}
    for (replica, size), old in convergence["banks"].items():
        scale = old["rates"][0] / context["rates"][0]
        banks[replica, size] = dict(
            fields=context["quad"]["nominal"] * scale * size, rates=old["rates"].copy(),
            normalization_divisor=size,
            metadata=dict(fingerprint=f"reference-{replica}-{size}", source_fingerprint=context["fingerprint"],
                          convergence_bank_fingerprint=old["metadata"]["fingerprint"],
                          replica=replica, generated_per_source=size))
    reference = dict(banks=banks, metadata=dict(fingerprint="reference-fixture", configuration=dict(largest_prefixes=2)))
    return context, selection, reference, calls, source


def test_saved_events_controls_free_fits_and_exact_decomposition(completed_reference):
    context, selection, reference, calls, _ = completed_reference
    rates = context["rates"].copy()
    study = run_mle_reference_diagnostics(context, reference, grid_size=17, n_scans=1)
    records, paired, summary = study["records"], study["paired"], study["summary"]
    assert len(calls) == len(selection)
    assert len(records) == len(selection) * len(reference["banks"]) * 5
    assert records.valid.all()
    assert records.count_reproduced.all()
    assert records.saved_fit_reproduced[records.saved_fit_available].all()
    assert (records.groupby(["mu_test", "toy_id"]).event_sha256.nunique() == 1).all()
    assert records.empty_observed_bins.eq(0).all()
    np.testing.assert_array_equal(context["rates"], rates)
    assert paired.paired_valid.all()
    np.testing.assert_allclose(paired.delta_q, paired.delta_pair + paired.refit_gain, atol=1e-12)
    assert (paired.refit_gain >= -1e-10).all()
    adaptive = records.loc[records.model.str.startswith("adaptive")]
    np.testing.assert_array_equal(adaptive.mu_reference, adaptive.analytic_mu_hat)
    np.testing.assert_array_equal(adaptive.exact_degenerate, adaptive.adaptive_exact_degenerate)
    assert records.query("model == 'fixed12'").mu_reference.eq(1.).all()
    for _, group in adaptive.groupby(["mu_test", "toy_id", "replica", "generated_per_source"]):
        assert group.mu_reference.nunique() == 1
    assert summary.query("stratum == 'all'").n_random.eq(3).all()
    assert set(summary.stratum) == {"all", "exact_degenerate", "nondegenerate"}
    assert set(study["changes"].comparison_type) == {"prefix", "replica"}
    assert "mu_reference" in set(study["changes"].term)
    assert len(study["scan_selection"]) == 2
    exact = adaptive.loc[adaptive.exact_degenerate]
    assert exact.q_pair.eq(0.).all()
    if len(exact):
        assert exact.yield_only_q_error.abs().max() < 1e-10
    for key, group in study["scans"].groupby(["mu_test", "toy_id", "replica", "generated_per_source", "model"]):
        assert group.mu_reference.nunique() == 1


def test_resume_authenticates_fields_and_preserves_previous_outputs(completed_reference):
    context, selection, reference, calls, _ = completed_reference
    directory = context["output_dir"] / "convergence/bank_convergence/refits"
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir() if path.is_file()}
    study = run_mle_reference_diagnostics(context, reference, grid_size=17, n_scans=0)
    again = run_mle_reference_diagnostics(context, reference, grid_size=17, n_scans=0)
    assert len(calls) == len(selection)
    pd.testing.assert_frame_equal(study["records"], again["records"], check_dtype=False)
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir() if path.is_file()}
    assert after == before
    changed = copy.deepcopy(reference)
    fields = changed["banks"][0, 100]["fields"]
    shift = fields[:, 0] * .01
    fields[:, 0] -= shift
    fields[:, 1] += shift
    with pytest.raises(ValueError, match="inputs changed"):
        run_mle_reference_diagnostics(context, changed, grid_size=17, n_scans=0)
    with pytest.raises(ValueError, match="inputs changed"):
        run_mle_reference_diagnostics(context, reference, grid_size=19, n_scans=0)


def test_resume_recovers_partial_scan_without_resampling_other_toys(completed_reference):
    context, selection, reference, calls, _ = completed_reference
    study = run_mle_reference_diagnostics(context, reference, grid_size=17, n_scans=1)
    path = context["output_dir"] / "convergence/bank_convergence/mle_reference/mle_reference/refits/scans.csv"
    scans = pd.read_csv(path, float_precision="round_trip")
    scans.drop(index=0).to_csv(path, index=False)
    again = run_mle_reference_diagnostics(context, reference, grid_size=17, n_scans=1)
    assert len(calls) == len(selection) + 1
    pd.testing.assert_frame_equal(study["records"], again["records"], check_dtype=False)
    keys = ["mu_test", "toy_id", "replica", "generated_per_source", "model", "mu_scan"]
    pd.testing.assert_frame_equal(study["scans"].sort_values(keys).reset_index(drop=True),
                                   again["scans"].sort_values(keys).reset_index(drop=True), check_dtype=False)


def test_saved_control_mismatch_is_explicit(completed_reference):
    context, _, reference, _, _ = completed_reference
    path = context["output_dir"] / "convergence/bank_convergence/refits/results.csv"
    source = pd.read_csv(path)
    target = ((source.model == "binned12") & (source.bank_kind == "independent")
              & (source.replica == 0) & (source.generated_per_source == 100)
              & (source.mu_test == 0.) & (source.toy_id == 0))
    source.loc[target, "q"] += 1.
    source.to_csv(path, index=False)
    study = run_mle_reference_diagnostics(context, reference, grid_size=17, n_scans=0)
    failed = study["records"].loc[~study["records"].valid]
    assert len(failed) == 1
    assert failed.model.eq("fixed12").all()
    assert failed.message.str.contains("differs from saved").all()
    row = study["summary"].query("model == 'fixed12' and replica == 0 and generated_per_source == 100 and mu_test == 0 and stratum == 'all'")
    assert row.n_failed.iloc[0] == 1 and row.n_paired_failed.iloc[0] == 1


def test_observed_empty_support_fails_without_floor(completed_reference, monkeypatch):
    import poodemo.toy_mle_reference as module
    context, _, reference, _, _ = completed_reference
    original = module._bank_templates

    def unsupported(*args, **kwargs):
        templates, occupancy = original(*args, **kwargs)
        for value in templates.values():
            occupied = np.flatnonzero(value.sum(axis=0))
            if len(occupied) > 1:
                value[:, occupied[1]] += value[:, occupied[0]]
                value[:, occupied[0]] = 0.
        return templates, occupancy

    monkeypatch.setattr(module, "_bank_templates", unsupported)
    study = run_mle_reference_diagnostics(context, reference, grid_size=17, n_scans=0)
    failed = study["records"].loc[study["records"].empty_observed_bins > 0]
    assert len(failed) > 0
    assert failed.model.str.startswith("adaptive").all()
    assert not failed.valid.any()
    assert failed.message.str.contains("support|zero intensity").all()
    assert failed.q.isna().all()
    assert study["records"].query("model == 'analytic' or model == 'fixed12' or model == 'fixed36'").valid.all()


def test_unverified_analytical_anchor_invalidates_adaptive_rows(completed_reference):
    context, _, reference, _, _ = completed_reference
    path = context["output_dir"] / "convergence/bank_convergence/refits/results.csv"
    source = pd.read_csv(path)
    target = ((source.model == "analytic") & (source.bank_kind == "independent")
              & (source.replica == 0) & (source.generated_per_source == 100)
              & (source.mu_test == 0.) & (source.toy_id == 0))
    source.loc[target, "q"] += 1.
    source.to_csv(path, index=False)
    study = run_mle_reference_diagnostics(context, reference, grid_size=17, n_scans=0)
    failed = study["records"].loc[~study["records"].valid]
    assert set(failed.model) == {"analytic", "adaptive12", "adaptive36"}
    assert len(failed) == 3
    analytical = failed.loc[failed.model == "analytic"].iloc[0]
    assert abs(analytical.saved_q_error) > .9
    assert failed.loc[failed.model.str.startswith("adaptive"), "message"].str.contains("no adaptive reference").all()
    for model in ("adaptive12", "adaptive36"):
        row = study["summary"].query("model == @model and replica == 0 and generated_per_source == 100 and mu_test == 0 and stratum == 'all'")
        assert row.n_failed.iloc[0] == 1 and row.n_paired_failed.iloc[0] == 1
