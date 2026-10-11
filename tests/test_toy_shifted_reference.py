"""Endpoint ablations retain physical tests, paired controls and authenticated toys."""
import copy
import hashlib

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from test_toy_convergence_refits import experiment
from test_toy_mle_reference import completed_reference
from poodemo.inference import component_coefficients
from poodemo.toy_integration_diagnostics import diagnostic_bin_edges
from poodemo.toy_likelihood import StatOnlyLikelihood
from poodemo.toy_mle_reference import _bank_templates, mle_reference_observable
from poodemo.toy_shifted_reference import (
    _contrasts, run_shifted_reference_diagnostics, shifted_endpoints,
)
from poodemo.toy_study import _rng, reference_observable


def test_endpoint_mapping_normalizers_and_actual_importance_histograms():
    fields = np.array([[2., 1., .3], [4., 5., .7], [3., 4., .5], [2., 1., 3.]])
    original_rates = fields.sum(axis=1)
    point_scaling = np.array([2., 7., .1])
    bank = dict(fields=fields * point_scaling, rates=(fields * point_scaling).sum(axis=1) / 17,
                normalization_divisor=17, metadata=dict(generated_per_source=17))
    edges = {n: diagnostic_bin_edges(n) for n in (12, 36)}
    for eta in (0., 1.4):
        for variant, (numerator, denominator) in shifted_endpoints(eta).items():
            ca, cb = component_coefficients(numerator), component_coefficients(denominator)
            direct = expit(100 * np.log((ca @ fields) / (cb @ fields)
                                       * (cb @ original_rates) / (ca @ original_rates)))
            result = mle_reference_observable(fields, original_rates, numerator, denominator, 100.)
            np.testing.assert_allclose(result, direct, atol=3e-15)
            np.testing.assert_allclose(mle_reference_observable(bank["fields"], original_rates, numerator,
                                                               denominator, 100.), result, atol=3e-15)
            if variant == "baseline":
                np.testing.assert_array_equal(result, reference_observable(fields, original_rates, eta, 100.))
            templates, occupancy = _bank_templates(bank, original_rates, numerator, denominator, 100., edges)
            for n, bins in edges.items():
                expected = np.stack([np.histogram(result, bins, weights=row)[0] for row in bank["fields"]]) / 17
                np.testing.assert_allclose(templates[n], expected, atol=1e-14)
                np.testing.assert_allclose(templates[n].sum(axis=1), bank["rates"], atol=1e-14)
                np.testing.assert_array_equal(occupancy[n], np.histogram(result, bins)[0])
    assert shifted_endpoints(0.)["denominator"] == (0., 1.1)
    with pytest.raises(ValueError, match="nonnegative"):
        shifted_endpoints(0., -.1)


def test_physical_hypotheses_free_fits_decomposition_and_template_reuse(completed_reference, monkeypatch):
    import poodemo.toy_shifted_reference as module
    context, selection, reference, calls, _ = completed_reference
    original = module._bank_templates
    template_calls = []

    def counted(bank, rates, numerator, denominator, power, edges):
        template_calls.append((bank["metadata"]["replica"], bank["metadata"]["generated_per_source"], numerator, denominator))
        return original(bank, rates, numerator, denominator, power, edges)

    monkeypatch.setattr(module, "_bank_templates", counted)
    rates = context["rates"].copy()
    study = run_shifted_reference_diagnostics(context, reference, grid_size=17, n_scans=1)
    records, paired = study["records"], study["paired"]
    assert len(calls) == len(selection)
    assert len(template_calls) == len(reference["banks"]) * selection.mu_test.nunique() * 4
    assert len(records) == len(selection) * len(reference["banks"]) * 9
    assert records.valid.all() and paired.paired_valid.all()
    assert records.saved_fit_reproduced[records.saved_fit_available].all()
    assert (records.groupby(["mu_test", "toy_id"]).event_sha256.nunique() == 1).all()
    assert records.count_reproduced.all() and records.empty_observed_bins.eq(0).all()
    np.testing.assert_array_equal(context["rates"], rates)
    np.testing.assert_allclose(paired.delta_q, paired.delta_pair + paired.refit_gain, atol=1e-12)
    assert (paired.refit_gain >= -1e-10).all()
    assert (records.D_eta >= records[["D_low", "D_high"]].min(axis=1) - 1e-10).all()
    assert np.isfinite(records.D_low).all() and np.isfinite(records.D_high).all()
    # Independently reconstruct a numerator-shifted likelihood: the tested
    # parameter in D_eta must remain physical eta, not the map's eta+epsilon.
    row = records.query("mu_test == 0 and model == 'both36' and replica == 0 and generated_per_source == 100").iloc[0]
    x = calls[int(row.toy_id)]
    fields = context["run"].model.component_densities(x).T
    bins = {36: diagnostic_bin_edges(36)}
    bank = reference["banks"][0, 100]
    template, _ = original(bank, rates, row.mu_numerator, row.mu_reference, 100., bins)
    z = mle_reference_observable(fields, rates, row.mu_numerator, row.mu_reference, 100.)
    likelihood = StatOnlyLikelihood.from_binned(template[36], np.histogram(z, bins[36])[0])
    assert row.mu_numerator == .1 and row.mu_reference == 1.1 and row.mu_test == 0.
    assert row.D_eta == pytest.approx(likelihood.nll(0., reference_mu=1.), abs=1e-12)
    assert abs(row.D_eta - likelihood.nll(.1, reference_mu=1.)) > 1e-5
    assert study["summary"].query("stratum == 'all'").n_random.eq(3).all()
    assert set(study["summary"].stratum) == {"all", "analytical_at_test", "analytical_away_from_test"}
    assert {"prefix", "replica"} == set(study["changes"].comparison_type)
    assert "squared_delta_q" in set(study["contrasts"].term)
    for _, group in study["scans"].groupby(["mu_test", "toy_id", "replica", "generated_per_source", "model"]):
        assert group.mu_numerator.nunique() == 1 and group.mu_reference.nunique() == 1


def test_resume_authentication_and_preservation(completed_reference):
    context, selection, reference, calls, _ = completed_reference
    directory = context["output_dir"] / "convergence/bank_convergence/refits"
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir() if path.is_file()}
    study = run_shifted_reference_diagnostics(context, reference, grid_size=17, n_scans=0)
    again = run_shifted_reference_diagnostics(context, reference, grid_size=17, n_scans=0)
    assert len(calls) == len(selection)
    pd.testing.assert_frame_equal(study["records"], again["records"], check_dtype=False)
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir() if path.is_file()}
    assert before == after
    altered = copy.deepcopy(reference)
    fields = altered["banks"][0, 100]["fields"]
    change = fields[:, 0] * .01
    fields[:, 0] -= change
    fields[:, 1] += change
    with pytest.raises(ValueError, match="inputs changed"):
        run_shifted_reference_diagnostics(context, altered, grid_size=17, n_scans=0)
    with pytest.raises(ValueError, match="inputs changed"):
        run_shifted_reference_diagnostics(context, reference, numerator_shift=.2, grid_size=17, n_scans=0)
    with pytest.raises(ValueError, match="inputs changed"):
        run_shifted_reference_diagnostics(context, reference, denominator_shift=.2, grid_size=17, n_scans=0)


def test_partial_scan_repaired_and_changed_coordinates_rejected(completed_reference, monkeypatch):
    from poodemo import toy_sources
    context, selection, reference, calls, _ = completed_reference
    study = run_shifted_reference_diagnostics(context, reference, grid_size=17, n_scans=1)
    directory = context["output_dir"] / "convergence/bank_convergence/shifted_reference/shifted_reference"
    path = directory / "scans.csv"
    scans = pd.read_csv(path, float_precision="round_trip")
    scans.drop(index=0).to_csv(path, index=False)
    again = run_shifted_reference_diagnostics(context, reference, grid_size=17, n_scans=1)
    assert len(calls) == len(selection) + 1
    pd.testing.assert_frame_equal(study["records"], again["records"], check_dtype=False)
    keys = ["mu_test", "toy_id", "replica", "generated_per_source", "model", "mu_scan"]
    pd.testing.assert_frame_equal(study["scans"].sort_values(keys).reset_index(drop=True),
                                  again["scans"].sort_values(keys).reset_index(drop=True), check_dtype=False)
    records = pd.read_csv(directory / "records.csv", float_precision="round_trip")
    records.drop(index=0).to_csv(directory / "records.csv", index=False)
    original = toy_sources.sample_simulator_poisson

    def changed(*args, **kwargs):
        x = original(*args, **kwargs)
        x[:, 1] += .25
        return x

    monkeypatch.setattr(toy_sources, "sample_simulator_poisson", changed)
    with pytest.raises(ValueError, match="counts and coordinates"):
        run_shifted_reference_diagnostics(context, reference, grid_size=17, n_scans=1)
    assert len(pd.read_csv(directory / "records.csv")) == len(records) - 1


def test_paired_improvement_uses_common_controls_and_exposes_ambiguous_branches(completed_reference):
    context, _, reference, _, _ = completed_reference
    study = run_shifted_reference_diagnostics(context, reference, grid_size=17, n_scans=0)
    pair = study["paired"].copy()
    pair["delta_q"] = np.where(pair.variant == "baseline", pair.toy_id + 1., 1.)
    pair["paired_valid"] = True
    pair["branch_comparable"] = pair.toy_id != 2
    pair["branch_disagreement"] = np.where(pair.variant == "baseline", 1., 0.)
    # Strongly different flagged-only examples may not affect population metrics.
    extra = pair.loc[pair.toy_id == 0].assign(toy_id=99, is_random=False, is_flagged=True, delta_q=1e9)
    contrasts = _contrasts(pd.concat([pair, extra], ignore_index=True)).query("stratum == 'all'")
    squared = contrasts.query("term == 'squared_delta_q'")
    np.testing.assert_allclose(squared.mean_change, np.mean([0., -3., -8.]))
    np.testing.assert_allclose(squared.paired_se, np.std([0., -3., -8.], ddof=1) / np.sqrt(3))
    assert squared.n_valid.eq(3).all()
    branch = contrasts.query("term == 'branch_disagreement'")
    assert branch.n_valid.eq(2).all() and branch.n_branch_ambiguous.eq(1).all()
    assert branch.n_paired_failed.eq(0).all() and branch.mean_change.eq(-1.).all()
    # A failed baseline excludes the same toy from every comparison, even if
    # the corresponding shifted fit was valid.
    pair.loc[(pair.variant == "baseline") & (pair.toy_id == 1), "paired_valid"] = False
    failed = _contrasts(pair).query("stratum == 'all' and term == 'squared_delta_q'")
    assert failed.n_valid.eq(2).all() and failed.n_paired_failed.eq(1).all()
    np.testing.assert_allclose(failed.mean_change, -4.)


def test_saved_baseline_mismatch_and_unverified_analytical_anchor(completed_reference):
    context, _, reference, _, _ = completed_reference
    path = context["output_dir"] / "convergence/bank_convergence/refits/results.csv"
    source = pd.read_csv(path)
    target = ((source.model == "binned12") & (source.bank_kind == "independent") & (source.replica == 0)
              & (source.generated_per_source == 100) & (source.mu_test == 0.) & (source.toy_id == 0))
    source.loc[target, "q"] += 1.
    source.to_csv(path, index=False)
    study = run_shifted_reference_diagnostics(context, reference, grid_size=17, n_scans=0)
    failed = study["records"].loc[~study["records"].valid]
    assert len(failed) == 1 and failed.model.eq("baseline12").all()
    assert failed.message.str.contains("differs from saved").all()
    source.loc[target, "q"] -= 1.
    target = target | ((source.model == "analytic") & (source.bank_kind == "independent") & (source.replica == 0)
                       & (source.generated_per_source == 100) & (source.mu_test == 0.) & (source.toy_id == 0))
    source.loc[target, "q"] += 1.
    source.to_csv(path, index=False)
    study = run_shifted_reference_diagnostics(context, reference, shift_tag="invalid_anchor", grid_size=17, n_scans=0)
    failed = study["records"].loc[~study["records"].valid]
    assert len(failed) == 9
    assert not study["paired"].query("replica == 0 and generated_per_source == 100 and mu_test == 0 and toy_id == 0").paired_valid.any()


def test_observed_empty_support_is_retained_without_floor(completed_reference, monkeypatch):
    import poodemo.toy_shifted_reference as module
    context, _, reference, _, _ = completed_reference
    original = module._bank_templates

    def unsupported(bank, rates, numerator, denominator, power, edges):
        templates, occupancy = original(bank, rates, numerator, denominator, power, edges)
        if denominator != 1.:
            for value in templates.values():
                occupied = np.flatnonzero(value.sum(axis=0))
                if len(occupied) > 1:
                    value[:, occupied[1]] += value[:, occupied[0]]
                    value[:, occupied[0]] = 0.
        return templates, occupancy

    monkeypatch.setattr(module, "_bank_templates", unsupported)
    study = run_shifted_reference_diagnostics(context, reference, grid_size=17, n_scans=0)
    failed = study["records"].query("empty_observed_bins > 0")
    assert len(failed) > 0 and not failed.valid.any()
    assert set(failed.variant) <= {"denominator", "both"}
    assert failed.q.isna().all()
    assert study["records"].query("variant == 'baseline' or model == 'analytic'").valid.all()
