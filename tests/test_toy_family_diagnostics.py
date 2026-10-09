"""Whole-family controls, branch competition and authenticated toy reuse."""
import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

from test_toy_convergence_refits import experiment
from poodemo.toy_convergence_refits import run_convergence_refits
from poodemo.toy_likelihood import StatOnlyLikelihood


def test_unbinned_family_coordinates_preserve_full_likelihood():
    from poodemo.toy_family_diagnostics import _family_fields

    # Deliberately different baselines: the cancellation must work per event,
    # rather than relying on an overall constant shared by all events.
    fields = np.array([[2., 1., .3], [4., 5., .7], [3., 4., .5], [2., 1., 3.]])
    rates = np.array([7., 13., 12., 11.])
    counts = np.array([3., 1., 5.])
    analytical = StatOnlyLikelihood(fields, rates, counts, exposure=1.7)
    family = StatOnlyLikelihood(_family_fields(fields), rates, counts, exposure=1.7)
    hypotheses = np.r_[0., np.geomspace(1e-8, 3., 80)]
    for mu in hypotheses:
        np.testing.assert_allclose(family.nll(mu, reference_mu=1.),
                                   analytical.nll(mu, reference_mu=1.), atol=1e-12)
        assert family.expected_count(mu) == analytical.expected_count(mu)
    for mu in (0., 1.4):
        expected = analytical.test_statistic(mu, mu_bounds=(0., 3.), grid_size=65)
        actual = family.test_statistic(mu, mu_bounds=(0., 3.), grid_size=65)
        np.testing.assert_allclose(actual["q"], expected["q"], atol=1e-10)
        np.testing.assert_allclose(actual["mu_hat"], expected["mu_hat"], atol=1e-6)


def test_branch_gap_sign_and_nested_fit_identity():
    from poodemo.toy_family_diagnostics import _fit_branches

    # lambda(mu)=4 mu + 1 gives known MLEs: mu=.25 at n=2 and mu=2.75
    # at n=12. Positive G means the low branch wins; negative G means high.
    fields = np.array([[4.], [4.], [0.], [1.]])
    for count, expected_mu, high in ((2, .25, False), (12, 2.75, True)):
        likelihood = StatOnlyLikelihood.from_binned(fields, np.array([count]))
        fit = _fit_branches(likelihood, 1.4, (0., 3.), 1., 65)
        assert fit["valid"]
        assert (fit["G"] < 0) == high
        assert not fit["ambiguous_branch"]
        assert fit["low_at_split"] == high
        assert fit["high_at_split"] != high
        assert not fit["two_interior_minima"]
        np.testing.assert_allclose(fit["mu_hat"], expected_mu, atol=1e-5)
        np.testing.assert_allclose(fit["G"], fit["D_high"] - fit["D_low"], atol=1e-12)
        np.testing.assert_allclose(fit["q"], fit["D_eta"] - min(fit["D_low"], fit["D_high"]), atol=1e-10)


def test_shared_branch_boundary_is_reported_as_ambiguous():
    from poodemo.toy_family_diagnostics import _fit_branches

    fields = np.array([[4.], [4.], [0.], [1.]])
    likelihood = StatOnlyLikelihood.from_binned(fields, np.array([5.]))
    fit = _fit_branches(likelihood, 1.4, (0., 3.), 1., 65)
    assert fit["valid"]
    assert fit["ambiguous_branch"]
    np.testing.assert_allclose(fit["G"], 0., atol=1e-9)
    np.testing.assert_allclose([fit["mu_low"], fit["mu_high"]], 1., atol=1e-6)


@pytest.fixture
def completed_family(experiment, monkeypatch):
    """Real saved section-6 rows plus analytically exact small 2D templates."""
    import poodemo.toy_likelihood_decomposition as decomposition
    from poodemo.toy_family_banks import family_bin_indices

    context, selection, convergence, calls = experiment
    source = run_convergence_refits(context, selection, convergence, grid_size=17)
    calls.clear()
    monkeypatch.setattr(decomposition, "load_completed_convergence_banks",
                        lambda *args, **kwargs: convergence)
    edges = {n: (np.r_[-np.inf, np.linspace(-.3, .1, n - 1), np.inf],
                 np.r_[-np.inf, np.linspace(0., .5, n - 1), np.inf]) for n in (12, 36)}
    fields = context["quad"]["nominal"]
    banks = {}
    for (replica, size), old in convergence["banks"].items():
        scale = old["rates"][0] / context["rates"][0]
        templates, occupancy = {}, {}
        for n, boundary in edges.items():
            indices = family_bin_indices(fields, boundary)
            occupancy[n] = np.bincount(indices, minlength=n*n)
            templates[n] = scale * np.stack([
                np.bincount(indices, weights=component, minlength=n*n) for component in fields])
        banks[replica, size] = dict(
            templates=templates, occupancy=occupancy, edges=edges, rates=old["rates"].copy(),
            component_covariances={n: np.zeros((4, 4, n*n)) for n in edges},
            rate_covariance=np.zeros((4, 4)),
            metadata=dict(fingerprint=f"family-{replica}-{size}",
                          source_fingerprint=context["fingerprint"],
                          convergence_bank_fingerprint=old["metadata"]["fingerprint"],
                          replica=replica, generated_per_source=size, selected=3))
    family = dict(banks=banks, convergence_banks=convergence["banks"],
                  closure=pd.DataFrame(), support=pd.DataFrame(),
                  metadata=dict(fingerprint="family-fixture", configuration=dict(
                      axis_bins=[12, 36], largest_prefixes=2)))
    return context, selection, family, calls, source


def test_same_saved_events_exact_2d_control_and_pairing(completed_family):
    from poodemo.toy_family_diagnostics import run_family_diagnostics

    context, selection, family, calls, _ = completed_family
    original_rates = context["rates"].copy()
    study = run_family_diagnostics(context, family, grid_size=17, n_scans=1)
    records, paired = study["records"], study["paired"]
    assert len(calls) == len(selection)
    assert len(records) == len(selection) * len(family["banks"]) * 5
    assert records.valid.all()
    assert records.count_reproduced.all()
    assert records.empty_observed_bins.eq(0).all()
    assert records.saved_fit_reproduced[records.saved_fit_available].all()
    assert (records.groupby(["mu_test", "toy_id"]).event_sha256.nunique() == 1).all()
    assert records.family_identity_max_abs.max() < 1e-10
    np.testing.assert_array_equal(context["rates"], original_rates)
    full_family = paired.loc[paired.model.str.startswith("family")]
    assert full_family.paired_valid.all()
    # Empty unobserved bins are legitimate; the nonempty cells retain the
    # whole family exactly, not just one fixed-pair likelihood ratio.
    np.testing.assert_allclose(full_family.delta_q, 0., atol=1e-7)
    np.testing.assert_allclose(full_family.delta_G, 0., atol=1e-7)
    np.testing.assert_allclose(full_family.delta_common, 0., atol=1e-7)
    np.testing.assert_allclose(paired.delta_G, paired.delta_common + paired.delta_refit, atol=1e-12)
    assert study["summary"].n_random.eq(3).all()
    assert study["summary"].n_failed.eq(0).all()
    assert set(study["changes"].comparison_type) == {"prefix", "replica"}
    assert len(study["scan_selection"]) == 2
    for _, group in study["scans"].groupby(["mu_test", "toy_id", "replica", "generated_per_source", "mu_scan"]):
        analytical = group.loc[group.model == "analytic", "D"].iloc[0]
        np.testing.assert_allclose(group.loc[group.model.str.startswith("family"), "D"], analytical, atol=1e-10)


def test_resume_preserves_source_and_authenticates_numerical_inputs(completed_family):
    from poodemo.toy_family_diagnostics import run_family_diagnostics

    context, selection, family, calls, _ = completed_family
    directory = context["output_dir"] / "convergence/bank_convergence/refits"
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir() if path.is_file()}
    study = run_family_diagnostics(context, family, grid_size=17, n_scans=0)
    repeated = run_family_diagnostics(context, family, grid_size=17, n_scans=0)
    assert len(calls) == len(selection)
    pd.testing.assert_frame_equal(study["records"], repeated["records"], check_dtype=False)
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir() if path.is_file()}
    assert after == before
    changed = copy.deepcopy(family)
    template = changed["banks"][0, 100]["templates"][12]
    # Move a little probability between two occupied bins without changing any
    # rates or claimed metadata. Actual templates must enter the fingerprint.
    occupied = np.flatnonzero(template.sum(axis=0))
    shift = template[:, occupied[0]] * .01
    template[:, occupied[0]] -= shift
    template[:, occupied[1]] += shift
    with pytest.raises(ValueError, match="inputs changed|different|fingerprint"):
        run_family_diagnostics(context, changed, grid_size=17, n_scans=0)
    with pytest.raises(ValueError, match="inputs changed|different|fingerprint"):
        run_family_diagnostics(context, family, grid_size=19, n_scans=0)


def test_resume_recovers_partial_scan_without_losing_completed_toys(completed_family):
    from poodemo.toy_family_diagnostics import run_family_diagnostics

    context, selection, family, calls, _ = completed_family
    study = run_family_diagnostics(context, family, grid_size=17, n_scans=1)
    path = context["output_dir"] / "convergence/bank_convergence/family/family_control/refits/scans.csv"
    scans = pd.read_csv(path, float_precision="round_trip")
    scans.drop(index=0).to_csv(path, index=False)
    repeated = run_family_diagnostics(context, family, grid_size=17, n_scans=1)
    assert len(calls) == len(selection) + 1
    pd.testing.assert_frame_equal(study["records"], repeated["records"], check_dtype=False)
    keys = ["mu_test", "toy_id", "replica", "generated_per_source", "model", "mu_scan"]
    expected = study["scans"].sort_values(keys).reset_index(drop=True)
    actual = repeated["scans"].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False)


def test_saved_1d_reproduction_failure_is_explicit(completed_family):
    from poodemo.toy_family_diagnostics import run_family_diagnostics

    context, _, family, _, _ = completed_family
    path = context["output_dir"] / "convergence/bank_convergence/refits/results.csv"
    saved = pd.read_csv(path)
    target = ((saved.model == "binned12") & (saved.bank_kind == "independent")
              & (saved.replica == 0) & (saved.generated_per_source == 100)
              & (saved.mu_test == 0.) & (saved.toy_id == 0))
    assert target.sum() == 1
    saved.loc[target, "q"] += 1.
    saved.to_csv(path, index=False)
    study = run_family_diagnostics(context, family, grid_size=17, n_scans=0)
    failed = study["records"].loc[~study["records"].valid]
    assert len(failed) == 1
    assert failed.saved_fit_available.all()
    assert not failed.saved_fit_reproduced.any()
    assert failed.message.str.contains("differs from saved").all()
    assert abs(failed.saved_q_error.iloc[0]) > .9
    summary = study["summary"].query("model == 'binned12' and replica == 0 and generated_per_source == 100 and mu_test == 0")
    assert summary.n_random.iloc[0] == 3
    assert summary.n_failed.iloc[0] == 1
    assert summary.n_saved_fit_mismatch.iloc[0] == 1
    assert summary.n_paired_failed.iloc[0] == 1


def test_unsupported_observed_bins_fail_without_floors_or_merging(completed_family):
    from poodemo.toy_family_diagnostics import run_family_diagnostics

    context, _, family, _, _ = completed_family
    changed = copy.deepcopy(family)
    for bank in changed["banks"].values():
        template = bank["templates"][12]
        occupied = np.flatnonzero(template.sum(axis=0))
        template[:, occupied[1]] += template[:, occupied[0]]
        template[:, occupied[0]] = 0.
    study = run_family_diagnostics(context, changed, grid_size=17, n_scans=0)
    unsupported = study["records"].loc[study["records"].empty_observed_bins > 0]
    assert len(unsupported) > 0
    assert unsupported.model.eq("family12x12").all()
    assert not unsupported.valid.any()
    assert unsupported.message.str.contains("support|zero intensity").all()
    assert unsupported.q.isna().all()
    assert study["records"].query("model != 'family12x12'").valid.all()
    summary = study["summary"].query("model == 'family12x12'")
    np.testing.assert_array_equal(summary.n_failed, summary.n_empty_support)
    assert summary.n_failed.sum() == len(unsupported)
    assert study["summary"].n_random.eq(3).all()


def test_paired_branch_decomposition_excludes_flag_only_controls(completed_family):
    from poodemo.toy_family_diagnostics import _tables, run_family_diagnostics

    context, _, family, _, _ = completed_family
    records = run_family_diagnostics(context, family, grid_size=17, n_scans=0)["records"].copy()
    records["G"] = 100 * (records.toy_id + 1.)
    records["common_point_G"] = records.G
    chosen = records.model == "family12x12"
    records.loc[chosen, "common_point_G"] += records.loc[chosen, "toy_id"]
    records.loc[chosen, "G"] += 3 * records.loc[chosen, "toy_id"]
    extra = records.loc[records.toy_id == 0].assign(toy_id=99, is_random=False, is_flagged=True,
                                                  G=-1e9, common_point_G=1e9)
    paired, summary, _ = _tables(pd.concat([records, extra], ignore_index=True))
    assert len(paired.loc[paired.toy_id == 99]) > 0
    rows = summary.query("model == 'family12x12'")
    assert rows.n_random.eq(3).all()
    np.testing.assert_allclose(rows.mean_delta_common, 1.)
    np.testing.assert_allclose(rows.mean_delta_common_se, 1 / np.sqrt(3))
    np.testing.assert_allclose(rows.mean_delta_refit, 2.)
    np.testing.assert_allclose(rows.mean_delta_G, 3.)
    np.testing.assert_allclose(rows.rms_delta_G, np.sqrt(15.))
    # A missing analytical partner excludes the pair from every delta summary.
    records.loc[(records.model == "analytic") & (records.toy_id == 2), "valid"] = False
    _, summary, changes = _tables(records)
    rows = summary.query("model == 'family12x12'")
    assert rows.n_paired_valid.eq(2).all()
    assert rows.n_paired_failed.eq(1).all()
    np.testing.assert_allclose(rows.mean_delta_common, .5)
    np.testing.assert_allclose(rows.mean_delta_common_se, .5)
    assert changes.n_failed.eq(1).all()


def test_same_count_changed_coordinates_are_rejected(completed_family, monkeypatch):
    from poodemo import toy_sources
    from poodemo.toy_family_diagnostics import run_family_diagnostics

    context, _, family, _, _ = completed_family
    original = toy_sources.sample_simulator_poisson

    def changed(*args, **kwargs):
        events = original(*args, **kwargs)
        events[:, 1] += .25
        return events

    monkeypatch.setattr(toy_sources, "sample_simulator_poisson", changed)
    with pytest.raises(ValueError, match="counts and coordinates"):
        run_family_diagnostics(context, family, grid_size=17, n_scans=0)
