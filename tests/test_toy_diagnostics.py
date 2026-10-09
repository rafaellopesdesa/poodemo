"""Saved-experiment provenance, paired branches and threshold uncertainty."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from poodemo.toy_diagnostics import (CASES, analyze_paired_toys, bootstrap_coverage,
                                    load_toy_study, plot_bootstrap_coverage, plot_paired_toys)


def _source(n=8, mus=(0., 1.4)):
    configuration = dict(version=2, template_method="direct_quadrature", seed=110923,
                         mu_values=list(mus), mu_bounds=[0., 2.], exposure_multiplier=1., n_bins=12)
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True, allow_nan=False).encode()).hexdigest()
    metadata = dict(configuration=configuration, fingerprint=fingerprint, requested_toys_per_hypothesis=n)
    rows = []
    for mu in mus:
        for toy_id in range(n):
            for case in CASES:
                rows.append(dict(mu_true=mu, mu_test=mu, eta=mu, toy_id=toy_id, case=case,
                                 q=float(toy_id % 4), mu_hat=float(mu), valid=True, at_lower=mu == 0.,
                                 at_upper=False, source_stream=0 if case.endswith("simulator") else
                                 1 if case == "learned_model" else 2, exposure_multiplier=1., n_observed=100))
    return dict(results=pd.DataFrame(rows), metadata=metadata)


def _write_source(tmp_path, source):
    result = tmp_path / "results"
    result.mkdir(exist_ok=True)
    (result / "toy_study_direct_config.json").write_text(json.dumps(source["metadata"]))
    source["results"].to_csv(result / "toy_study_direct_results.csv", index=False)
    return SimpleNamespace(path=lambda *parts: tmp_path.joinpath(*parts))


def test_loader_reads_only_saved_tables_and_reports_missing_failed_and_extra(tmp_path):
    source = _source(n=6)
    frame = source["results"]
    mask = frame.mu_test.eq(0.) & frame.toy_id.eq(1) & frame.case.eq("binned_simulator")
    frame = frame.loc[~mask].copy()
    mask = frame.mu_test.eq(1.4) & frame.toy_id.eq(2) & frame.case.eq("learned_simulator")
    frame.loc[mask, ["valid", "q", "mu_hat"]] = [False, np.nan, np.nan]
    source["results"] = frame
    source["metadata"]["requested_toys_per_hypothesis"] = 4
    loaded = load_toy_study(_write_source(tmp_path, source))
    assert loaded["ignored_rows"] == 20
    assert len(loaded["results"]) == 39
    complete = loaded["completeness"].set_index(["mu_test", "case"])
    assert complete.loc[(0., "binned_simulator"), "n_missing"] == 1
    assert complete.loc[(1.4, "learned_simulator"), "n_failed"] == 1
    assert len(loaded["results_sha256"]) == len(loaded["config_sha256"]) == 64
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "results" / "reference_ratio_spline_templates.npz").exists()


@pytest.mark.parametrize("problem", ["fingerprint", "spline", "duplicate", "case", "eta", "seed", "counts", "valid"])
def test_loader_rejects_inconsistent_provenance_or_rows(tmp_path, problem):
    source = _source()
    if problem == "fingerprint":
        source["metadata"]["configuration"]["seed"] += 1
    elif problem == "spline":
        source["metadata"]["configuration"]["version"] = 1
    elif problem == "duplicate":
        source["results"] = pd.concat([source["results"], source["results"].iloc[:1]])
    elif problem == "case":
        source["results"].loc[0, "case"] = "unknown"
    elif problem == "eta":
        source["results"].loc[0, "eta"] = .1
    elif problem == "seed":
        source["results"]["seed"] = 123
    elif problem == "counts":
        source["results"].loc[0, "n_observed"] = 99
    else:
        source["results"]["valid"] = source["results"]["valid"].astype(object)
        source["results"].loc[0, "valid"] = "not a bool"
    with pytest.raises(ValueError):
        load_toy_study(_write_source(tmp_path, source))


def test_pairing_uses_experiment_ids_handles_both_truth_branches_and_failures():
    source = _source(n=4)
    frame = source["results"]
    for mu in (0., 1.4):
        near, remote = (0., 1.8) if mu == 0. else (1.4, .1)
        for case, hats in (("analytic_simulator", [near, near, remote, remote]),
                           ("binned_simulator", [near, remote, near, remote])):
            for toy_id, hat in enumerate(hats):
                select = frame.mu_test.eq(mu) & frame.case.eq(case) & frame.toy_id.eq(toy_id)
                frame.loc[select, "mu_hat"] = hat
        frame.loc[frame.mu_test.eq(mu) & frame.case.eq("binned_simulator"), "q"] += 2.
    # Shuffle rows to ensure pairing does not depend on CSV order.
    source["results"] = frame.sample(frac=1., random_state=44)
    analysis = analyze_paired_toys(source)
    binned = analysis["paired"].loc[lambda x: x.comparison_case.eq("binned_simulator")]
    expected = ["same_near_truth", "only_comparison_remote", "only_analytic_remote", "both_remote"]
    for mu in (0., 1.4):
        assert binned.loc[binned.mu_test.eq(mu)].sort_values("toy_id").branch_group.tolist() == expected
    np.testing.assert_allclose(binned.delta_q, 2.)
    summary = analysis["summary"].loc[lambda x: x.comparison_case.eq("binned_simulator")]
    np.testing.assert_allclose(summary.mean_delta_q, 2.)
    np.testing.assert_allclose(summary.paired_se_delta_q, 0.)
    assert summary.branch_disagreement_fraction.eq(.5).all()
    # Missing counterpart is not silently removed from the paired table.
    source["results"] = frame.loc[~(frame.mu_test.eq(0.) & frame.toy_id.eq(0) & frame.case.eq("binned_simulator"))]
    analysis = analyze_paired_toys(source)
    row = analysis["summary"].loc[lambda x: x.mu_test.eq(0.) & x.comparison_case.eq("binned_simulator")].iloc[0]
    assert row.n_requested == 4 and row.n_paired_valid == 3 and row.n_comparison_missing == 1


def test_bootstrap_includes_critical_value_uncertainty_and_is_reproducible():
    source = _source(n=4, mus=(0.,))
    # Even calibration toys [0,10] give higher 68% quantile 10; all odd
    # evaluation toys [1,5] pass. Binomial SE is zero but threshold uncertainty
    # is substantial, because bootstrap calibration can draw [0,0].
    source["results"]["q"] = source["results"].toy_id.map({0: 0., 1: 1., 2: 10., 3: 5.})
    first = bootstrap_coverage(source, n_bootstrap=300, levels=(.68,))
    second = bootstrap_coverage(source, n_bootstrap=300, levels=(.68,))
    pd.testing.assert_frame_equal(first, second)
    assert first.coverage.eq(1.).all() and first.binomial_se.eq(0.).all()
    assert first.bootstrap_se.gt(.3).all()
    assert first.critical_bootstrap_se.gt(3.).all()
    assert first.ci_low.eq(0.).all() and first.ci_high.eq(1.).all()


def test_bootstrap_preserves_simulator_pairing_and_boundary_atoms():
    source = _source(n=12, mus=(0.,))
    # The calibration sample is entirely the atom q=0. The same simulator
    # experiments have identical q across fits. With a shared ID bootstrap,
    # all three simulator acceptance estimates have exactly equal errors.
    source["results"]["q"] = np.where(source["results"].toy_id.mod(2).eq(0), 0.,
                                       source["results"].toy_id.mod(4).eq(1).astype(float))
    result = bootstrap_coverage(source, n_bootstrap=150, levels=(.68, .95))
    simulator = result.loc[result.case.str.endswith("simulator")]
    for _, group in simulator.groupby("nominal"):
        assert group.bootstrap_se.nunique() == 1
        assert group.ci_low.nunique() == group.ci_high.nunique() == 1
        assert group.critical_q.eq(0.).all()
        assert group.coverage.eq(.5).all()  # ties q=0 accepted, no randomized atom
    analytic = result.loc[result.case.eq("analytic_simulator")]
    assert analytic.simulator_minus_model.eq(0.).all()
    assert analytic.difference_bootstrap_se.eq(0.).all()
    assert result.loc[result.case.eq("learned_simulator"), "difference_bootstrap_se"].gt(0.).all()


def test_bootstrap_keeps_failed_missing_counts_and_worst_case_bounds():
    source = _source(n=6, mus=(0.,))
    frame = source["results"]
    frame["q"] = 0.
    bad = frame.case.eq("binned_simulator") & frame.toy_id.eq(1)
    frame.loc[bad, ["valid", "q", "mu_hat"]] = [False, np.nan, np.nan]
    source["results"] = frame.loc[~(frame.case.eq("binned_simulator") & frame.toy_id.eq(3))]
    result = bootstrap_coverage(source, n_bootstrap=30, levels=(.95,))
    row = result.loc[result.case.eq("binned_simulator")].iloc[0]
    assert row.n_evaluation == 1 and row.n_evaluation_failed == 1 and row.n_evaluation_missing == 1
    assert row.coverage == 1.
    assert row.failure_lower_bound == pytest.approx(1/3) and row.failure_upper_bound == 1.


def test_diagnostic_plot_survival_is_logarithmic_and_outputs_are_explicit(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    source = _source(n=8, mus=(0.,))
    figures = plot_paired_toys(analyze_paired_toys(source), tmp_path)
    assert all(ax.get_yscale() == "log" for ax in figures["branch_q_survival"].axes)
    coverage = bootstrap_coverage(source, n_bootstrap=20, levels=(.68, .95))
    figures.update(plot_bootstrap_coverage(coverage, tmp_path))
    assert len(list(tmp_path.glob("12_*.pdf"))) == 5
    for figure in figures.values():
        plt.close(figure)
