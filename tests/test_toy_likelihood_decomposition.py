"""Fixed-pair versus fitted-minimum diagnostics on identical saved toy fits."""
import copy

import numpy as np
import pandas as pd
import pytest

from test_toy_convergence_refits import experiment
from poodemo.toy_convergence_refits import run_convergence_refits
from poodemo.toy_likelihood import StatOnlyLikelihood


@pytest.fixture
def completed(experiment, monkeypatch):
    """Write real section-6 fits, then forbid any optimization in section 7."""
    import poodemo.toy_likelihood_decomposition as decomposition

    context, selection, banks, calls = experiment
    source = run_convergence_refits(context, selection, banks, grid_size=17)
    calls.clear()
    monkeypatch.setattr(decomposition, "load_completed_convergence_banks",
                        lambda *args, **kwargs: banks)

    def no_fitting(*args, **kwargs):
        raise AssertionError("The decomposition must reuse saved minima without fitting")

    monkeypatch.setattr(StatOnlyLikelihood, "fit", no_fitting)
    monkeypatch.setattr(StatOnlyLikelihood, "test_statistic", no_fitting)
    return context, selection, banks, calls, source


def test_decomposition_identity_and_fixed_pair_rate_cancellation(completed):
    from poodemo.toy_likelihood_decomposition import run_likelihood_decomposition

    context, selection, banks, calls, source = completed
    study = run_likelihood_decomposition(context)
    records, paired = study["records"], study["paired"]
    assert len(calls) == len(selection)
    assert len(records) == len(source["refits"])
    assert records.decomposition_valid.all()
    assert paired.paired_valid.all()
    np.testing.assert_allclose(records.min_D, records.D_eta - records.q, atol=1e-12)
    np.testing.assert_allclose(records.D_at_muhat, records.min_D, atol=1e-7)
    np.testing.assert_allclose(paired.delta_q, paired.delta_pair - paired.delta_min, atol=1e-12)
    np.testing.assert_allclose(records.q_reproduction_error, 0., atol=1e-7)
    # These banks differ only by a uniform yield scale. Their rate terms alter
    # each D separately, but cancel from the binned-minus-analytic fixed pair.
    group = paired.groupby(["mu_test", "toy_id", "n_bins"])
    np.testing.assert_allclose(group.delta_pair.max() - group.delta_pair.min(), 0., atol=1e-10)
    # At eta=1.4 the fine bins separate every physical cell: no coarsening loss.
    fine = paired.query("mu_test == 1.4 and n_bins == 36")
    np.testing.assert_allclose(fine.delta_pair, 0., atol=1e-10)
    np.testing.assert_allclose(fine.delta_min, 0., atol=1e-6)


def test_resume_is_read_only_and_changed_source_fits_are_rejected(completed):
    from poodemo.toy_likelihood_decomposition import run_likelihood_decomposition

    context, selection, _, calls, _ = completed
    fit_path = context["output_dir"] / "convergence/bank_convergence/refits/results.csv"
    source_bytes = fit_path.read_bytes()
    study = run_likelihood_decomposition(context)
    repeated = run_likelihood_decomposition(context)
    assert len(calls) == len(selection)
    pd.testing.assert_frame_equal(study["records"], repeated["records"], check_dtype=False)
    assert fit_path.read_bytes() == source_bytes
    fits = pd.read_csv(fit_path)
    fits.loc[0, "q"] += 1.
    fits.to_csv(fit_path, index=False)
    with pytest.raises(ValueError, match="different|changed|fingerprint|inputs"):
        run_likelihood_decomposition(context)


def test_mismatched_saved_q_is_visible_and_not_refitted(completed):
    from poodemo.toy_likelihood_decomposition import run_likelihood_decomposition

    context, _, _, _, _ = completed
    fit_path = context["output_dir"] / "convergence/bank_convergence/refits/results.csv"
    fits = pd.read_csv(fit_path)
    fits.loc[0, "q"] += 1.
    fits.to_csv(fit_path, index=False)
    study = run_likelihood_decomposition(context)
    bad = study["records"].loc[~study["records"].decomposition_valid]
    assert len(bad) == 1
    assert abs(bad.q_reproduction_error.iloc[0]) > .9
    assert not study["paired"].paired_valid.all()


def test_failed_saved_fit_remains_in_population_denominator(completed):
    from poodemo.toy_likelihood_decomposition import run_likelihood_decomposition

    context, _, _, _, _ = completed
    fit_path = context["output_dir"] / "convergence/bank_convergence/refits/results.csv"
    fits = pd.read_csv(fit_path)
    target = ((fits.mu_test == 0.) & (fits.toy_id == 0) &
              (fits.bank_kind == "original") & (fits.model == "analytic"))
    assert target.sum() == 1
    fits.loc[target, ["q", "mu_hat"]] = np.nan
    fits.loc[target, "valid"] = False
    fits.loc[target, "message"] = "Deliberate missing-support failure"
    fits.to_csv(fit_path, index=False)
    study = run_likelihood_decomposition(context)
    rows = study["summary"].query("mu_test == 0 and bank_kind == 'original' and branch_pair == 'all'")
    assert rows.n_random.eq(3).all()
    assert rows.n_paired_valid.eq(2).all()
    assert rows.n_paired_failed.eq(1).all()
    bad = study["paired"].query("mu_test == 0 and toy_id == 0 and bank_kind == 'original'")
    assert not bad.paired_valid.any()
    assert bad.delta_pair.isna().all()


def test_same_count_different_coordinates_are_rejected(completed, monkeypatch):
    from poodemo import toy_sources
    from poodemo.toy_likelihood_decomposition import run_likelihood_decomposition

    context, _, _, _, _ = completed
    original = toy_sources.sample_simulator_poisson

    def altered(*args, **kwargs):
        events = original(*args, **kwargs)
        events[:, 1] += .01
        return events

    monkeypatch.setattr(toy_sources, "sample_simulator_poisson", altered)
    with pytest.raises(ValueError, match="event|coordinates|digest"):
        run_likelihood_decomposition(context)


def test_changed_bank_arrays_cannot_hide_behind_metadata(completed, monkeypatch):
    import poodemo.toy_likelihood_decomposition as decomposition

    context, _, banks, _, _ = completed
    altered = copy.deepcopy(banks)
    bank = altered["banks"][0, 100]
    bank["rates"] *= 1.01
    bank["templates"] = {key: value * 1.01 for key, value in bank["templates"].items()}
    monkeypatch.setattr(decomposition, "load_completed_convergence_banks",
                        lambda *args, **kwargs: altered)
    with pytest.raises(ValueError, match="bank|fingerprint|digest|different|match"):
        decomposition.run_likelihood_decomposition(context)


def test_random_only_paired_summaries_and_branch_labels(completed):
    from poodemo.toy_likelihood_decomposition import run_likelihood_decomposition, _tables

    context, _, _, _, _ = completed
    records = run_likelihood_decomposition(context)["records"].copy()
    # Large common values must cancel before calculating the paired error.
    records["D_eta"] = 100 * (records.toy_id + 1.)
    records["min_D"] = -100 * (records.toy_id + 1.)
    binned = records.model == "binned12"
    records.loc[binned, "D_eta"] += records.loc[binned, "toy_id"]
    records.loc[binned, "min_D"] -= 2 * records.loc[binned, "toy_id"]
    records["q"] = records.D_eta - records.min_D
    records["D_at_muhat"] = records.min_D
    records["mu_hat"] = np.where(records.model == "analytic", .5, 1.5)
    extra = records.loc[records.toy_id == 0].assign(toy_id=99, is_random=False,
                                                  is_flagged=True, D_eta=1e9)
    paired, summary, _ = _tables(pd.concat([records, extra], ignore_index=True))
    # All-branch rows explicitly summarize the random sample, not flag-only toys.
    rows = summary.loc[(summary.model == "binned12") & (summary.branch_pair == "all")]
    assert len(rows)
    assert rows.n_random.eq(3).all()
    np.testing.assert_allclose(rows.mean_delta_pair, 1.)
    np.testing.assert_allclose(rows.mean_delta_pair_se, 1 / np.sqrt(3))
    np.testing.assert_allclose(rows.rms_delta_pair, np.sqrt(5 / 3))
    np.testing.assert_allclose(rows.mean_delta_min, -2.)
    np.testing.assert_allclose(rows.mean_delta_q, 3.)
    assert set(paired.branch_pair) == {"low / high"}


def test_signed_decomposition_plots_render_on_linear_y_axes(completed, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from poodemo.toy_likelihood_decomposition import (run_likelihood_decomposition,
                                                     plot_likelihood_decomposition)

    context, _, _, _, _ = completed
    study = run_likelihood_decomposition(context)
    figures = plot_likelihood_decomposition(study, tmp_path / "plots")
    assert len(figures) == 6
    for name, figure in figures.items():
        assert all(axis.get_yscale() == "linear" for axis in figure.axes)
        assert (tmp_path / "plots" / f"12_{name}.png").stat().st_size > 1000
        assert (tmp_path / "plots" / f"12_{name}.pdf").stat().st_size > 1000
        plt.close(figure)

