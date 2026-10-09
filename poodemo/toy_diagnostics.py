"""Read-only diagnostics of saved notebook 11 experiments.

No simulation, neural-network loading, or integration is performed here.  Toy
IDs identify complete experiments, so bootstrap draws preserve the pairing of
all likelihood fits to a common simulator experiment.  Reported fit summaries
and acceptance probabilities are conditional on valid fits, with failures and
missing rows reported explicitly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

CASES = {
    "analytic_simulator": "Analytical / simulator",
    "learned_simulator": "Learned / simulator",
    "learned_model": "Learned / learned bank",
    "binned_simulator": "Binned / simulator",
    "binned_model": "Binned / model",
}
SELF_CASE = {"analytic_simulator": "analytic_simulator", "learned_simulator": "learned_model",
             "learned_model": "learned_model", "binned_simulator": "binned_model",
             "binned_model": "binned_model"}
BRANCH_GROUPS = ("same_near_truth", "only_comparison_remote", "only_analytic_remote", "both_remote")
BRANCH_LABELS = ("Both on truth branch", "Only comparison remote", "Only analytical remote", "Both remote")
COLORS = {"analytic_simulator": "black", "learned_simulator": "#d55e00",
          "learned_model": "#009e73", "binned_simulator": "#0072b2", "binned_model": "#cc79a7"}


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _booleans(series, name):
    converted = series.map({True: True, False: False, "True": True, "False": False,
                            "true": True, "false": False, 1: True, 0: False})
    if converted.isna().any():
        raise ValueError(f"Toy column {name} contains invalid booleans")
    return converted.astype(bool)


def load_toy_study(run, source_tag="toy_study_direct"):
    """Load authenticated direct-yield notebook 11 CSV/configuration only.

    The saved configuration fingerprint is verified; model and bank hashes are
    retained as provenance rather than reloading those large artifacts.  Extra
    completed IDs above the last requested toy count are ignored, as in 11.
    Missing rows and failed fits remain visible in ``completeness``.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", source_tag):
        raise ValueError("source_tag must contain only letters, digits, underscores or hyphens")
    config_path = run.path("results", f"{source_tag}_config.json")
    results_path = run.path("results", f"{source_tag}_results.csv")
    config_bytes, result_bytes = config_path.read_bytes(), results_path.read_bytes()
    metadata = json.loads(config_bytes)
    config = metadata.get("configuration", {})
    if config.get("version") != 2 or config.get("template_method") != "direct_quadrature":
        raise ValueError("Notebook 12 requires notebook 11 version 2 direct_quadrature results (no spline templates)")
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError("Saved notebook 11 configuration fingerprint does not match its contents")
    requested = _integer(metadata.get("requested_toys_per_hypothesis"), "requested_toys_per_hypothesis", 1)
    _integer(config.get("seed"), "seed")
    mus = np.asarray(config.get("mu_values", []), float)
    bounds = np.asarray(config.get("mu_bounds", []), float)
    if (mus.ndim != 1 or not len(mus) or len(np.unique(mus)) != len(mus) or not np.all(np.isfinite(mus))
            or bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] != 0
            or bounds[1] <= bounds[0] or np.any((mus < bounds[0]) | (mus > bounds[1]))):
        raise ValueError("Invalid saved mu hypotheses or fit bounds")
    exposure = float(config.get("exposure_multiplier", np.nan))
    if not np.isfinite(exposure) or exposure <= 0 or config.get("n_bins") != 12:
        raise ValueError("Expected positive exposure and 12 direct bins in notebook 11")
    import io
    results = pd.read_csv(io.BytesIO(result_bytes))
    required = {"mu_true", "mu_test", "eta", "toy_id", "case", "q", "mu_hat", "valid",
                "at_lower", "at_upper", "source_stream", "exposure_multiplier"}
    if missing := required.difference(results.columns):
        raise ValueError(f"Toy CSV is missing required columns: {sorted(missing)}")
    for name in ("valid", "at_lower", "at_upper"):
        results[name] = _booleans(results[name], name)
    ids = pd.to_numeric(results.toy_id, errors="coerce").to_numpy(float)
    if np.any(~np.isfinite(ids)) or np.any(ids < 0) or np.any(ids != np.floor(ids)):
        raise ValueError("Toy IDs must be nonnegative integers")
    results["toy_id"] = ids.astype(np.int64)
    if results.duplicated(["mu_test", "toy_id", "case"]).any():
        raise ValueError("Toy CSV contains duplicate experiment/case keys")
    if not results.case.isin(CASES).all() or not results.mu_test.isin(mus).all():
        raise ValueError("Toy CSV contains incompatible cases or mu hypotheses")
    if not (results.mu_true.eq(results.mu_test) & results.eta.eq(results.mu_test)).all():
        raise ValueError("Toy CSV must have mu_true = mu_test = eta")
    expected_stream = results.case.map({case: 0 if case.endswith("simulator") else
                                       1 if case == "learned_model" else 2 for case in CASES})
    if not results.source_stream.eq(expected_stream).all() or not results.exposure_multiplier.eq(exposure).all():
        raise ValueError("Toy CSV source streams or exposure disagree with the saved configuration")
    if "seed" in results and not results.seed.eq(config["seed"]).all():
        raise ValueError("Toy CSV seed disagrees with the saved configuration")
    finite = np.isfinite(results.q) & np.isfinite(results.mu_hat)
    if (results.valid & (~finite | (results.q < -1e-9)
                         | (results.mu_hat < bounds[0] - 1e-9) | (results.mu_hat > bounds[1] + 1e-9))).any():
        raise ValueError("A valid toy fit contains invalid q or mu_hat")
    # Successful fits to paired experiments must agree on the observed count.
    if "n_observed" in results:
        paired_counts = results.loc[results.case.str.endswith("simulator")].groupby(["mu_test", "toy_id"]).n_observed.nunique()
        if (paired_counts > 1).any():
            raise ValueError("Simulator fits with the same toy ID have different observed counts")
    ignored = int(np.sum(results.toy_id >= requested))
    results = results.loc[results.toy_id < requested].sort_values(["mu_test", "toy_id", "case"]).reset_index(drop=True)
    rows = []
    for mu in mus:
        for case in CASES:
            group = results.loc[results.mu_test.eq(mu) & results.case.eq(case)]
            good = group.valid & np.isfinite(group.q) & np.isfinite(group.mu_hat)
            rows.append(dict(mu_test=mu, case=case, requested=requested, n_present=len(group),
                             n_valid=int(good.sum()), n_failed=int((~good).sum()), n_missing=requested-len(group)))
    # Pure table summarization; this call does not prepare quadrature or models.
    from .toy_study import summarize_toys
    summary, coverage = summarize_toys(results)
    return dict(results=results, metadata=metadata, summary=summary, coverage=coverage,
                source_tag=source_tag, completeness=pd.DataFrame(rows), ignored_rows=ignored,
                results_sha256=hashlib.sha256(result_bytes).hexdigest(),
                config_sha256=hashlib.sha256(config_bytes).hexdigest())


def _source_grid(source):
    metadata = source["metadata"]
    count = int(metadata["requested_toys_per_hypothesis"])
    mus = metadata["configuration"]["mu_values"]
    return count, mus


def _se(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan


def analyze_paired_toys(source, branch_split=1.0):
    """Pair by experiment ID, preserving missing and failed counterparts.

    A fixed split classifies minima for descriptive diagnostics only.  The
    branch containing the tested truth is called the truth branch; this is not
    a proof that either fit selected a particular local stationary point.
    """
    if not np.isfinite(branch_split) or branch_split < 0:
        raise ValueError("branch_split must be finite and nonnegative")
    results, (count, mus) = source["results"], _source_grid(source)
    index = pd.MultiIndex.from_product([mus, range(count)], names=["mu_test", "toy_id"])
    columns = ["q", "mu_hat", "valid", "at_lower", "at_upper"]
    def case_frame(case, prefix):
        frame = results.loc[results.case.eq(case)].set_index(["mu_test", "toy_id"])[columns]
        frame["present"] = True
        return frame.reindex(index).add_prefix(prefix)
    analytic = case_frame("analytic_simulator", "analytic_")
    all_pairs, summaries, branches = [], [], []
    for case in ("learned_simulator", "binned_simulator"):
        pair = analytic.join(case_frame(case, "comparison_")).reset_index()
        pair["comparison_case"] = case
        for prefix in ("analytic_", "comparison_"):
            for name in ("present", "valid", "at_lower", "at_upper"):
                pair[prefix + name] = pair[prefix + name].eq(True)
        pair["paired_valid"] = (pair.analytic_valid & pair.comparison_valid
                                  & np.isfinite(pair.analytic_q) & np.isfinite(pair.comparison_q)
                                  & np.isfinite(pair.analytic_mu_hat) & np.isfinite(pair.comparison_mu_hat))
        pair["delta_q"] = (pair.comparison_q - pair.analytic_q).where(pair.paired_valid)
        truth_high = pair.mu_test.ge(branch_split)
        for prefix in ("analytic_", "comparison_"):
            high = pair[prefix + "mu_hat"].ge(branch_split)
            pair[prefix + "branch"] = np.where(high, "high", "low")
            pair.loc[~pair[prefix + "valid"], prefix + "branch"] = "invalid"
            pair[prefix + "remote"] = high.ne(truth_high) & pair[prefix + "valid"]
            pair[prefix + "zero_q"] = pair[prefix + "q"].le(1e-10) & pair[prefix + "valid"]
        pair["branch_group"] = np.select(
            [~pair.analytic_remote & ~pair.comparison_remote,
             ~pair.analytic_remote & pair.comparison_remote,
             pair.analytic_remote & ~pair.comparison_remote], BRANCH_GROUPS[:3], default=BRANCH_GROUPS[3])
        pair.loc[~pair.paired_valid, "branch_group"] = "invalid_or_missing"
        all_pairs.append(pair)
        for mu, group in pair.groupby("mu_test", sort=False):
            good = group.loc[group.paired_valid]
            n = len(good)
            row = dict(mu_test=mu, comparison_case=case, n_requested=len(group), n_paired_valid=n,
                       n_invalid_or_missing=len(group)-n,
                       n_analytic_missing=int((~group.analytic_present).sum()),
                       n_comparison_missing=int((~group.comparison_present).sum()),
                       n_analytic_failed=int((group.analytic_present & ~group.analytic_valid).sum()),
                       n_comparison_failed=int((group.comparison_present & ~group.comparison_valid).sum()),
                       mean_delta_q=float(good.delta_q.mean()), paired_se_delta_q=_se(good.delta_q),
                       median_delta_q=float(good.delta_q.median()),
                       mean_absolute_delta_q=float(good.delta_q.abs().mean()),
                       branch_disagreement_fraction=float(good.analytic_remote.ne(good.comparison_remote).mean()),
                       analytic_zero_q_fraction=float(good.analytic_zero_q.mean()),
                       comparison_zero_q_fraction=float(good.comparison_zero_q.mean()))
            zero_difference = good.comparison_zero_q.astype(float)-good.analytic_zero_q.astype(float)
            row.update(zero_q_fraction_difference=float(zero_difference.mean()),
                       paired_se_zero_q_difference=_se(zero_difference))
            summaries.append(row)
            for branch in BRANCH_GROUPS:
                selected = good.loc[good.branch_group.eq(branch)]
                fraction = len(selected)/n if n else np.nan
                branches.append(dict(mu_test=mu, comparison_case=case, branch_group=branch,
                                     n_paired=len(selected), fraction_of_valid_pairs=fraction,
                                     fraction_se=np.sqrt(fraction*(1-fraction)/n) if n else np.nan,
                                     mean_delta_q=float(selected.delta_q.mean()),
                                     paired_se_delta_q=_se(selected.delta_q)))
    return dict(paired=pd.concat(all_pairs, ignore_index=True), summary=pd.DataFrame(summaries),
                branch_summary=pd.DataFrame(branches), branch_split=float(branch_split),
                interpretation="All fractions and differences condition on valid paired fits; the branch split is descriptive.")


def _valid_q_arrays(source, mu):
    count, _ = _source_grid(source)
    arrays = {}
    for case in CASES:
        frame = source["results"].loc[source["results"].mu_test.eq(mu) & source["results"].case.eq(case)]
        values = np.full(count, np.nan)
        good = frame.valid.astype(bool) & np.isfinite(frame.q)
        values[frame.loc[good, "toy_id"].to_numpy(int)] = frame.loc[good, "q"].to_numpy(float)
        arrays[case] = values
    return arrays


def _finite_quantiles(values, levels):
    good = values[np.isfinite(values)]
    return np.quantile(good, levels, method="higher") if len(good) else np.full(len(levels), np.nan)


def _acceptance(values, thresholds):
    good = values[np.isfinite(values)]
    if not len(good):
        return np.full(len(thresholds), np.nan)
    return np.asarray([np.mean(good <= q) if np.isfinite(q) else np.nan for q in thresholds])


def _replicate_interval(values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return (np.nan, np.nan, np.nan, 0)
    low, high = np.quantile(finite, [.025, .975])
    return (float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
            float(low), float(high), len(finite))


def bootstrap_coverage(source, n_bootstrap=1000, seed=120923, levels=(.68, .90, .95, .99)):
    """Whole-experiment bootstrap, including estimated critical values.

    Even IDs calibrate, odd IDs evaluate.  Each replicate draws independent
    even/odd cohorts for each source; the three simulator fits share their ID
    draw.  Learned and binned self-model sources draw independently.  Failed
    and missing rows are sampled with their experiment ID and then omitted
    from valid-fit-conditional probabilities.  Their original counts and
    worst-case acceptance bounds are returned.  Percentile intervals are
    descriptive bootstrap intervals, particularly at discrete boundary atoms.
    """
    n_bootstrap = _integer(n_bootstrap, "n_bootstrap", 2)
    seed = _integer(seed, "seed")
    levels = np.asarray(levels, float)
    if levels.ndim != 1 or not len(levels) or np.any(~np.isfinite(levels)) or np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("levels must be finite probabilities strictly between zero and one")
    count, mus = _source_grid(source)
    even, odd = np.arange(0, count, 2), np.arange(1, count, 2)
    if not len(odd):
        raise ValueError("At least two requested toys are needed for disjoint calibration and evaluation")
    rows = []
    for mu in mus:
        q = _valid_q_arrays(source, mu)
        bits = np.float64(mu).view(np.uint64).item()
        rng = np.random.default_rng(np.random.SeedSequence([seed, bits & 0xffffffff, bits >> 32]))
        critical = {case: _finite_quantiles(q[SELF_CASE[case]][even], levels) for case in CASES}
        coverage = {case: _acceptance(q[case][odd], critical[case]) for case in CASES}
        draws = {case: np.full((n_bootstrap, len(levels)), np.nan) for case in CASES}
        critical_draws = {case: np.full((n_bootstrap, len(levels)), np.nan) for case in CASES}
        for replicate in range(n_bootstrap):
            # Distinct sources are independent; paired simulator fits share
            # the exact same sampled toy IDs, including failures/missing rows.
            source_ids = {stream: (rng.choice(even, len(even), replace=True),
                                  rng.choice(odd, len(odd), replace=True)) for stream in range(3)}
            for case in CASES:
                self_case = SELF_CASE[case]
                calibration_stream = 0 if self_case == "analytic_simulator" else 1 if self_case == "learned_model" else 2
                evaluation_stream = 0 if case.endswith("simulator") else 1 if case == "learned_model" else 2
                threshold = _finite_quantiles(q[self_case][source_ids[calibration_stream][0]], levels)
                critical_draws[case][replicate] = threshold
                draws[case][replicate] = _acceptance(q[case][source_ids[evaluation_stream][1]], threshold)
        for case in CASES:
            calibration_case = SELF_CASE[case]
            frame = source["results"].loc[source["results"].mu_test.eq(mu) & source["results"].case.eq(case)]
            cal_frame = source["results"].loc[source["results"].mu_test.eq(mu) & source["results"].case.eq(calibration_case)]
            ncal, neval = int(np.isfinite(q[calibration_case][even]).sum()), int(np.isfinite(q[case][odd]).sum())
            ncal_present, neval_present = int(cal_frame.toy_id.mod(2).eq(0).sum()), int(frame.toy_id.mod(2).eq(1).sum())
            for k, level in enumerate(levels):
                se, low, high, nrep = _replicate_interval(draws[case][:, k])
                critical_se, _, _, _ = _replicate_interval(critical_draws[case][:, k])
                difference_draws = draws[case][:, k] - draws[calibration_case][:, k]
                difference_se, difference_low, difference_high, _ = _replicate_interval(difference_draws)
                fraction = coverage[case][k]
                accepted = int(np.sum(q[case][odd] <= critical[case][k])) if np.isfinite(critical[case][k]) else np.nan
                rows.append(dict(mu_test=mu, case=case, label=CASES[case], nominal=float(level),
                                 calibration_case=calibration_case, critical_q=critical[case][k],
                                 critical_bootstrap_se=critical_se, coverage=fraction,
                                 binomial_se=np.sqrt(fraction*(1-fraction)/neval) if neval else np.nan,
                                 bootstrap_se=se, ci_low=low, ci_high=high, n_bootstrap=n_bootstrap,
                                 n_bootstrap_valid=nrep, bootstrap_seed=seed,
                                 own_model_coverage=coverage[calibration_case][k],
                                 simulator_minus_model=fraction-coverage[calibration_case][k],
                                 difference_bootstrap_se=difference_se,
                                 difference_ci_low=difference_low, difference_ci_high=difference_high,
                                 n_calibration=ncal, n_calibration_failed=ncal_present-ncal,
                                 n_calibration_missing=len(even)-ncal_present,
                                 n_evaluation=neval, n_evaluation_failed=neval_present-neval,
                                 n_evaluation_missing=len(odd)-neval_present,
                                 failure_lower_bound=accepted/len(odd),
                                 failure_upper_bound=(accepted+len(odd)-neval)/len(odd),
                                 conditioning="valid fits; missing/failed fits shown separately"))
    return pd.DataFrame(rows)


def _save_figures(figures, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        for extension in ("png", "pdf"):
            figure.savefig(output / f"12_{name}.{extension}", dpi=160, bbox_inches="tight")
    return figures


def plot_paired_toys(analysis, output):
    """Plot paired fits and branch-conditional q survival (logarithmic y)."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    paired = analysis["paired"]
    mus = list(paired.mu_test.unique())
    comparisons = ("learned_simulator", "binned_simulator")
    figures = {}
    with plt.style.context(hep.style.ATLAS):
        for name in ("paired_mu_hat", "paired_delta_q", "branch_q_survival"):
            fig, axes = plt.subplots(2, len(mus), figsize=(7*len(mus), 10), squeeze=False)
            for row, case in enumerate(comparisons):
                for col, mu in enumerate(mus):
                    ax = axes[row, col]
                    group = paired.loc[paired.mu_test.eq(mu) & paired.comparison_case.eq(case) & paired.paired_valid]
                    label = "Learned" if case == "learned_simulator" else "Binned"
                    ax.set_title(rf"$\mu={mu:g}$: {label} versus analytical")
                    if name == "paired_mu_hat":
                        for branch, branch_label, color in zip(BRANCH_GROUPS, BRANCH_LABELS, ("0.4", "#d55e00", "#0072b2", "#009e73")):
                            part = group.loc[group.branch_group.eq(branch)]
                            ax.scatter(part.analytic_mu_hat, part.comparison_mu_hat, s=9, alpha=.35,
                                       color=color, label=f"{branch_label} ({len(part)})", rasterized=True)
                        top = max(2., float(group[["analytic_mu_hat", "comparison_mu_hat"]].max().max())) if len(group) else 2.
                        ax.plot([0, top], [0, top], ":", color="black", linewidth=1.)
                        ax.axvline(analysis["branch_split"], color="0.6", linestyle="--", linewidth=.8)
                        ax.axhline(analysis["branch_split"], color="0.6", linestyle="--", linewidth=.8)
                        ax.set(xlabel=r"Analytical $\widehat\mu$", ylabel=label+r" $\widehat\mu$", xlim=(-.03, top+.03), ylim=(-.03, top+.03))
                    elif name == "paired_delta_q":
                        values = group.delta_q.to_numpy()
                        lo, hi = (min(-.1, values.min()), max(.1, values.max())) if len(values) else (-1., 1.)
                        edges = np.linspace(lo, hi, 41)
                        for branch, branch_label, color in zip(BRANCH_GROUPS, BRANCH_LABELS, ("0.4", "#d55e00", "#0072b2", "#009e73")):
                            part = group.loc[group.branch_group.eq(branch)]
                            if len(part):
                                # Normalize to all valid pairs: area displays the branch fraction.
                                counts = np.histogram(part.delta_q, edges)[0]/max(1, len(group))/np.diff(edges)
                                hep.histplot(counts, edges, histtype="step", color=color, label=f"{branch_label} ({len(part)})", ax=ax)
                        ax.axvline(0, color="black", linestyle=":", linewidth=1.)
                        ax.set(xlabel=r"$\Delta q=q_{\rm comparison}-q_{\rm analytical}$", ylabel="Density per valid pair")
                    else:
                        for branch, branch_label, color in zip(BRANCH_GROUPS, BRANCH_LABELS, ("0.4", "#d55e00", "#0072b2", "#009e73")):
                            part = group.loc[group.branch_group.eq(branch)]
                            if not len(part):
                                continue
                            for key, style in (("analytic_q", "--"), ("comparison_q", "-")):
                                q = np.sort(part[key].to_numpy())
                                thresholds = np.unique(np.r_[0., q])
                                survival = (len(q)-np.searchsorted(q, thresholds, side="left"))/len(q)
                                ax.step(thresholds, survival, where="pre", linestyle=style, color=color,
                                        label=f"{branch_label} ({len(part)})" if style == "-" else None)
                        ax.set_yscale("log")
                        ax.set(xlabel=r"$q_\mu$", ylabel=r"$P(q_\mu\geq q\mid\mathrm{branch})$",
                               ylim=(.5/max(1, len(group)), 1.3))
                        ax.text(.97, .97, "Solid: comparison\nDashed: analytical", ha="right", va="top", transform=ax.transAxes, fontsize=11)
                    ax.legend(fontsize=9, frameon=False, loc="best")
                    attempted = len(paired.loc[paired.mu_test.eq(mu) & paired.comparison_case.eq(case)])
                    if len(group) < attempted:
                        ax.text(.02, .02, f"{attempted-len(group)} invalid/missing pairs", transform=ax.transAxes, fontsize=10)
            fig.tight_layout()
            figures[name] = fig
    return _save_figures(figures, output)


def plot_bootstrap_coverage(coverage, output):
    """Show bootstrap intervals including calibration-threshold uncertainty."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    mus = list(coverage.mu_test.unique())
    figures = {}
    with plt.style.context(hep.style.ATLAS):
        for name in ("bootstrap_coverage", "bootstrap_transfer_difference"):
            fig, axes = plt.subplots(1, len(mus), figsize=(7*len(mus), 5.5), squeeze=False)
            for ax, mu in zip(axes[0], mus):
                cases = list(CASES) if name == "bootstrap_coverage" else ["learned_simulator", "binned_simulator"]
                for j, case in enumerate(cases):
                    group = coverage.loc[coverage.mu_test.eq(mu) & coverage.case.eq(case)]
                    x = group.nominal+(j-(len(cases)-1)/2)*.002
                    if name == "bootstrap_coverage":
                        y, lo, hi = group.coverage, group.ci_low, group.ci_high
                    else:
                        y, lo, hi = group.simulator_minus_model, group.difference_ci_low, group.difference_ci_high
                    ax.vlines(x, lo, hi, color=COLORS[case], linewidth=1.4)
                    ax.plot(x, y, "o", color=COLORS[case], label=CASES[case], markersize=4)
                if name == "bootstrap_coverage":
                    ax.plot([.6, 1.], [.6, 1.], ":", color="0.5")
                    ax.set(ylabel="Acceptance conditional on valid fits", ylim=(0, 1.03))
                else:
                    ax.axhline(0., linestyle=":", color="0.5")
                    ax.set(ylabel="Simulator minus own-model acceptance")
                ax.set(xlabel="Nominal acceptance probability", title=rf"$\mu={mu:g}$", xlim=(.65, 1.01))
                ax.legend(fontsize=10, frameon=False)
            fig.suptitle("95% bootstrap intervals; recalibrated thresholds in every replicate", fontsize=13)
            fig.tight_layout()
            figures[name] = fig
    return _save_figures(figures, output)
