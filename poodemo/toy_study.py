"""Paired frequentist toys for the analytical, learned and 12-bin models.

The observable anchor eta is fixed at the tested hypothesis throughout each
fit. Simulator event toys are shared by all three likelihood descriptions;
histogramming a Poisson point process already gives independent Poisson bin
counts, so these counts are never fluctuated a second time.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.special import expit

from .data import PROCESSES, file_digest, save_json
from .inference import component_coefficients, histogram_components
from .pipeline import observable_bin_edges, prepare_quadrature
from .toy_likelihood import StatOnlyLikelihood

CASES = {
    "analytic_simulator": "Analytical / simulator toys",
    "learned_simulator": "Learned / simulator toys",
    "learned_model": "Learned / learned-bank toys",
    "binned_simulator": "12-bin / simulator toys",
    "binned_model": "12-bin / binned-model toys",
}
_SELF_CASE = {"analytic_simulator": "analytic_simulator",
              "learned_simulator": "learned_model", "learned_model": "learned_model",
              "binned_simulator": "binned_model", "binned_model": "binned_model"}


def _array_digest(array):
    array = np.asarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    # Iterate in rows to avoid copying an entire memory-mapped bank.
    for start in range(0, len(array), 8192):
        digest.update(np.ascontiguousarray(array[start:start + 8192]).tobytes())
    return digest.hexdigest()


def _rng(seed, mu, toy_id, stream):
    """Streams depend on hypothesis values and toy IDs, never loop lengths."""
    mu_bits = np.float64(mu).view(np.uint64).item()
    return np.random.default_rng(np.random.SeedSequence(
        [seed, mu_bits & 0xffffffff, mu_bits >> 32, int(toy_id), int(stream)]))


def reference_observable(fields, rates, eta, power):
    """Apply the analytic observable with frozen selected-rate normalizers.

    Rates MUST come from the original integration bank, not the toy events.
    The returned coordinates agree with notebook 08's reference-ratio bins.
    """
    fields, rates = np.asarray(fields, float), np.asarray(rates, float)
    if not np.isfinite(eta) or eta < 0 or not np.isfinite(power) or power <= 0:
        raise ValueError("eta must be nonnegative and power must be positive")
    if fields.ndim != 2 or fields.shape[0] != 4 or rates.shape != (4,):
        raise ValueError("fields and frozen rates must have shapes (4,N) and (4,)")
    tested = component_coefficients(eta)
    reference = component_coefficients(1.)
    density, base = tested @ fields, reference @ fields
    total, base_total = float(tested @ rates), float(reference @ rates)
    if (np.any(~np.isfinite(density)) or np.any(density < 0)
            or np.any(~np.isfinite(base)) or np.any(base <= 0)
            or not np.isfinite(total + base_total) or total <= 0 or base_total <= 0):
        raise ValueError("Reference observable requires positive physical rates and reference density")
    if eta == 1.:
        return np.full(fields.shape[1], .5)
    with np.errstate(divide="ignore", over="ignore"):
        return expit(power * (np.log(density) - np.log(total)
                              - np.log(base) + np.log(base_total)))


def _direct_templates(quad, mus, n_bins, power):
    """Integrate process yields directly in the frozen observable bins.

    Each eta gets its own histogram of the original analytical quadrature,
    including eta=0. As in notebook 08, these process yields remain fixed
    throughout a fit; only the exact physical mu coefficients change.
    No notebook 09 output or interpolated template is read.
    """
    edges = observable_bin_edges(n_bins, "reference_ratio")
    rates = np.asarray(quad["nominal"]) @ quad["weights"]
    templates, rows, provenance = {}, [], {}
    source = "direct analytical quadrature histogram (notebook 08 method)"
    for mu in mus:
        z = reference_observable(quad["nominal"], rates, mu, power)
        yields = histogram_components(z, quad["nominal"], quad["weights"], edges)
        templates[mu] = yields
        provenance[str(mu)] = source
        for j, process in enumerate(PROCESSES):
            for bi in range(n_bins):
                rows.append(dict(mu_test=mu, eta=mu, process=process, bin=bi,
                                 expected_yield=yields[j, bi], template_source=source))
    return templates, edges, pd.DataFrame(rows), {
        "template_method": "direct_quadrature", "template_sources": provenance,
    }


def _atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _failure(message):
    return dict(q=np.nan, mu_hat=np.nan, nll=np.nan, valid=False, message=str(message),
                at_lower=False, at_upper=False, n_evaluations=0, optimizer_failures=0)


def _fit(likelihood, mu, bounds, grid):
    try:
        return likelihood.test_statistic(mu, mu_bounds=bounds, grid_size=grid)
    except (ValueError, FloatingPointError, RuntimeError) as error:
        return _failure(f"{type(error).__name__}: {error}")


def summarize_toys(results, levels=(.68, .90, .95, .99)):
    """Report failed fits and use disjoint calibration/evaluation toy IDs.

    Quantiles use all valid toys for descriptive summaries. Coverage uses only
    even self-model toys for critical values and odd toys for evaluation.
    Binomial errors are conditional on those estimated critical values;
    finite calibration-sample uncertainty is not included in that error.
    """
    summaries, coverage = [], []
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (mu, case), group in results.groupby(["mu_test", "case"], sort=False):
        good = group.loc[group.valid.astype(bool) & np.isfinite(group.q)]
        row = dict(mu_test=mu, case=case, label=CASES[case], n_toys=len(group),
                   n_valid=len(good), n_failed=len(group) - len(good),
                   failure_fraction=1. - len(good) / len(group),
                   mean_q=float(good.q.mean()), mean_mu_hat=float(good.mu_hat.mean()),
                   lower_boundary_fraction=float(good.at_lower.mean()),
                   upper_boundary_fraction=float(good.at_upper.mean()),
                   zero_q_fraction=float((good.q <= 1e-10).mean()) if len(good) else np.nan)
        for level in levels:
            row[f"q{int(round(100 * level))}"] = (
                float(np.quantile(good.q, level, method="higher")) if len(good) else np.nan)
        summaries.append(row)
        calibration_case = _SELF_CASE[case]
        calibration = results.loc[(results.mu_test == mu) & (results.case == calibration_case)
                                  & (results.toy_id % 2 == 0)]
        calibration_good = calibration.loc[calibration.valid.astype(bool) & np.isfinite(calibration.q)]
        evaluation = group.loc[group.toy_id % 2 == 1]
        evaluation_good = evaluation.loc[evaluation.valid.astype(bool) & np.isfinite(evaluation.q)]
        for level in levels:
            critical = (float(np.quantile(calibration_good.q, level, method="higher"))
                        if len(calibration_good) else np.nan)
            accepted = int(np.sum(evaluation_good.q <= critical)) if np.isfinite(critical) else 0
            n_eval, n_attempt = len(evaluation_good), len(evaluation)
            fraction = accepted / n_eval if n_eval and np.isfinite(critical) else np.nan
            coverage.append(dict(mu_test=mu, case=case, label=CASES[case], nominal=level,
                                 calibration_case=calibration_case, critical_q=critical,
                                 n_calibration=len(calibration_good),
                                 n_calibration_failed=len(calibration)-len(calibration_good),
                                 n_evaluation=n_eval, n_evaluation_failed=n_attempt-n_eval,
                                 coverage=fraction,
                                 binomial_se=np.sqrt(fraction * (1 - fraction) / n_eval) if n_eval else np.nan,
                                 failure_lower_bound=accepted / n_attempt if n_attempt and np.isfinite(critical) else np.nan,
                                 failure_upper_bound=(accepted + n_attempt-n_eval) / n_attempt if n_attempt and np.isfinite(critical) else np.nan))
    return pd.DataFrame(summaries), pd.DataFrame(coverage)


def run_toy_study(run, *, n_toys=500, mu_values=(0., 1.4), n_bins=12,
                  reference_power=100., exposure=1., seed=None, mu_bounds=(0., 2.),
                  grid_size=65, output_tag="toy_study_direct", progress_every=10):
    """Run/resume five stat-only ensembles with paired simulator event toys.

    ``exposure`` is a multiplier of the existing run's physical exposure.
    Increasing ``n_toys`` extends deterministic streams. Other scientific
    settings or changed model/bank content reject stale cached toy results.
    Every completed toy ID is checkpointed atomically. No event arrays are
    persisted. The learned-bank source is a finite quadrature approximation,
    whose effective sample size is recorded rather than hidden. Binned process
    yields are direct quadrature histograms at each fixed eta, without interpolation.
    """
    from .toy_sources import (QuadraturePoissonSource, load_frozen_selector,
                              make_frozen_workspace, sample_simulator_poisson)
    for value, name, minimum in ((n_toys, "n_toys", 1), (n_bins, "n_bins", 2),
                                 (grid_size, "grid_size", 5), (progress_every, "progress_every", 1)):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if n_bins != 12:
        raise ValueError("This comparison uses direct reference-ratio histograms with exactly 12 bins")
    bounds = np.asarray(mu_bounds, float)
    mus = tuple(float(mu) for mu in mu_values)
    if (bounds.shape != (2,) or np.any(~np.isfinite(bounds)) or bounds[0] != 0
            or bounds[1] <= bounds[0] or not mus or len(set(mus)) != len(mus)
            or any(not np.isfinite(mu) or not bounds[0] <= mu <= bounds[1] for mu in mus)):
        raise ValueError("Use unique finite mu values within fit bounds starting at exact zero")
    if not np.isfinite(exposure) or exposure <= 0 or not np.isfinite(reference_power) or reference_power <= 0:
        raise ValueError("exposure and reference_power must be finite and positive")
    seed = int(run.config["seed"]) + 110000 if seed is None else seed
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output_tag):
        raise ValueError("output_tag must contain only letters, digits, underscores or hyphens")
    quad = prepare_quadrature(run)
    templates, edges, bin_yields, template_metadata = _direct_templates(quad, mus, n_bins, reference_power)
    selector, threshold, selection_provenance = load_frozen_selector(run)
    workspace = make_frozen_workspace(run, quad)
    learned = np.asarray(workspace.learned_quadrature[0])
    rates = np.asarray(quad["nominal"]) @ quad["weights"]
    learned_rates = learned @ quad["weights"]
    sources, source_diagnostics, source_errors = {}, {}, {}
    for mu in mus:
        try:
            sources[mu] = QuadraturePoissonSource(quad, learned, mu, exposure=exposure)
            source_diagnostics[str(mu)] = sources[mu].diagnostics
        except (ValueError, FloatingPointError) as error:
            # An invalid learned density cannot define a toy generator. Keep
            # other comparisons available and record every affected row.
            source_errors[mu] = str(error)
            source_diagnostics[str(mu)] = dict(valid=False, continuous_generator=False,
                                                error=str(error))
    learned_guard = StatOnlyLikelihood(np.empty((4, 0)), learned_rates,
                                       exposure=exposure, validity_fields=learned)
    dependencies = {}
    model_dir = run.path("models")
    if model_dir.exists():
        for path in sorted(model_dir.rglob("*")):
            if path.is_file() and path.name in {"model.pt", "ensemble.json", "calibrator.joblib"}:
                dependencies[str(path.relative_to(run.root))] = file_digest(path)
    configuration = dict(version=2, statistic="two-sided q_mu; exact physical boundary; eta fixed during fit",
                         mu_values=list(mus), n_bins=int(n_bins), reference_power=float(reference_power),
                         exposure_multiplier=float(exposure), seed=int(seed), mu_bounds=bounds.tolist(),
                         grid_size=int(grid_size), bin_edges=edges.tolist(), physics=run.model.to_dict(),
                         selection=selection_provenance, model_artifacts=dependencies,
                         bank_hashes={key: _array_digest(quad[key]) for key in ("x", "weights", "nominal")},
                         learned_bank_sha256=_array_digest(learned), **template_metadata)
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True, allow_nan=False).encode()).hexdigest()
    config_path = run.path("results", f"{output_tag}_config.json")
    result_path = run.path("results", f"{output_tag}_results.csv")
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved.get("fingerprint") != fingerprint:
            raise ValueError("Cached toy-study settings/models/bank differ. Use a new OUTPUT_TAG or restore matching inputs.")
    elif result_path.exists():
        raise ValueError("Toy CSV has no configuration fingerprint; use a new OUTPUT_TAG")
    metadata = dict(configuration=configuration, fingerprint=fingerprint,
                    requested_toys_per_hypothesis=int(n_toys),
                    source_diagnostics=source_diagnostics,
                    simulation_note="Fresh continuous physical Poisson events with frozen preselection; shared by analytic, learned and binned fits. No second Poisson fluctuation of histogram counts.",
                    learned_source_note="Poisson sampling from finite quadrature weights times frozen learned intensity. Repeated bank points are integer multiplicities; this is not an exact continuous neural sampler.",
                    normalization_note="All fitted integrals and observable normalizers are frozen from the original quadrature. Analytical likelihood rates also have finite integration error.",
                    fit_note="All minima resolved by a grid in sqrt(mu), their local refinements, and exact bounds are compared; eta remains fixed at mu_test.",
                    coverage_note="Even self-model toy IDs calibrate critical values; odd IDs evaluate coverage. Errors are binomial conditional on estimated critical values, excluding finite calibration uncertainty. Invalid fits are retained and coverage bounds include them. Atoms/ties may make coverage conservative.",
                    validity_note="Learned intensities are checked on the quadrature and each observed sample; finite-bank checks do not establish positivity everywhere.")
    save_json(config_path, metadata)
    records = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    keys = [(float(row["mu_test"]), int(row["toy_id"]), row["case"]) for row in records]
    if len(set(keys)) != len(keys) or any(mu not in mus or case not in CASES for mu, _, case in keys):
        raise ValueError("Toy results contain duplicate or incompatible row keys")
    completed = set(keys)
    for mu in mus:
        bin_fields = templates[mu]
        bin_means = exposure * (component_coefficients(mu) @ bin_fields)
        bin_source_invalid = np.any(~np.isfinite(bin_means)) or np.any(bin_means < 0)
        for toy_id in range(int(n_toys)):
            needed = [case for case in CASES if (mu, toy_id, case) not in completed]
            if not needed:
                continue
            paired = None
            paired_error = None
            if any(case in needed for case in ("analytic_simulator", "learned_simulator", "binned_simulator")):
                try:
                    x = sample_simulator_poisson(run, mu, _rng(seed, mu, toy_id, 0),
                                                 exposure=exposure, selector=selector, threshold=threshold)
                    analytic_events = run.model.component_densities(x).T
                    paired = (x, analytic_events)
                except (ValueError, FloatingPointError, RuntimeError) as error:
                    paired_error = f"Simulator toy generation failed: {error}"
            for case in needed:
                n_observed = np.nan
                source_stream = 0 if case.endswith("simulator") else (1 if case == "learned_model" else 2)
                try:
                    if case.endswith("simulator"):
                        if paired_error:
                            raise ValueError(paired_error)
                        x, analytic_events = paired
                        n_observed = len(x)
                        if case == "analytic_simulator":
                            likelihood = StatOnlyLikelihood(analytic_events, rates, exposure=exposure)
                        elif case == "learned_simulator":
                            likelihood = StatOnlyLikelihood(workspace.evaluate(x)[0], learned_rates,
                                                            exposure=exposure, validity_intervals=learned_guard.negative_intervals)
                        else:
                            z = reference_observable(analytic_events, rates, mu, reference_power)
                            counts = np.histogram(z, edges)[0]
                            likelihood = StatOnlyLikelihood.from_binned(bin_fields, counts, exposure=exposure)
                    elif case == "learned_model":
                        if mu in source_errors:
                            raise ValueError("Learned toy generator is invalid: " + source_errors[mu])
                        indices, multiplicities = sources[mu].draw_counts(_rng(seed, mu, toy_id, 1))
                        n_observed = int(np.sum(multiplicities))
                        likelihood = StatOnlyLikelihood(learned[:, indices], learned_rates, multiplicities,
                                                        exposure=exposure, validity_intervals=learned_guard.negative_intervals)
                    else:
                        if bin_source_invalid:
                            raise ValueError(f"Binned model has invalid Poisson means at mu={mu}")
                        counts = _rng(seed, mu, toy_id, 2).poisson(bin_means)
                        n_observed = int(np.sum(counts))
                        likelihood = StatOnlyLikelihood.from_binned(bin_fields, counts, exposure=exposure)
                    result = _fit(likelihood, mu, bounds, grid_size)
                except (ValueError, FloatingPointError, RuntimeError) as error:
                    result = _failure(f"{type(error).__name__}: {error}")
                records.append(dict(mu_true=mu, mu_test=mu, eta=mu, toy_id=toy_id, case=case,
                                    label=CASES[case], source_stream=source_stream,
                                    n_observed=n_observed, exposure_multiplier=exposure,
                                    template_source=template_metadata["template_sources"][str(mu)] if case.startswith("binned") else "unbinned",
                                    **result))
                completed.add((mu, toy_id, case))
            _atomic_csv(pd.DataFrame(records), result_path)
            if (toy_id + 1) % progress_every == 0 or toy_id + 1 == n_toys:
                print(f"mu={mu:g}: {toy_id + 1}/{n_toys} toys complete for all five likelihood/source combinations", flush=True)
    all_results = pd.DataFrame(records)
    results = all_results.loc[all_results.toy_id < n_toys].sort_values(["mu_test", "toy_id", "case"]).reset_index(drop=True)
    summary, coverage = summarize_toys(results)
    for suffix, frame in (("summary", summary), ("coverage", coverage), ("bin_yields", bin_yields)):
        _atomic_csv(frame, run.path("results", f"{output_tag}_{suffix}.csv"))
    return dict(results=results, summary=summary, coverage=coverage, bin_yields=bin_yields, metadata=metadata)


def plot_toy_study(study, output):
    """Save mplhep-style distributions, inclusive survival curves and coverage."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    from scipy.stats import chi2
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results, coverage = study["results"], study["coverage"]
    mus = sorted(results.mu_test.unique())
    colors = dict(zip(CASES, ("black", "#d55e00", "#009e73", "#0072b2", "#cc79a7")))
    figures = {}
    with plt.style.context(hep.style.ATLAS):
        for name in ("q_histograms", "q_survival", "mu_hat"):
            fig, axes = plt.subplots(1, len(mus), figsize=(7 * len(mus), 5.5), squeeze=False)
            for ax, mu in zip(axes[0], mus):
                subset = results.loc[results.mu_test == mu]
                finite = subset.loc[subset.valid.astype(bool) & np.isfinite(subset.q)]
                if name == "mu_hat":
                    lo, hi = study["metadata"]["configuration"]["mu_bounds"]
                    edges = np.linspace(lo, hi, 31)
                else:
                    top = max(5., float(finite.q.max()) * 1.02) if len(finite) else 5.
                    edges = np.linspace(0., top, 31)
                for case, label in CASES.items():
                    group = finite.loc[finite.case == case]
                    if group.empty:
                        continue
                    n = len(group)
                    if name == "q_survival":
                        q = np.sort(group.q.to_numpy())
                        thresholds = np.unique(np.r_[0., q, edges[-1]])
                        # Inclusive survival displays the boundary atom at zero.
                        survival = (n - np.searchsorted(q, thresholds, side="left")) / n
                        ax.step(thresholds, survival, where="pre", color=colors[case], label=label, linewidth=1.7)
                    else:
                        values = group.mu_hat if name == "mu_hat" else group.q
                        counts = np.histogram(values, edges)[0]
                        norm = n * np.diff(edges)
                        hep.histplot(counts / norm, edges, yerr=np.sqrt(counts) / norm,
                                     histtype="step", color=colors[case], label=label, ax=ax, linewidth=1.6)
                ax.set_title(rf"$\mu_{{\mathrm{{true}}}}=\mu_{{\mathrm{{test}}}}={mu:g}$")
                if name == "mu_hat":
                    ax.axvline(mu, color="0.4", linestyle=":", linewidth=1.)
                    ax.set(xlabel=r"$\widehat{\mu}$", ylabel="Toy probability density", xlim=(lo, hi))
                elif name == "q_survival":
                    grid = np.linspace(.0001, edges[-1], 300)
                    factor = .5 if mu == 0. else 1.
                    asym_label = (r"$\frac{1}{2}\delta_0+\frac{1}{2}\chi^2_1$ reference" if mu == 0. else r"$\chi^2_1$ reference")
                    ax.plot(grid, factor * chi2.sf(grid, 1), "--", color="0.5", label=asym_label)
                    ax.set(xlabel=r"$q_\mu=-2\log[L(\mu)/L(\widehat{\mu})]$", ylabel=r"$P(q_\mu\geq q)$", xlim=(0, edges[-1]), ylim=(0, 1.05))
                else:
                    ax.set(xlabel=r"$q_\mu=-2\log[L(\mu)/L(\widehat{\mu})]$", ylabel="Toy probability density", xlim=(0, edges[-1]))
                ax.legend(fontsize=10, frameon=False)
                failed = len(subset) - len(finite)
                if failed:
                    ax.text(.02, .96, f"{failed} failed fits; see failure table", transform=ax.transAxes,
                            va="top", fontsize=10, color="#d55e00")
            fig.tight_layout()
            figures[name] = fig
        fig, axes = plt.subplots(1, len(mus), figsize=(7 * len(mus), 5.5), squeeze=False)
        for ax, mu in zip(axes[0], mus):
            for i, (case, label) in enumerate(CASES.items()):
                group = coverage.loc[(coverage.mu_test == mu) & (coverage.case == case)]
                ax.errorbar(group.nominal + (i - 2) * .0015, group.coverage, yerr=group.binomial_se,
                            marker="o", linestyle="none", color=colors[case], label=label, markersize=4)
            ax.plot([.60, 1.], [.60, 1.], ":", color="0.5", linewidth=1)
            ax.set(xlabel="Nominal acceptance probability", ylabel="Empirical acceptance probability",
                   title=rf"$\mu_{{\mathrm{{true}}}}=\mu_{{\mathrm{{test}}}}={mu:g}$", xlim=(.65, 1.01), ylim=(0, 1.03))
            ax.legend(fontsize=10, frameon=False, loc="lower right")
        fig.tight_layout()
        figures["coverage"] = fig
        for name, fig in figures.items():
            for extension in ("png", "pdf"):
                fig.savefig(output / f"11_{name}.{extension}", dpi=160, bbox_inches="tight")
    return figures
