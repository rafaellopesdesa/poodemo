"""Learned compression with independent, physical simulation-derived yields.

The neural networks assign bins only. Every nominal/varied bin expectation and
every Asimov event weight is calculated from the analytical physical simulator.
The two banks are used once with matched template/data integration (A/A), then
with independent Asimov integration (A/B); neither bank is renormalized to force
closure. The observable's original selected-rate normalizers remain frozen.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from . import pipeline
from .closure_validation import IndependentValidationLikelihood
from .data import PROCESSES, save_json
from .diagnostics import local_asimov_information, local_width
from .inference import TemplateLikelihood, component_coefficients, histogram_components
from .toy_study import reference_observable


ANALYTIC_MODEL = "analytic unbinned"
MAP_MODELS = {"analytic": "binned analytic map", "learned": "binned learned map"}


def _check_bank(bank, name):
    weights = np.asarray(bank["weights"], dtype=float)
    n = len(weights)
    if weights.shape != (n,) or not n or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError(f"{name} requires positive finite integration weights")
    for field, shape in (("nominal", (4, n)), ("down", (1, 4, n)), ("up", (1, 4, n)),
                         ("truth", (n,))):
        values = np.asarray(bank[field])
        if values.shape != shape or np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{name}.{field} has invalid shape or physical values")
    if np.asarray(bank["learned_nominal"]).shape != (4, n):
        raise ValueError(f"{name}.learned_nominal must have shape (4,N)")


def _map(bank, rates, eta, power, floor, source, bank_name):
    """Audit invalid learned interference before assigning any event to a bin."""
    endpoint = pipeline.reference_ratio_eta(eta, floor)
    fields = np.asarray(bank["nominal" if source == "analytic" else "learned_nominal"])
    with np.errstate(invalid="ignore", over="ignore"):
        density = component_coefficients(endpoint) @ fields
        reference = component_coefficients(1.) @ fields
    nonfinite = ~np.isfinite(density) | ~np.isfinite(reference) | np.any(~np.isfinite(fields), axis=0)
    negative = (density < 0) | (reference <= 0) | np.any(fields < 0, axis=0)
    invalid = nonfinite | negative
    expected = np.asarray(bank["weights"]) * bank["truth"]
    record = dict(bank=bank_name, model=MAP_MODELS[source], eta=float(eta),
                  eta_effective=endpoint, valid=False,
                  negative_fraction=float(np.mean(negative)),
                  nonfinite_fraction=float(np.mean(nonfinite)),
                  invalid_fraction=float(np.mean(invalid)),
                  invalid_expected_fraction=float(expected[invalid].sum() / expected.sum()),
                  message="")
    if np.any(invalid):
        record["message"] = "Invalid map intensity; no events were clipped or discarded"
        return None, record
    try:
        values = reference_observable(fields, rates, eta, power, reference_eta_floor=floor)
    except ValueError as error:
        record["message"] = str(error)
        return None, record
    if np.any(~np.isfinite(values)):
        record["message"] = "Nonfinite transformed coordinate"
        return None, record
    record.update(valid=True, message="All events retained")
    return values, record


def _bin_moments(bank, values, edges):
    """Physical expected counts, MC variance and descriptive weighted ESS.

    The variance is the fixed-size stratified importance-sampling estimator.
    Rejected events are accounted for through generated_counts, contributing
    zero to both sums; sum(w^2) alone is not the variance of this estimator.
    """
    contributions = np.asarray(bank["weights"]) * bank["truth"]
    counts = np.histogram(values, edges, weights=contributions)[0]
    squares = np.histogram(values, edges, weights=contributions**2)[0]
    ess = np.divide(counts**2, squares, out=np.zeros_like(counts), where=squares > 0)
    metadata = bank.get("metadata", {})
    generated = metadata.get("generated_counts")
    sources = metadata.get("sources", ("S", "B", "NI"))
    variance = np.full_like(counts, np.nan)
    if generated is not None and "stratum" in bank:
        variance = np.zeros_like(counts)
        for label, source in enumerate(sources):
            n_generated = int(generated[source])
            if n_generated <= 1:
                raise ValueError("MC variance requires at least two generated events per stratum")
            mask = np.asarray(bank["stratum"]) == label
            if np.count_nonzero(mask) > n_generated:
                raise ValueError("Selected stratum size exceeds generated count")
            first = np.histogram(values[mask], edges, weights=contributions[mask])[0]
            second = np.histogram(values[mask], edges, weights=contributions[mask]**2)[0]
            variance += n_generated / (n_generated - 1) * np.maximum(second - first**2 / n_generated, 0.)
    return counts, np.sqrt(variance), ess


def _fresh_unbinned_scans(run, integration, observations, grid, systematics, comparison):
    """Recompute this bank's baseline, caching the common unrestricted fit."""
    fields = lambda bank: (bank["nominal"], bank["down"][0], bank["up"][0])
    likelihood = IndependentValidationLikelihood(
        fields(integration), integration["weights"], fields(observations),
        np.asarray(observations["weights"]) * observations["truth"],
        reference_intensity=observations["truth"])
    options = dict(systematics=systematics, mu_bounds=tuple(run.config["mu_fit_bounds"]),
                   alpha_bounds=tuple(run.config["nuisance_bounds"]),
                   mu_starts=pipeline.mu_fit_starts(run))
    best = likelihood.fit(initial_mu=run.config["asimov_mu"], **options)
    conditional, previous_alpha = [], np.zeros(likelihood.n_nuisance)
    for mu in grid:
        null = likelihood.fit(mu_fixed=float(mu), initial_alpha=previous_alpha, **options)
        conditional.append(null)
        if null.success:
            previous_alpha = null.alpha
        # A newly discovered lower basin is a feasible seed for the free fit.
        if null.success and (not best.success or null.nll < best.nll - 1e-6):
            candidate = likelihood.fit(initial_mu=float(mu), initial_alpha=null.alpha, **options)
            if candidate.success and (not best.success or candidate.nll < best.nll):
                best = candidate
    rows = []
    for mu, null in zip(grid, conditional):
        difference = null.nll - best.nll
        valid = bool(null.success and best.success and np.isfinite(difference) and difference >= -1e-6)
        rows.append(dict(comparison=comparison, model=ANALYTIC_MODEL, mu=float(mu), eta=float(mu),
                         eta_effective=np.nan, systematics=bool(systematics),
                         q=max(0., float(difference)) if valid else np.nan,
                         valid=valid, mu_hat=best.mu, alpha_NI=float(null.alpha[0]),
                         fit_message=null.message + "; " + best.message,
                         baseline_source="Recomputed with this study's physical simulation banks"))
    information = local_width(local_asimov_information(likelihood, run.config["asimov_mu"])["matrix"],
                              systematics=systematics)
    return rows, information


def _fit_bins(run, nominal, down, up, counts, eta, systematics):
    """Make unsupported validation bins explicit; never add pseudocounts."""
    unsupported = (nominal.sum(axis=0) == 0) & (counts > 0)
    if np.any(unsupported):
        return dict(q=np.nan, valid=False, mu_hat=np.nan, alpha_NI=np.nan,
                    fit_message=f"{int(unsupported.sum())} positive data bins have zero template support"), None
    variations = (down, up) if systematics else (np.empty((0,) + nominal.shape),) * 2
    try:
        likelihood = TemplateLikelihood(nominal, *variations, np.ones(len(counts)), counts)
        result = pipeline._fit_test(run, likelihood, eta, systematics)
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return dict(q=np.nan, valid=False, mu_hat=np.nan, alpha_NI=np.nan,
                    fit_message=str(error)), None
    return result, likelihood


def run_learned_compression_study(run, banks, *, n_bins=36, reference_power=100.,
                                  reference_eta_floor=.2, scan_grid=None,
                                  systematics=(False, True),
                                  distribution_etas=(0., .3, .6, 1., 1.5),
                                  output_tag="13_learned_compression_36_floor02"):
    """Compare two compression maps with the same physical bin likelihoods.

    A/A isolates compression (and the chosen bin-level nuisance morph). A/B
    independently tests finite-simulation closure, including the unbinned
    integral. The trained networks and their original normalizers stay frozen.
    No learned density is used to predict yields or construct Asimov weights.
    """
    started = perf_counter()
    if not isinstance(output_tag, str) or not output_tag or output_tag in (".", "..") or Path(output_tag).name != output_tag:
        raise ValueError("output_tag must be a nonempty directory name")
    edges = pipeline.observable_bin_edges(n_bins, "reference_ratio")
    floor = pipeline.reference_ratio_eta(0., reference_eta_floor)
    power = float(reference_power)
    if not np.isfinite(power) or power <= 0:
        raise ValueError("reference_power must be finite and positive")
    modes = tuple(dict.fromkeys(systematics))
    if not modes or any(not isinstance(mode, (bool, np.bool_)) for mode in modes):
        raise ValueError("systematics must contain booleans")
    truth = float(run.config["asimov_mu"])
    if not np.isfinite(truth) or truth <= 0:
        raise ValueError("Local-width diagnostics require positive asimov_mu")
    grid = np.asarray(pipeline.mu_grid(run) if scan_grid is None else scan_grid, dtype=float)
    low, high = run.config["mu_fit_bounds"]
    if grid.ndim != 1 or not len(grid) or np.any(~np.isfinite(grid)) or np.any(grid < max(low, 0.)) or np.any(grid > high):
        raise ValueError("scan_grid must lie inside the physical mu_fit_bounds")
    if not low <= truth <= high:
        raise ValueError("asimov_mu must lie inside mu_fit_bounds")
    grid = np.unique(np.r_[grid, truth])
    distributions = np.asarray(distribution_etas, dtype=float)
    if distributions.ndim != 1 or np.any(~np.isfinite(distributions)) or np.any(distributions < 0):
        raise ValueError("distribution_etas must be finite and nonnegative")
    etas = np.unique(np.r_[grid, distributions])
    a, b = banks["template"], banks["validation"]
    _check_bank(a, "template")
    _check_bank(b, "validation")
    rates = np.asarray(banks["rates"], dtype=float)
    if rates.shape != (4,) or np.any(~np.isfinite(rates)) or np.any(rates <= 0):
        raise ValueError("Original frozen selected rates must be positive and finite")
    scans, information, closure, components, diagnostics, timing = [], [], [], [], [], []
    baseline_widths = {}
    for syst in modes:
        for comparison, data in (("matched", a), ("independent", b)):
            stage = perf_counter()
            rows, width = _fresh_unbinned_scans(run, a, data, grid, syst, comparison)
            scans.extend(rows)
            if comparison == "matched":
                baseline_widths[syst] = width
                information.append(dict(model=ANALYTIC_MODEL, eta=np.nan, eta_effective=np.nan,
                                        information_mu=truth, systematics=bool(syst),
                                        information_fraction=1., **width))
            timing.append(dict(stage="analytic unbinned", comparison=comparison,
                               systematics=bool(syst), seconds=perf_counter() - stage))
            print(f"Analytical unbinned: {comparison}, {'NI profiled' if syst else 'NI fixed'} fitted", flush=True)
    for eta in etas:
        endpoint = pipeline.reference_ratio_eta(eta, floor)
        is_scan = bool(np.any(grid == eta))
        for source, label in MAP_MODELS.items():
            za, diagnostic_a = _map(a, rates, eta, power, floor, source, "template")
            zb, diagnostic_b = _map(b, rates, eta, power, floor, source, "validation")
            diagnostics.extend((diagnostic_a, diagnostic_b))
            nominal = down = up = counts_a = counts_b = None
            if za is not None:
                nominal = histogram_components(za, a["nominal"], a["weights"], edges)
                down = np.stack([histogram_components(za, values, a["weights"], edges) for values in a["down"]])
                up = np.stack([histogram_components(za, values, a["weights"], edges) for values in a["up"]])
                counts_a, se_a, ess_a = _bin_moments(a, za, edges)
                for j, process in enumerate(PROCESSES):
                    for index in range(n_bins):
                        components.append(dict(model=label, eta=float(eta), eta_effective=endpoint,
                                               process=process, bin=index, low=edges[index], high=edges[index+1],
                                               expected_yield=nominal[j, index]))
            if zb is not None:
                counts_b, se_b, ess_b = _bin_moments(b, zb, edges)
            if counts_a is not None and counts_b is not None:
                for index in range(n_bins):
                    ca, cb = counts_a[index], counts_b[index]
                    sa, sb = se_a[index], se_b[index]
                    ratio = cb / ca if ca > 0 else np.nan
                    ratio_se = np.sqrt((sb / ca)**2 + (cb * sa / ca**2)**2) if ca > 0 else np.nan
                    uncertainty = np.hypot(sa, sb)
                    closure.append(dict(model=label, eta=float(eta), eta_effective=endpoint, bin=index,
                                        low=edges[index], high=edges[index+1], template=ca, validation=cb,
                                        template_se=sa, validation_se=sb, ratio=ratio, ratio_se=ratio_se,
                                        pull=(cb-ca)/uncertainty if uncertainty > 0 else np.nan,
                                        template_ess=ess_a[index], validation_ess=ess_b[index],
                                        valid=bool(ca > 0 or cb == 0)))
            if not is_scan:
                continue
            for syst in modes:
                matched_model = None
                for comparison, counts in (("matched", counts_a), ("independent", counts_b)):
                    if nominal is None or counts is None:
                        result = dict(q=np.nan, valid=False, mu_hat=np.nan, alpha_NI=np.nan,
                                      fit_message="Invalid compression map; inspect map_diagnostics")
                    else:
                        result, likelihood = _fit_bins(run, nominal, down, up, counts, eta, syst)
                        if comparison == "matched":
                            matched_model = likelihood
                    scans.append(dict(comparison=comparison, model=label, mu=float(eta), eta=float(eta),
                                      eta_effective=endpoint, systematics=bool(syst), **result))
                width = dict(information_mu_effective=np.nan, sigma_mu=np.nan,
                             width_valid=False, width_message="Invalid map or histogram fit")
                if matched_model is not None:
                    try:
                        width = local_width(local_asimov_information(matched_model, truth)["matrix"],
                                            systematics=syst)
                    except (ValueError, FloatingPointError) as error:
                        width["width_message"] = str(error)
                denominator = baseline_widths[syst]["information_mu_effective"]
                information.append(dict(model=label, eta=float(eta), eta_effective=endpoint,
                                        information_mu=truth, systematics=bool(syst),
                                        information_fraction=width["information_mu_effective"]/denominator,
                                        **width))
        if is_scan:
            print(f"Frozen physical templates fitted at eta={eta:g} (map endpoint {endpoint:g})", flush=True)
    frames = dict(scans=pd.DataFrame(scans), information=pd.DataFrame(information),
                  bin_closure=pd.DataFrame(closure), component_templates=pd.DataFrame(components),
                  map_diagnostics=pd.DataFrame(diagnostics), timings=pd.DataFrame(timing))
    baseline = frames["scans"].loc[frames["scans"].model == ANALYTIC_MODEL,
                                    ["comparison", "systematics", "mu", "q"]].rename(columns={"q": "q_analytic"})
    frames["scans"] = frames["scans"].merge(baseline, on=["comparison", "systematics", "mu"], validate="many_to_one")
    frames["scans"]["delta_q"] = frames["scans"].q - frames["scans"].q_analytic
    metadata = dict(n_bins=int(n_bins), bin_edges=edges.tolist(), reference_ratio_power=power,
                    reference_ratio_eta_floor=floor, eta_reference=1., asimov_mu=truth,
                    scan_grid=grid.tolist(), distribution_etas=distributions.tolist(),
                    systematics=list(map(bool, modes)), frozen_rates=rates.tolist(),
                    frozen_network_normalizers=banks.get("metadata", {}).get("frozen_normalizers", {}),
                    output_tag=output_tag, bank_metadata=banks.get("metadata", {}),
                    likelihood_source="Physical simulator integrals on template bank A",
                    asimov_source="A/A matched and independently integrated A/B validation",
                    learned_use="Observable coordinates only; never expected-yield or Asimov weights",
                    normalization="Original notebook 03 rates frozen for both banks and both maps",
                    morph_order="Binned NI anchors morphed after integration; unbinned NI morph before integration",
                    finite_mc_uncertainty="Stratified integration standard errors shown diagnostically, not profiled",
                    coverage_scope="Asimov compression/closure study; no repeated-experiment coverage claim",
                    mu_fit_bounds=list(run.config["mu_fit_bounds"]),
                    nuisance_bounds=list(run.config["nuisance_bounds"]),
                    elapsed_seconds=perf_counter()-started)
    output = run.path("results", output_tag)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output / (name + ".csv"), index=False)
    save_json(output / "metadata.json", metadata)
    return {**frames, "metadata": metadata}
