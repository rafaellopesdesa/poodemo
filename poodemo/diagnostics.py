"""Estimator and local-width diagnostics for a frozen observable.

Every eta compresses the SAME Asimov experiment generated at ``asimov_mu``.
The information is evaluated at that fixed truth, not at mu=eta. Reported
widths are local Fisher/Asimov widths, not global confidence intervals; they
do not describe a remote secondary minimum or the fluctuations of one toy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .inference import FitResult, TemplateLikelihood


def local_asimov_information(model, mu, *, alpha=None, nuisance_step=1e-4):
    """Return extended Fisher information, including auxiliary constraints.

    Parameter order is (mu, alpha_1, ...). The POI derivative is analytical;
    nuisance derivatives differentiate the actual normalized exp-poly model
    with central differences and Richardson refinement. Comparing the two
    step sizes provides a numerical derivative diagnostic.

    For bins, weights are one and intensities are bin yields. For an
    unbinned model, the same expression uses its integration quadrature.
    This is also half the Hessian of the correctly specified Asimov -2 log L.
    """
    mu = float(mu)
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("Local information requires an interior positive mu")
    if not np.isfinite(nuisance_step) or nuisance_step <= 0:
        raise ValueError("nuisance_step must be positive and finite")
    alpha = model._alpha(alpha)
    weights = np.asarray(model.quadrature_weights, dtype=float)
    intensity = np.asarray(model.intensity(mu, alpha), dtype=float)
    if np.any(~np.isfinite(intensity)) or np.any(intensity < 0):
        raise ValueError("Fisher information requires nonnegative finite intensities")

    inverse = 0.5 / np.sqrt(mu)
    derivatives = [np.array([1 - inverse, inverse, -inverse, 0.]) @ model.components(alpha)]
    relative_errors = []
    for k in range(model.n_nuisance):
        shift = np.zeros(model.n_nuisance)
        shift[k] = nuisance_step
        coarse = (model.intensity(mu, alpha + shift) - model.intensity(mu, alpha - shift)) / (2 * nuisance_step)
        fine = (model.intensity(mu, alpha + shift / 2) - model.intensity(mu, alpha - shift / 2)) / nuisance_step
        derivative = (4 * fine - coarse) / 3
        derivatives.append(derivative)
        norm = np.linalg.norm(derivative)
        relative_errors.append(float(np.linalg.norm(fine - coarse) / norm) if norm else 0.)
    derivatives = np.asarray(derivatives)
    if np.any(~np.isfinite(derivatives)):
        raise ValueError("Intensity derivatives are not finite")
    active = (weights > 0) & (intensity > 0)
    if np.any((weights > 0) & (intensity == 0) & np.any(derivatives != 0, axis=0)):
        raise ValueError("Zero intensity with nonzero derivative is not a regular Fisher point")
    rate = float(intensity @ weights)
    if rate <= 0 or not np.isfinite(rate):
        raise ValueError("The expected event count must be positive and finite")
    weighted_derivatives = derivatives[:, active] * np.sqrt(weights[active] / intensity[active])
    data_matrix = weighted_derivatives @ weighted_derivatives.T
    rate_derivative = derivatives @ weights
    rate_matrix = np.outer(rate_derivative, rate_derivative) / rate
    auxiliary_matrix = np.zeros_like(data_matrix)
    auxiliary_matrix[1:, 1:] = np.diag(1. / model.aux_sigma**2)
    return {
        "matrix": data_matrix + auxiliary_matrix,
        "data_matrix": data_matrix,
        "rate_matrix": rate_matrix,
        "shape_matrix": data_matrix - rate_matrix,
        "auxiliary_matrix": auxiliary_matrix,
        "expected_events": rate,
        "nuisance_derivative_relative_error": max(relative_errors, default=0.),
    }


def local_width(information, *, systematics=False, relative_tolerance=1e-12):
    """Convert Fisher information into a local POI width, profiling if asked.

    A Schur complement includes nuisance constraints already present in the
    supplied matrix. Unresolved or zero information gives an infinite/invalid
    width, never an artificially precise result from flooring the information.
    """
    matrix = np.asarray(information, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("information must be a nonempty square matrix")
    if np.any(~np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
        return dict(information_mu_effective=np.nan, sigma_mu=np.nan,
                    width_valid=False, width_message="Nonfinite or asymmetric information matrix")
    information_mu = float(matrix[0, 0])
    effective = information_mu
    if systematics and len(matrix) > 1:
        nuisance = matrix[1:, 1:]
        eigenvalues = np.linalg.eigvalsh(nuisance)
        if eigenvalues[0] <= relative_tolerance * max(eigenvalues[-1], np.finfo(float).tiny):
            return dict(information_mu_effective=np.nan, sigma_mu=np.inf,
                        width_valid=False, width_message="Nuisance information is singular or numerically unresolved")
        effective -= float(matrix[0, 1:] @ np.linalg.solve(nuisance, matrix[1:, 0]))
    scale = max(abs(information_mu), np.finfo(float).tiny)
    if effective <= relative_tolerance * scale:
        message = "No resolved positive local POI information"
        if effective < -relative_tolerance * scale:
            message = "Negative efficient information; inspect the model or numerical derivatives"
        return dict(information_mu_effective=effective, sigma_mu=np.inf,
                    width_valid=False, width_message=message)
    return dict(information_mu_effective=effective, sigma_mu=float(1 / np.sqrt(effective)),
                width_valid=True, width_message="Local Fisher/Asimov width")


def _estimator_record(run, likelihood, information, systematics, *, fit=None,
                      nll_at_truth=None, fit_source="NumPy/SciPy fit"):
    """Attach truth-local information to a fitted, already frozen model."""
    truth = float(run.config["asimov_mu"])
    bounds = tuple(run.config["mu_fit_bounds"])
    # Interference can produce additional minima far from one. Span the actual
    # configured interval instead of assuming the former narrow default bounds.
    starts = np.unique(np.r_[truth, np.linspace(*bounds, 7)]).tolist()
    if fit is None:
        fit = likelihood.fit(initial_mu=truth, initial_alpha=np.zeros(likelihood.n_nuisance),
                             systematics=systematics, mu_bounds=bounds,
                             alpha_bounds=tuple(run.config["nuisance_bounds"]), mu_starts=starts)
    width = local_width(information["matrix"], systematics=systematics)
    if nll_at_truth is None:
        nll_at_truth = likelihood.nll(truth, np.zeros(likelihood.n_nuisance))
    fit_valid = bool(fit.success and np.isfinite(fit.nll) and fit.nll <= nll_at_truth + 1e-6)
    fit_message = fit.message
    if np.isfinite(fit.nll) and fit.nll > nll_at_truth + 1e-6:
        fit_message += "; fit is worse than the feasible generating truth"
    return {
        "truth": truth, "information_mu": truth,
        "mu_hat": fit.mu, "mu_hat_minus_truth": fit.mu - truth,
        "alpha_NI_hat": fit.alpha[0] if len(fit.alpha) else 0.,
        "systematics": bool(systematics), "nll": fit.nll,
        "nll_at_truth": nll_at_truth,
        "fit_valid": fit_valid, "fit_message": fit_message, "fit_source": fit_source,
        "information_mu_mu": float(information["matrix"][0, 0]),
        "rate_information_mu_mu": float(information["rate_matrix"][0, 0]),
        "shape_information_mu_mu": float(information["shape_matrix"][0, 0]),
        "expected_events": information["expected_events"],
        "nuisance_derivative_relative_error": information["nuisance_derivative_relative_error"],
        "width_method": "Fisher/Asimov local curvature at fixed generating truth",
        **width, "valid": bool(fit_valid and width["width_valid"]),
    }


def _unbinned_baselines(run, parent, information):
    """Reuse notebook 3 toolkit fits after checking their current-model NLL.

    The run/selection state checks own provenance. This check verifies the
    stored fit parameters and likelihood convention against the current NumPy
    model; it is not another optimization or a proof about every remote minimum.
    An invalid or incomplete saved result must be fixed in notebook 3 instead
    of being silently replaced by a different optimizer here.
    """
    path = run.path("results", "unbinned_fits.csv")
    if not path.exists():
        return {syst: _estimator_record(run, parent, information, syst)
                for syst in (False, True)}
    saved = pd.read_csv(path)
    required = {"model", "systematics", "mu_hat", "alpha_NI_hat", "nll", "valid"}
    if not required.issubset(saved.columns):
        raise ValueError("Saved unbinned fits are incomplete; rerun notebook 3")
    saved = saved.loc[saved.model == "analytic unbinned"].copy()
    for column in ("systematics", "valid"):
        parsed = saved[column].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if parsed.isna().any():
            raise ValueError(f"Saved unbinned {column} flags are invalid; rerun notebook 3")
        saved[column] = parsed
    truth_nll = parent.nll(float(run.config["asimov_mu"]), np.zeros(parent.n_nuisance))
    results = {}
    for syst in (False, True):
        matching = saved.loc[saved.systematics == syst]
        if len(matching) != 1:
            raise ValueError(f"Expected one saved analytic unbinned fit for systematics={syst}; rerun notebook 3")
        row = matching.iloc[0]
        if not row.valid:
            raise ValueError(f"Saved analytic unbinned fit is invalid for systematics={syst}; rerun notebook 3")
        mu, alpha, saved_nll = float(row.mu_hat), float(row.alpha_NI_hat), float(row.nll)
        if not np.all(np.isfinite([mu, alpha, saved_nll])):
            raise ValueError("Saved analytic unbinned fit has nonfinite values; rerun notebook 3")
        if not syst and not np.isclose(alpha, 0., rtol=0., atol=1e-10):
            raise ValueError("Saved stat-only fit did not fix the NI nuisance to zero; rerun notebook 3")
        nll = parent.nll(mu, np.array([alpha]))
        if not np.isfinite(nll) or not np.isclose(nll, saved_nll, rtol=1e-7, atol=1e-6):
            raise ValueError("Saved toolkit fit NLL disagrees with the current analytical model; rerun notebook 3")
        fit = FitResult(mu, np.array([alpha]), nll, True,
                        "Reused notebook 3 toolkit fit; NumPy likelihood consistency checked", 0)
        results[syst] = _estimator_record(run, parent, information, syst, fit=fit,
                                         nll_at_truth=truth_nll,
                                         fit_source="Saved notebook 3 toolkit fit")
    return results


def run_estimator_study(run, *, observable="score"):
    """Save/return mu_hat(eta) and local sigma_mu(eta) for one Asimov truth.

    Uses the finest configured ``bin_counts`` and the existing eta scan grid.
    At each eta the observable and observed Asimov histogram are frozen before
    either fit. Stat-only and NI-profiled results share those same data.
    ``observable`` selects the local score or the density-ratio observable;
    both use the same configured bin count, generating truth and baselines.
    Ratio results are saved separately as ``ratio_estimator_eta.csv``.
    The exact unbinned baselines reuse notebook 3 toolkit fits, checking their
    likelihood values before repeating them over eta. If no saved fits exist,
    standalone runs fit these baselines once. Identified Asimov fits stay at truth;
    a flat mu_hat curve does not imply equal estimates for fluctuated data.

    Nuisance anchors are morphed AFTER histogramming, matching the existing
    direct-template study. Its profiled comparison therefore includes the
    morph/integration approximation, as well as compression and binning.
    The Fisher width is not a global interval across possible secondary minima.
    """
    from .pipeline import (prepare_quadrature, observable_metadata, observable_values,
                           observable_configuration, observable_bin_edges, _positive_study_grid)
    from .data import save_json

    options = observable_metadata(observable)
    settings = observable_configuration(run, observable)
    etas = _positive_study_grid(run)
    quad = prepare_quadrature(run)
    parent = TemplateLikelihood(quad["nominal"], quad["down"], quad["up"],
                                quad["weights"], quad["truth"])
    truth = float(run.config["asimov_mu"])
    if not run.config["mu_fit_bounds"][0] < truth < run.config["mu_fit_bounds"][1]:
        raise ValueError("A local Asimov width requires truth inside the mu fit bounds")
    if parent.n_nuisance != 1:
        raise ValueError("This study expects the single NI nuisance")
    n_bins = max(run.config["bin_counts"])
    edges = observable_bin_edges(n_bins, observable)
    full_information = local_asimov_information(parent, truth)
    baselines = _unbinned_baselines(run, parent, full_information)
    rows = []
    for eta in etas:
        z = observable_values(run, quad, float(eta), observable=observable)
        histogram = parent.binned(z, edges, morph_order="after", eta=float(eta))
        information = local_asimov_information(histogram, truth)
        for syst in (False, True):
            rows.append({"eta": float(eta), "model": options["direct_label"], "n_bins": n_bins,
                         "morph_order": "after", **_estimator_record(run, histogram, information, syst)})
            rows.append({"eta": float(eta), "model": "analytic unbinned", "n_bins": 0,
                         "morph_order": "unbinned", **baselines[syst]})
        print(f"Estimator and local-width study at eta={eta:.3g} ready", flush=True)
    result = pd.DataFrame(rows)
    for key, value in settings.items():
        result[key] = value
    filename = f"{options['prefix']}estimator_eta.csv"
    path = run.path("results", filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    save_json(run.path("results", f"{options['prefix']}estimator_config.json"),
              {**settings, "n_bins": int(n_bins), "bin_edges": edges.tolist(),
               "scan_grid": etas.tolist(), "asimov_mu": truth})
    return result
