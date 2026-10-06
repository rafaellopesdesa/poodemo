"""The five notebook stages, with all persistent products under a run directory."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

from .data import (Run, create_run, PROCESSES, SAMPLES, generate_samples,
                   raw_split, selected_split, materialize_selection, save_json, save_array)


def generate_data(run):
    return pd.DataFrame(generate_samples(run))


def _training_config(run, preselection=False):
    from .training import TrainingConfig
    c = run.config
    return TrainingConfig(epochs=c["preselection_epochs" if preselection else "epochs"], patience=c["patience"],
                          batch_size=c["batch_size"], width=c["n_neurons"], depth=c["n_hidden"],
                          max_train_per_class=c["preselection_train_cap" if preselection else "ratio_train_cap"],
                          ensemble_size=1 if preselection else c["ensemble_size"],
                          seed=c["seed"], progress_bar=c["mode"] != "smoke")


def run_preselection(run):
    from .training import fit_preselection, choose_preselection_threshold
    train = {p: raw_split(run, p, "preselection_train") for p in ("S", "B", "NI")}
    validation = {p: raw_split(run, p, "preselection_validation") for p in train}
    predictor = fit_preselection(train, validation, run.path("models", "preselection"),
                                 _training_config(run, preselection=True))
    calibration = {p: raw_split(run, p, "selection_calibration") for p in ("S", "NI")}
    metrics = choose_preselection_threshold(
        predictor, calibration, {p: run.model.component_yield(p) for p in calibration},
        target_ratio=run.config["target_s_over_ni"],
        min_ni_events=20 if run.config["mode"] == "smoke" else 200)
    selection = pd.DataFrame(materialize_selection(run, predictor, metrics["threshold"]))
    heldout = selection.query("split == 'integration'").set_index("sample")
    metrics["heldout_s_over_ni"] = float(heldout.loc["S", "expected_yield"] / heldout.loc["NI", "expected_yield"])
    save_json(run.path("results", "preselection.json"), metrics)
    selection.to_csv(run.path("results", "selection.csv"), index=False)
    return {**metrics, "selection": selection}


RATIO_TASKS = [("SBI", "S"), ("B", "S"), ("NI", "S"),
               ("NI_up", "NI"), ("NI_down", "NI")]


def _ratio_path(run, numerator, denominator):
    return run.path("models", "ratios", f"{numerator}_over_{denominator}")


def train_ratios(run):
    from .training import fit_density_ratio, normalize_ratio, ratio_diagnostics
    reports = []
    manifest = json.loads(run.path("selected", "manifest.json").read_text())
    # Independent acceptance estimates for a diagnostic analytic ratio. These
    # estimates are not targets supplied to the neural-network training.
    accepted_rates = {r["sample"]: r["expected_yield"] for r in manifest["records"]
                      if r["split"] == "integration"}
    for number, (num, den) in enumerate(RATIO_TASKS):
        directory = _ratio_path(run, num, den)
        print(f"Training/loading {num}/{den}", flush=True)
        cfg = replace(_training_config(run), seed=run.config["seed"] + 103*number)
        predictor = fit_density_ratio(
            selected_split(run, num, "train"), selected_split(run, den, "train"),
            selected_split(run, num, "validation"), selected_split(run, den, "validation"),
            directory, cfg, numerator_label=num, denominator_label=den)
        normalization = normalize_ratio(predictor, selected_split(run, den, "calibration"))
        def exact_log_ratio(x):
            cp, an = SAMPLES[num]
            dp, dn = SAMPLES[den]
            numerator = run.model.component_pdf(x, cp, alpha_ni=an)*run.model.component_yield(cp, alpha_ni=an)/accepted_rates[num]
            denominator = run.model.component_pdf(x, dp, alpha_ni=dn)*run.model.component_yield(dp, alpha_ni=dn)/accepted_rates[den]
            return np.log(numerator)-np.log(denominator)
        report = ratio_diagnostics(predictor, selected_split(run, num, "integration"),
                                   selected_split(run, den, "integration"), exact_log_ratio=exact_log_ratio)
        reliability = report.pop("reliability")
        pd.DataFrame({"score_low": reliability["edges"][:-1], "score_high": reliability["edges"][1:],
                      **{k: v for k, v in reliability.items() if k != "edges"}}).to_csv(directory / "reliability.csv", index=False)
        record = {"ratio": f"{num}/{den}", **report, **normalization}
        save_json(directory / "diagnostics.json", record)
        reports.append(record)
    result = pd.DataFrame(reports)
    result.to_csv(run.path("results", "ratio_diagnostics.csv"), index=False)
    return result


def prepare_quadrature(run):
    """Independent, stratified S/B/NI mixture; S remains the ratio reference.

    Integrals are computed on the same positive nodes as the weighted Asimov
    data, so an exact-model Asimov minimum is stationary at the input truth.
    No generated training or calibration event enters these integrals.
    """
    names = ("x", "weights", "nominal", "down", "up", "truth", "stratum", "half")
    paths = {name: run.path("quadrature", name + ".npy") for name in names}
    selection_state = json.loads(run.path("selected", "selection_state.json").read_text())
    quad_state = run.path("quadrature", "selection_state.json")
    if all(path.exists() for path in paths.values()):
        if not quad_state.exists() or json.loads(quad_state.read_text()) != selection_state:
            raise ValueError("Cached quadrature has a different selection. Choose a new run directory.")
        return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    from .training import load_predictor
    predictor = load_predictor(run.path("models", "preselection"))
    threshold = json.loads(run.path("results", "preselection.json").read_text())["threshold"]
    n = min(run.config["quadrature_per_process"], *(len(raw_split(run, p, "integration")) for p in ("S", "B", "NI")))
    x = np.concatenate([np.asarray(raw_split(run, p, "integration")[:n]) for p in ("S", "B", "NI")])
    stratum = np.repeat(np.arange(3), n)
    half = np.tile(np.arange(n) % 2, 3)
    selected = predictor.predict_proba(x)[:, 0] >= threshold
    x = x[selected]
    stratum, half = stratum[selected], half[selected]
    proposal = sum(run.model.component_pdf(x, p) for p in ("S", "B", "NI"))/3.
    weights = 1./(3*n*proposal)
    nominal = run.model.component_densities(x).T
    down = run.model.component_densities(x, alpha_ni=-1.).T[None, ...]
    up = run.model.component_densities(x, alpha_ni=1.).T[None, ...]
    truth = run.model.intensity(x, run.config["asimov_mu"])
    result = dict(x=x, weights=weights, nominal=nominal, down=down, up=up, truth=truth,
                  stratum=stratum, half=half)
    for name, value in result.items():
        save_array(paths[name], value)
    save_json(quad_state, selection_state)
    event_weights = weights*truth
    save_json(run.path("quadrature", "diagnostics.json"),
              dict(proposed_events=3*n, selected_events=len(x),
                   expected_events=float(event_weights.sum()),
                   effective_sample_size=float(event_weights.sum()**2/np.sum(event_weights**2)),
                   selected_yields=dict(zip(PROCESSES, (nominal @ weights).tolist())),
                   proposal="equal mixture of nominal S, B and NI", reference="selected nominal S",
                   note="Finite quadrature is shared by model rates and Asimov data, not by network training."))
    return result


def quadrature_diagnostics(run, quad=None):
    """Estimate integration uncertainty, including rejected draws as zeros.

    Halves are split within each proposal stratum BEFORE selection, rather than
    incorrectly splitting the concatenated mixture sample down its middle.
    """
    from .inference import component_coefficients
    quad = prepare_quadrature(run) if quad is None else quad
    n = min(run.config["quadrature_per_process"], *(len(raw_split(run, p, "integration")) for p in ("S", "B", "NI")))
    records = []
    truth = quad["truth"]
    for mu in mu_grid(run):
        expected = component_coefficients(mu) @ quad["nominal"]
        delta = (expected-truth)/truth
        contribution = 2*quad["weights"]*truth*(delta-np.log1p(delta))
        variance = 0.
        for s in range(3):
            a = contribution[quad["stratum"] == s]
            variance += max(0., (n*np.sum(a*a)-np.sum(a)**2)/(n-1))
        half_values = [float(np.sum(contribution[quad["half"] == h])*n/len(np.arange(h, n, 2))) for h in (0, 1)]
        records.append(dict(mu=mu, q=float(np.sum(contribution)), standard_error=float(np.sqrt(variance)),
                            q_half_even=half_values[0], q_half_odd=half_values[1]))
    frame = pd.DataFrame(records)
    frame.to_csv(run.path("results", "quadrature_diagnostics.csv"), index=False)
    return frame


def _learned_fields(run, quad):
    from .training import load_predictor
    x, w = quad["x"], quad["weights"]
    rates = quad["nominal"] @ w
    q_s = quad["nominal"][0]/rates[0]
    nominal = np.empty_like(quad["nominal"])
    nominal[0] = quad["nominal"][0]
    normalization = []
    for j, p in enumerate(PROCESSES[1:], 1):
        predictor = load_predictor(_ratio_path(run, p, "S"))
        shape = q_s * predictor.predict_ratio(x)
        integral = float(shape @ w)
        normalization.append(dict(sample=p, variation="nominal", raw_integral=integral))
        nominal[j] = rates[j]*shape/integral
    down = nominal[None, ...].copy()
    up = down.copy()
    j = PROCESSES.index("NI")
    for label, array, exact in (("up", up, quad["up"]), ("down", down, quad["down"])):
        predictor = load_predictor(_ratio_path(run, f"NI_{label}", "NI"))
        shape = nominal[j]/rates[j]*predictor.predict_ratio(x)
        integral = float(shape @ w)
        normalization.append(dict(sample="NI", variation=label, raw_integral=integral))
        array[0, j] = (exact[0, j] @ w)*shape/integral
    pd.DataFrame(normalization).to_csv(run.path("results", "workspace_normalization.csv"), index=False)
    return nominal, down, up


def mu_grid(run):
    c = run.config
    return np.unique(np.r_[np.linspace(c["mu_min"], c["mu_max"], c["mu_points"]), c["asimov_mu"]])


def mu_fit_starts(run):
    """Cover both interference branches over the configured fit domain."""
    low, high = run.config["mu_fit_bounds"]
    return np.unique(np.r_[np.linspace(low, high, 7),
                           np.clip(run.config["asimov_mu"], low, high)]).tolist()


def run_unbinned(run):
    from .toolkit import build_workspace, load_model, make_fitter
    q = prepare_quadrature(run)
    learned = _learned_fields(run, q)
    records, diagnostics = [], []
    for label, fields in (("analytic unbinned", (q["nominal"], q["down"], q["up"])),
                          ("learned unbinned", learned)):
        workspace = build_workspace(run.path("workspaces", label.split()[0]), *fields,
                                    q["weights"], q["truth"],
                                    parameter_bounds=[run.config["mu_fit_bounds"],
                                                      run.config["nuisance_bounds"]])
        fitter = make_fitter(load_model(workspace))
        for syst in (False, True):
            scan = fitter.scan(mu_grid(run), systematics=syst)
            for i, mu in enumerate(scan["mu"]):
                records.append(dict(mu=mu, q=scan["t_mu"][i], model=label, systematics=syst,
                                    alpha_NI=scan["parameters"][i, 1],
                                    valid=bool(scan["valid"][i]), edm=scan["edm"][i]))
            best = fitter.last_fit
            diagnostics.append(dict(model=label, systematics=syst, mu_hat=best.parameters[0],
                                    alpha_NI_hat=best.parameters[1],
                                    nll=best.nll, valid=best.valid, edm=best.edm))
    scans, diagnostic_frame = pd.DataFrame(records), pd.DataFrame(diagnostics)
    scans.to_csv(run.path("results", "unbinned_scans.csv"), index=False)
    diagnostic_frame.to_csv(run.path("results", "unbinned_fits.csv"), index=False)
    return dict(scans=scans, diagnostics=diagnostic_frame, quadrature=quadrature_diagnostics(run, q))


def _score(run, quad, eta):
    # Algebraically identical to PhysicsModel.score, evaluated from cached
    # analytic intensity fields to avoid re-evaluating Gaussians in every scan.
    from .inference import component_coefficients
    if eta <= 0:
        raise ValueError("The sqrt(mu) score requires eta > 0")
    s, sbi, b, _ = quad["nominal"]
    derivative = s + (sbi-s-b)/(2*np.sqrt(eta))
    density = component_coefficients(eta) @ quad["nominal"]
    return derivative/density - (derivative @ quad["weights"])/(density @ quad["weights"])


def _record_scan_result(result, *, mu, model, systematics, **extra):
    """Internal schema helper used by histogram studies."""
    return {"mu": float(mu), "model": model, "systematics": systematics, **result, **extra}


def _template_likelihood(quad, systematics=True):
    from .inference import TemplateLikelihood
    n = len(quad["weights"])
    down = quad["down"] if systematics else np.empty((0, 4, n))
    up = quad["up"] if systematics else np.empty((0, 4, n))
    return TemplateLikelihood(quad["nominal"], down, up, quad["weights"], quad["truth"])


def _fit_test(run, model, mu, systematics):
    result = model.test_statistic(mu, systematics=systematics,
                                 mu_bounds=tuple(run.config["mu_fit_bounds"]),
                                 alpha_bounds=tuple(run.config["nuisance_bounds"]),
                                 mu_starts=mu_fit_starts(run))
    null, best = result["null_fit"], result["best_fit"]
    valid = bool(null.success and best.success)
    return dict(q=result["t"] if valid else np.nan, valid=valid,
                mu_hat=best.mu, alpha_NI=null.alpha[0] if len(null.alpha) else 0.,
                fit_message=null.message + "; " + best.message)


def _saved_unbinned(run, systematics=None):
    path = run.path("results", "unbinned_scans.csv")
    if not path.exists():
        raise FileNotFoundError("Run notebook 3 (run_unbinned) before the histogram comparisons.")
    scans = pd.read_csv(path)
    if systematics is not None:
        scans = scans.loc[scans.systematics == systematics].copy()
    return scans


def run_binning_study(run):
    """Refill and freeze each matched-eta histogram before BOTH likelihood fits."""
    from .inference import component_coefficients, histogram_components
    quad = prepare_quadrature(run)
    parent = _template_likelihood(quad, systematics=False)
    records, fidelity = [], []
    for mu in mu_grid(run):
        eta = float(mu)
        score = _score(run, quad, eta)
        z = expit(score/run.config["score_scale"])
        density = component_coefficients(eta) @ quad["nominal"]
        s, sbi, b, _ = quad["nominal"]
        derivative = s+(sbi-s-b)/(2*np.sqrt(eta))
        rate = density @ quad["weights"]
        drate = derivative @ quad["weights"]
        full_info = np.sum(quad["weights"]*derivative**2/density)
        rate_info = drate**2/rate
        for nbins in run.config["bin_counts"]:
            edges = np.linspace(0., 1., nbins+1)
            hist = parent.binned(z, edges, eta=eta)
            result = _fit_test(run, hist, mu, systematics=False)
            records.append(_record_scan_result(result, mu=mu, eta=eta, n_bins=nbins,
                                               model="direct score histogram", systematics=False))
            expected = component_coefficients(eta) @ hist.nominal
            # Signed derivative weights are valid; only event means must be nonnegative.
            dbin = np.histogram(z, edges, weights=quad["weights"]*derivative)[0]
            info = np.sum(np.divide(dbin**2, expected, out=np.zeros_like(expected), where=expected>0))
            fidelity.append(dict(eta=eta, n_bins=nbins, information_fraction=info/full_info,
                                 shape_information_fraction=(info-rate_info)/(full_info-rate_info),
                                 full_information=full_info, binned_information=info,
                                 occupied_bins=int(np.count_nonzero(expected))))
        print(f"Matched eta={eta:.3g}: all binnings fitted", flush=True)
    # A fixed score makes the distinction from the moving observable visible.
    eta = run.config["asimov_mu"]
    z = expit(_score(run, quad, eta)/run.config["score_scale"])
    nbins = max(run.config["bin_counts"])
    hist = parent.binned(z, np.linspace(0, 1, nbins+1), eta=eta)
    for mu in mu_grid(run):
        result = _fit_test(run, hist, mu, systematics=False)
        records.append(_record_scan_result(result, mu=mu, eta=eta, n_bins=nbins,
                                           model="fixed score histogram", systematics=False))
    scans = pd.concat([pd.DataFrame(records), _saved_unbinned(run, False)], ignore_index=True)
    fidelity = pd.DataFrame(fidelity)
    scans.to_csv(run.path("results", "binning_scans.csv"), index=False)
    fidelity.to_csv(run.path("results", "binning_fisher.csv"), index=False)
    return dict(scans=scans, fidelity=fidelity)


def _histogram_anchors(quad, z, edges):
    from .inference import histogram_components
    fields = [quad["nominal"], quad["down"][0], quad["up"][0]]
    return np.stack([histogram_components(z, value, quad["weights"], edges) for value in fields])


def _histogram_model(anchors, counts):
    from .inference import TemplateLikelihood
    nominal = anchors[0]
    down, up = anchors[[1]], anchors[[2]]
    return TemplateLikelihood(nominal, down, up, np.ones(len(counts)), counts)


def run_spline_study(run):
    """Fit eta-dependent EXPECTED templates; always refill observed Asimov bins."""
    from .inference import YieldFractionSpline, component_coefficients
    quad = prepare_quadrature(run)
    c = run.config
    # Choose the smallest tested binning that retains >=99% extended Fisher
    # information over the tested anchors, if it exists. Otherwise use the
    # finest tested binning and explicitly report the residual information loss.
    fisher_path = run.path("results", "binning_fisher.csv")
    if not fisher_path.exists():
        raise FileNotFoundError("Run notebook 4 before selecting the spline binning.")
    fisher = pd.read_csv(fisher_path)
    worst = fisher.groupby("n_bins").information_fraction.min()
    qualified = worst[worst >= .99]
    nbins = int(qualified.index.min() if len(qualified) else worst.index.max())
    edges = np.linspace(0, 1, nbins+1)
    etas = np.unique(np.r_[np.linspace(c["mu_min"], c["mu_max"], c["spline_anchors"]), c["asimov_mu"]])
    anchor_values = np.stack([_histogram_anchors(quad, expit(_score(run, quad, eta)/c["score_scale"]), edges) for eta in etas])
    spline = YieldFractionSpline(etas, anchor_values)
    np.savez_compressed(run.path("results", "spline_templates.npz"), etas=etas,
                        edges=edges, bin_yields=anchor_values, totals=spline.totals)
    save_json(run.path("results", "spline_choice.json"),
              dict(n_bins=nbins, worst_information_fraction=float(worst.loc[nbins]),
                   passed_99_percent=bool(len(qualified)), n_anchors=len(etas),
                   note="PCHIP acts on nonnegative bin fractions, renormalized to sum one; totals are eta-independent."))
    variations = ("nominal", "NI_down", "NI_up")
    records_yields, validation = [], []
    for ai, eta in enumerate(etas):
        for vi, variation in enumerate(variations):
            for pi, process in enumerate(PROCESSES):
                for bi, value in enumerate(anchor_values[ai, vi, pi]):
                    records_yields.append(dict(eta=eta, sample=process, variation=variation, bin=bi, yield_value=value))
    for eta in (etas[:-1]+etas[1:])/2:
        exact = _histogram_anchors(quad, expit(_score(run, quad, eta)/c["score_scale"]), edges)
        predicted = spline(eta)
        delta = (predicted-exact)/spline.totals[..., None]
        for vi, variation in enumerate(variations):
            for pi, process in enumerate(PROCESSES):
                validation.append(dict(eta=eta, sample=process, variation=variation,
                                       max_fraction_error=np.max(np.abs(delta[vi, pi])),
                                       l1_fraction_error=np.sum(np.abs(delta[vi, pi]))))
    scan_records, morph_comparison = [], []
    parent = _template_likelihood(quad, systematics=True)
    check_mus = mu_grid(run)[[0, len(mu_grid(run))//2, -1]]
    for mu in mu_grid(run):
        eta = float(mu)
        z = expit(_score(run, quad, eta)/c["score_scale"])
        counts = np.histogram(z, edges, weights=quad["weights"]*quad["truth"])[0]
        exact = _histogram_anchors(quad, z, edges)
        predicted = spline(eta)
        for label, anchors in (("direct score histogram", exact), ("spline score histogram", predicted)):
            model = _histogram_model(anchors, counts)
            for syst in (False, True):
                result = _fit_test(run, model, mu, systematics=syst)
                scan_records.append(_record_scan_result(result, mu=mu, eta=eta, n_bins=nbins,
                                                        model=label, systematics=syst))
        # Integrating the morphed unbinned density need not equal morphing bin
        # integrals. Quantify this model difference separately at fixed anchors.
        if np.any(np.isclose(mu, check_mus)):
            before, after = parent.binned(z, edges, "before", eta), parent.binned(z, edges, "after", eta)
            for alpha in ([-1.], [-.5], [.5], [1.], [1.5]):
                a, b = before.intensity(mu, alpha), after.intensity(mu, alpha)
                morph_comparison.append(dict(eta=eta, alpha_NI=alpha[0],
                                             l1_relative_difference=float(np.sum(np.abs(a-b))/np.sum(a)),
                                             nll_difference=after.nll(mu, alpha)-before.nll(mu, alpha)))
        print(f"Spline and direct profile fits at eta={eta:.3g} ready", flush=True)
    # Physical mu dependence is separate: fix eta and scale S/SBI/B exactly.
    fixed = _histogram_anchors(quad, expit(_score(run, quad, c["asimov_mu"])/c["score_scale"]), edges)[0]
    physical = pd.DataFrame([dict(mu=mu, eta=c["asimov_mu"], bin=bi, yield_value=value)
                             for mu in mu_grid(run)
                             for bi, value in enumerate(component_coefficients(mu) @ fixed)])
    scans = pd.concat([pd.DataFrame(scan_records), _saved_unbinned(run)], ignore_index=True)
    direct = scans.loc[scans.model == "direct score histogram", ["mu", "systematics", "q", "mu_hat"]]
    interpolated = scans.loc[scans.model == "spline score histogram", ["mu", "systematics", "q", "mu_hat"]]
    comparison = direct.merge(interpolated, on=["mu", "systematics"], suffixes=("_direct", "_spline"))
    comparison["delta_q"] = comparison.q_spline-comparison.q_direct
    comparison["is_spline_anchor"] = [bool(np.any(np.isclose(mu, etas, atol=1e-12, rtol=0))) for mu in comparison.mu]
    yields, validation = pd.DataFrame(records_yields), pd.DataFrame(validation)
    morph = pd.DataFrame(morph_comparison)
    for filename, frame in (("spline_scans", scans), ("spline_yields", yields),
                            ("spline_validation", validation), ("physical_bin_yields", physical),
                            ("morph_order_diagnostic", morph), ("spline_likelihood_validation", comparison)):
        frame.to_csv(run.path("results", filename+".csv"), index=False)
    return dict(scans=scans, yields=yields, validation=validation, physical_yields=physical,
                morph_comparison=morph, comparison=comparison, n_bins=nbins, spline=spline)
