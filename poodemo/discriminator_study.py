"""Fixed process-ratio compression versus the moving physical likelihood ratio.

All templates and Asimov data use the same analytical integration bank.  The
optional learned mode changes only the axes, keeping density-estimation error
separate from the information lost by compression.  No network is trained.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from . import pipeline
from .data import PROCESSES, Run, save_json
from .diagnostics import local_asimov_information, local_width


AXES = ("NI", "SBI", "B")
DEFAULT_POWERS = {"NI": 1., "SBI": 2., "B": 2.}
FIXED_LABELS = ("NI only", "NI × SBI", "NI × SBI × B")
REFERENCE_LABEL = "moving reference ratio"


def process_ratio_values(quad, powers=None, *, nominal=None):
    """Return z_j = sigmoid(a_j log(p_j/p_S)), with selected normalized PDFs.

    ``nominal`` can supply learned component fields evaluated on the same
    quadrature; all four integrals are computed anew on that bank.  Extreme
    ratios are handled in log space and zero numerator density maps to zero.
    """
    powers = dict(DEFAULT_POWERS if powers is None else powers)
    if set(powers) != set(AXES):
        raise ValueError("powers must contain exactly NI, SBI and B")
    for name in AXES:
        try:
            powers[name] = float(powers[name])
        except (TypeError, ValueError):
            raise ValueError("Each discriminator power must be finite and positive") from None
        if not np.isfinite(powers[name]) or powers[name] <= 0:
            raise ValueError("Each discriminator power must be finite and positive")
    fields = np.asarray(quad["nominal"] if nominal is None else nominal, dtype=float)
    weights = np.asarray(quad["weights"], dtype=float)
    if fields.shape != (4, len(weights)) or weights.ndim != 1:
        raise ValueError("Expected nominal shape (4,N) and weights shape (N,)")
    if (np.any(~np.isfinite(fields)) or np.any(fields < 0)
            or np.any(~np.isfinite(weights)) or np.any(weights < 0)):
        raise ValueError("Component fields and quadrature weights must be finite and nonnegative")
    rates = fields @ weights
    if np.any(rates <= 0) or np.any(~np.isfinite(rates)) or np.any(fields[0] <= 0):
        raise ValueError("Selected rates and the S reference density must be positive")
    reference_logpdf = np.log(fields[0]) - np.log(rates[0])
    values = {}
    for name in AXES:
        index = PROCESSES.index(name)
        with np.errstate(divide="ignore", over="ignore"):
            log_ratio = np.log(fields[index]) - np.log(rates[index]) - reference_logpdf
            values[name] = expit(powers[name] * log_ratio)
    return values


def multidimensional_histogram(parent, coordinates, edges, *, morph_order="after"):
    """Freeze a Cartesian histogram, retaining empty cells and all endpoints.

    Flattened cell identifiers are only storage coordinates.  Their assignment
    never changes when mu or nuisance parameters are varied in a likelihood fit.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if coordinates.ndim == 1:
        coordinates = coordinates[:, None]
    if (coordinates.ndim != 2 or coordinates.shape[0] != len(parent.quadrature_weights)
            or not 1 <= coordinates.shape[1] <= 3):
        raise ValueError("coordinates must have shape (N,d), with 1 <= d <= 3")
    if (edges.ndim != 1 or len(edges) < 2 or np.any(~np.isfinite(edges))
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("edges must be finite and strictly increasing")
    if (np.any(~np.isfinite(coordinates)) or np.any(coordinates < edges[0])
            or np.any(coordinates > edges[-1])):
        raise ValueError("Every coordinate must lie inside the histogram range")
    n_bins = len(edges) - 1
    indices = np.searchsorted(edges, coordinates, side="right") - 1
    indices[coordinates == edges[-1]] = n_bins - 1
    flat = np.ravel_multi_index(indices.T, (n_bins,) * coordinates.shape[1])
    histogram = parent.binned(flat + .5, np.arange(n_bins**coordinates.shape[1] + 1),
                              morph_order=morph_order)
    histogram.coordinate_edges = edges.copy()
    histogram.histogram_shape = (n_bins,) * coordinates.shape[1]
    return histogram


def _weighted_quantile(values, weights, probabilities):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return np.interp(probabilities, cumulative / cumulative[-1], values[order])


def _occupancy(histogram, *, label, eta=np.nan):
    counts = histogram.data_counts
    squares = np.bincount(histogram._index, weights=histogram.parent.asimov_weights**2,
                          minlength=len(counts))
    ess = np.divide(counts**2, squares, out=np.zeros_like(counts), where=squares > 0)
    occupied = counts > 0
    return dict(model=label, eta=eta, total_cells=len(counts),
                occupied_cells=int(occupied.sum()), empty_cells=int((~occupied).sum()),
                expected_events=float(counts.sum()),
                minimum_cell_bank_ess=float(ess[occupied].min()),
                median_cell_bank_ess=float(np.median(ess[occupied])),
                expected_fraction_in_ess_below_10=float(counts[ess < 10].sum()/counts.sum()))


def _information(model, *, label, truth, systematics, baseline, eta=np.nan):
    information = local_asimov_information(model, truth)
    width = local_width(information["matrix"], systematics=systematics)
    denominator = baseline["information_mu_effective"]
    return dict(model=label, eta=eta, information_mu=truth, systematics=bool(systematics),
                information_fraction=width["information_mu_effective"]/denominator,
                **width)


def _analytic_scan(run, parent, grid, systematics):
    """Reuse matching notebook-03 analytical fits; otherwise compute them."""
    try:
        saved = pipeline._saved_unbinned(run, systematics)
    except FileNotFoundError:
        saved = pd.DataFrame()
    if not saved.empty and "model" in saved:
        saved = saved.loc[saved.model == "analytic unbinned"].copy()
        if "asimov_mu" in saved:
            saved = saved.loc[np.isclose(saved.asimov_mu, run.config["asimov_mu"], rtol=0, atol=1e-12)]
        else:
            saved = pd.DataFrame()
    records = []
    for mu in grid:
        rows = saved.loc[np.isclose(saved.mu, mu, rtol=0, atol=1e-12)] if not saved.empty else saved
        if len(rows) == 1 and bool(rows.iloc[0].valid) and np.isfinite(rows.iloc[0].q):
            record = rows.iloc[0].to_dict()
            record["baseline_source"] = "saved notebook 03"
        else:
            record = pipeline._record_scan_result(pipeline._fit_test(run, parent, mu, systematics),
                                                  mu=mu, model="analytic unbinned", systematics=systematics)
            record["baseline_source"] = "recomputed analytical likelihood"
        records.append(record)
    return records


def run_discriminator_study(run, n_bins=12, powers=None, reference_power=100.,
                            ratio_source="analytic", systematics=(False, True),
                            scan_grid=None, morph_order="after"):
    """Compare 12, 12² and 12³ fixed cells with 12 moving-ratio bins.

    At each tested mu the reference-ratio anchor eta equals mu; its histogram
    is frozen before the numerator and denominator fits, as in notebooks 08/09.
    All fixed-axis histograms remain frozen throughout the complete scan.
    Local information is measured at the Asimov truth for every compression.

    Nuisance interpolation defaults to morphing integrated anchors (notebook09's
    model). ``morph_order='before'`` instead compresses the exact parent morph
    at each evaluation, isolating binning loss at greater computational cost.
    """
    edges = pipeline.observable_bin_edges(n_bins, "reference_ratio")
    powers = dict(DEFAULT_POWERS if powers is None else powers)
    if ratio_source not in ("analytic", "learned"):
        raise ValueError("ratio_source must be 'analytic' or 'learned'")
    if morph_order not in ("before", "after"):
        raise ValueError("morph_order must be 'before' or 'after'")
    modes = tuple(dict.fromkeys(systematics))
    if not modes or any(not isinstance(mode, (bool, np.bool_)) for mode in modes):
        raise ValueError("systematics must be a nonempty sequence of booleans")
    config = {**run.config, "reference_ratio_power": reference_power}
    current = Run(run.root, config, run.model)
    settings = pipeline.observable_configuration(current, "reference_ratio")
    truth = float(run.config["asimov_mu"])
    if not np.isfinite(truth) or truth <= 0:
        raise ValueError("The local-information comparison requires positive asimov_mu")
    grid = np.asarray(pipeline.mu_grid(run) if scan_grid is None else scan_grid, dtype=float)
    low, high = run.config["mu_fit_bounds"]
    if (grid.ndim != 1 or not len(grid) or np.any(~np.isfinite(grid))
            or np.any(grid < max(0., low)) or np.any(grid > high)):
        raise ValueError("scan_grid must be finite, nonnegative and inside mu_fit_bounds")
    grid = np.unique(np.r_[grid, truth])
    quad = pipeline.prepare_quadrature(run)
    axis_fields = None
    if ratio_source == "learned":
        axis_fields = pipeline._learned_fields(run, quad)[0]
    values = process_ratio_values(quad, powers, nominal=axis_fields)
    powers = {name: float(powers[name]) for name in AXES}
    axis_quad = quad if axis_fields is None else {**quad, "nominal": axis_fields}
    event_weights = quad["weights"]*quad["truth"]
    summaries, marginal_rows = [], []
    for name in AXES:
        z = values[name]
        quantiles = _weighted_quantile(z, event_weights, [.01, .1, .5, .9, .99])
        summaries.append(dict(axis=name, power=powers[name],
                              **dict(zip(("q01", "q10", "q50", "q90", "q99"), quantiles)),
                              endpoint_event_fraction=float(event_weights[(z < edges[1]) |
                                                                           (z >= edges[-2])].sum()/event_weights.sum()),
                              saturated_event_fraction=float(event_weights[(z == 0.) |
                                                                            (z == 1.)].sum()/event_weights.sum())))
        for j, process in enumerate(PROCESSES):
            counts = np.histogram(z, edges, weights=quad["weights"]*quad["nominal"][j])[0]
            for index, count in enumerate(counts):
                marginal_rows.append(dict(axis=name, process=process, bin=index,
                                          low=edges[index], high=edges[index+1], expected=count,
                                          probability=count/counts.sum()))
    joints = {}
    for first, second in (("NI", "SBI"), ("NI", "B"), ("SBI", "B")):
        joints[first + "_" + second] = np.histogram2d(values[first], values[second], bins=(edges, edges),
                                                     weights=event_weights)[0]
    scans, information, occupancy, templates = [], [], [], {}
    for syst in modes:
        parent = pipeline._template_likelihood(quad, systematics=syst)
        baseline = local_width(local_asimov_information(parent, truth)["matrix"], systematics=syst)
        information.append(_information(parent, label="analytic unbinned", truth=truth,
                                         systematics=syst, baseline=baseline))
        scans.extend(_analytic_scan(current, parent, grid, syst))
        for dimension, label in enumerate(FIXED_LABELS, 1):
            coordinates = np.column_stack([values[name] for name in AXES[:dimension]])
            histogram = multidimensional_histogram(parent, coordinates, edges, morph_order=morph_order)
            information.append(_information(histogram, label=label, truth=truth,
                                             systematics=syst, baseline=baseline))
            if syst == modes[0]:
                occupancy.append(_occupancy(histogram, label=label))
                templates[f"{dimension}d_nominal"] = histogram.nominal
                templates[f"{dimension}d_asimov"] = histogram.data_counts
            for mu in grid:
                result = pipeline._fit_test(current, histogram, mu, syst)
                scans.append(pipeline._record_scan_result(result, mu=mu, model=label, systematics=syst,
                                                          dimension=dimension, n_bins=n_bins,
                                                          total_cells=n_bins**dimension))
            print(f"Fixed {label}: {n_bins**dimension} cells, {'profiled' if syst else 'stat only'} fitted", flush=True)
        for mu in grid:
            z = pipeline.observable_values(current, axis_quad, mu, "reference_ratio")
            histogram = parent.binned(z, edges, morph_order=morph_order, eta=mu)
            result = pipeline._fit_test(current, histogram, mu, syst)
            scans.append(pipeline._record_scan_result(result, mu=mu, model=REFERENCE_LABEL,
                                                      systematics=syst, eta=mu, dimension=1,
                                                      n_bins=n_bins, total_cells=n_bins))
            information.append(_information(histogram, label=REFERENCE_LABEL, truth=truth,
                                             systematics=syst, baseline=baseline, eta=mu))
            if syst == modes[0]:
                occupancy.append(_occupancy(histogram, label=REFERENCE_LABEL, eta=mu))
        print(f"Moving reference ratio: {len(grid)} anchors, {'profiled' if syst else 'stat only'} fitted", flush=True)
    result_config = dict(n_bins=int(n_bins), powers=powers, ratio_source=ratio_source,
                         reference_ratio_power=settings["reference_ratio_power"],
                         binning_policy=settings["binning_policy"], bin_edges=edges.tolist(),
                         asimov_mu=truth, scan_grid=grid.tolist(), systematics=list(map(bool, modes)),
                         morph_order=morph_order, likelihood_source="analytic", asimov_source="analytic",
                         nuisance_note="After-integration morphing is an additional approximation to the unbinned nuisance model.",
                         information_note="Local information is evaluated at the fixed Asimov truth, not at eta.")
    result = dict(scans=pd.DataFrame(scans), information=pd.DataFrame(information),
                  occupancy=pd.DataFrame(occupancy), axis_summary=pd.DataFrame(summaries),
                  axis_histograms=pd.DataFrame(marginal_rows), joint_histograms=joints,
                  config=result_config)
    output = run.path("results", "10_discriminator_study")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", result_config)
    for key in ("scans", "information", "occupancy", "axis_summary", "axis_histograms"):
        result[key].to_csv(output / (key + ".csv"), index=False)
    np.savez_compressed(output / "histograms.npz", edges=edges, **joints, **templates)
    return result


def plot_discriminator_study(study, output=None):
    """Return mplhep figures for axes, correlations, scans and local information."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import mplhep as hep

    figures = {}
    settings = study["config"]
    edges = np.asarray(settings["bin_edges"])
    colors = {"analytic unbinned": "black", REFERENCE_LABEL: "C0",
              FIXED_LABELS[0]: "C1", FIXED_LABELS[1]: "C2", FIXED_LABELS[2]: "C3"}
    labels = {"analytic unbinned": "Analytic unbinned",
              REFERENCE_LABEL: f"Moving reference ratio ({settings['n_bins']} bins)",
              **{name: f"{name} ({settings['n_bins']}" + (f"$^{i}$" if i > 1 else "") + " cells)"
                 for i, name in enumerate(FIXED_LABELS, 1)}}
    with plt.style.context(hep.style.ATLAS):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
        for axis, name in zip(axes, AXES):
            subset = study["axis_histograms"].query("axis == @name")
            for process in PROCESSES:
                counts = subset.loc[subset.process == process].sort_values("bin").probability.to_numpy()
                hep.histplot(counts, edges, histtype="step", label=process, ax=axis, linewidth=1.7)
            axis.set(xlim=(0, 1), xlabel=rf"$z_{{\mathrm{{{name}}}}}$", ylabel="Probability / bin",
                     title=rf"$a_{{\mathrm{{{name}}}}}={settings['powers'][name]:g}$")
        axes[0].legend(fontsize=12)
        figures["axes"] = fig
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
        for axis, (first, second) in zip(axes, (("NI", "SBI"), ("NI", "B"), ("SBI", "B"))):
            counts = study["joint_histograms"][first + "_" + second]
            positive = counts[counts > 0]
            norm = LogNorm(vmin=positive.min(), vmax=positive.max()) if len(positive) and positive.max() > positive.min() else None
            mesh = axis.pcolormesh(edges, edges, np.ma.masked_less_equal(counts.T, 0.), norm=norm, cmap="viridis")
            fig.colorbar(mesh, ax=axis, label="Expected events / cell")
            axis.set(xlabel=rf"$z_{{\mathrm{{{first}}}}}$", ylabel=rf"$z_{{\mathrm{{{second}}}}}$")
        figures["joint"] = fig
        modes = settings["systematics"]
        fig, axes = plt.subplots(1, len(modes), figsize=(8*len(modes), 6), squeeze=False, layout="constrained")
        for axis, syst in zip(axes[0], modes):
            for label, color in colors.items():
                rows = study["scans"].loc[(study["scans"].systematics == syst) &
                                           (study["scans"].model == label)].sort_values("mu")
                valid = rows.valid.to_numpy(dtype=bool)
                axis.plot(rows.mu, np.where(valid, rows.q, np.nan), color=color,
                          ls="--" if label == "analytic unbinned" else "-", label=labels[label], linewidth=1.8)
            axis.set(xlabel=r"Tested $\mu$", ylabel=r"$-2\log\lambda(\mu)$",
                     title="Profiled NI nuisance" if syst else "Stat only", ylim=(0, None))
            axis.legend(fontsize=11)
        figures["scans"] = fig
        fig, axes = plt.subplots(1, len(modes), figsize=(8*len(modes), 5.5), squeeze=False, layout="constrained")
        for axis, syst in zip(axes[0], modes):
            for label, color in colors.items():
                rows = study["information"].loc[(study["information"].systematics == syst) &
                                                 (study["information"].model == label)]
                if label == REFERENCE_LABEL:
                    rows = rows.sort_values("eta")
                    axis.plot(rows.eta, rows.information_fraction, color=color, label=labels[label])
                else:
                    axis.axhline(rows.information_fraction.iloc[0], color=color,
                                 ls="--" if label == "analytic unbinned" else "-", label=labels[label])
            axis.set(xlabel=r"Moving-observable anchor $\eta$", ylabel="Retained local information",
                     title=("Profiled" if syst else "Stat only") + rf", truth $\mu={settings['asimov_mu']:g}$",
                     xlim=(min(settings["scan_grid"]), max(settings["scan_grid"])), ylim=(0, None))
            axis.legend(fontsize=11)
        figures["information"] = fig
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            for extension in ("png", "pdf"):
                figure.savefig(output / f"10_discriminators_{name}.{extension}", dpi=160, bbox_inches="tight")
    return figures
