"""Direct-fill controls for notebook 09: floor, bin resolution and morph order."""
from dataclasses import asdict, dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from . import pipeline
from .data import save_json
from .reference_splines import TemplateBank, _fingerprint, _safe_nll


@dataclass(frozen=True)
class FloorStudySettings:
    comparison_floor: float = .2
    resolution_multiplier: int = 4
    alpha_grid_points: int = 17
    alpha_xtol: float = 1e-7

    def __post_init__(self):
        if not np.isfinite(self.comparison_floor) or not 0 <= self.comparison_floor < 1:
            raise ValueError("comparison_floor must lie in [0, 1)")
        for name, minimum in (("resolution_multiplier", 2), ("alpha_grid_points", 5)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(self.alpha_xtol) or self.alpha_xtol <= 0:
            raise ValueError("alpha_xtol must be finite and positive")


def profile_nuisance(objective, bounds, settings):
    """Search every resolved 1D basin and endpoints; record failed searches.

    The coarse grid includes zero and the exp-poly joins. This is a finite
    multistart numerical check, not a proof of a global conditional minimum.
    """
    low, high = map(float, bounds)
    if not np.isfinite([low, high]).all() or not low < high or not low <= 0 <= high:
        raise ValueError("Nuisance bounds must be finite, ordered and include zero")
    points = np.unique(np.r_[np.linspace(low, high, settings.alpha_grid_points),
                            [a for a in (-1., 0., 1.) if low <= a <= high]])
    evaluations = {}
    def evaluate(alpha):
        alpha = float(alpha)
        if alpha not in evaluations:
            value = float(objective(alpha))
            evaluations[alpha] = value if np.isfinite(value) else np.inf
        return evaluations[alpha]
    values = np.array([evaluate(a) for a in points])
    candidates = [(values[0], points[0]), (values[-1], points[-1])]
    failures, basins = 0, 0
    for i in range(1, len(points)-1):
        if np.isfinite(values[i]) and values[i] <= values[i-1] and values[i] <= values[i+1]:
            basins += 1
            result = minimize_scalar(evaluate, bounds=(points[i-1], points[i+1]),
                                     method="bounded", options={"xatol": settings.alpha_xtol})
            if result.success and np.isfinite(result.fun):
                candidates.extend([(float(result.fun), float(result.x)), (values[i], points[i])])
            else:
                failures += 1
    best, alpha = min(candidates)
    # A failed basin search is never hidden by a better-looking successful one.
    valid = bool(np.isfinite(best) and failures == 0 and best <= np.min(values)+1e-9)
    return dict(nll=best if valid else np.nan, alpha=alpha if valid else np.nan,
                valid=valid, basins=basins, failed_searches=failures, nfev=len(evaluations),
                boundary=bool(valid and min(abs(alpha-low), abs(alpha-high)) < 10*settings.alpha_xtol),
                message="Resolved basins and endpoints searched" if valid else "Conditional profile search failed")


def direct_asimov_fits(run, model, mu, settings):
    """Use the checked global Asimov minimum, then profile only the NI nuisance.

    All models here are direct fills of the generating bank. A feasible truth
    reaching the nonnegative KL+constraint lower bound certifies the denominator
    of q, including rate-only degeneracies. No certificate is used for the
    approximate interpolated templates in the other study.
    """
    truth = float(run.config["asimov_mu"])
    bounds = run.config["mu_fit_bounds"]
    alpha_bounds = run.config["nuisance_bounds"]
    truth_nll = _safe_nll(model, truth, 0.)
    certified = bool(bounds[0] <= truth <= bounds[1] and alpha_bounds[0] <= 0 <= alpha_bounds[1]
                     and 0 <= truth_nll <= 1e-10)
    fixed_nll = _safe_nll(model, mu, 0.)
    profiled = profile_nuisance(lambda alpha: _safe_nll(model, mu, alpha), alpha_bounds, settings)
    records = []
    for systematics in (False, True):
        nll = profiled["nll"] if systematics else fixed_nll
        valid = bool(certified and np.isfinite(nll) and nll-truth_nll >= -1e-8
                     and (not systematics or profiled["valid"]))
        records.append(dict(systematics=systematics, valid=valid,
            q=max(0., float(nll-truth_nll)) if valid else np.nan,
            nll_null=nll, nll_best=truth_nll, truth_certified=certified,
            alpha_null=profiled["alpha"] if systematics else 0.,
            alpha_boundary=profiled["boundary"] if systematics else False,
            profile_basins=profiled["basins"] if systematics else 0,
            failed_searches=profiled["failed_searches"] if systematics else 0,
            nfev=profiled["nfev"] if systematics else 1,
            message=(profiled["message"] if systematics else "NI fixed at zero")
                    if certified else "Direct model did not attain the nonnegative NLL bound at truth"))
    return records


def run_reference_floor_study(run, n_bins=36, *, quadrature=None, settings=None, progress=print):
    """Matched direct scans; vary only compression, reuse the selected bank.

    Morph-before-bin is the actual compression of the saved analytic unbinned
    model. Bin-before-morph is the extra template approximation used by 08/09.
    Every variant retains the same physical model, Asimov truth and constraint.
    """
    settings = settings or FloorStudySettings()
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")
    observable = pipeline.observable_configuration(run, "reference_ratio")
    floors = np.unique([0., observable["reference_ratio_eta_floor"], settings.comparison_floor])
    bins = [n_bins, settings.resolution_multiplier*n_bins]
    scan = pipeline._positive_study_grid(run)
    unbinned = pipeline._saved_unbinned(run)
    quad = pipeline.prepare_quadrature(run) if quadrature is None else quadrature
    parent = pipeline._template_likelihood(quad, systematics=True)
    output = run.path("results", "reference_ratio_floor_study")
    output.mkdir(parents=True, exist_ok=True)
    bank = TemplateBank(run, quad, pipeline.observable_bin_edges(n_bins, "reference_ratio"))
    metadata = dict(schema_version=2, comparison_floor=settings.comparison_floor, settings=asdict(settings), observable=observable,
                    floors=floors.tolist(), bin_counts=bins, scan=scan.tolist(),
                    bin_edges={str(n): pipeline.observable_bin_edges(n, "reference_ratio").tolist() for n in bins},
                    asimov_mu=run.config["asimov_mu"], mu_fit_bounds=run.config["mu_fit_bounds"],
                    nuisance_bounds=run.config["nuisance_bounds"],
                    quadrature_fingerprint=_fingerprint(bank), evaluation_root=str(run.path("results").parent),
                    denominator="Checked feasible generating point attains nonnegative NLL bound",
                    conditional_profile="1D grid with bounded refinement of every resolved basin and endpoints",
                    status="running")
    provenance = run.path("results", "reference_ratio_model_source.json")
    if provenance.exists():
        metadata["model_source"] = json.loads(provenance.read_text())
    save_json(output/"metadata.json", metadata)
    rows = []
    for index, eta in enumerate(scan):
        # Floor choices have identical partitions above the floor. Reuse those
        # exact results, but never reuse a fit at a different physical mu.
        cached = {}
        for floor in floors:
            variant = run.with_config(reference_ratio_eta_floor=float(floor))
            effective = max(float(eta), float(floor))
            z = pipeline.observable_values(variant, quad, eta, "reference_ratio")
            for count in bins:
                edges = pipeline.observable_bin_edges(count, "reference_ratio")
                for order in ("before", "after"):
                    key = (effective, count, order)
                    if key not in cached:
                        model = parent.binned(z, edges, order, eta)
                        cached[key] = direct_asimov_fits(run, model, float(eta), settings)
                    for fit in cached[key]:
                        rows.append(dict(eta=float(eta), mu=float(eta), eta_effective=effective,
                                         floor=float(floor), n_bins=count, morph_order=order, **fit))
        progress(f"Direct floor/resolution/morph controls: {index+1}/{len(scan)}, eta={eta:.7g}")
        pd.DataFrame(rows).to_csv(output/"scans.csv", index=False)
    scans = pd.DataFrame(rows)
    summary = scans.groupby(["floor", "n_bins", "morph_order", "systematics"]).agg(
        fit_checks=("valid", "size"), valid_fits=("valid", "sum"),
        truth_certificates=("truth_certified", "sum"), alpha_boundaries=("alpha_boundary", "sum")).reset_index()
    summary["invalid_fits"] = summary.fit_checks-summary.valid_fits
    # Explicitly matched comparisons; no interpolation of reference q curves.
    after = scans.loc[scans.morph_order == "after"]
    before = scans.loc[scans.morph_order == "before"]
    morph = after.merge(before, on=["eta", "mu", "floor", "n_bins", "systematics"], suffixes=("_after", "_before"))
    morph["delta_q_after_minus_before"] = morph.q_after-morph.q_before
    floored = scans.loc[scans.floor == settings.comparison_floor]
    no_floor = scans.loc[scans.floor == 0.]
    floor_effect = no_floor.merge(floored, on=["eta", "mu", "n_bins", "morph_order", "systematics"], suffixes=("_no_floor", "_floor"))
    floor_effect["delta_q_no_floor_minus_floor"] = floor_effect.q_no_floor-floor_effect.q_floor
    analytic = unbinned.loc[unbinned.model == "analytic unbinned", ["mu", "systematics", "q"]].copy()
    if "valid" in unbinned:
        analytic.loc[~unbinned.loc[analytic.index, "valid"], "q"] = np.nan
    # CSV parsing can move an identical scan point by one ULP. Match only
    # roundoff-equivalent points, retaining the actual saved coordinate.
    tolerance = 32*np.finfo(float).eps*max(1., float(np.max(np.abs(scan))))
    reference = pd.merge_asof(scans.sort_values("mu"),
        analytic.rename(columns={"mu": "mu_unbinned", "q": "q_unbinned"}).sort_values("mu_unbinned"),
        left_on="mu", right_on="mu_unbinned", by="systematics", direction="nearest", tolerance=tolerance)
    reference["delta_q_binned_minus_unbinned"] = reference.q-reference.q_unbinned
    frames = dict(scans=scans, summary=summary, morph_comparison=morph,
                  floor_comparison=floor_effect, reference_comparison=reference, unbinned=unbinned)
    for name, frame in frames.items():
        frame.to_csv(output/f"{name}.csv", index=False)
    metadata.update(status="complete", all_fits_valid=bool(scans.valid.all()))
    save_json(output/"metadata.json", metadata)
    progress("Direct control summary:\n"+summary.to_string(index=False))
    return dict(**frames, output=output, metadata=metadata)


def plot_reference_floor_study(run, study, *, scan_xlim=(.02, 1.15), scan_ylim=(0., 8.),
                              secondary_xlim=(.02, .3)):
    import matplotlib.pyplot as plt
    import mplhep as hep
    destination = run.path("plots", "09_floor_study")
    destination.mkdir(parents=True, exist_ok=True)
    figures = []
    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(destination/f"09_{name}.{extension}", bbox_inches="tight", dpi=150)
        figures.append(fig)
    scans = study["scans"]
    floors, bins = study["metadata"]["floors"], study["metadata"]["bin_counts"]
    colors = {floor: color for floor, color in zip(floors, ("#0072B2", "#D55E00"))}
    with plt.style.context(hep.style.ATLAS):
        for order, title in (("before", "Morph then bin"), ("after", "Bin then morph")):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for column, syst in enumerate((False, True)):
                for floor in floors:
                    for count in bins:
                        frame = scans.loc[(scans.morph_order == order) & (scans.floor == floor)
                                          & (scans.n_bins == count) & (scans.systematics == syst)].sort_values("eta")
                        for ax in axes[:, column]:
                            ax.plot(frame.eta, frame.q.where(frame.valid), color=colors[floor],
                                    ls="-" if count == bins[0] else "--", lw=1.7,
                                    label=f"Floor {floor:g}, {count} bins")
                reference = study["unbinned"].query("model == 'analytic unbinned' and systematics == @syst").sort_values("mu")
                for row, ax in enumerate(axes[:, column]):
                    ax.plot(reference.mu, reference.q, color="black", lw=1.5, label="Analytic unbinned")
                    ax.axhline(1., color=".5", ls=":", lw=1)
                    ax.set(xlabel=r"Tested $\mu_0=\eta$", ylabel=r"$q(\mu_0)$",
                           title=("NI profiled" if syst else "NI fixed")+(" — secondary region" if row else ""),
                           xlim=secondary_xlim if row else scan_xlim, ylim=scan_ylim)
                    ax.legend(fontsize=9)
            fig.suptitle(title+"; direct fills throughout", fontsize=16)
            save(fig, f"floor_{order}")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for floor in floors:
            for count in bins:
                frame = scans.loc[(scans.floor == floor) & (scans.n_bins == count)
                                   & (scans.morph_order == "before") & scans.systematics].sort_values("eta")
                axes[0].plot(frame.eta, frame.alpha_null.where(frame.valid), color=colors[floor],
                             ls="-" if count == bins[0] else "--", label=f"Floor {floor:g}, {count} bins")
                comparison = study["morph_comparison"]
                # `floor` is also a NumExpr built-in; boolean selection works
                # whether or not pandas enables that optional query engine.
                morph = comparison.loc[(comparison["floor"] == floor)
                    & (comparison.n_bins == count) & comparison.systematics].sort_values("eta")
                axes[1].semilogy(morph.eta, np.maximum(abs(morph.delta_q_after_minus_before), 1e-12),
                                 color=colors[floor], ls="-" if count == bins[0] else "--", label=f"Floor {floor:g}, {count} bins")
        axes[0].set(xlabel=r"Tested $\mu_0=\eta$", ylabel=r"Conditional $\hat\alpha_{NI}$", xlim=secondary_xlim,
                    title="Morph then bin: nuisance response")
        axes[1].set(xlabel=r"Tested $\mu_0=\eta$", ylabel=r"$|q_{\rm bin\ then\ morph}-q_{\rm morph\ then\ bin}|$",
                    xlim=scan_xlim, title="Profiled effect of morph order")
        for ax in axes:
            ax.legend(fontsize=9)
        save(fig, "floor_nuisance_and_morph_order")
    return figures
