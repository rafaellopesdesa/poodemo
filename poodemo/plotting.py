"""ATLAS-style process plots with exact, positive coherent marginal templates.

No experiment label is added. The Gaussian amplitude model permits exact bin
integrals, avoiding cancellations between independently generated MC samples.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import ndtr

from .physics import COMPONENTS


def _normal_bin_probabilities(edges, mean, variance):
    z = (np.asarray(edges, dtype=float) - mean) / np.sqrt(variance)
    # Survival probabilities preserve precision in the positive tail.
    return np.where(z[:-1] >= 0, ndtr(-z[:-1]) - ndtr(-z[1:]),
                    ndtr(z[1:]) - ndtr(z[:-1]))


def marginal_component_yields(model, coordinate, edges, alpha_ni=0.):
    """Exact full-space marginal bin counts, in S/SBI/B/NI order.

    The product sqrt(phi_S * phi_B) is another Gaussian, times the overlap
    integral. Its covariance is 2*(Sigma_S^-1 + Sigma_B^-1)^-1. Infinite
    endpoints are allowed for normalization checks; plot edges must be finite.
    This function describes the unselected model, not classifier-selected data.
    """
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or np.any(np.isnan(edges))
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("edges must be a strictly increasing one-dimensional array")
    if coordinate not in (0, 1, 2):
        raise ValueError("coordinate must be 0, 1, or 2")
    primitive = {}
    for name in ("S", "B", "NI"):
        covariance = np.asarray(getattr(model, f"cov_{name.lower()}"))
        primitive[name] = model.component_yield(name) * _normal_bin_probabilities(
            edges, model.mean(name, alpha_ni=alpha_ni)[coordinate],
            covariance[coordinate, coordinate])
    precision_s = np.linalg.inv(np.asarray(model.cov_s))
    precision_b = np.linalg.inv(np.asarray(model.cov_b))
    covariance_i = 2 * np.linalg.inv(precision_s + precision_b)
    mean_i = .5 * covariance_i @ (
        precision_s @ model.mean("S") + precision_b @ model.mean("B"))
    integral_i = (2 * model.phase_cos
                  * np.sqrt(model.component_yield("S") * model.component_yield("B"))
                  * model.gaussian_overlap())
    interference = integral_i * _normal_bin_probabilities(
        edges, mean_i[coordinate], covariance_i[coordinate, coordinate])
    sbi = primitive["S"] + primitive["B"] + interference
    tolerance = 2048 * np.finfo(float).eps * (primitive["S"] + primitive["B"])
    if np.any(sbi < -tolerance):
        raise ValueError("Coherent SBI marginal has a negative bin")
    return np.stack((primitive["S"], np.maximum(sbi, 0.), primitive["B"], primitive["NI"]))


def coherent_marginal_yields(model, coordinate, edges, mu):
    """Return the positive coherent SBI(mu) and nominal NI bin counts."""
    coefficients = model.coefficients(mu)  # Also checks finite, nonnegative mu.
    components = marginal_component_yields(model, coordinate, edges)
    terms = coefficients[:3, None] * components[:3]
    coherent = terms.sum(axis=0)
    tolerance = 2048 * np.finfo(float).eps * np.abs(terms).sum(axis=0)
    if np.any(coherent < -tolerance):
        raise ValueError("Coherent physical marginal has a negative bin")
    return np.maximum(coherent, 0.), components[3]


def _save(fig, output, stem):
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            fig.savefig(output / f"{stem}.{extension}", bbox_inches="tight", dpi=160)


def plot_process_marginals(model, edges, samples=None, output=None):
    """Normalized process overlays, linear above logarithmic for each variable.

    Exact model bin averages are shown as steps. Optional sampled events are
    overlaid as markers with Poisson MC errors; normalization uses the complete
    plotting sample, so plotting-range losses are not normalized away.
    """
    import matplotlib.pyplot as plt
    import mplhep as hep

    edges = np.asarray(edges, dtype=float)
    if not np.all(np.isfinite(edges)):
        raise ValueError("Plot edges must be finite")
    colors = ("#D55E00", "#0072B2", "#009E73", "#CC79A7")
    with plt.style.context(hep.style.ATLAS):
        fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex="col", sharey="row")
        max_density = 0.
        for coordinate in range(3):
            counts = marginal_component_yields(model, coordinate, edges)
            for name, expected, color in zip(COMPONENTS, counts, colors):
                density = expected / model.component_yield(name) / np.diff(edges)
                max_density = max(max_density, float(np.max(density)))
                for ax in axes[:, coordinate]:
                    hep.histplot(density, edges, ax=ax, color=color, label=name,
                                 histtype="step", linewidth=1.8)
                if samples is not None:
                    points = np.asarray(samples[name])
                    observed = np.histogram(points[:, coordinate], edges)[0]
                    scale = len(points) * np.diff(edges)
                    for ax in axes[:, coordinate]:
                        hep.histplot(observed / scale, edges, ax=ax,
                                     yerr=np.sqrt(observed) / scale,
                                     color=color, histtype="errorbar", marker=".",
                                     markersize=3, elinewidth=.65, alpha=.65)
            axes[0, coordinate].set_title(fr"$x_{coordinate+1}$", fontsize=17)
            axes[1, coordinate].set_yscale("log")
            for ax in axes[:, coordinate]:
                ax.set_xlim(edges[0], edges[-1])
                ax.set_xlabel(fr"$x_{coordinate+1}$")
                ax.tick_params(labelsize=12)
        axes[0, 0].set_ylim(0, max_density * 1.25)
        axes[1, 0].set_ylim(1e-6, max_density * 3.)
        axes[0, 0].set_ylabel("Normalized marginal density", fontsize=15)
        axes[1, 0].set_ylabel("Normalized marginal density", fontsize=15)
        axes[0, 0].legend(fontsize=13, ncol=2)
        description = "Full phase space, before preselection. Steps: exact model."
        if samples is not None:
            description += " Markers: generated MC subset."
        fig.suptitle(description, fontsize=16, y=1.)
        fig.tight_layout()
        _save(fig, output, "01_process_marginals")
    return fig, axes


def plot_coherent_stacks(model, edges, mus=(0., .5, 1., 2.), *, log=False, output=None):
    """Stack coherent SBI(mu) plus NI with exact expected events per bin.

    Signed S/SBI/B basis contributions are combined before plotting. All
    coordinates are integrated over the other two dimensions, without cuts.
    """
    import matplotlib.pyplot as plt
    import mplhep as hep

    edges = np.asarray(edges, dtype=float)
    if not np.all(np.isfinite(edges)):
        raise ValueError("Plot edges must be finite")
    mus = tuple(mus)
    if not mus:
        raise ValueError("Choose at least one signal strength")
    with plt.style.context(hep.style.ATLAS):
        fig, axes = plt.subplots(3, len(mus), figsize=(4.5 * len(mus), 11),
                                 sharex=True, sharey="row", squeeze=False)
        for coordinate in range(3):
            max_height = 0.
            for mu, ax in zip(mus, axes[coordinate]):
                coherent, ni = coherent_marginal_yields(model, coordinate, edges, mu)
                hep.histplot([ni, coherent], edges, ax=ax, stack=True, histtype="fill",
                             color=["#B8C7DF", "#E69F00"], edgecolor="black", linewidth=.5,
                             label=["NI", r"Coherent SBI$(\mu)$"])
                max_height = max(max_height, float(np.max(coherent + ni)))
                ax.set(xlim=(edges[0], edges[-1]), xlabel=fr"$x_{coordinate+1}$")
                ax.tick_params(labelsize=11)
                if coordinate == 0:
                    ax.set_title(fr"$\mu={mu:g}$", fontsize=18)
                if log:
                    ax.set_yscale("log")
            for ax in axes[coordinate]:
                ax.set_ylim((max(max_height * 1e-7, 1e-8), max_height * 8.)
                            if log else (0, max_height * 1.32))
            axes[coordinate, 0].set_ylabel("Expected events / bin", fontsize=14)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=13, ncol=2, loc="upper center",
                   bbox_to_anchor=(.5, .97))
        fig.suptitle("Full phase space, before preselection; exact marginal bin integrals",
                     fontsize=16, y=1.)
        fig.tight_layout(rect=(0, 0, 1, .94))
        _save(fig, output, "01_coherent_stacks_log" if log else "01_coherent_stacks_linear")
    return fig, axes
