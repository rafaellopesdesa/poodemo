"""Closure figures for notebook 09; every figure is saved under its evaluation."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from .reference_splines import LABELS, METHODS

COLORS = dict(direct="black", ecdf_u="#1f77b4")


def plot_reference_spline_comparison(run, study, *, scan_xlim=(.02, 1.15),
                                     scan_ylim=(0., 8.), focus_xlim=(.94, 1.06)):
    import mplhep as hep
    destination = run.path("plots", "09_interpolation")
    destination.mkdir(parents=True, exist_ok=True)
    figures = []

    def save(fig, name):
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(destination / f"09_{name}.{extension}", bbox_inches="tight", dpi=150)
        figures.append(fig)

    with plt.style.context(hep.style.ATLAS):
        scans = study["scans"]
        main = scans.loc[scans.on_scan]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, syst in zip(axes, (False, True)):
            for method in ("direct", *METHODS):
                values = main.loc[(main.method == method) & (main.systematics == syst)].sort_values("eta")
                ax.plot(values.eta, values.q.where(values.valid), color=COLORS[method],
                        label=LABELS[method], lw=2 if method == "direct" else 1.5)
            unbinned = study["unbinned"]
            if not unbinned.empty:
                for name, values in unbinned.loc[unbinned.systematics == syst].groupby("model"):
                    values = values.sort_values("mu")
                    ax.plot(values.mu, values.q, ls="--", lw=1, alpha=.7, label=name)
            ax.axhline(1., color=".5", ls=":", lw=1)
            ax.set(xlabel=r"Tested signal strength $\mu_0=\eta$", ylabel=r"$q(\mu_0)$",
                   title="NI profiled" if syst else "NI fixed")
            if scan_xlim is not None:
                ax.set_xlim(scan_xlim)
            if scan_ylim is not None:
                ax.set_ylim(scan_ylim)
            ax.legend(fontsize=9)
        save(fig, "scans")

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        tolerance = study["metadata"]["settings"]["q_tolerance"]
        for col, syst in enumerate((False, True)):
            for method in METHODS:
                values = study["comparison"].loc[(study["comparison"].method == method)
                         & (study["comparison"].systematics == syst)].sort_values("eta")
                # Plot absolute error on a logarithmic scale so failures remain
                # visible without clipping a signed-difference panel.
                axes[0, col].semilogy(values.eta, np.maximum(values.delta_q.abs(), 1e-12),
                                      ".-", ms=3, color=COLORS[method], label=LABELS[method])
                axes[1, col].plot(values.eta, values.mu_hat.where(values.valid),
                                  ".-", ms=3, color=COLORS[method], label=LABELS[method])
            direct = scans.loc[(scans.method == "direct") & (scans.systematics == syst)].sort_values("eta")
            axes[1, col].plot(direct.eta, direct.mu_hat.where(direct.valid), color="black", label="Direct fills")
            axes[0, col].axhline(tolerance, color=".4", ls="--", label="Closure tolerance")
            axes[0, col].set(ylabel=r"$|q_{\rm interpolated}-q_{\rm direct}|$",
                             title="NI profiled" if syst else "NI fixed")
            axes[1, col].set(ylabel=r"Fitted $\hat\mu$", xlim=focus_xlim)
            for row in (0, 1):
                axes[row, col].set_xlabel(r"Frozen observable $\eta$")
                axes[row, col].legend(fontsize=9)
        save(fig, "likelihood_closure")

        fig, axes = plt.subplots(1, 3, figsize=(19, 5))
        validation = study["validation"]
        for method in METHODS:
            for ax, role in zip(axes[:2], ("heldout", "anchor_coordinate_check")):
                values = validation.loc[(validation.method == method) & (validation.role == role)].sort_values("eta")
                ax.semilogy(values.eta, np.maximum(values.max_deviance, 1e-15), ".",
                            color=COLORS[method], label=LABELS[method])
            values = study["fixed_checks"].loc[(study["fixed_checks"].method == method)
                & (study["fixed_checks"].mu == run.config["asimov_mu"])
                & (study["fixed_checks"].alpha_NI == 0.)].sort_values("eta")
            axes[2].semilogy(values.eta, np.maximum(values.nll_interpolated, 1e-15), ".",
                             color=COLORS[method], label=LABELS[method])
        for ax, title in zip(axes[:2], ("Held-out templates", r"Coordinate checks at $\eta$ anchors")):
            ax.axhline(study["metadata"]["settings"]["max_deviance"], color=".4", ls="--")
            ax.set_ylabel("Largest probe deviance", fontsize=20)
            ax.set_title(title, fontsize=18)
        axes[2].set_ylabel(r"$-2\log L(\mu_*,0)$", fontsize=20)
        axes[2].set_title("Generating point (direct: 0)", fontsize=18)
        for ax in axes:
            ax.set_xlabel(r"Frozen observable $\eta$")
            ax.legend(fontsize=9)
        save(fig, "template_closure")

        history = study["refinement"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history.n_anchors, history.n_failed, "o-", label="Template midpoint failures")
        axes[0].plot(history.n_anchors, history.q_failed, "s-", label="Likelihood refinement failures")
        axes[0].set(xlabel="Number of anchors", ylabel="Failed checks", title="Refinement convergence")
        axes[0].legend(fontsize=9)
        for syst, label in ((False, "NI fixed"), (True, "NI profiled")):
            checks = study["likelihood_refinement"].loc[study["likelihood_refinement"].systematics == syst]
            worst = checks.groupby("round").delta_q.apply(lambda x: x.abs().max())
            axes[1].semilogy(worst.index, np.maximum(worst, 1e-12), "o-", label=label)
        axes[1].axhline(tolerance, color=".4", ls="--", label="Closure tolerance")
        axes[1].set(xlabel="Refinement round", ylabel=r"Largest $|\Delta q|$", title="Finite valid refinement fits")
        axes[1].legend(fontsize=9)
        save(fig, "refinement")

        # Inspect the bins at the worst finite likelihood discrepancy.
        comparison = study["comparison"]
        finite = comparison.loc[np.isfinite(comparison.delta_q)]
        if len(finite):
            eta = float(finite.loc[finite.delta_q.abs().idxmax(), "eta"])
            selected = study["bin_errors"].loc[study["bin_errors"].eta == eta]
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
            direct = selected.loc[selected.method == METHODS[0]]
            axes[0].step(direct.bin, direct.observed, where="mid", color="black", label="Direct Asimov counts")
            outside, empty_counts = 0, []
            for method in METHODS:
                values = selected.loc[selected.method == method]
                axes[0].plot(values.bin, values.expected, ".", color=COLORS[method], label=LABELS[method])
                good = values.observed > 0
                ratio = values.loc[good, "expected"] / values.loc[good, "observed"]
                axes[1].plot(values.loc[good, "bin"], ratio, ".", color=COLORS[method])
                outside += int(((ratio < .95) | (ratio > 1.05)).sum())
                empty_counts.append(f"{LABELS[method]}: {values.loc[~good, 'expected'].sum():.3g}")
            axes[0].set(yscale="symlog", ylabel="Expected events", title=fr"Rare-bin check at $\eta={eta:.7g}$")
            axes[0].legend(fontsize=10)
            axes[1].axhline(1., color=".5", ls="--")
            axes[1].set(xlabel="Original observable bin", ylim=(.95, 1.05))
            axes[1].set_ylabel("Interpolated / direct", fontsize=20)
            axes[1].text(.02, .06, f"{outside} ratios outside displayed range; empty-bin ratios omitted\n"
                          + "Events predicted in empty bins — " + "; ".join(empty_counts),
                          transform=axes[1].transAxes, fontsize=8)
            save(fig, "rare_bins")

        surfaces = study["surfaces"]
        for index, (eta, group) in enumerate(surfaces.groupby("eta")):
            columns = 1 + len(METHODS)
            fig, axes = plt.subplots(2, columns, figsize=(5 * columns, 9), sharex=True, sharey=True)
            reference = group.loc[group.method == "direct"].pivot(index="alpha_NI", columns="mu", values="nll")
            reference_values = reference.to_numpy()
            offset = np.nanmin(np.where(np.isfinite(reference_values), reference_values, np.nan))
            for col, method in enumerate(("direct", *METHODS)):
                mesh = group.loc[group.method == method].pivot(index="alpha_NI", columns="mu", values="nll")
                values = mesh.to_numpy()
                top_image = axes[0, col].contourf(mesh.columns, mesh.index, np.ma.masked_invalid(values-offset),
                                      levels=[0., 1., 4., 9., 25., 100.], extend="max", cmap="viridis_r")
                axes[0, col].set_title(LABELS[method].replace("empirical CDF", "empirical\nCDF"), fontsize=15)
                difference = np.full_like(values, np.nan)
                np.subtract(values, reference_values, out=difference,
                            where=np.isfinite(values) & np.isfinite(reference_values))
                im = axes[1, col].pcolormesh(mesh.columns, mesh.index, np.ma.masked_invalid(difference),
                    shading="auto", cmap="coolwarm", norm=SymLogNorm(linthresh=.01, vmin=-10., vmax=10.))
                fit = scans.loc[(scans.method == method) & (scans.eta == eta) & scans.systematics & scans.valid]
                if len(fit):
                    axes[0, col].plot(fit.mu_hat, fit.alpha_hat, "r*", ms=9)
                axes[1, col].set_xlabel(r"Physical $\mu$")
            axes[0, 0].set_ylabel(r"$\alpha_{NI}$")
            axes[1, 0].set_ylabel(r"$\alpha_{NI}$")
            fig.suptitle(fr"Frozen $\eta={eta:.7g}$; stars: profiled minima; blank: invalid", fontsize=16)
            # Colorbar axes are not compatible with tight_layout.
            fig.subplots_adjust(wspace=.3, hspace=.25, top=.9, right=.86)
            fig.colorbar(top_image, cax=fig.add_axes([.90, .56, .012, .28]),
                         label="NLL, common offset")
            fig.colorbar(im, cax=fig.add_axes([.90, .14, .012, .28]), label="NLL minus direct", extend="both")
            for extension in ("pdf", "png"):
                fig.savefig(destination / f"09_surface_{index}.{extension}", bbox_inches="tight", dpi=150)
            figures.append(fig)
    return figures
