"""Plots for notebook 13's independent-bank learned-compression comparison."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


MODELS = ("analytic unbinned", "binned analytic map", "binned learned map")
LABELS = {
    "analytic unbinned": "Analytical unbinned",
    "binned analytic map": "Analytical map, binned",
    "binned learned map": "Learned map, binned",
}
COLORS = dict(zip(MODELS, ("black", "#4477AA", "#CC6677")))
PROCESS_COLORS = dict(zip(("S", "SBI", "B", "NI"),
                          ("#228833", "#4477AA", "#AA3377", "#EE7733")))


def _style():
    import mplhep as hep
    hep.style.use("ATLAS")
    # Typography and ticks only; deliberately no experiment label.
    plt.rcParams.update({"axes.grid": False, "figure.dpi": 110})


def _save(figures, output):
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
            figure.savefig(output / f"{name}.png", bbox_inches="tight", dpi=160)
    return figures


def plot_compression_scans(study, *, xlim=None, ylim=None, output=None):
    """One q/delta-q figure per Asimov-bank comparison and nuisance treatment."""
    _style()
    figures = {}
    for (comparison, systematics), frame in study["scans"].groupby(
            ["comparison", "systematics"], sort=False):
        fig, (ax, residual) = plt.subplots(
            2, 1, figsize=(8.2, 7.0), sharex=True,
            gridspec_kw={"height_ratios": (3, 1), "hspace": .08})
        for model in MODELS:
            part = frame.loc[frame.model == model].sort_values("mu")
            if part.empty:
                continue
            valid = part.valid.to_numpy(dtype=bool)
            q = np.where(valid, part.q, np.nan)
            ax.plot(part.mu, q, color=COLORS[model], label=LABELS[model],
                    linewidth=2, linestyle="-" if model != MODELS[2] else "--")
            if model != MODELS[0]:
                residual.plot(part.mu, np.where(valid, part.delta_q, np.nan),
                              color=COLORS[model], linewidth=1.8,
                              linestyle="-" if model == MODELS[1] else "--")
        residual.axhline(0, color="0.5", linewidth=1)
        ax.axhline(1., color="0.65", linewidth=.8, linestyle=":")
        ax.set_ylabel(r"$q_\mu$")
        residual.set_ylabel(r"$\Delta q_\mu$")
        residual.set_xlabel(r"Tested physical $\mu$ ($\eta=\mu$)")
        ax.legend(frameon=False, fontsize=11)
        title = ("Matched Asimov: A data / A templates" if comparison == "matched"
                 else "Independent Asimov: B data / A templates")
        ax.set_title(title + ("\nNI profiled" if systematics else "\nNI fixed"), fontsize=14)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        elif len(frame):
            ax.set_ylim(bottom=0)
        name = f"scan_{comparison}_{'profiled' if systematics else 'fixed'}"
        figures[name] = fig
    return _save(figures, output)


def plot_compression_information(study, *, xlim=None, output=None):
    """Local widths at the physical Asimov truth, for each frozen partition."""
    _style()
    figures = {}
    for systematics, frame in study["information"].groupby("systematics", sort=False):
        fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True,
                                 gridspec_kw={"height_ratios": (2, 1), "hspace": .08})
        for model in MODELS:
            part = frame.loc[frame.model == model].sort_values("eta")
            if part.empty:
                continue
            if model == MODELS[0] and (len(part) == 1 or part.eta.isna().all()):
                axes[0].axhline(part.sigma_mu.iloc[0], color=COLORS[model],
                                label=LABELS[model], linewidth=2)
                axes[1].axhline(1., color=COLORS[model], linewidth=1)
            else:
                valid = part.width_valid.to_numpy(dtype=bool)
                for ax, value in zip(axes, ("sigma_mu", "information_fraction")):
                    ax.plot(part.eta, np.where(valid, part[value], np.nan),
                            color=COLORS[model], label=LABELS[model], linewidth=2,
                            linestyle="--" if model == MODELS[2] else "-")
        axes[0].set_ylabel(r"Local width $\sigma_\mu$")
        axes[1].set_ylabel("Information\nfraction")
        axes[1].set_xlabel(r"Observable construction parameter $\eta$")
        axes[0].legend(frameon=False, fontsize=11)
        axes[0].set_title("At the physical Asimov truth; " +
                          ("NI profiled" if systematics else "NI fixed"), fontsize=14)
        if xlim is not None:
            axes[0].set_xlim(*xlim)
        figures[f"local_information_{'profiled' if systematics else 'fixed'}"] = fig
    return _save(figures, output)


def _eta_rows(frame, eta):
    return frame.loc[np.isclose(frame.eta, eta, rtol=0, atol=1e-12)]


def _edges(frame):
    return np.r_[frame.low.to_numpy(dtype=float), float(frame.high.iloc[-1])]


def plot_compression_distributions(study, *, etas=(0., .3, .6, 1., 1.5), output=None):
    """Physical component shapes on analytical/learned axes, normalized for display."""
    _style()
    figures = {}
    for eta in etas:
        frame = _eta_rows(study["component_templates"], eta)
        if frame.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8), sharey=True)
        for ax, model in zip(axes, MODELS[1:]):
            available = not frame.loc[frame.model == model].empty
            for process, color in PROCESS_COLORS.items():
                part = frame.loc[(frame.model == model) &
                                 (frame.process == process)].sort_values("bin")
                if part.empty:
                    continue
                counts = part.expected_yield.to_numpy(dtype=float)
                if counts.sum() > 0:
                    ax.stairs(counts / counts.sum(), _edges(part),
                              label=process, color=color, linewidth=1.7)
            ax.set_title(LABELS[model], fontsize=14)
            ax.set_xlabel(r"$t_\eta(x)$")
            ax.set_xlim(0, 1)
            ax.set_yscale("log")
            if available:
                ax.legend(frameon=False, fontsize=11)
            else:
                ax.text(.5, .5, "Invalid map; inspect diagnostics", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11)
        axes[0].set_ylabel("Process fraction / bin")
        fig.suptitle(rf"Physical templates, $\eta={eta:g}$", fontsize=16)
        fig.tight_layout()
        figures[f"component_shapes_eta_{eta:g}".replace(".", "p")] = fig
    return _save(figures, output)


def plot_compression_closure(study, *, etas=(0., .3, .6, 1., 1.5), output=None):
    """Unrescaled independent-bank yields and their B/A closure ratios."""
    _style()
    figures = {}
    for eta in etas:
        frame = _eta_rows(study["bin_closure"], eta)
        if frame.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(13.4, 7.), sharex="col",
                                 gridspec_kw={"height_ratios": (3, 1), "hspace": .08,
                                             "wspace": .28})
        for column, model in enumerate(MODELS[1:]):
            part = frame.loc[frame.model == model].sort_values("bin")
            ax, residual = axes[:, column]
            ax.set_title(LABELS[model], fontsize=14)
            ax.set_ylabel("Expected events / bin")
            residual.set_ylabel("B / A")
            residual.set_xlabel(r"$t_\eta(x)$")
            residual.set_xlim(0, 1)
            if part.empty:
                axes[0, column].text(.5, .5, "Invalid map; inspect diagnostics",
                                     transform=axes[0, column].transAxes,
                                     ha="center", va="center", fontsize=11)
                continue
            centers = .5 * (part.low.to_numpy() + part.high.to_numpy())
            template = part.template.to_numpy(dtype=float)
            validation = part.validation.to_numpy(dtype=float)
            if "template_se" in part:
                template_error = part.template_se.to_numpy(dtype=float)
                ax.stairs(template + template_error, _edges(part),
                          baseline=np.maximum(template-template_error, 0), fill=True,
                          color=COLORS[model], alpha=.2, linewidth=0)
            ax.stairs(template, _edges(part), color=COLORS[model], linewidth=2,
                      label="Template bank A")
            error_column = next((c for c in ("validation_mc_error", "validation_se", "validation_error")
                                 if c in part), None)
            validation_error = None if error_column is None else part[error_column].to_numpy()
            ax.errorbar(centers, validation, yerr=validation_error, fmt="o",
                        markersize=3, color="black", linewidth=1,
                        label="Independent bank B")
            ratio_error_column = next((c for c in ("ratio_mc_error", "ratio_se", "ratio_error")
                                       if c in part), None)
            ratio_error = None if ratio_error_column is None else part[ratio_error_column].to_numpy()
            # Empty reference cells remain undefined and are never filled by hand.
            ratio = np.where(template > 0, part.ratio, np.nan)
            residual.errorbar(centers, ratio, yerr=ratio_error, fmt="o", markersize=3,
                              color="black", linewidth=1)
            residual.axhline(1., color="0.5", linewidth=1)
            ax.set_yscale("log")
            ax.legend(frameon=False, fontsize=11)
        fig.suptitle(rf"Independent simulation closure, $\eta={eta:g}$", fontsize=16)
        fig.subplots_adjust(top=.85)
        figures[f"bin_closure_eta_{eta:g}".replace(".", "p")] = fig
    return _save(figures, output)
