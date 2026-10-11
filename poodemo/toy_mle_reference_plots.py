"""Plots for the per-toy MLE-reference diagnostic in notebook 12.

Ensemble panels use only the preselected random controls.  The reference is
recomputed for each integration bank and then frozen during the binned fit.
Error bars are conditional toy-sampling errors, not integration uncertainties.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd


_LABELS = {
    "analytic": "Analytical unbinned",
    "fixed12": "Fixed reference, 12 bins",
    "fixed36": "Fixed reference, 36 bins",
    "adaptive12": "MLE reference, 12 bins",
    "adaptive36": "MLE reference, 36 bins",
}
_COLORS = {
    "analytic": "black", "fixed12": "#0072b2", "fixed36": "#56b4e9",
    "adaptive12": "#d55e00", "adaptive36": "#009e73",
}
_MODELS = tuple(_LABELS)


def _bool(values):
    """Accept freshly calculated booleans and their CSV representations."""
    return values.astype(str).str.lower().isin(("true", "1", "1.0"))


def _stat(values, rms=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    if rms:
        return float(np.sqrt(np.mean(values * values))), np.nan
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
    return float(values.mean()), se


def _mu_tag(mu):
    return f"{mu:g}".replace("-", "m").replace(".", "p")


def _separation_coordinates(values):
    """Log spacing for positive separations; exact zero has a separate slot.

    No epsilon is added to physical separations.  The zero column is one full
    decade to the left of the positive domain and is explicitly labeled.
    """
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0.)]
    if len(positive):
        low = int(np.floor(np.log10(positive.min())))
        high = int(np.ceil(np.log10(positive.max())))
        high = max(high, low + 1)
    else:
        low, high = -2, 0
    zero_x = float(low - 1)
    exponents = list(range(low, high + 1, max(1, int(np.ceil((high - low) / 4)))))
    if exponents[-1] != high:
        exponents.append(high)

    def transform(array):
        array = np.asarray(array, dtype=float)
        result = np.full(array.shape, np.nan, dtype=float)
        result[array == 0.] = zero_x
        mask = np.isfinite(array) & (array > 0.)
        result[mask] = np.log10(array[mask])
        return result

    ticks = [zero_x] + exponents
    labels = ["Exact\nzero"] + [rf"$10^{{{exponent}}}$" for exponent in exponents]
    return transform, ticks, labels, (zero_x - .35, high + .25), zero_x + .5


def plot_mle_reference_diagnostics(study, output):
    """Save and return the section-9 MLE-reference comparison figures.

    ``study`` is returned by ``run_mle_reference_diagnostics``.  Plotting never
    changes templates, refits toys, or regenerates data.  Signed differences
    and likelihood scans use linear y axes.  Exact reference degeneracy is
    displayed separately from small, positive reference separations.
    """
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, paired = study["records"].copy(), study["paired"].copy()
    if records.empty or paired.empty:
        raise ValueError("MLE-reference plots require fit records and analytical comparisons")
    random = records.loc[_bool(records.is_random)].copy()
    pairs = paired.loc[_bool(paired.is_random)].copy()
    if random.empty or pairs.empty:
        raise ValueError("MLE-reference ensemble plots require random-control toys")
    mus = sorted(random.mu_test.unique())
    replicas = sorted(random.replica.unique())
    largest = int(random.generated_per_source.max())
    models = [model for model in _MODELS[1:] if model in set(pairs.model)]
    all_models = [model for model in _MODELS if model in set(random.model)]
    figures = {}

    def finish(fig, name, note="", top=.92):
        if note:
            fig.text(.5, .012, note, ha="center", va="bottom", fontsize=10)
        fig.tight_layout(rect=(0., .055 if note else 0., 1., top))
        figures[name] = fig

    def model_handles(selected=models, *, distinguish_reference=False):
        return [Line2D([], [], color=_COLORS[model],
                       marker="^" if distinguish_reference and model.startswith("adaptive") else "o", markersize=5,
                       linestyle="none", label=_LABELS[model]) for model in selected]

    style = {"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
             "xtick.labelsize": 10, "ytick.labelsize": 11, "legend.fontsize": 10}
    with plt.style.context(hep.style.ATLAS), plt.rc_context(style):
        # Identity and matched points remain on a linear scale: q can be zero.
        fig, axes = plt.subplots(len(replicas), len(mus),
                                 figsize=(7.0 * len(mus), 4.9 * len(replicas)), squeeze=False)
        for row, replica in enumerate(replicas):
            for col, mu in enumerate(mus):
                ax = axes[row, col]
                frame = pairs.loc[(pairs.replica == replica) & (pairs.mu_test == mu)
                                  & (pairs.generated_per_source == largest)]
                finite = _bool(frame.paired_valid) & np.isfinite(frame.q) & np.isfinite(frame.analytic_q)
                good = frame.loc[finite]
                values = good[["q", "analytic_q"]].to_numpy().ravel()
                upper = max(float(values.max()) * 1.05, 1.) if len(values) else 1.
                lower = min(float(values.min()) * 1.05, 0.) if len(values) else 0.
                ax.plot([lower, upper], [lower, upper], linestyle=":", color="black", linewidth=1)
                counts = []
                for model in models:
                    part = good.loc[good.model == model]
                    total = int((frame.model == model).sum())
                    ax.scatter(part.analytic_q, part.q, s=22, color=_COLORS[model], alpha=.55,
                               marker="o" if model.startswith("fixed") else "^", edgecolors="none")
                    counts.append(f"{_LABELS[model]}: {len(part)}/{total}")
                ax.text(.03, .97, "Valid pairs\n" + "\n".join(counts), transform=ax.transAxes,
                        va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .83})
                ax.set(xlim=(lower, upper), ylim=(lower, upper), xlabel=r"Analytical $q_\eta$",
                       ylabel=r"Binned $q_\eta$", title=rf"$\eta={mu:g}$; replica {int(replica) + 1}")
        fig.legend(handles=model_handles(distinguish_reference=True), loc="upper center", ncol=2, bbox_to_anchor=(.5, .995))
        finish(fig, "mle_reference_paired_q",
               f"Random controls; {largest:,} generated/source. Every binned denominator is freely fitted; the reference is frozen within each fit.")

        metrics = (("delta_q", r"$\Delta q$", "Total difference"),
                   ("delta_pair", r"$\Delta q_{\rm pair}$", "Same two parameter values"),
                   ("refit_gain", r"$g_{\rm refit}$", "Gain from the free binned fit"))
        line_handles = [Line2D([], [], color=_COLORS[model], label=_LABELS[model]) for model in models]
        line_handles += [Line2D([], [], color=".35", marker="o" if index % 2 == 0 else "s",
                               linestyle="-" if index % 2 == 0 else "--", label=f"Replica {int(replica) + 1}")
                         for index, replica in enumerate(replicas)]
        decomposition_sizes = sorted(pairs.generated_per_source.unique())
        single_size = len(decomposition_sizes) == 1
        for rms in (False, True):
            fig, axes = plt.subplots(len(metrics), len(mus), figsize=(7.0 * len(mus), 11.7), squeeze=False)
            for col, mu in enumerate(mus):
                frame = pairs.loc[(pairs.mu_test == mu) & _bool(pairs.paired_valid)]
                for row, (metric, label, description) in enumerate(metrics):
                    ax = axes[row, col]
                    for model in models:
                        for index, replica in enumerate(replicas):
                            part = frame.loc[(frame.model == model) & (frame.replica == replica)]
                            values = [(size, *_stat(group[metric], rms=rms))
                                      for size, group in part.groupby("generated_per_source", sort=True)]
                            if not values:
                                continue
                            x, y, error = np.asarray(values, dtype=float).T
                            if single_size:
                                # The default run uses one prefix: separate the
                                # observables instead of hiding eight estimates
                                # at the same integration-bank size.
                                x = np.full(len(x), models.index(model) +
                                            .18 * (index - (len(replicas) - 1) / 2.))
                            kwargs = dict(color=_COLORS[model], marker="o" if index % 2 == 0 else "s",
                                          linestyle="none" if single_size else ("-" if index % 2 == 0 else "--"),
                                          markersize=5)
                            if rms:
                                ax.plot(x, y, **kwargs)
                            else:
                                ax.errorbar(x, y, yerr=error, capsize=3, **kwargs)
                    if not rms:
                        ax.axhline(0., color=".6", linewidth=.8)
                    ax.set(ylabel=("RMS " if rms else "Mean ") + label,
                           title=rf"$\eta={mu:g}$; {description}")
                    if single_size:
                        ax.set_xticks(np.arange(len(models)),
                                      [_LABELS[model].replace(" reference, ", "\n") for model in models])
                        ax.set(xlim=(-.5, len(models) - .5), xlabel="Observable")
                        ax.minorticks_off()
                    else:
                        ax.set(xscale="log", xlabel="Generated events per source")
                        sizes = sorted(frame.generated_per_source.unique())
                        if sizes:
                            ax.set_xticks(sizes, [f"{int(size):,}" for size in sizes])
                            ax.set_xlim(min(sizes) / 1.15, max(sizes) * 1.15)
                            ax.minorticks_off()
            fig.legend(handles=line_handles, loc="upper center", bbox_to_anchor=(.5, .998), ncol=3)
            finish(fig, "mle_reference_decomposition_rms" if rms else "mle_reference_decomposition_mean",
                   (f"Bank: {int(decomposition_sizes[0]):,} generated/source. " if single_size else "") +
                   r"$\Delta q=\Delta q_{\rm pair}+g_{\rm refit}$, with $g_{\rm refit}\geq0$. Random controls; valid pairs only." + "\n" +
                   ("RMS includes bias and toy variation; component RMS values do not add. References are recomputed for each bank."
                    if rms else "Bars: paired toy-sampling SE conditional on each bank. Prefixes share MC events; each bank has its own unbinned reference."), top=.925)

        categories = (("low", "Low branch", "#0072b2"), ("high", "High branch", "#d55e00"),
                      ("ambiguous", "Tie/shared endpoint", "#cc79a7"), ("invalid", "Failed fit", "#bdbdbd"))
        fig, axes = plt.subplots(len(replicas), len(mus),
                                 figsize=(7.2 * len(mus), 5.0 * len(replicas)), squeeze=False)
        for row, replica in enumerate(replicas):
            for col, mu in enumerate(mus):
                ax = axes[row, col]
                frame = random.loc[(random.replica == replica) & (random.mu_test == mu)
                                   & (random.generated_per_source == largest)]
                bottoms = np.zeros(len(all_models))
                for category, label, color in categories:
                    fractions = []
                    for model in all_models:
                        part = frame.loc[frame.model == model]
                        valid = _bool(part.valid)
                        if category == "invalid":
                            selected = ~valid
                        elif category == "ambiguous":
                            selected = valid & ~part.branch_label.isin(("low", "high"))
                        else:
                            selected = valid & (part.branch_label == category)
                        fractions.append(float(selected.sum() / len(part)) if len(part) else 0.)
                    bars = ax.bar(np.arange(len(all_models)), fractions, bottom=bottoms,
                                  color=color, edgecolor="white", linewidth=.5)
                    for bar, fraction, bottom in zip(bars, fractions, bottoms):
                        if fraction >= .07:
                            ax.text(bar.get_x() + bar.get_width()/2, bottom + fraction/2,
                                    f"{100*fraction:.0f}%", ha="center", va="center", fontsize=10,
                                    color="white" if category in ("low", "high") else "black")
                    bottoms += fractions
                for index, model in enumerate(all_models):
                    part = frame.loc[frame.model == model]
                    ax.text(index, 1.015, f"n={len(part)}", ha="center", fontsize=9)
                ax.set(xticks=np.arange(len(all_models)), xticklabels=[_LABELS[model] for model in all_models],
                       ylim=(0., 1.1), ylabel="Fraction of all random controls",
                       title=rf"$\eta={mu:g}$; replica {int(replica) + 1}")
                ax.tick_params(axis="x", labelrotation=25)
        fig.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in categories],
                   loc="upper center", bbox_to_anchor=(.5, .99), ncol=4)
        finish(fig, "mle_reference_branch_fractions",
               f"{largest:,} generated/source. Failed and ambiguous fits remain in the denominator. References and fits are calculated separately for each bank.", top=.94)

        # The same row population drives the separation and occupancy panels.
        # Exact-zero references are not jittered, clipped, or put on log axes.
        adaptive = pairs.loc[pairs.model.str.startswith("adaptive")
                             & (pairs.generated_per_source == largest)].copy()
        if not adaptive.empty:
            transform, ticks, labels, limits, divider = _separation_coordinates(adaptive.reference_separation_kappa)
            row_keys = [(mu, replica) for mu in mus for replica in replicas]
            fig, axes = plt.subplots(len(row_keys), 3, figsize=(17.5, 4.0 * len(row_keys)), squeeze=False)
            for row, (mu, replica) in enumerate(row_keys):
                frame = adaptive.loc[(adaptive.mu_test == mu) & (adaptive.replica == replica)]
                for col, (metric, ylabel) in enumerate((("delta_q", r"$\Delta q$"),
                                                         ("refit_gain", r"$g_{\rm refit}$"),
                                                         ("occupied_toy_bins", "Occupied bins"))):
                    ax = axes[row, col]
                    counts = []
                    for model in [item for item in models if item.startswith("adaptive")]:
                        full = frame.loc[frame.model == model]
                        good = full if col == 2 else full.loc[_bool(full.paired_valid)]
                        finite = np.isfinite(good[metric]) & np.isfinite(good.reference_separation_kappa)
                        part = good.loc[finite]
                        x = transform(part.reference_separation_kappa)
                        ax.scatter(x, part[metric], color=_COLORS[model], s=22, alpha=.5, edgecolors="none")
                        if col == 2:
                            predicted = full.loc[np.isfinite(full.occupied_mc_bins)
                                                 & np.isfinite(full.reference_separation_kappa)]
                            ax.scatter(transform(predicted.reference_separation_kappa), predicted.occupied_mc_bins,
                                       color=_COLORS[model], s=25, alpha=.45, marker="x", linewidths=.8)
                        zeros = int((full.reference_separation_kappa == 0.).sum())
                        counts.append(f"{int(full.n_bins.iloc[0]) if len(full) else '?'} bins: zero {zeros}/{len(full)}, plotted {len(part)}")
                    ax.axvline(divider, color=".65", linestyle=":", linewidth=.8)
                    if col < 2:
                        ax.axhline(0., color=".65", linewidth=.8)
                    ax.set(xlim=limits, xticks=ticks, xticklabels=labels,
                           xlabel=r"$|\sqrt{\eta}-\sqrt{\hat\mu_{\rm ana}}|$ (positive values: log spacing)",
                           ylabel=ylabel, title=rf"$\eta={mu:g}$; replica {int(replica) + 1}")
                    ax.text(.98, .97, "\n".join(counts), transform=ax.transAxes, ha="right", va="top", fontsize=9,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
            handles = model_handles([model for model in models if model.startswith("adaptive")])
            handles += [Line2D([], [], marker="o", linestyle="none", color=".35", label="Observed occupied bins"),
                        Line2D([], [], marker="x", linestyle="none", color=".35", label="MC occupied bins")]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995), ncol=4)
            finish(fig, "mle_reference_separation",
                   f"Random controls; {largest:,} generated/source. Exact zero has its own left-hand column; overlapping points share the same separation.\n"
                   "Likelihood panels use valid pairs; occupancy panels retain all finite records, including failed fits. MC occupancy is template-bank support.", top=.96)

        scans = study.get("scans", pd.DataFrame())
        if not scans.empty:
            selected = study.get("scan_selection", pd.DataFrame())
            for mu in sorted(scans.mu_test.unique()):
                frame = scans.loc[(scans.mu_test == mu) & (scans.generated_per_source == largest)]
                if not selected.empty and {"mu_test", "toy_id"} <= set(selected):
                    toy_ids = list(dict.fromkeys(selected.loc[selected.mu_test == mu, "toy_id"].astype(int)))[:3]
                else:
                    toy_ids = sorted(frame.toy_id.unique())[:3]
                if not toy_ids:
                    continue
                fig, axes = plt.subplots(len(replicas), len(toy_ids),
                                         figsize=(6.1 * len(toy_ids), 5.0 * len(replicas)), squeeze=False)
                for row, replica in enumerate(replicas):
                    for col, toy_id in enumerate(toy_ids):
                        ax = axes[row, col]
                        part = frame.loc[(frame.replica == replica) & (frame.toy_id == toy_id)]
                        fits = records.loc[(records.mu_test == mu) & (records.toy_id == toy_id)
                                           & (records.replica == replica) & (records.generated_per_source == largest)]
                        failed = []
                        for model in all_models:
                            curve = part.loc[part.model == model].sort_values("mu_scan")
                            ax.plot(curve.mu_scan, curve.D, color=_COLORS[model], linewidth=1.6,
                                    linestyle="--" if model.endswith("12") else "-")
                            fit = fits.loc[fits.model == model]
                            if not fit.empty and bool(_bool(fit.valid).iloc[0]):
                                first = fit.iloc[0]
                                ax.scatter([first.mu_low, first.mu_high], [first.D_low, first.D_high],
                                           color=_COLORS[model], marker="o", s=24, zorder=4)
                            else:
                                failed.append(_LABELS[model])
                        anchor = fits.loc[fits.model == "analytic", "mu_hat"]
                        anchor = float(anchor.iloc[0]) if len(anchor) else np.nan
                        ax.axvline(1., color=".6", linestyle=":", linewidth=.8)
                        if np.isfinite(anchor):
                            ax.axvline(anchor, color=".3", linestyle="-.", linewidth=.8)
                        ax.axhline(0., color=".6", linewidth=.8)
                        reason = str(part.selection_reason.iloc[0]) if len(part) and "selection_reason" in part else "selected example"
                        reason = reason.replace("_", " ")
                        ax.set(xlabel=r"$\mu$", ylabel=r"$D(\mu)=-2\log[L(\mu)/L(1)]$",
                               title=f"Toy {int(toy_id)}, replica {int(replica) + 1}; " + rf"$\hat\mu_{{\rm ana}}={anchor:.5g}$" +
                               "\n" + fill(reason, 45))
                        if failed:
                            ax.text(.03, .97, "Failed fits: " + ", ".join(failed), transform=ax.transAxes,
                                    va="top", color="#9c3028", fontsize=9,
                                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
                handles = [Line2D([], [], color=_COLORS[model], linestyle="--" if model.endswith("12") else "-",
                                  label=_LABELS[model]) for model in all_models]
                fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .96), ncol=3)
                fig.suptitle(rf"$\eta={mu:g}$; selected full likelihood scans", y=1., fontsize=17)
                finish(fig, f"mle_reference_scans_mu{_mu_tag(mu)}",
                       f"{largest:,} generated/source. All curves are relative to μ=1; dots mark branch minima. Dash-dot line: this bank's analytical MLE.\n"
                       "Selected examples are not an ensemble summary. The MLE reference is frozen while each binned curve is scanned.", top=.87)

        for name, fig in figures.items():
            for extension in ("pdf", "png"):
                fig.savefig(output / f"12_{name}.{extension}", dpi=160, bbox_inches="tight")
    return figures
