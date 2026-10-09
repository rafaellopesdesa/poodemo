"""Plots for notebook 12's two-dimensional family and branch diagnostics.

Ensemble panels use the saved random controls only.  Error bars describe toy
sampling conditional on the integration banks; the support panels instead show
the propagated finite-bank errors with the quantile binning held fixed.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd


_LABELS = {
    "analytic": "Analytical unbinned",
    "binned12": "1D, 12 bins",
    "binned36": "1D, 36 bins",
    "family12x12": r"2D, $12\times12$ bins",
    "family36x36": r"2D, $36\times36$ bins",
}
_COLORS = {
    "analytic": "black", "binned12": "#0072b2", "binned36": "#56b4e9",
    "family12x12": "#d55e00", "family36x36": "#009e73",
}
_MODELS = tuple(_LABELS)


def _bool(values):
    """Accept freshly computed booleans as well as CSV round trips."""
    return values.astype(str).str.lower().isin(("true", "1", "1.0"))


def _models(frame, *, analytic=False):
    return [model for model in _MODELS if model in set(frame.model)
            and (analytic or model != "analytic")]


def _mu_tag(mu):
    return f"{mu:g}".replace("-", "m").replace(".", "p")


def _stat(values, rms=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    if rms:
        # This is descriptive RMS, not a fitted bias correction.
        return float(np.sqrt(np.mean(values * values))), np.nan
    error = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
    return float(np.mean(values)), error


def plot_family_diagnostics(study, bank_study, output):
    """Save and return the section-8 family, branch and support figures.

    ``study`` is the output of ``run_family_diagnostics`` and ``bank_study``
    the output of its bank preparation.  No fits, event generation or template
    modification occur here.  Signed likelihood differences use linear axes.
    Representative scans show only the largest available integration prefix;
    the paired-summary figures retain both prefixes and both replicas.
    """
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, paired = study["records"].copy(), study["paired"].copy()
    if records.empty or paired.empty:
        raise ValueError("Family plots require saved fit records and matched analytical comparisons")
    random = records.loc[_bool(records.is_random)].copy()
    pairs = paired.loc[_bool(paired.is_random)].copy()
    if random.empty or pairs.empty:
        raise ValueError("Family ensemble plots require the saved random-control population")
    mus = sorted(random.mu_test.unique())
    replicas = sorted(random.replica.unique())
    largest = int(random.generated_per_source.max())
    models = _models(pairs)
    all_models = _models(random, analytic=True)
    figures = {}

    def finish(fig, name, note=""):
        if note:
            fig.text(.5, .012, note, ha="center", va="bottom", fontsize=10)
        fig.tight_layout(rect=(0., .04 if note else 0., 1., .96))
        figures[name] = fig

    style = {"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
             "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10}
    with plt.style.context(hep.style.ATLAS), plt.rc_context(style):
        # Positive G prefers the low branch.  This sign is deliberately stated
        # beside every scatter so an upward or downward displacement is clear.
        scatter_colors = {"same": "#777777", "low_high": "#d55e00", "high_low": "#0072b2",
                          "ambiguous": "#cc79a7"}
        for mu in mus:
            frame = pairs.loc[(pairs.mu_test == mu) & (pairs.generated_per_source == largest)]
            good = frame.loc[_bool(frame.paired_valid) & np.isfinite(frame.G) & np.isfinite(frame.analytic_G)]
            all_values = good[["G", "analytic_G"]].to_numpy().ravel()
            lo, hi = (min(float(all_values.min()), 0.), max(float(all_values.max()), 0.)) if len(all_values) else (-1., 1.)
            padding = .07 * max(hi - lo, 1.)
            limits = (lo - padding, hi + padding)
            fig, axes = plt.subplots(len(replicas), len(models),
                                     figsize=(5.3 * len(models), 4.6 * len(replicas)), squeeze=False)
            for row, replica in enumerate(replicas):
                for col, model in enumerate(models):
                    ax = axes[row, col]
                    full = frame.loc[(frame.replica == replica) & (frame.model == model)]
                    part = good.loc[(good.replica == replica) & (good.model == model)]
                    kinds = np.full(len(part), "same", dtype=object)
                    kinds[(part.analytic_G.to_numpy() > 0.) & (part.G.to_numpy() < 0.)] = "low_high"
                    kinds[(part.analytic_G.to_numpy() < 0.) & (part.G.to_numpy() > 0.)] = "high_low"
                    if "branch_label" in part and "analytic_branch_label" in part:
                        ambiguous = ~part.branch_label.isin(("low", "high")) | ~part.analytic_branch_label.isin(("low", "high"))
                        kinds[ambiguous.to_numpy()] = "ambiguous"
                    for kind, color in scatter_colors.items():
                        points = part.loc[kinds == kind]
                        ax.scatter(points.analytic_G, points.G, s=18, c=color, alpha=.68, edgecolors="none")
                    ax.plot(limits, limits, color="black", linewidth=1, linestyle=":")
                    ax.axhline(0., color=".65", linewidth=.7)
                    ax.axvline(0., color=".65", linewidth=.7)
                    ax.set(xlim=limits, ylim=limits, xlabel=r"Analytical $G$", ylabel=r"Binned $G$",
                           title=f"{_LABELS[model]}; replica {int(replica) + 1}")
                    ax.text(.04, .96, f"Valid pairs: {len(part)}/{len(full)}", transform=ax.transAxes,
                            va="top", fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
            handles = [Line2D([], [], marker="o", linestyle="none", color=scatter_colors[key], label=label)
                       for key, label in (("same", "Same branch"), ("low_high", "Analytical low → binned high"),
                                          ("high_low", "Analytical high → binned low"), ("ambiguous", "Ambiguous branch"))]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.025), ncol=4, fontsize=11)
            fig.suptitle(rf"$\mu_{{\rm test}}={mu:g}$; {largest:,} generated/source; random controls", y=1.07, fontsize=17)
            finish(fig, f"family_branch_preference_mu{_mu_tag(mu)}",
                   r"$G=\min_{\rm high}D-\min_{\rm low}D$: negative values prefer the high branch. Each point is one matched toy.")

        metrics = (("delta_G", r"$\Delta G$", "Total branch-preference difference"),
                   ("delta_common", r"$C-G_{\rm ana}$", "Same two parameter values"),
                   ("delta_refit", r"$G_{\rm bin}-C$", "Within-branch refitting"))
        legend_handles = [Line2D([], [], color=_COLORS[m], label=_LABELS[m]) for m in models]
        legend_handles.extend(Line2D([], [], color=".35", marker="o" if int(r) % 2 == 0 else "s",
                                     linestyle="-" if int(r) % 2 == 0 else "--", label=f"Replica {int(r) + 1}")
                              for r in replicas)
        for rms in (False, True):
            fig, axes = plt.subplots(len(metrics), len(mus), figsize=(7.0 * len(mus), 11.4), squeeze=False)
            for col, mu in enumerate(mus):
                frame = pairs.loc[(pairs.mu_test == mu) & _bool(pairs.paired_valid)]
                for row, (metric, label, description) in enumerate(metrics):
                    ax = axes[row, col]
                    for model in models:
                        for replica in replicas:
                            part = frame.loc[(frame.model == model) & (frame.replica == replica)]
                            values = [(size, *_stat(group[metric], rms=rms))
                                      for size, group in part.groupby("generated_per_source", sort=True)]
                            if not values:
                                continue
                            x, y, err = np.asarray(values).T
                            kwargs = dict(color=_COLORS[model], marker="o" if int(replica) % 2 == 0 else "s",
                                          linestyle="-" if int(replica) % 2 == 0 else "--", markersize=5)
                            if rms:
                                ax.plot(x, y, **kwargs)
                            else:
                                ax.errorbar(x, y, yerr=err, capsize=3, **kwargs)
                    if not rms:
                        ax.axhline(0., color=".6", linewidth=.8)
                    ax.set(xscale="log", xlabel="Generated events per source", ylabel=("RMS " if rms else "Mean ") + label)
                    ax.set_title(rf"$\mu_{{\rm test}}={mu:g}$; {description}")
                    sizes = sorted(frame.generated_per_source.unique())
                    if sizes:
                        ax.set_xticks(sizes, [f"{int(size):,}" for size in sizes])
                        ax.set_xlim(min(sizes) / 1.15, max(sizes) * 1.15)
                        ax.minorticks_off()
            fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(.5, 1.02), ncol=3, fontsize=11)
            finish(fig, "family_branch_decomposition_rms" if rms else "family_branch_decomposition_mean",
                   "Random controls; valid paired fits only. " +
                   ("RMS includes both bias and toy-to-toy variation; no RMS error bars are inferred."
                    if rms else "Bars: paired toy-sampling SE conditional on each bank; prefixes share MC events."))

        # Include failed and ambiguous cases in the denominator instead of
        # quietly presenting a success-conditioned branch fraction.
        categories = (("low", "Low", "#0072b2"), ("high", "High", "#d55e00"),
                      ("ambiguous", "Tie/shared endpoint", "#cc79a7"), ("invalid", "Failed", "#bdbdbd"))
        fig, axes = plt.subplots(len(replicas), len(mus), figsize=(7.1 * len(mus), 4.7 * len(replicas)), squeeze=False)
        for row, replica in enumerate(replicas):
            for col, mu in enumerate(mus):
                ax = axes[row, col]
                frame = random.loc[(random.mu_test == mu) & (random.replica == replica)
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
                    bars = ax.bar(np.arange(len(all_models)), fractions, bottom=bottoms, color=color,
                                  label=label, edgecolor="white", linewidth=.5)
                    for bar, fraction, bottom in zip(bars, fractions, bottoms):
                        if fraction >= .07:
                            ax.text(bar.get_x() + bar.get_width()/2, bottom + fraction/2,
                                    f"{100*fraction:.0f}%", ha="center", va="center", fontsize=10,
                                    color="white" if category in ("low", "high") else "black")
                    bottoms += fractions
                for index, model in enumerate(all_models):
                    n = int((frame.model == model).sum())
                    ax.text(index, 1.015, f"n={n}", ha="center", fontsize=9)
                ax.set(xticks=np.arange(len(all_models)), xticklabels=[_LABELS[m] for m in all_models],
                       ylim=(0., 1.10), ylabel="Fraction of all random controls",
                       title=rf"$\mu_{{\rm test}}={mu:g}$; replica {int(replica) + 1}")
                ax.tick_params(axis="x", labelrotation=25)
        fig.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in categories],
                   loc="upper center", bbox_to_anchor=(.5, 1.025), ncol=4)
        finish(fig, "family_branch_fractions",
               f"Largest bank: {largest:,} generated/source. Ambiguous cases and failures remain in the denominator.")

        scans = study.get("scans", pd.DataFrame())
        if not scans.empty:
            for mu in sorted(scans.mu_test.unique()):
                frame = scans.loc[(scans.mu_test == mu) & (scans.generated_per_source == largest)]
                selected = study.get("scan_selection", pd.DataFrame())
                if not selected.empty and {"mu_test", "toy_id"} <= set(selected):
                    toy_ids = list(dict.fromkeys(selected.loc[selected.mu_test == mu, "toy_id"].astype(int)))[:3]
                else:
                    toy_ids = sorted(frame.toy_id.unique())[:3]
                if not toy_ids:
                    continue
                fig, axes = plt.subplots(len(replicas), len(toy_ids),
                                         figsize=(6.0 * len(toy_ids), 4.8 * len(replicas)), squeeze=False)
                for row, replica in enumerate(replicas):
                    for col, toy_id in enumerate(toy_ids):
                        ax = axes[row, col]
                        part = frame.loc[(frame.replica == replica) & (frame.toy_id == toy_id)]
                        failed_models = []
                        for model in all_models:
                            curve = part.loc[part.model == model].sort_values("mu_scan")
                            ax.plot(curve.mu_scan, curve.D, color=_COLORS[model], linewidth=1.7,
                                    linestyle="-" if model in ("analytic", "binned36", "family36x36") else "--")
                            fit = records.loc[(records.mu_test == mu) & (records.toy_id == toy_id)
                                              & (records.replica == replica) & (records.generated_per_source == largest)
                                              & (records.model == model)]
                            if not fit.empty and bool(_bool(fit.valid).iloc[0]):
                                first = fit.iloc[0]
                                ax.scatter([first.mu_low, first.mu_high], [first.D_low, first.D_high],
                                           color=_COLORS[model], marker="o", s=24, zorder=4)
                            else:
                                failed_models.append(_LABELS[model])
                        ax.axhline(0., color=".65", linewidth=.8)
                        ax.axvline(1., color=".65", linewidth=.8, linestyle=":")
                        reason = str(part.selection_reason.iloc[0]) if len(part) and "selection_reason" in part else "representative"
                        reason = reason.replace("_", " ")
                        ax.set(xlabel=r"$\mu$", ylabel=r"$D(\mu)=-2\log[L(\mu)/L(1)]$",
                               title=f"Toy {int(toy_id)}, replica {int(replica) + 1}\n{fill(reason, 42)}")
                        ax.set_yscale("linear")
                        if failed_models:
                            ax.text(.04, .96, "Failed fits: " + ", ".join(failed_models), transform=ax.transAxes,
                                    va="top", fontsize=9, color="#9c3028",
                                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
                fig.legend(handles=[Line2D([], [], color=_COLORS[m],
                                           linestyle="-" if m in ("analytic", "binned36", "family36x36") else "--",
                                           label=_LABELS[m]) for m in all_models],
                           loc="upper center", bbox_to_anchor=(.5, 1.025), ncol=3)
                fig.suptitle(rf"$\mu_{{\rm test}}={mu:g}$; representative full scans", y=1.075, fontsize=17)
                finish(fig, f"family_scans_mu{_mu_tag(mu)}",
                       f"{largest:,} generated/source. All curves are relative to μ=1; dots mark branch minima. These selected toys are not an ensemble summary.")

        support = bank_study.get("support", pd.DataFrame())
        if not support.empty:
            for mu in sorted(support.mu.unique()):
                frame = support.loc[(support.mu == mu) & (support.generated_per_source == largest)]
                axis_bins = sorted(frame.n_axis.unique())
                fig, axes = plt.subplots(len(replicas), 2 * len(axis_bins),
                                         figsize=(5.3 * 2 * len(axis_bins), 4.8 * len(replicas)), squeeze=False)
                for j, n in enumerate(axis_bins):
                    for offset, (metric, label) in enumerate((("predicted_yield", "Expected events"),
                                                              ("relative_mc_se", "Relative MC standard error"))):
                        column = 2*j + offset
                        values = frame.loc[frame.n_axis == n, metric].to_numpy(dtype=float)
                        positive = values[np.isfinite(values) & (values > 0.)]
                        if len(positive):
                            lower, upper = float(positive.min()), float(positive.max())
                            if lower == upper:
                                lower, upper = lower / 2., upper * 2.
                            norm = LogNorm(vmin=lower, vmax=upper)
                        else:
                            norm = None
                        for row, replica in enumerate(replicas):
                            ax = axes[row, column]
                            part = frame.loc[(frame.replica == replica) & (frame.n_axis == n)]
                            matrix = np.full((int(n), int(n)), np.nan)
                            for cell in part.itertuples():
                                matrix[int(cell.u_bin), int(cell.v_bin)] = float(getattr(cell, metric))
                            masked = np.ma.masked_where(~np.isfinite(matrix) | (matrix <= 0.), matrix)
                            cmap = plt.get_cmap("viridis").copy()
                            cmap.set_bad("#dedede")
                            im = ax.imshow(masked, origin="lower", interpolation="nearest", aspect="equal",
                                           cmap=cmap, norm=norm, extent=(-.5, n-.5, -.5, n-.5))
                            fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label=label)
                            empty = int(_bool(part.empty_mc).sum()) if "empty_mc" in part else 0
                            nonpositive = int(_bool(part.nonpositive_yield).sum()) if "nonpositive_yield" in part else 0
                            ax.set(xlabel=r"$v$ quantile-bin index", ylabel=r"$u$ quantile-bin index",
                                   title=f"{int(n)}×{int(n)}; replica {int(replica) + 1}\nEmpty MC: {empty}; nonpositive yield: {nonpositive}")
                fig.suptitle(rf"$\mu={mu:g}$; 2D template support; {largest:,} generated/source", y=1.035, fontsize=17)
                finish(fig, f"family_support_mu{_mu_tag(mu)}",
                       "Gray cells have zero/nonfinite values. MC errors are propagated at fixed quantile edges; they are not profiled nuisance parameters.")

        for name, fig in figures.items():
            for extension in ("pdf", "png"):
                fig.savefig(output / f"12_{name}.{extension}", dpi=160, bbox_inches="tight")
    return figures
