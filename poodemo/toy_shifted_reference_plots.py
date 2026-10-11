"""Figures for fixed-endpoint shifts in notebook 12.

Only the saved random controls enter ensemble summaries.  Replicas are shown
separately because they contain the same toys and different integration banks.
The discriminator endpoints stay fixed during every physical likelihood fit.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd

_VARIANTS = ("baseline", "numerator", "denominator", "both")
_COLORS = {"analytic": "black", "baseline": "#777777", "numerator": "#0072b2",
           "denominator": "#d55e00", "both": "#009e73"}


def _endpoint_labels(numerator_shift, denominator_shift):
    """Describe the actual saved discriminator settings, including custom shifts."""
    numerator = r"\eta" if numerator_shift == 0. else rf"\eta{numerator_shift:+g}"
    reference = 1. + denominator_shift
    return {
        "analytic": "Analytical unbinned",
        "baseline": r"Baseline $(\eta,1)$",
        "numerator": rf"Numerator $({numerator},1)$",
        "denominator": rf"Denominator $(\eta,{reference:g})$",
        "both": rf"Both $({numerator},{reference:g})$",
    }


def _bool(values):
    return values.astype(str).str.lower().isin(("true", "1", "1.0"))


def _variant(model):
    return str(model).rstrip("0123456789")


def _stat(values, rms=False):
    """Conditional toy SE; RMS uncertainty uses the delta method."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    if rms:
        second = values * values
        estimate = float(np.sqrt(second.mean()))
        error = (float(second.std(ddof=1) / np.sqrt(len(values)) / (2. * estimate))
                 if len(values) > 1 and estimate > 0. else (0. if estimate == 0. else np.nan))
        return estimate, error
    error = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
    return float(values.mean()), error


def _branch_categories(frame):
    """Partition all paired records; ambiguous and failed rows stay visible."""
    valid = _bool(frame.paired_valid)
    comparable = valid & frame.branch_label.isin(("low", "high")) & frame.analytic_branch_label.isin(("low", "high"))
    agreement = frame.branch_label == frame.analytic_branch_label
    return {
        "low": comparable & agreement & (frame.branch_label == "low"),
        "high": comparable & agreement & (frame.branch_label == "high"),
        "disagree": comparable & ~agreement,
        "ambiguous": valid & ~comparable,
        "failed": ~valid,
    }


def plot_shifted_reference_diagnostics(study, output):
    """Save PNG/PDF comparisons and return a mapping of names to figures.

    This follows the plotting API of the earlier notebook-12 sections.  No
    fitting, simulation, endpoint tuning, or change of bin edges is performed.
    ``contrasts`` supplies common-toy changes relative to the unshifted
    observable, separately for each integration bank and number of bins. Endpoint labels
    are derived from the saved metadata, including nondefault shift settings.
    """
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configuration = study["metadata"]["configuration"]
    labels = _endpoint_labels(float(configuration["numerator_shift"]), float(configuration["denominator_shift"]))
    records = study["records"].copy()
    paired = study["paired"].copy()
    if records.empty or paired.empty:
        raise ValueError("Shifted-reference plots require fit records and analytical comparisons")
    random = records.loc[_bool(records.is_random)].copy()
    pairs = paired.loc[_bool(paired.is_random)].copy()
    if random.empty or pairs.empty:
        raise ValueError("Shifted-reference ensemble plots require random-control toys")
    mus = sorted(random.mu_test.unique())
    replicas = sorted(random.replica.unique())
    largest = int(random.generated_per_source.max())
    random = random.loc[random.generated_per_source == largest]
    pairs = pairs.loc[pairs.generated_per_source == largest]
    bins = sorted(int(value) for value in pairs.n_bins.dropna().unique())
    models = [f"{variant}{n_bins}" for variant in _VARIANTS for n_bins in bins
              if f"{variant}{n_bins}" in set(pairs.model)]
    main_bins = 36 if 36 in bins else max(bins)
    main_models = [f"{variant}{main_bins}" for variant in _VARIANTS if f"{variant}{main_bins}" in models]
    contrasts = study.get("contrasts", pd.DataFrame()).copy()
    if not contrasts.empty:
        contrasts = contrasts.loc[(contrasts.generated_per_source == largest) & (contrasts.stratum == "all")]
    figures = {}

    def finish(fig, name, note, top=.92):
        fig.text(.5, .012, note, ha="center", va="bottom", fontsize=10)
        fig.tight_layout(rect=(0., .062, 1., top))
        figures[name] = fig

    def handles(analytic=False):
        variants = (("analytic",) if analytic else ()) + _VARIANTS
        return [Line2D([], [], color=_COLORS[variant], linewidth=2.4 if variant == "denominator" else 1.7,
                       label=labels[variant]) for variant in variants]

    def set_model_axis(ax):
        ax.set_xticks(range(len(models)), [f"{_variant(model).capitalize()}\n{int(pairs.loc[pairs.model == model, 'n_bins'].iloc[0])} bins"
                                          for model in models])
        ax.set_xlim(-.6, len(models)-.4)
        ax.minorticks_off()

    def point(ax, x, y, error, variant, replica_index):
        shift = .19 * (replica_index - (len(replicas)-1)/2.)
        kwargs = {"color": _COLORS[variant], "marker": "o" if replica_index % 2 == 0 else "s",
                  "linestyle": "none", "markersize": 5, "capsize": 3}
        ax.errorbar(x + shift, y, yerr=error if np.isfinite(error) else None, **kwargs)

    bank_handles = [Line2D([], [], color=".25", marker="o" if index % 2 == 0 else "s", linestyle="none",
                           label=f"Bank replica {int(replica)+1}") for index, replica in enumerate(replicas)]
    common_note = (f"Random controls; {largest:,} generated/source. Banks reuse the same toys and are displayed separately.\n"
                   "Error bars: toy-sampling SE conditional on the bank; they do not measure finite-MC uncertainty.")
    style = {"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
             "xtick.labelsize": 9, "ytick.labelsize": 11, "legend.fontsize": 10}
    with plt.style.context(hep.style.ATLAS), plt.rc_context(style):
        # Counts make failed fits explicit in legends, without normalizing each
        # valid subset to unit area.  Zero q belongs to the first histogram bin.
        row_keys = [(mu, replica) for mu in mus for replica in replicas]
        fig, axes = plt.subplots(len(row_keys), len(bins), figsize=(7.2 * len(bins), 4.0 * len(row_keys)), squeeze=False)
        for row, (mu, replica) in enumerate(row_keys):
            full = random.loc[(random.mu_test == mu) & (random.replica == replica)]
            for col, n_bins in enumerate(bins):
                ax = axes[row, col]
                selected_models = ["analytic"] + [f"{variant}{n_bins}" for variant in _VARIANTS]
                frame = full.loc[full.model.isin(selected_models)]
                finite = frame.loc[_bool(frame.valid) & np.isfinite(frame.q)]
                upper = max(1., float(finite.q.max())*1.03) if len(finite) else 1.
                lower = min(0., float(finite.q.min())) if len(finite) else 0.
                edges = np.linspace(lower, upper, 31)
                maximum = 1
                for model in selected_models:
                    part = finite.loc[finite.model == model]
                    denominator = int((frame.model == model).sum())
                    if not denominator:
                        continue
                    counts, _ = np.histogram(part.q, bins=edges)
                    maximum = max(maximum, int(counts.max()))
                    variant = _variant(model)
                    ax.stairs(counts, edges, color=_COLORS[variant],
                              linewidth=2.4 if variant == "denominator" else 1.6,
                              label=f"{labels[variant]}: {len(part)}/{denominator}")
                ax.set(yscale="log", ylim=(.6, maximum*2.), xlim=(lower, upper),
                       xlabel=r"$q_\eta$", ylabel="Random controls / bin",
                       title=rf"$\eta={mu:g}$; {n_bins} bins; bank {int(replica)+1}")
                ax.legend(fontsize=9, loc="upper right", title="Valid / all controls", title_fontsize=9)
        finish(fig, "shifted_reference_q_distributions",
               f"{largest:,} generated/source. Logarithmic count axis; zero-count bins are invisible. Zero test statistics enter the first bin.\n"
               "Each binned statistic uses a free fit over both physical branches; shifts affect only the fixed observable.", top=.985)

        fig, axes = plt.subplots(len(replicas), len(mus), figsize=(7.2*len(mus), 5.0*len(replicas)), squeeze=False)
        for row, replica in enumerate(replicas):
            for col, mu in enumerate(mus):
                ax = axes[row, col]
                frame = pairs.loc[(pairs.replica == replica) & (pairs.mu_test == mu) & (pairs.n_bins == main_bins)]
                good = frame.loc[_bool(frame.paired_valid) & np.isfinite(frame.q) & np.isfinite(frame.analytic_q)]
                values = good[["q", "analytic_q"]].to_numpy().ravel()
                limits = (min(0., float(values.min())) if len(values) else 0., max(1., float(values.max())*1.05) if len(values) else 1.)
                ax.plot(limits, limits, color="black", linewidth=1, linestyle=":")
                counts = []
                for model in main_models:
                    part = good.loc[good.model == model]
                    variant = _variant(model)
                    ax.scatter(part.analytic_q, part.q, color=_COLORS[variant], s=23,
                               marker="^" if variant == "denominator" else "o", edgecolors="none", alpha=.52)
                    counts.append(f"{variant.capitalize()}: {len(part)}/{int((frame.model == model).sum())}")
                ax.text(.03, .97, "Valid pairs / all controls\n"+"\n".join(counts), transform=ax.transAxes,
                        va="top", fontsize=9, bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
                ax.set(xlim=limits, ylim=limits, xlabel=r"Analytical $q_\eta$", ylabel=r"Binned $q_\eta$",
                       title=rf"$\eta={mu:g}$; {main_bins} bins; bank {int(replica)+1}")
        fig.legend(handles=handles(), loc="upper center", bbox_to_anchor=(.5,.995), ncol=2)
        finish(fig, "shifted_reference_paired_q",
               f"Random controls; {largest:,} generated/source. Banks reuse the same toys and are displayed separately.\n"
               "Each point compares the same physical toy. Dotted line: equality. All likelihood denominators are freely fitted.", top=.92)

        # Paired squared-error changes answer improvement more directly than
        # a signed q shift.  Every contrast uses the same baseline toy subset.
        metrics = (("delta_q", False, r"Mean $\Delta q$"), ("delta_q", True, r"RMS $\Delta q$"),
                   ("contrast_q", False, r"Mean $(q_{\rm shift}-q_{\rm baseline})$"),
                   ("contrast_squared", False, r"Mean $[(\Delta q_{\rm shift})^2-(\Delta q_{\rm baseline})^2]$"))
        fig, axes = plt.subplots(len(metrics), len(mus), figsize=(7.2*len(mus), 15.2), squeeze=False)
        for col, mu in enumerate(mus):
            for row, (metric, rms, ylabel) in enumerate(metrics):
                ax = axes[row, col]
                for x, model in enumerate(models):
                    for index, replica in enumerate(replicas):
                        if metric.startswith("contrast"):
                            if _variant(model) == "baseline":
                                continue
                            term = "delta_q" if metric == "contrast_q" else "squared_delta_q"
                            part = contrasts.loc[(contrasts.mu_test == mu) & (contrasts.replica == replica)
                                                 & (contrasts.model == model) & (contrasts.term == term)] if not contrasts.empty else pd.DataFrame()
                            if part.empty:
                                continue
                            estimate, error = float(part.mean_change.iloc[0]), float(part.paired_se.iloc[0])
                        else:
                            part = pairs.loc[(pairs.mu_test == mu) & (pairs.replica == replica)
                                             & (pairs.model == model) & _bool(pairs.paired_valid)]
                            estimate, error = _stat(part[metric], rms=rms)
                        if np.isfinite(estimate):
                            point(ax, x, estimate, error, _variant(model), index)
                ax.axhline(0., color=".65", linewidth=.8)
                set_model_axis(ax)
                ax.set(ylabel=ylabel, title=rf"$\eta={mu:g}$")
                if metric == "contrast_squared":
                    ax.text(.98,.04,"Negative = smaller error",transform=ax.transAxes,ha="right",fontsize=10)
        fig.legend(handles=handles()+bank_handles, loc="upper center", bbox_to_anchor=(.5,.999), ncol=3)
        finish(fig, "shifted_reference_closure", common_note+"\n"
               "Δq = binned − analytical. Bottom two rows use joint-valid, matched baseline toys; signed q shifts alone do not imply improvement.", top=.94)

        terms = (("delta_pair", r"Mean $\Delta q_{\rm pair}$", "Same analytical-MLE anchor"),
                 ("refit_gain", r"Mean $g_{\rm refit}$", "Gain from the free binned fit"))
        fig, axes = plt.subplots(2,len(mus),figsize=(7.2*len(mus),8.8),squeeze=False)
        for col, mu in enumerate(mus):
            for row, (term,ylabel,title) in enumerate(terms):
                ax = axes[row,col]
                for x,model in enumerate(models):
                    for index,replica in enumerate(replicas):
                        part = pairs.loc[(pairs.mu_test==mu)&(pairs.replica==replica)&(pairs.model==model)&_bool(pairs.paired_valid)]
                        estimate,error = _stat(part[term])
                        if np.isfinite(estimate):
                            point(ax,x,estimate,error,_variant(model),index)
                ax.axhline(0.,color=".65",linewidth=.8)
                set_model_axis(ax)
                ax.set(ylabel=ylabel,title=rf"$\eta={mu:g}$; {title}")
        fig.legend(handles=handles()+bank_handles,loc="upper center",bbox_to_anchor=(.5,.999),ncol=3)
        finish(fig,"shifted_reference_decomposition",common_note+"\n"+
               r"$\Delta q=\Delta q_{\rm pair}+g_{\rm refit}$. The analytical MLE is a decomposition anchor, not a discriminator endpoint.",top=.9)

        categories = (("low","Agree: low branch","#0072b2"),("high","Agree: high branch","#009e73"),
                      ("disagree","Different branches","#d55e00"),("ambiguous","Ambiguous in either fit","#cc79a7"),
                      ("failed","Failed pair","#bdbdbd"))
        fig,axes = plt.subplots(len(replicas),len(mus),figsize=(7.4*len(mus),5.0*len(replicas)),squeeze=False)
        for row,replica in enumerate(replicas):
            for col,mu in enumerate(mus):
                ax=axes[row,col]
                frame=pairs.loc[(pairs.replica==replica)&(pairs.mu_test==mu)]
                bottoms=np.zeros(len(models))
                for category,label,color in categories:
                    fractions=[]
                    for model in models:
                        part=frame.loc[frame.model==model]
                        masks=_branch_categories(part)
                        fractions.append(float(masks[category].sum()/len(part)) if len(part) else 0.)
                    bars=ax.bar(np.arange(len(models)),fractions,bottom=bottoms,color=color,edgecolor="white",linewidth=.4)
                    for bar,fraction,bottom in zip(bars,fractions,bottoms):
                        if fraction>=.095:
                            ax.text(bar.get_x()+bar.get_width()/2.,bottom+fraction/2.,f"{fraction:.0%}",ha="center",va="center",fontsize=9,
                                    color="white" if category in ("low","high","disagree") else "black")
                    bottoms+=fractions
                for x,model in enumerate(models):
                    part=frame.loc[frame.model==model]
                    ax.text(x,1.025,f"n={len(part)}",ha="center",fontsize=8)
                set_model_axis(ax)
                ax.set(ylim=(0.,1.12),ylabel="Fraction of all random controls",title=rf"$\eta={mu:g}$; bank {int(replica)+1}")
        fig.legend(handles=[Patch(facecolor=color,label=label) for _,label,color in categories],loc="upper center",bbox_to_anchor=(.5,.995),ncol=3)
        finish(fig,"shifted_reference_branches",
               f"{largest:,} generated/source. Every bar includes all controls: failed pairs and ties/shared endpoints stay in the denominator.\n"
               "Disagreement requires both analytical and binned fits to have an unambiguous branch; all other valid pairs are labelled ambiguous.",top=.91)

        fig,axes=plt.subplots(len(replicas),len(mus),figsize=(7.4*len(mus),4.8*len(replicas)),squeeze=False)
        for row,replica in enumerate(replicas):
            for col,mu in enumerate(mus):
                ax=axes[row,col]
                frame=pairs.loc[(pairs.replica==replica)&(pairs.mu_test==mu)]
                for x,model in enumerate(models):
                    part=frame.loc[frame.model==model]
                    for offset,metric,marker in ((-.12,"occupied_toy_bins","o"),(.12,"occupied_mc_bins","x")):
                        estimate,error=_stat(part[metric])
                        if np.isfinite(estimate):
                            ax.errorbar(x+offset,estimate,yerr=error if np.isfinite(error) else None,
                                        color=_COLORS[_variant(model)],marker=marker,linestyle="none",capsize=3,markersize=6)
                    n_bins=int(part.n_bins.iloc[0]) if len(part) else 0
                    ax.plot([x-.32,x+.32],[n_bins,n_bins],color=".6",linewidth=1,linestyle=":")
                set_model_axis(ax)
                ax.set(ylim=(0.,max(bins)*1.13),ylabel="Mean occupied bins",title=rf"$\eta={mu:g}$; bank {int(replica)+1}")
        occupancy_handles=[Line2D([],[],color=".25",marker="o",linestyle="none",label="Observed toy occupancy"),
                           Line2D([],[],color=".25",marker="x",linestyle="none",label="MC component-template support"),
                           Line2D([],[],color=".6",linestyle=":",label="Total number of bins")]
        fig.legend(handles=occupancy_handles,loc="upper center",bbox_to_anchor=(.5,.995),ncol=3)
        finish(fig,"shifted_reference_occupancy",
               f"Random controls; {largest:,} generated/source. All finite occupancy records are retained, including failed fits.\n"
               "MC support counts bins with a nonzero sum of component templates. Observed-occupancy bars show toy SE; greater occupancy alone is not evidence of better inference.",top=.93)

        scans=study.get("scans",pd.DataFrame())
        selected=study.get("scan_selection",pd.DataFrame())
        if not scans.empty:
            for mu in sorted(scans.mu_test.unique()):
                frame=scans.loc[(scans.mu_test==mu)&(scans.generated_per_source==largest)]
                if not selected.empty and {"mu_test","toy_id"}<=set(selected):
                    toy_ids=list(dict.fromkeys(selected.loc[selected.mu_test==mu,"toy_id"].astype(int)))[:3]
                else:
                    toy_ids=sorted(frame.toy_id.unique())[:3]
                if not toy_ids:
                    continue
                fig,axes=plt.subplots(len(replicas),len(toy_ids),figsize=(6.2*len(toy_ids),5.*len(replicas)),squeeze=False)
                for row,replica in enumerate(replicas):
                    for col,toy_id in enumerate(toy_ids):
                        ax=axes[row,col]
                        part=frame.loc[(frame.replica==replica)&(frame.toy_id==toy_id)]
                        fits=records.loc[(records.mu_test==mu)&(records.toy_id==toy_id)&(records.replica==replica)
                                         &(records.generated_per_source==largest)]
                        failed=[]
                        for model in ["analytic"]+main_models:
                            curve=part.loc[part.model==model].sort_values("mu_scan")
                            variant=_variant(model)
                            ax.plot(curve.mu_scan,curve.D,color=_COLORS[variant],linewidth=2.2 if variant=="denominator" else 1.6)
                            fit=fits.loc[fits.model==model]
                            if not fit.empty and bool(_bool(fit.valid).iloc[0]):
                                first=fit.iloc[0]
                                ax.scatter([first.mu_low,first.mu_high],[first.D_low,first.D_high],color=_COLORS[variant],s=21,zorder=4)
                            else:
                                failed.append(variant)
                        anchor=fits.loc[fits.model=="analytic","mu_hat"]
                        anchor=float(anchor.iloc[0]) if len(anchor) else np.nan
                        ax.axvline(1.,color=".6",linestyle=":",linewidth=.8)
                        if np.isfinite(anchor):
                            ax.axvline(anchor,color=".3",linestyle="-.",linewidth=.8)
                        ax.axhline(0.,color=".6",linewidth=.8)
                        reason=str(part.selection_reason.iloc[0]).replace("_"," ") if len(part) and "selection_reason" in part else "selected example"
                        ax.set(xlabel=r"Physical $\mu$",ylabel=r"$D(\mu)=-2\log[L(\mu)/L(1)]$",
                               title=f"Toy {int(toy_id)}, bank {int(replica)+1}; "+rf"$\hat\mu_{{\rm ana}}={anchor:.4g}$"+"\n"+fill(reason,45))
                        if failed:
                            ax.text(.03,.97,"Failed fits: "+", ".join(failed),transform=ax.transAxes,va="top",fontsize=9,color="#9c3028",
                                    bbox={"facecolor":"white","edgecolor":"none","alpha":.85})
                fig.legend(handles=handles(analytic=True),loc="upper center",bbox_to_anchor=(.5,.96),ncol=3)
                fig.suptitle(rf"$\eta={mu:g}$; {main_bins}-bin fixed-observable scans",y=1.,fontsize=17)
                finish(fig,f"shifted_reference_scans_mu{mu:g}".replace(".","p"),
                       f"{largest:,} generated/source. All curves use physical μ=1 as the likelihood origin; dots mark branch minima. Dash-dot: analytical MLE.\n"
                       "Discriminator endpoints stay fixed throughout each scan. Selected examples are not an ensemble summary.",top=.87)

        for name,fig in figures.items():
            for extension in ("pdf","png"):
                fig.savefig(output/f"12_{name}.{extension}",dpi=160,bbox_inches="tight")
    return figures
