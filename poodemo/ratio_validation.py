"""Independent density-ratio reweighting and balanced-class calibration checks.

The fitted ratios are never renormalized on the diagnostic sample.  A fixed
number of iid events is used for each class, so errors use fixed-sample moments
rather than a Poisson event-count convention.  No experiment label is added.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


RATIO_TASKS = (("SBI", "S"), ("B", "S"), ("NI", "S"),
               ("NI_up", "NI"), ("NI_down", "NI"))
STAGE_LABELS = {"raw": "Raw network", "normalized": "Mean-normalized"}
STAGE_COLORS = {"raw": "#D55E00", "normalized": "#0072B2"}


def _edges(edges, *, finite=True):
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or np.any(np.isnan(edges))
            or np.any(np.diff(edges) <= 0)
            or (finite and np.any(~np.isfinite(edges)))):
        raise ValueError("Bin edges must be a strictly increasing finite 1D array")
    return edges


def _sample(x):
    x = np.asarray(x)
    if x.ndim != 2 or len(x) < 2 or np.any(~np.isfinite(x)):
        raise ValueError("Samples must be finite 2D arrays with at least two events")
    return x


def _mean_variance(total, squared_total, size):
    """Variance of a sample mean, allowing zero observations outside a bin."""
    return np.maximum(squared_total - total * total / size, 0.) / (size * (size - 1))


def reweighting_table(numerator, denominator, denominator_ratios, edges, *, min_count=5):
    """Compare target density with independently reweighted reference density.

    ``denominator_ratios`` are p_numerator / p_denominator at reference events.
    Histograms divide by the FULL sample size and bin width, retaining both
    tail mass and any global normalization error.  Errors on reweighted/target
    include independent numerator and denominator fixed-sample fluctuations.
    Bins with fewer than ``min_count`` events in either class are flagged and
    omitted from the ratio panel, never displayed as successful zero closure.
    """
    numerator, denominator = _sample(numerator), _sample(denominator)
    if numerator.shape[1] != denominator.shape[1]:
        raise ValueError("Target and reference samples must have the same dimension")
    edges = _edges(edges)
    weights = np.asarray(denominator_ratios, dtype=float)
    if (weights.shape != (len(denominator),) or np.any(~np.isfinite(weights))
            or np.any(weights < 0)):
        raise ValueError("Ratios must be finite nonnegative weights, one per reference event")
    if np.any(weights > np.sqrt(np.finfo(float).max / len(weights))):
        raise ValueError("Ratios are too large for finite histogram second moments")
    if min_count < 2:
        raise ValueError("min_count must be at least two for an uncertainty estimate")
    width = np.diff(edges)
    rows = []
    for coordinate in range(numerator.shape[1]):
        hn = np.histogram(numerator[:, coordinate], edges)[0]
        hd = np.histogram(denominator[:, coordinate], edges)[0]
        hw = np.histogram(denominator[:, coordinate], edges, weights=weights)[0]
        hw2 = np.histogram(denominator[:, coordinate], edges, weights=weights * weights)[0]
        # Weighted histogram accumulation can leave roundoff in empty bins.
        hw[hd == 0] = 0.
        hw2[hd == 0] = 0.
        pn, pd_, pw = hn / len(numerator), hd / len(denominator), hw / len(denominator)
        vn = _mean_variance(hn, hn, len(numerator))
        vd = _mean_variance(hd, hd, len(denominator))
        vw = _mean_variance(hw, hw2, len(denominator))
        valid = (hn >= min_count) & (hd >= min_count) & (hw2 > 0)
        ratio = np.divide(pw, pn, out=np.full(len(pn), np.nan), where=valid)
        ratio_var = np.full(len(pn), np.nan)
        ratio_var[valid] = (vw[valid] / pn[valid] ** 2
                            + pw[valid] ** 2 * vn[valid] / pn[valid] ** 4)
        ess = np.divide(hw * hw, hw2, out=np.zeros(len(hw)), where=hw2 > 0)
        rows.append(pd.DataFrame({
            "coordinate": coordinate, "bin_low": edges[:-1], "bin_high": edges[1:],
            "numerator_count": hn, "denominator_count": hd,
            "numerator_density": pn / width, "denominator_density": pd_ / width,
            "reweighted_density": pw / width,
            "numerator_se": np.sqrt(vn) / width,
            "denominator_se": np.sqrt(vd) / width,
            "reweighted_se": np.sqrt(vw) / width,
            "reweighted_over_target": ratio, "ratio_se": np.sqrt(ratio_var),
            "reweighted_ess": ess, "ratio_valid": valid,
            "status": np.where(valid, "available", "unavailable: low class count or zero weights"),
            "numerator_events": len(numerator), "denominator_events": len(denominator),
            "numerator_plot_mass": pn.sum(), "denominator_plot_mass": pd_.sum(),
            "reweighted_plot_mass": pw.sum(), "denominator_mean_ratio": weights.mean(),
        }))
    return pd.concat(rows, ignore_index=True)


def calibration_table(numerator_scores, denominator_scores, edges=None, *, min_count=5):
    """Reliability and residual tables at an equal numerator/reference prior.

    Classes are weighted by inverse total class size, so unequal selected
    sample sizes do not alter the intended 1:1 prior.  The residual is
    ``empirical numerator fraction - mean predicted score``.  Its uncertainty
    includes the covariance of both quantities, using fixed-sample influence
    moments separately for the two independently simulated classes.  Empty or
    sparsely populated class bins have unavailable estimates, not zero error.
    """
    edges = _edges(np.linspace(0., 1., 21) if edges is None else edges)
    if edges[0] != 0. or edges[-1] != 1.:
        raise ValueError("Calibration bins must cover the whole [0, 1] score range")
    scores = [np.asarray(s, dtype=float) for s in (numerator_scores, denominator_scores)]
    for s in scores:
        if s.ndim != 1 or len(s) < 2 or np.any(~np.isfinite(s)) or np.any((s < 0) | (s > 1)):
            raise ValueError("Scores must be finite 1D arrays in [0, 1], with at least two events")
    if min_count < 2:
        raise ValueError("min_count must be at least two for an uncertainty estimate")
    n_num, n_den = map(len, scores)
    hn, hd = [np.histogram(s, edges)[0] for s in scores]
    sn, sd = [np.histogram(s, edges, weights=s)[0] for s in scores]
    s2n, s2d = [np.histogram(s, edges, weights=s * s)[0] for s in scores]
    a, b = hn / n_num, hd / n_den
    total = a + b
    occupied = total > 0
    empirical = np.divide(a, total, out=np.full(len(a), np.nan), where=occupied)
    predicted = np.divide(sn / n_num + sd / n_den, total,
                          out=np.full(len(a), np.nan), where=occupied)
    residual = empirical - predicted
    va, vb = _mean_variance(hn, hn, n_num), _mean_variance(hd, hd, n_den)
    empirical_var = np.divide(b * b * va + a * a * vb, total ** 4,
                             out=np.full(len(a), np.nan), where=occupied)
    # Influence numerator: I_bin (label - score - residual); its class-wise
    # variance is divided by the squared total balanced bin probability.
    zn = (1 - residual) * hn - sn
    zd = -residual * hd - sd
    z2n = (1 - residual) ** 2 * hn - 2 * (1 - residual) * sn + s2n
    z2d = residual ** 2 * hd + 2 * residual * sd + s2d
    residual_var = np.divide(_mean_variance(zn, z2n, n_num) + _mean_variance(zd, z2d, n_den),
                            total * total, out=np.full(len(a), np.nan), where=occupied)
    valid = (hn >= min_count) & (hd >= min_count)
    # Retain the raw point estimates for inspection in the CSV, but never
    # attach a misleading Gaussian uncertainty to an unpopulated class.
    empirical_var[~valid] = np.nan
    residual_var[~valid] = np.nan
    return pd.DataFrame({
        "bin_low": edges[:-1], "bin_high": edges[1:],
        "numerator_count": hn, "denominator_count": hd,
        "balanced_bin_probability": total / 2,
        "predicted": predicted, "empirical": empirical, "residual": residual,
        "empirical_se": np.sqrt(empirical_var), "residual_se": np.sqrt(residual_var),
        "calibration_valid": valid,
        "status": np.where(valid, "available", "unavailable: low class count"),
        "numerator_events": n_num, "denominator_events": n_den,
    })


def _save_figure(figure, output):
    if output is None:
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        figure.savefig(output.with_suffix(suffix), dpi=160, bbox_inches="tight")


def plot_reweighting(table, numerator, denominator, *, output=None, log=False, label=None):
    """Three marginal reweighting checks with independent-MC ratio errors."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    coordinates = sorted(table.coordinate.unique())
    with plt.style.context(hep.style.ATLAS):
        fig, axes = plt.subplots(2, len(coordinates), figsize=(6.1 * len(coordinates), 6.4),
                                 sharex="col", squeeze=False,
                                 gridspec_kw={"height_ratios": [3, 1], "hspace": .06})
        for column, coordinate in enumerate(coordinates):
            ax, ratio_ax = axes[:, column]
            field = table.loc[table.coordinate == coordinate]
            reference = field.loc[field.stage == field.stage.iloc[0]]
            edges = np.r_[reference.bin_low.to_numpy(), reference.bin_high.iloc[-1]]
            centers = .5 * (edges[:-1] + edges[1:])
            keep = reference.numerator_count.to_numpy() > 0
            ax.errorbar(centers[keep], reference.numerator_density.to_numpy()[keep],
                        yerr=reference.numerator_se.to_numpy()[keep], fmt="o", markersize=3,
                        color="black", label=f"Target {numerator}", zorder=4)
            hep.histplot(reference.denominator_density.to_numpy(), edges, ax=ax, color="0.65",
                         linestyle=":", linewidth=1.5, label=f"Reference {denominator}")
            for stage, part in field.groupby("stage", sort=False):
                color = STAGE_COLORS[stage]
                hep.histplot(part.reweighted_density.to_numpy(), edges, ax=ax, color=color,
                             linewidth=1.7, label=STAGE_LABELS[stage])
                lo = part.reweighted_density.to_numpy() - part.reweighted_se.to_numpy()
                hi = part.reweighted_density.to_numpy() + part.reweighted_se.to_numpy()
                ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]], step="post",
                                color=color, alpha=.12)
                available = part.ratio_valid.to_numpy()
                ratio_ax.errorbar(centers[available], part.reweighted_over_target.to_numpy()[available],
                                  yerr=part.ratio_se.to_numpy()[available], fmt="o", markersize=2.5,
                                  color=color, linewidth=1.)
            ax.set_ylabel("Selected probability density")
            if log:
                ax.set_yscale("log")
                central = field[["numerator_density", "denominator_density",
                                 "reweighted_density"]].to_numpy().ravel()
                positive = central[central > 0]
                if len(positive):
                    # Error bands that touch zero must not expand a logarithmic
                    # axis down to floating-point roundoff.
                    ax.set_ylim(positive.min() * .25, positive.max() * 3.)
            else:
                ax.set_ylim(bottom=0)
            ax.legend(fontsize=10, frameon=False)
            ratio_ax.axhline(1., color="black", linewidth=1.)
            ratio_ax.set_xlabel(fr"$x_{coordinate + 1}$")
            ratio_ax.set_ylabel("Reweighted\n/ target", fontsize=12)
            ratio_ax.tick_params(labelsize=11)
            ratio_ax.grid(axis="y", alpha=.2)
        heading = f"{numerator}/{denominator}" + (f" ({label})" if label else "")
        fig.suptitle(f"{heading}: independent selected-sample reweighting", fontsize=16)
        fig.text(.5, .005, "Fixed sample sizes; no diagnostic-sample renormalization. "
                 "Ratio errors include both samples; sparse bins are omitted.", ha="center", fontsize=10)
        fig.subplots_adjust(top=.88, bottom=.13, wspace=.3)
        _save_figure(fig, output)
    return fig


def plot_calibration(table, numerator, denominator, *, output=None, label=None):
    """Balanced-class reliability and correlated residual uncertainty panels."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    with plt.style.context(hep.style.ATLAS):
        fig, (ax, residual_ax) = plt.subplots(2, 1, figsize=(7.6, 7.2), sharex=True,
                    gridspec_kw={"height_ratios": [3, 1], "hspace": .06})
        ax.plot([0, 1], [0, 1], color="black", linewidth=1., linestyle="--")
        for stage, part in table.groupby("stage", sort=False):
            good = part.calibration_valid.to_numpy()
            color = STAGE_COLORS[stage]
            ax.errorbar(part.predicted.to_numpy()[good], part.empirical.to_numpy()[good],
                        yerr=part.empirical_se.to_numpy()[good], fmt="o-", markersize=4,
                        color=color, linewidth=1., label=STAGE_LABELS[stage])
            residual_ax.errorbar(part.predicted.to_numpy()[good], part.residual.to_numpy()[good],
                        yerr=part.residual_se.to_numpy()[good], fmt="o", markersize=3,
                        color=color, linewidth=1.)
        ax.set(ylabel="Empirical numerator fraction", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=12, frameon=False)
        residual_ax.axhline(0., color="black", linewidth=1.)
        residual_ax.set(xlabel=r"Mean predicted $r/(1+r)$", ylabel="Empirical\n− predicted")
        residual_ax.grid(axis="y", alpha=.2)
        heading = f"{numerator}/{denominator}" + (f" ({label})" if label else "")
        fig.suptitle(f"{heading}: balanced-class calibration", fontsize=16)
        fig.text(.5, .018, "Independent selected samples; equal class prior.\n"
                 "Sparse bins omitted; residual errors include score–fraction covariance.",
                 ha="center", fontsize=10)
        fig.subplots_adjust(top=.89, bottom=.17, left=.17, right=.96)
        _save_figure(fig, output)
    return fig


def plot_ratio_task(numerator, denominator, samples, predictor, output, *, edges=None,
                    score_bins=20, max_events=None, min_count=5, log=False,
                    results=None, prefix="03", label=None, stages=("raw", "normalized")):
    """Save one completed classifier's independent validation figures/tables.

    This synchronous helper can run after each ensemble member, before training
    the next one. Use separate ``output`` / ``results`` member directories (or a
    different ``prefix``) to retain every result; ``label`` identifies the member
    or ensemble in figure titles. The selected diagnostic bank is never used to
    fit a calibration or normalization. Set ``stages=("raw",)`` for individual
    members before the ensemble normalization. Return tables and figures for immediate
    notebook display, with the same keys as :func:`plot_ratio_validation`.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = output if results is None else Path(results)
    results.mkdir(parents=True, exist_ok=True)
    edges = _edges(np.linspace(-5., 6., 56) if edges is None else edges)
    if score_bins < 2:
        raise ValueError("score_bins must be at least two")
    if max_events is not None and max_events < 2:
        raise ValueError("max_events must be at least two or None")
    key = f"{numerator}_over_{denominator}"
    num = samples[numerator] if max_events is None else samples[numerator][:max_events]
    den = samples[denominator] if max_events is None else samples[denominator][:max_events]
    stages = tuple(stages)
    if not stages or len(set(stages)) != len(stages) or any(stage not in STAGE_LABELS for stage in stages):
        raise ValueError("stages must be a nonempty sequence of unique raw/normalized entries")
    reweighting, calibration = [], []
    for stage in stages:
        method = predictor.predict_raw_log_ratio if stage == "raw" else predictor.predict_log_ratio
        log_den = np.asarray(method(den), dtype=float)
        log_num = np.asarray(method(num), dtype=float)
        if (log_den.shape != (len(den),) or log_num.shape != (len(num),)
                or np.any(~np.isfinite(log_den)) or np.any(~np.isfinite(log_num))
                or np.any(np.abs(log_den) > 350)):
            raise ValueError(f"Nonfinite or extreme {stage} predictions for {key}; inspect the network")
        rw = reweighting_table(num, den, np.exp(log_den), edges, min_count=min_count)
        cal = calibration_table(expit(log_num), expit(log_den),
                                np.linspace(0., 1., score_bins + 1), min_count=min_count)
        for table in (rw, cal):
            table.insert(0, "stage", stage)
            table.insert(0, "task", key)
        reweighting.append(rw)
        calibration.append(cal)
    rw, cal = pd.concat(reweighting, ignore_index=True), pd.concat(calibration, ignore_index=True)
    rw.to_csv(results / f"{prefix}_reweighting_{key}.csv", index=False)
    cal.to_csv(results / f"{prefix}_calibration_{key}.csv", index=False)
    figures = {
        f"reweighting_{key}": plot_reweighting(
            rw, numerator, denominator, output=output / f"{prefix}_reweighting_{key}",
            log=log, label=label),
        f"calibration_{key}": plot_calibration(
            cal, numerator, denominator, output=output / f"{prefix}_calibration_{key}",
            label=label),
    }
    suffix = f" ({label})" if label else ""
    print(f"Independent reweighting and calibration plots ready: {numerator}/{denominator}{suffix}",
          flush=True)
    return {"reweighting": rw, "calibration": cal, "figures": figures}


def plot_ratio_validation(samples, predictors, output, *, edges=None, score_bins=20,
                          max_events=None, min_count=5, log=False, results=None):
    """Save all five ratio tasks' reweighting and calibration figures/tables.

    ``samples`` maps S/SBI/B/NI/NI_up/NI_down to an independent, selected
    simulation bank. ``predictors`` uses keys such as ``SBI_over_S``. CSVs go
    to ``results`` (or ``output`` when omitted), figures to ``output``. Return
    combined tables and a figure mapping for notebook display. Existing mean
    normalization is read from predictors; no calibration is fitted here.
    """
    rw_tables, cal_tables, figures = [], [], {}
    for numerator, denominator in RATIO_TASKS:
        key = f"{numerator}_over_{denominator}"
        report = plot_ratio_task(
            numerator, denominator, samples, predictors[key], output, edges=edges,
            score_bins=score_bins, max_events=max_events, min_count=min_count,
            log=log, results=results)
        rw_tables.append(report["reweighting"])
        cal_tables.append(report["calibration"])
        figures.update(report["figures"])
    return {"reweighting": pd.concat(rw_tables, ignore_index=True),
            "calibration": pd.concat(cal_tables, ignore_index=True), "figures": figures}
