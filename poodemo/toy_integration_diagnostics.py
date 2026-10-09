"""Independent integration checks with a frozen notebook-11 compression.

Fresh S/B/NI proposal strata are sampled before selection. Rejected draws
contribute zeros to fixed-size stratum means and covariance estimates. The
learned fields and observable normalizers are never re-estimated here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PROCESSES
from .inference import component_coefficients, histogram_components
from .pipeline import observable_bin_edges
from .toy_study import reference_observable


def diagnostic_bin_edges(n_bins, base_bins=12):
    """Nested odd subdivisions preserve 0.5 inside its original bin."""
    if any(isinstance(v, bool) or int(v) != v for v in (n_bins, base_bins)):
        raise ValueError("Bin counts must be integers")
    n_bins, base_bins = int(n_bins), int(base_bins)
    if n_bins < base_bins or n_bins % base_bins or (n_bins // base_bins) % 2 != 1:
        raise ValueError("Use the baseline bin count or an odd integer refinement")
    base = observable_bin_edges(base_bins, "reference_ratio")
    subdivisions = n_bins // base_bins
    return np.r_[np.concatenate([np.linspace(lo, hi, subdivisions + 1)[:-1]
                                for lo, hi in zip(base[:-1], base[1:])]), base[-1]]


def _bin_indices(z, edges):
    # np.histogram includes the rightmost endpoint.
    return np.minimum(np.searchsorted(edges, z, side="right") - 1, len(edges) - 2)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _atomic_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False))
    os.replace(temporary, path)


def _atomic_csv(path, frame):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _moments(sums, second, n):
    """Mean/covariance of a fixed-generated-size stratum, including zeros."""
    if n < 2:
        raise ValueError("At least two generated events per stratum are required")
    sums, second = np.asarray(sums), np.asarray(second)
    outer = np.einsum("i...,j...->ij...", sums, sums)
    return sums / n, (second - outer / n) / (n * (n - 1))


def _blank(specs):
    output = dict(rate_sum=np.zeros(8), rate_second=np.zeros((8, 8)),
                  score_sum=np.zeros(len(specs)), score_second=np.zeros(len(specs)),
                  selected=np.array(0), negative_learned=np.zeros(len(specs), dtype=np.int64))
    for k, spec in enumerate(specs):
        b = len(spec["edges"]) - 1
        output[f"bin_sum_{k}"] = np.zeros((4, b))
        output[f"bin_second_{k}"] = np.zeros((4, 4, b))
    return output


def _accumulate(state, fields, learned, inverse_proposal, specs, rates, power, exposure):
    """Accumulate selected draws; generated count is supplied separately."""
    if not fields.shape[1]:
        return
    a = fields * inverse_proposal
    ell = learned * inverse_proposal
    values = np.vstack((a, ell))
    if np.any(~np.isfinite(values)):
        raise ValueError("Nonfinite integration weight; inspect proposal tail coverage")
    state["rate_sum"] += values.sum(axis=1)
    state["rate_second"] += values @ values.T
    state["selected"] += fields.shape[1]
    for k, spec in enumerate(specs):
        z = reference_observable(fields, rates, spec["mu"], power)
        indices = _bin_indices(z, spec["edges"])
        n_bins = len(spec["edges"]) - 1
        for p in range(4):
            state[f"bin_sum_{k}"][p] += np.bincount(indices, weights=a[p], minlength=n_bins)
            for q in range(p, 4):
                moment = np.bincount(indices, weights=a[p] * a[q], minlength=n_bins)
                state[f"bin_second_{k}"][p, q] += moment
                if p != q:
                    state[f"bin_second_{k}"][q, p] += moment
        physical = spec["coefficients"] @ a
        score = exposure * physical * spec["score_weights"][indices]
        state["score_sum"][k] += score.sum()
        state["score_second"][k] += score @ score
        state["negative_learned"][k] += np.count_nonzero(spec["coefficients"] @ learned < 0)


def _combine(states, n, specs):
    result = dict(rates=np.zeros(8), rate_covariance=np.zeros((8, 8)),
                  score_means=np.zeros(len(specs)), score_variances=np.zeros(len(specs)),
                  selected=0, negative_learned=np.zeros(len(specs), dtype=np.int64), bins=[], covariance=[])
    for k, spec in enumerate(specs):
        b = len(spec["edges"]) - 1
        result["bins"].append(np.zeros((4, b)))
        result["covariance"].append(np.zeros((4, 4, b)))
    for state in states:
        mean, covariance = _moments(state["rate_sum"], state["rate_second"], n)
        result["rates"] += mean
        result["rate_covariance"] += covariance
        result["score_means"] += state["score_sum"] / n
        result["score_variances"] += (state["score_second"] - state["score_sum"]**2 / n) / (n * (n - 1))
        result["selected"] += int(state["selected"])
        result["negative_learned"] += state["negative_learned"]
        for k in range(len(specs)):
            mean, covariance = _moments(state[f"bin_sum_{k}"], state[f"bin_second_{k}"], n)
            result["bins"][k] += mean
            result["covariance"][k] += covariance
    return result


def _specifications(context, n_bins):
    config, quad = context["configuration"], context["quad"]
    exposure, power = config["exposure_multiplier"], config["reference_power"]
    specs = []
    for b in n_bins:
        edges = diagnostic_bin_edges(b, config["n_bins"])
        for mu in config["mu_values"]:
            z = reference_observable(quad["nominal"], context["rates"], mu, power)
            baseline = histogram_components(z, quad["nominal"], quad["weights"], edges)
            coefficients = component_coefficients(mu)
            derivative = np.array([2 * np.sqrt(mu) - 1, 1, -1, 0.])
            predicted = exposure * coefficients @ baseline
            differentiated = exposure * derivative @ baseline
            if np.any(predicted < 0):
                raise ValueError("Original direct template has negative physical bin yield")
            score_weights = np.divide(differentiated, predicted, out=np.zeros_like(predicted), where=predicted > 0)
            fisher = float(np.sum(differentiated * score_weights))
            specs.append(dict(n_bins=int(b), mu=float(mu), edges=edges, baseline=baseline,
                              coefficients=coefficients, derivative=derivative,
                              predicted=predicted, differentiated=differentiated,
                              score_weights=score_weights, fisher=fisher))
    return specs


def run_integration_diagnostics(context, *, generated_per_source=200_000,
                                multiples=(1, 2, 5), replicas=2, seed=120524,
                                n_bins=(12, 36), chunk_size=100_000):
    """Integrate fresh fixed-size proposal strata and compare frozen predictions.

    Prefix sizes within a replica are correlated; different replicas and the
    original bank are independent. Checkpoints store sufficient statistics,
    including cross-component covariance, rather than huge event arrays.
    Returned templates use the largest prefix of replica 0 and retain the
    notebook-11 observable, original rate normalizers and preselection.
    """
    for value, name, minimum in ((generated_per_source, "generated_per_source", 2),
                                 (replicas, "replicas", 1), (chunk_size, "chunk_size", 1)):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if any(int(value) != value or value < 1 for value in multiples) or not multiples:
        raise ValueError("multiples must contain positive integers")
    if len(set(n_bins)) != len(n_bins) or not n_bins:
        raise ValueError("n_bins must be nonempty and unique")
    sizes = sorted(set(int(generated_per_source * value) for value in multiples))
    config = context["configuration"]
    specs = _specifications(context, n_bins)
    directory = Path(context["output_dir"]) / "integration"
    directory.mkdir(parents=True, exist_ok=True)
    metadata = dict(version=1, source_fingerprint=context["fingerprint"], seed=int(seed),
                    generated_per_source=int(generated_per_source), sizes=sizes,
                    replicas=int(replicas), n_bins=list(map(int, n_bins)), chunk_size=int(chunk_size),
                    mu_values=list(map(float, config["mu_values"])),
                    exposure_multiplier=float(config["exposure_multiplier"]),
                    reference_power=float(config["reference_power"]),
                    frozen_observable_rates=np.asarray(context["rates"]).tolist(),
                    original_n_per_source=context.get("original_n_per_source"),
                    note="Independent fixed-generated-size S/B/NI strata; rejected events are zeros. Prefixes are correlated. All original normalizers are frozen.",
                    score_interpretation="The local score diagnoses boundary tendency; distant competing minima can also determine q=0 and require the paired branch tests.",
                    score_uncertainty="Conditional on original bin predictions and score weights; includes cross-bin stratum covariance. It is not an uncertainty on an independently re-fitted model.",
                    template_choice="Largest prefix, replica 0; unscaled component yields (fit multiplies exposure).")
    metadata["fingerprint"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    manifest = directory / "configuration.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("Integration diagnostics changed; choose another diagnostic output tag")
    _atomic_json(manifest, metadata)
    snapshots = {}
    model, selector, threshold = context["run"].model, context["selector"], context["threshold"]
    for replica in range(int(replicas)):
        for source_id, source in enumerate(("S", "B", "NI")):
            state, completed = _blank(specs), 0
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), 120524, replica, source_id]))
            for size in sizes:
                checkpoint = directory / f"replica_{replica}_{source}_{size}.npz"
                if checkpoint.exists():
                    with np.load(checkpoint, allow_pickle=False) as saved:
                        if str(saved["fingerprint"]) != metadata["fingerprint"]:
                            raise ValueError("Integration checkpoint has a different fingerprint")
                        state = {key: saved[key].copy() for key in state}
                        rng.bit_generator.state = json.loads(str(saved["rng_state"]))
                    completed = size
                else:
                    while completed < size:
                        count = min(int(chunk_size), size - completed)
                        x = model.sample_component(source, count, rng)
                        if selector is not None:
                            x = x[selector.predict_proba(x)[:, 0] >= threshold]
                        if len(x):
                            proposal = sum(model.component_pdf(x, p) for p in ("S", "B", "NI"))
                            if np.any(proposal <= 0) or np.any(~np.isfinite(proposal)):
                                raise ValueError("Invalid mixture proposal density")
                            fields = model.component_densities(x).T
                            learned = np.asarray(context["workspace"].evaluate(x)[0])
                            _accumulate(state, fields, learned, 1. / proposal, specs,
                                        context["rates"], config["reference_power"], config["exposure_multiplier"])
                        completed += count
                    _atomic_npz(checkpoint, **state, fingerprint=np.array(metadata["fingerprint"]),
                                rng_state=np.array(json.dumps(rng.bit_generator.state)))
                    print(f"Integration replica {replica + 1}/{replicas}, {source}: {size:,} generated, {int(state['selected']):,} selected", flush=True)
                snapshots[replica, source_id, size] = {key: value.copy() for key, value in state.items()}
    baseline_stats = None
    original_n = context.get("original_n_per_source")
    quad = context["quad"]
    if original_n is not None and "stratum" in quad:
        original_n = int(original_n)
        baseline_states = []
        for source_id in range(3):
            mask = np.asarray(quad["stratum"]) == source_id
            if np.sum(mask) > original_n:
                raise ValueError("Original generated count is smaller than a selected stratum")
            state = _blank(specs)
            # Original weights are 1/(n * sum_p p_p). Avoid evaluating NNs
            # again: baseline analytic covariance does not need learned rates.
            fields = np.asarray(quad["nominal"])[:, mask]
            _accumulate(state, fields, fields, original_n * np.asarray(quad["weights"])[mask], specs,
                        context["rates"], config["reference_power"], config["exposure_multiplier"])
            baseline_states.append(state)
        baseline_stats = _combine(baseline_states, original_n, specs)
    rows = {name: [] for name in ("bin_closure", "rate_closure", "convergence", "score_closure", "component_yields")}
    templates, covariances = {}, {}
    chosen_rates = None
    exposure = config["exposure_multiplier"]
    original_rates = np.r_[context["rates"], context["learned_rates"]]
    for replica in range(int(replicas)):
        for size in sizes:
            stats = _combine([snapshots[replica, s, size] for s in range(3)], size, specs)
            if replica == 0 and size == sizes[-1]:
                chosen_rates = stats["rates"].copy()
            # Save full rate covariance and 4x4 per-bin component covariance.
            covariance_arrays = dict(rate_covariance=stats["rate_covariance"], rates=stats["rates"])
            for k, spec in enumerate(specs):
                covariance_arrays[f"component_covariance_{k}"] = stats["covariance"][k]
                covariance_arrays[f"component_yields_{k}"] = stats["bins"][k]
                covariance_arrays[f"bin_edges_{k}"] = spec["edges"]
            _atomic_npz(directory / f"moments_replica_{replica}_{size}.npz", **covariance_arrays)
            for kind, offset in (("analytic", 0), ("learned", 4)):
                for mu in config["mu_values"]:
                    coefficients = component_coefficients(mu)
                    rate = exposure * float(coefficients @ stats["rates"][offset:offset + 4])
                    baseline = exposure * float(coefficients @ original_rates[offset:offset + 4])
                    covariance = stats["rate_covariance"][offset:offset + 4, offset:offset + 4]
                    se = exposure * np.sqrt(max(0., coefficients @ covariance @ coefficients))
                    baseline_se = (exposure * np.sqrt(max(0., coefficients @ baseline_stats["rate_covariance"][:4, :4] @ coefficients))
                                   if kind == "analytic" and baseline_stats is not None else np.nan)
                    row = dict(replica=replica, generated_per_source=size, selected=stats["selected"],
                               model=kind, mu_test=mu, estimate=rate, standard_error=se, baseline=baseline,
                               baseline_se_fixed_partition=baseline_se,
                               relative=rate / baseline - 1 if baseline else np.nan,
                               ratio=rate / baseline if baseline else np.nan,
                               ratio_se=se / baseline if baseline else np.nan)
                    rows["convergence"].append(row)
                    if size == sizes[-1]:
                        rows["rate_closure"].append(row)
            for k, spec in enumerate(specs):
                components, covariance = stats["bins"][k], stats["covariance"][k]
                coefficients = spec["coefficients"]
                physical = exposure * coefficients @ components
                variance = exposure**2 * np.einsum("i,ijb,j->b", coefficients, covariance, coefficients)
                validation_se = np.sqrt(np.maximum(variance, 0.))
                baseline_se = (exposure * np.sqrt(np.maximum(0., np.einsum("i,ijb,j->b", coefficients, baseline_stats["covariance"][k], coefficients)))
                               if baseline_stats is not None else np.full(len(physical), np.nan))
                prefix = dict(replica=replica, generated_per_source=size, n_bins=spec["n_bins"], mu_test=spec["mu"])
                unsupported = float(physical[spec["predicted"] == 0].sum())
                score = stats["score_means"][k] - spec["differentiated"].sum()
                score_se = np.sqrt(max(0., stats["score_variances"][k]))
                fisher = spec["fisher"]
                rows["score_closure"].append(dict(**prefix, expected_score=score, standard_error=score_se,
                    model_fisher=fisher, standardized_score=score / np.sqrt(fisher) if fisher > 0 and unsupported == 0 else np.nan,
                    standardized_se=score_se / np.sqrt(fisher) if fisher > 0 and unsupported == 0 else np.nan,
                    unsupported_expected_yield=unsupported, negative_learned_nodes=int(stats["negative_learned"][k])))
                if size == sizes[-1]:
                    for b, (observed, predicted, se, old_se) in enumerate(zip(physical, spec["predicted"], validation_se, baseline_se)):
                        # The old-observable partition depends on its old bank.
                        # old_se is the plug-in, fixed-partition diagnostic only.
                        combined = np.sqrt(se**2 + old_se**2)
                        rows["bin_closure"].append(dict(**prefix, bin=b, left=spec["edges"][b], right=spec["edges"][b+1],
                            estimate=observed, baseline=predicted, delta=observed-predicted,
                            validation_se=se, baseline_se_fixed_partition=old_se, combined_se_fixed_partition=combined,
                            relative=observed/predicted-1 if predicted > 0 else np.nan,
                            ratio=observed/predicted if predicted > 0 else np.nan,
                            ratio_se=se/predicted if predicted > 0 else np.nan,
                            residual_validation_only=(observed-predicted)/se if se > 0 else np.nan,
                            residual_combined_fixed_partition=(observed-predicted)/combined if combined > 0 else np.nan))
                    for p, name in enumerate(PROCESSES):
                        for b in range(spec["n_bins"]):
                            rows["component_yields"].append(dict(**prefix, process=name, bin=b,
                                expected_yield=components[p, b], standard_error=np.sqrt(max(0., covariance[p, p, b]))))
                    if replica == 0:
                        templates[spec["n_bins"], spec["mu"]] = components
                        covariances[spec["n_bins"], spec["mu"]] = covariance
    result = {name: pd.DataFrame(values) for name, values in rows.items()}
    for name, frame in result.items():
        _atomic_csv(directory / (name + ".csv"), frame)
    result.update(templates=templates, component_covariances=covariances,
                  rates=chosen_rates[:4], learned_rates=chosen_rates[4:],
                  edges={int(b): diagnostic_bin_edges(b, config["n_bins"]) for b in n_bins},
                  metadata=metadata)
    return result


def plot_integration_diagnostics(study, output=None):
    """Yield ratios, conditional residuals, rate convergence, and kappa score."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    from .plotting import _save
    figures = {}
    with plt.style.context(hep.style.ATLAS):
        table = study["bin_closure"]
        for (b, mu), group in table.groupby(["n_bins", "mu_test"]):
            fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1, 1]}, constrained_layout=True)
            first = group[group.replica == group.replica.min()]
            edges = np.r_[first.left.to_numpy(), first.right.iloc[-1]]
            hep.histplot(first.baseline.to_numpy(), bins=edges, ax=axes[0], label="Original prediction", color="black")
            for replica, points in group.groupby("replica"):
                center = .5 * (points.left + points.right)
                label = f"Independent bank {replica + 1}"
                axes[0].errorbar(center, points.estimate, points.validation_se, fmt=".", label=label)
                axes[1].errorbar(center, points.ratio, points.ratio_se, fmt=".")
                axes[2].plot(center, points.residual_validation_only, ".")
            axes[0].set_yscale("log")
            axes[0].set_ylabel("Expected events")
            axes[0].set_title(rf"Frozen observable: $\eta={mu:g}$, {b} bins")
            axes[0].legend(fontsize="small")
            axes[1].axhline(1., color="gray", linestyle="--")
            axes[1].set_ylabel("New / original")
            axes[2].axhline(0., color="gray", linestyle="--")
            axes[2].set_ylabel(r"$\Delta/\sigma_{\rm new}$")
            axes[2].set_xlabel("Frozen reference-ratio observable")
            _save(fig, output, f"integration_bins_{b}_mu_{mu:g}")
            figures[f"bins_{b}_mu_{mu:g}"] = fig
        convergence = study["convergence"]
        mus = sorted(convergence.mu_test.unique())
        fig, axes = plt.subplots(1, len(mus), figsize=(7 * len(mus), 5), squeeze=False, constrained_layout=True)
        for ax, mu in zip(axes[0], mus):
            for (kind, replica), group in convergence[convergence.mu_test == mu].groupby(["model", "replica"]):
                ax.errorbar(group.generated_per_source, group.ratio - 1, group.ratio_se, marker="o", label=f"{kind}, bank {replica + 1}")
            ax.axhline(0., color="gray", linestyle="--")
            ax.set_xscale("log")
            ax.set_xlabel("Generated events per S/B/NI stratum")
            ax.set_ylabel("Independent rate / fitted rate − 1")
            ax.set_title(rf"$\mu={mu:g}$; frozen normalizers")
            ax.legend(fontsize="small")
        _save(fig, output, "integration_rate_convergence")
        figures["rate_convergence"] = fig
        fig, axes = plt.subplots(1, len(mus), figsize=(7 * len(mus), 5), squeeze=False, constrained_layout=True)
        score = study["score_closure"]
        for ax, mu in zip(axes[0], mus):
            for (b, replica), group in score[score.mu_test == mu].groupby(["n_bins", "replica"]):
                ax.errorbar(group.generated_per_source, group.standardized_score, group.standardized_se,
                            marker="o", label=f"{b} bins, bank {replica + 1}")
            ax.axhline(0., color="gray", linestyle="--")
            ax.set_xscale("log")
            ax.set_xlabel("Generated events per S/B/NI stratum")
            ax.set_ylabel(r"$E_{\rm true}[U_\kappa]/\sqrt{I_{\kappa\kappa}}$")
            ax.set_title(rf"$\mu={mu:g}$; original model fixed")
            ax.legend(fontsize="small")
        _save(fig, output, "integration_expected_kappa_score")
        figures["expected_kappa_score"] = fig
    return figures
