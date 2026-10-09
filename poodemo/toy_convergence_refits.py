"""Paired finite-integration checks on the saved notebook-12 toy selection.

Every bank describes the same frozen observable. Event densities, event samples
and fit settings are shared; only the selected rate integrals and bin yields
change. Reported standard errors describe toy sampling conditional on the banks.
They are not estimates of the uncertainty from finite integration samples.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import save_json
from .inference import histogram_components
from .toy_likelihood import StatOnlyLikelihood
from .toy_refit_diagnostics import _bool
from .toy_study import _array_digest, _atomic_csv, _failure, _fit, _rng, reference_observable


def load_convergence_selection(context):
    """Load exactly the original diagnostic selection, verifying its provenance."""
    output = Path(context["output_dir"])
    path, config_path = output / "refit_selection.csv", output / "refit_config.json"
    if not path.exists() or not config_path.exists():
        raise FileNotFoundError("The original notebook-12 refit selection is missing; run its controlled-refit stage first")
    config = json.loads(config_path.read_text())["configuration"]
    if config.get("context_fingerprint") != context["fingerprint"]:
        raise ValueError("The saved selection belongs to a different notebook-11 context")
    if hashlib.sha256(path.read_bytes()).hexdigest() != config.get("selection_sha256"):
        raise ValueError("The saved refit selection does not match its original fingerprint")
    selected = pd.read_csv(path, float_precision="round_trip")
    return _selection(selected, context)


def _selection(selection, context):
    required = {"mu_test", "toy_id", "is_random", "is_flagged", "branch_split"}
    if not required <= set(selection) or selection.empty:
        raise ValueError("Use the nonempty saved notebook-12 selection with its random/flagged labels")
    selected = selection.sort_values(["mu_test", "toy_id"]).reset_index(drop=True).copy()
    if selected.duplicated(["mu_test", "toy_id"]).any():
        raise ValueError("Duplicate toy IDs in the saved selection")
    for key in ("is_random", "is_flagged"):
        selected[key] = _bool(selected[key])
    if "branch_disagreement" in selected:
        selected["branch_disagreement"] = _bool(selected.branch_disagreement)
    if (not np.isfinite(selected[["mu_test", "toy_id", "branch_split"]].to_numpy()).all()
            or (selected.toy_id < 0).any() or (selected.toy_id != selected.toy_id.astype(int)).any()
            or (selected.branch_split <= 0).any()):
        raise ValueError("Invalid hypothesis, toy ID or branch split in saved selection")
    if any(group.branch_split.nunique() != 1 for _, group in selected.groupby("mu_test")):
        raise ValueError("Each tested hypothesis must use one documented branch split")
    source = context["source"]["results"]
    original = source.loc[source.case == "analytic_simulator"]
    available = set(zip(original.mu_test.astype(float), original.toy_id.astype(int)))
    if not set(zip(selected.mu_test.astype(float), selected.toy_id.astype(int))) <= available:
        raise ValueError("Selected toys are absent from the original simulator results")
    return selected


def _validated_banks(context, bank_study, selection, bin_counts):
    from .toy_integration_diagnostics import diagnostic_bin_edges
    config = context["configuration"]
    edges = {n: diagnostic_bin_edges(n, base_bins=int(config["n_bins"])) for n in bin_counts}
    rates = np.asarray(context["rates"], dtype=float)
    quad = context["quad"]
    original = {}
    for mu in selection.mu_test.unique():
        z = reference_observable(quad["nominal"], rates, float(mu), config["reference_power"])
        for n, bins in edges.items():
            original[n, float(mu)] = histogram_components(z, quad["nominal"], quad["weights"], bins)
    prepared = [dict(bank_kind="original", replica=-1,
                     generated_per_source=int(context["original_n_per_source"]),
                     templates=original, rates=rates,
                     fingerprint=context["fingerprint"])]
    if not bank_study.get("banks"):
        raise ValueError("No independent convergence banks were supplied")
    for (replica, size), bank in sorted(bank_study["banks"].items()):
        metadata = bank.get("metadata", {})
        if (metadata.get("source_fingerprint") != context["fingerprint"]
                or metadata.get("replica") != replica or metadata.get("generated_per_source") != size
                or not metadata.get("fingerprint")):
            raise ValueError("Independent convergence bank provenance does not match its key/source context")
        if int(replica) != replica or replica < 0 or int(size) != size or size < 1:
            raise ValueError("Bank replica and generated sample size must be nonnegative/positive integers")
        bank_rates = np.asarray(bank["rates"], dtype=float)
        if bank_rates.shape != (4,) or not np.isfinite(bank_rates).all() or (bank_rates < 0).any():
            raise ValueError("Independent bank rates must be four finite nonnegative integrals")
        templates = {}
        for key, expected in original.items():
            n, _ = key
            if n not in bank.get("edges", {}) or not np.array_equal(np.asarray(bank["edges"][n]), edges[n]):
                raise ValueError("Independent convergence banks must use the original frozen bin edges")
            value = np.asarray(bank["templates"].get(key), dtype=float)
            if value.shape != expected.shape or not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"Invalid or missing independent component template {key}")
            if not np.allclose(value.sum(axis=1), bank_rates, rtol=2e-9, atol=1e-9):
                raise ValueError("Analytical rate integrals and binned templates must come from the same bank")
            templates[key] = value
        prepared.append(dict(bank_kind="independent", replica=int(replica), generated_per_source=int(size),
                             templates=templates, rates=bank_rates, fingerprint=metadata["fingerprint"]))
    return prepared, edges


def _mean_se(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan


def _fraction_se(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values)) if len(values) else np.nan
    return mean, float(np.sqrt(mean * (1 - mean) / len(values))) if len(values) else np.nan


_BANK_KEYS = ["mu_test", "bank_kind", "replica", "generated_per_source"]
_PAIR_KEYS = _BANK_KEYS + ["toy_id"]


def _tables(refits):
    """Summaries use random controls; all selected rows remain in paired output."""
    analytic = refits.loc[refits.model == "analytic", _PAIR_KEYS + ["q", "mu_hat", "valid"]].rename(
        columns={name: f"analytic_{name}" for name in ("q", "mu_hat", "valid")})
    paired = refits.loc[refits.model != "analytic"].merge(analytic, on=_PAIR_KEYS, validate="many_to_one")
    paired["paired_valid"] = (_bool(paired.valid) & _bool(paired.analytic_valid)
                              & np.isfinite(paired.q) & np.isfinite(paired.analytic_q)
                              & np.isfinite(paired.mu_hat) & np.isfinite(paired.analytic_mu_hat))
    paired["delta_q"] = (paired.q - paired.analytic_q).where(paired.paired_valid)
    paired["branch_disagreement"] = ((paired.mu_hat >= paired.branch_split)
                                     != (paired.analytic_mu_hat >= paired.branch_split)).where(paired.paired_valid)
    paired["zero_q_difference"] = ((paired.q <= 1e-10).astype(int)
                                   - (paired.analytic_q <= 1e-10).astype(int)).where(paired.paired_valid)
    summaries = []
    for key, group in refits.loc[_bool(refits.is_random)].groupby(_BANK_KEYS + ["model", "n_bins"], sort=True):
        good = group.loc[_bool(group.valid) & np.isfinite(group.q) & np.isfinite(group.mu_hat)]
        row = dict(zip(_BANK_KEYS + ["model", "n_bins"], key))
        row.update(n_random=len(group), n_valid=len(good), n_failed=len(group) - len(good),
                   n_at_upper=int(_bool(good.at_upper).sum()), mu_upper=float(group.mu_upper.iloc[0]),
                   summary_population="random controls only; conditional on valid fits")
        row["mean_q"], row["mean_q_se"] = _mean_se(good.q)
        for label, values in (("zero_q", good.q <= 1e-10),
                              ("high_branch", good.mu_hat >= good.branch_split),
                              ("upper_boundary", _bool(good.at_upper))):
            row[f"{label}_fraction"], row[f"{label}_se"] = _fraction_se(values)
        if row["model"] != "analytic":
            match = paired.loc[_bool(paired.is_random)]
            for name in _BANK_KEYS + ["model"]:
                match = match.loc[match[name] == row[name]]
            pg = match.loc[match.paired_valid]
            row.update(n_paired_valid=len(pg), n_paired_failed=len(match) - len(pg))
            row["mean_delta_q"], row["mean_delta_q_se"] = _mean_se(pg.delta_q)
            row["branch_disagreement_fraction"], row["branch_disagreement_se"] = _fraction_se(pg.branch_disagreement)
            row["zero_q_difference"], row["zero_q_difference_se"] = _mean_se(pg.zero_q_difference)
        summaries.append(row)
    return paired, pd.DataFrame(summaries), _bank_changes(refits, paired)


def _bank_changes(refits, paired):
    """Compare whole toy IDs, preserving covariance for every bank contrast."""
    metrics = []
    for model, group in refits.loc[_bool(refits.is_random)].groupby("model"):
        base = group.copy()
        base["metric_valid"] = _bool(base.valid) & np.isfinite(base.q) & np.isfinite(base.mu_hat)
        for metric, values in (("q", base.q), ("zero_q", (base.q <= 1e-10).astype(float)),
                               ("high_branch", (base.mu_hat >= base.branch_split).astype(float))):
            metrics.append((model, metric, base.assign(value=values)))
    for model, group in paired.loc[_bool(paired.is_random)].groupby("model"):
        metrics.append((model, "gap", group.assign(value=group.delta_q, metric_valid=group.paired_valid)))
    rows = []
    for model, metric, frame in metrics:
        for mu, group in frame.groupby("mu_test"):
            bankkeys = sorted(set(zip(group.bank_kind, group.replica, group.generated_per_source)))
            independent = [key for key in bankkeys if key[0] == "independent"]
            contrasts = []
            for replica in sorted({key[1] for key in independent}):
                chain = [key for key in independent if key[1] == replica]
                contrasts.extend(("prefix", left, right) for left, right in zip(chain[:-1], chain[1:]))
            for size in sorted({key[2] for key in independent}):
                chain = [key for key in independent if key[2] == size]
                contrasts.extend(("replica", chain[0], key) for key in chain[1:])
            originals = [key for key in bankkeys if key[0] == "original"]
            if originals:
                contrasts.extend(("original", originals[0], key) for key in independent)
            for kind, left, right in contrasts:
                def values(key):
                    return group.loc[(group.bank_kind == key[0]) & (group.replica == key[1])
                                     & (group.generated_per_source == key[2]), ["toy_id", "value", "metric_valid"]]
                both = values(left).merge(values(right), on="toy_id", suffixes=("_left", "_right"), validate="one_to_one")
                good = both.loc[_bool(both.metric_valid_left) & _bool(both.metric_valid_right)
                                & np.isfinite(both.value_left) & np.isfinite(both.value_right)]
                change, se = _mean_se(good.value_right - good.value_left)
                rows.append(dict(mu_test=float(mu), model=model, metric=metric, comparison_type=kind,
                                 from_replica=left[1], to_replica=right[1], from_size=left[2], to_size=right[2],
                                 n_random=len(both), n_valid=len(good), n_failed=len(both) - len(good),
                                 mean_change=change, paired_se=se,
                                 uncertainty="paired toy-sampling SE, conditional on both banks and valid fits"))
    return pd.DataFrame(rows)


def run_convergence_refits(context, selection, banks, *, mu_bounds=(0., 3.), grid_size=129,
                           bin_counts=(12, 36), progress_every=10, convergence_tag="bank_convergence"):
    """Refit exactly the saved toy IDs across all integration banks, without NNs.

    Checkpoints are separate from the original notebook-12 refits and contain
    one record per toy/bank/model. A completed failure is retained explicitly.
    Changing inputs requires another ``convergence_tag``; no caches are erased.
    """
    from .toy_sources import sample_simulator_poisson
    if not re.fullmatch(r"[A-Za-z0-9_-]+", convergence_tag):
        raise ValueError("convergence_tag must contain only letters, digits, underscores or hyphens")
    selection = _selection(selection, context)
    config = context["configuration"]
    bounds = np.asarray(mu_bounds, dtype=float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] != 0.
            or bounds[1] < max(config["mu_bounds"][1], float(selection.mu_test.max()))):
        raise ValueError("Convergence refits must retain zero and at least the original fit range")
    for value, name, minimum in ((grid_size, "grid_size", int(config["grid_size"])),
                                 (progress_every, "progress_every", 1)):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if (not bin_counts or len(set(bin_counts)) != len(bin_counts)
            or any(isinstance(n, bool) or int(n) != n for n in bin_counts)
            or int(config["n_bins"]) not in bin_counts):
        raise ValueError("bin_counts must be distinct integers including the original bin count")
    bin_counts = tuple(int(n) for n in bin_counts)
    prepared, edges = _validated_banks(context, banks, selection, bin_counts)
    identity = dict(version=1, source_fingerprint=context["fingerprint"],
                    selection_sha256=hashlib.sha256(selection.to_csv(index=False).encode()).hexdigest(),
                    mu_bounds=bounds.tolist(), grid_size=int(grid_size), bin_counts=list(bin_counts),
                    edges={str(n): value.tolist() for n, value in edges.items()},
                    banks=[dict(bank_kind=b["bank_kind"], replica=b["replica"], generated_per_source=b["generated_per_source"],
                                fingerprint=b["fingerprint"], rates=_array_digest(b["rates"]),
                                templates={f"{n}:{mu}": _array_digest(value) for (n, mu), value in sorted(b["templates"].items())})
                           for b in prepared])
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    output = Path(context["output_dir"]) / "convergence" / convergence_tag / "refits"
    config_path, results_path = output / "config.json", output / "results.csv"
    if config_path.exists():
        if json.loads(config_path.read_text()).get("fingerprint") != fingerprint:
            raise ValueError("Convergence refits use different inputs/settings; choose a new CONVERGENCE_TAG")
    elif results_path.exists():
        raise ValueError("Convergence refit checkpoint is missing its configuration fingerprint")
    metadata = dict(configuration=identity, fingerprint=fingerprint, output_dir=str(output),
                    normalization_note="Observable and its original rate normalizers stay frozen. Each bank updates both analytical selected rates and binned yields.",
                    sampling_note="Exactly the saved toy selection, stream 0. Events are generated once per unfinished toy and shared by every bank; no learned ratios are evaluated.",
                    uncertainty_note="Error bars are paired toy-sampling errors conditional on the chosen banks and valid fits. Prefixes share events; two replicas do not establish a finite-MC nuisance model.",
                    failure_note="Failures are retained. Population summaries use is_random only and condition on the explicitly counted valid fits.")
    save_json(config_path, metadata)
    _atomic_csv(selection, output / "selection.csv")
    records = pd.read_csv(results_path, float_precision="round_trip").to_dict("records") if results_path.exists() else []
    models = [("analytic", 0)] + [(f"binned{n}", n) for n in bin_counts]
    keycols = ["mu_test", "toy_id", "bank_kind", "replica", "generated_per_source", "model"]
    expected = {(float(row.mu_test), int(row.toy_id), b["bank_kind"], b["replica"], b["generated_per_source"], model)
                for row in selection.itertuples() for b in prepared for model, _ in models}
    keys = [tuple(record[name] for name in keycols) for record in records]
    if len(set(keys)) != len(keys) or not set(keys) <= expected:
        raise ValueError("Convergence checkpoint has duplicate or unexpected toy/bank/model records")
    completed = set(keys)
    exposure = float(config["exposure_multiplier"])
    source = context["source"]["results"]
    saved_counts = source.loc[source.case == "analytic_simulator"].set_index(["mu_test", "toy_id"]).n_observed
    for number, selected in enumerate(selection.to_dict("records"), start=1):
        mu, toy_id = float(selected["mu_test"]), int(selected["toy_id"])
        needed = [(b, model, n) for b in prepared for model, n in models
                  if (mu, toy_id, b["bank_kind"], b["replica"], b["generated_per_source"], model) not in completed]
        if not needed:
            continue
        error, n_observed, event_digest, analytic, counts = None, np.nan, "", None, {}
        try:
            x = sample_simulator_poisson(context["run"], mu, _rng(int(config["seed"]), mu, toy_id, 0),
                                         exposure=exposure, selector=context["selector"], threshold=context["threshold"])
            n_observed, event_digest = len(x), _array_digest(x)
            if n_observed != saved_counts.loc[mu, toy_id]:
                raise ValueError("Reconstructed event count does not reproduce notebook 11; stop interpreting this toy")
            analytic = context["run"].model.component_densities(x).T
            z = reference_observable(analytic, context["rates"], mu, config["reference_power"])
            counts = {n: np.histogram(z, bins)[0] for n, bins in edges.items()}
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            error = str(exc)
        earlier_digests = {record.get("event_sha256") for record in records
                           if record["mu_test"] == mu and record["toy_id"] == toy_id
                           and isinstance(record.get("event_sha256"), str) and record["event_sha256"]}
        if event_digest and earlier_digests and earlier_digests != {event_digest}:
            raise ValueError("Regenerated event coordinates differ from this toy's partial checkpoint; "
                             "resolve the simulator mismatch before resuming")
        for bank, model, n in needed:
            try:
                if error is not None:
                    raise ValueError(error)
                likelihood = (StatOnlyLikelihood(analytic, bank["rates"], exposure=exposure) if n == 0 else
                              StatOnlyLikelihood.from_binned(bank["templates"][n, mu], counts[n], exposure=exposure))
                result = _fit(likelihood, mu, bounds, int(grid_size))
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                result = _failure(exc)
            row = dict(**selected, bank_kind=bank["bank_kind"], replica=bank["replica"],
                       generated_per_source=bank["generated_per_source"], bank_fingerprint=bank["fingerprint"],
                       model=model, n_bins=n, n_observed=n_observed, event_sha256=event_digest,
                       count_reproduced=bool(n_observed == saved_counts.loc[mu, toy_id]),
                       mu_lower=float(bounds[0]), mu_upper=float(bounds[1]), **result)
            records.append(row)
            completed.add(tuple(row[name] for name in keycols))
        _atomic_csv(pd.DataFrame(records), results_path)
        if number % progress_every == 0 or number == len(selection):
            print(f"Integration convergence refits: {number}/{len(selection)} paired experiments complete", flush=True)
    refits = pd.DataFrame(records).sort_values(keycols).reset_index(drop=True)
    refits["message"] = refits.message.fillna("")
    paired, summary, convergence = _tables(refits)
    for name, frame in (("paired", paired), ("summary", summary), ("convergence", convergence)):
        _atomic_csv(frame, output / f"{name}.csv")
    return dict(refits=refits, summary=summary, paired=paired, convergence=convergence, metadata=metadata)


def plot_convergence_refits(study, output):
    """Plot conditional paired errors; bank replicas retain separate curves."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary, changes = study["summary"], study["convergence"]
    if summary.empty:
        raise ValueError("Convergence plots require at least one saved random control")
    mus = sorted(summary.mu_test.unique())
    models = sorted(summary.model.unique(), key=lambda name: (name != "analytic", name))
    colors = {model: color for model, color in zip(models, ("black", "#0072b2", "#d55e00", "#009e73"))}
    figures = {}
    labels = {"analytic": "Analytical", **{model: model.replace("binned", "Binned, ") + " bins" for model in models if model != "analytic"}}
    with plt.style.context(hep.style.ATLAS):
        def curves(name, value, error, ylabel, *, binned_only=False):
            fig, axes = plt.subplots(1, len(mus), figsize=(7.2 * len(mus), 5.9), squeeze=False)
            for ax, mu in zip(axes[0], mus):
                frame = summary.loc[summary.mu_test == mu]
                for model in models:
                    if binned_only and model == "analytic":
                        continue
                    subset = frame.loc[frame.model == model]
                    original = subset.loc[subset.bank_kind == "original"]
                    if len(original) and np.isfinite(original[value].iloc[0]):
                        ax.axhline(original[value].iloc[0], color=colors[model], linestyle=":", alpha=.7,
                                   label=f"{labels[model]}, original")
                    for replica, group in subset.loc[subset.bank_kind == "independent"].groupby("replica"):
                        group = group.sort_values("generated_per_source")
                        ax.errorbar(group.generated_per_source, group[value], yerr=group[error],
                                    fmt="o-" if int(replica) % 2 == 0 else "s--", color=colors[model],
                                    markersize=4, capsize=2, label=f"{labels[model]}, replica {int(replica) + 1}")
                if binned_only:
                    ax.axhline(0., color=".6", linewidth=1)
                ax.set(xscale="log", xlabel="Generated events per source in integration bank", ylabel=ylabel,
                       title=rf"$\mu_{{\rm test}}={mu:g}$; random controls")
                ax.legend(fontsize=8, frameon=False)
            fig.text(.5, -.01, "Bars: toy-sampling SE conditional on banks and valid fits; prefixes are correlated.", ha="center", fontsize=10)
            fig.tight_layout()
            figures[name] = fig
        curves("convergence_mean_q", "mean_q", "mean_q_se", r"Mean $q_\mu$")
        curves("convergence_paired_gap", "mean_delta_q", "mean_delta_q_se",
               r"Mean paired $q_\mu^{\rm binned}-q_\mu^{\rm analytical}$", binned_only=True)
        curves("convergence_zero_q", "zero_q_fraction", "zero_q_se", r"Fraction with $q_\mu=0$")
        curves("convergence_branches", "branch_disagreement_fraction", "branch_disagreement_se",
               "Branch disagreement with matched analytical fit", binned_only=True)
        fig, axes = plt.subplots(2, len(mus), figsize=(7.2 * len(mus), 10.5), squeeze=False)
        for column, mu in enumerate(mus):
            for row, kind in enumerate(("prefix", "replica")):
                ax = axes[row, column]
                group = changes.loc[(changes.mu_test == mu) & (changes.metric == "gap")
                                    & (changes.comparison_type == kind)]
                for model in models:
                    for replica, part in group.loc[group.model == model].groupby("to_replica"):
                        part = part.sort_values("to_size")
                        ax.errorbar(part.to_size, part.mean_change, yerr=part.paired_se,
                                    fmt="o-" if int(replica) % 2 == 0 else "s--", color=colors[model], capsize=2,
                                    label=f"{labels[model]}, replica {int(replica) + 1}" if kind == "prefix" else labels[model])
                ax.axhline(0., color=".5", linestyle=":")
                ax.set(xscale="log", xlabel="Generated events per source",
                       ylabel="Change in mean paired gap")
                ax.set_title(rf"$\mu_{{\rm test}}={mu:g}$; " +
                             ("adjacent prefixes" if kind == "prefix" else "replicas: later minus first"),
                             fontsize=18)
                ax.legend(fontsize=9, frameon=False)
        fig.text(.5, -.005, "Same toys in each difference. Two bank replicas are a stability check, not a finite-MC error model.", ha="center", fontsize=10)
        fig.tight_layout()
        figures["convergence_bank_changes"] = fig
        for name, fig in figures.items():
            for extension in ("pdf", "png"):
                fig.savefig(output / f"12_{name}.{extension}", dpi=160, bbox_inches="tight")
    return figures
