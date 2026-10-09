"""Separate fixed-pair compression from the fitted denominator of saved toy q.

The observable stays frozen at eta, and every likelihood difference uses mu=1
as reference. No optimization, integration-bank generation or ratio-network
inference is performed. The saved simulator toys are reconstructed exactly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import save_json
from .toy_convergence_banks import load_completed_convergence_banks
from .toy_convergence_refits import _mean_se, _selection, _validated_banks
from .toy_likelihood import StatOnlyLikelihood
from .toy_refit_diagnostics import _bool
from .toy_study import _array_digest, _atomic_csv, _rng, reference_observable


_BANK_KEYS = ["mu_test", "bank_kind", "replica", "generated_per_source"]
_PAIR_KEYS = _BANK_KEYS + ["toy_id"]
_RECORD_KEYS = _PAIR_KEYS + ["model"]
_TERMS = ("delta_pair", "delta_min", "delta_q")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _rms_se(values):
    """RMS and its first-order, conditional toy-sampling standard error."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rms = float(np.sqrt(np.mean(values ** 2)))
    if len(values) < 2:
        return rms, np.nan
    se = float(np.std(values ** 2, ddof=1) / np.sqrt(len(values)) / (2 * rms)) if rms > 0 else 0.
    return rms, se


def _tables(records):
    """Keep every failure; only random, jointly valid toys enter summaries."""
    names = ["D_eta", "min_D", "q", "mu_hat", "decomposition_valid"]
    analytical = records.loc[records.model == "analytic", _PAIR_KEYS + names].rename(
        columns={name: f"analytic_{name}" for name in names})
    paired = records.loc[records.model != "analytic"].merge(analytical, on=_PAIR_KEYS, validate="many_to_one")
    paired["paired_valid"] = (_bool(paired.decomposition_valid) & _bool(paired.analytic_decomposition_valid))
    paired["delta_pair"] = (paired.D_eta - paired.analytic_D_eta).where(paired.paired_valid)
    paired["delta_min"] = (paired.min_D - paired.analytic_min_D).where(paired.paired_valid)
    paired["delta_q"] = (paired.q - paired.analytic_q).where(paired.paired_valid)
    paired["reconstruction_error"] = (paired.delta_q - paired.delta_pair + paired.delta_min).where(paired.paired_valid)
    paired["branch_pair"] = "invalid"
    good = paired.paired_valid
    paired.loc[good, "branch_pair"] = [
        f"{'high' if a >= split else 'low'} / {'high' if b >= split else 'low'}"
        for a, b, split in zip(paired.loc[good, "analytic_mu_hat"], paired.loc[good, "mu_hat"], paired.loc[good, "branch_split"])]
    paired["branch_disagreement"] = ((paired.mu_hat >= paired.branch_split)
                                     != (paired.analytic_mu_hat >= paired.branch_split)).where(good)
    summaries = []
    for key, group in paired.loc[_bool(paired.is_random)].groupby(_BANK_KEYS + ["model", "n_bins"], sort=True):
        valid = group.loc[group.paired_valid]
        strata = [("all", valid)] + [(name, part) for name, part in valid.groupby("branch_pair", sort=True)]
        for branch, part in strata:
            row = dict(zip(_BANK_KEYS + ["model", "n_bins"], key))
            row.update(branch_pair=branch, n_random=len(group), n_paired_valid=len(valid),
                       n_paired_failed=len(group) - len(valid), n_stratum=len(part),
                       branch_fraction=len(part) / len(valid) if len(valid) else np.nan,
                       summary_population="random controls; conditional on jointly valid decompositions")
            for term in _TERMS:
                row[f"mean_{term}"], row[f"mean_{term}_se"] = _mean_se(part[term])
                row[f"rms_{term}"], row[f"rms_{term}_se"] = _rms_se(part[term])
            row["max_abs_reconstruction_error"] = float(part.reconstruction_error.abs().max()) if len(part) else np.nan
            summaries.append(row)
    return paired, pd.DataFrame(summaries), _bank_changes(paired)


def _bank_changes(paired):
    rows = []
    for (mu, model), group in paired.loc[_bool(paired.is_random)].groupby(["mu_test", "model"]):
        bankkeys = sorted(set(zip(group.bank_kind, group.replica, group.generated_per_source)))
        independent = [key for key in bankkeys if key[0] == "independent"]
        contrasts = []
        for replica in sorted({key[1] for key in independent}):
            chain = [key for key in independent if key[1] == replica]
            contrasts.extend(("prefix", a, b) for a, b in zip(chain[:-1], chain[1:]))
        for size in sorted({key[2] for key in independent}):
            chain = [key for key in independent if key[2] == size]
            contrasts.extend(("replica", chain[0], b) for b in chain[1:])
        originals = [key for key in bankkeys if key[0] == "original"]
        if originals:
            contrasts.extend(("original", originals[0], b) for b in independent)
        for kind, left, right in contrasts:
            def subset(key):
                return group.loc[(group.bank_kind == key[0]) & (group.replica == key[1])
                                 & (group.generated_per_source == key[2]),
                                 ["toy_id", "paired_valid", *_TERMS]]
            joined = subset(left).merge(subset(right), on="toy_id", suffixes=("_left", "_right"), validate="one_to_one")
            valid = joined.loc[_bool(joined.paired_valid_left) & _bool(joined.paired_valid_right)]
            for term in _TERMS:
                values = valid[f"{term}_right"] - valid[f"{term}_left"]
                mean, se = _mean_se(values)
                rms, rms_se = _rms_se(values)
                rows.append(dict(mu_test=float(mu), model=model, term=term, comparison_type=kind,
                                 from_replica=left[1], to_replica=right[1], from_size=left[2], to_size=right[2],
                                 n_random=len(joined), n_valid=len(valid), n_failed=len(joined) - len(valid),
                                 mean_change=mean, paired_se=se, rms_change=rms, rms_change_se=rms_se,
                                 uncertainty="paired toy-sampling SE conditional on both banks and valid decompositions"))
    return pd.DataFrame(rows)


def _saved_inputs(context, convergence_tag, largest_prefixes):
    """Authenticate the completed refits and requested bank predictions."""
    directory = Path(context["output_dir"]) / "convergence" / convergence_tag / "refits"
    paths = {name: directory / name for name in ("config.json", "selection.csv", "results.csv")}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("Complete notebook 12 section 6 before the likelihood decomposition")
    metadata = json.loads(paths["config.json"].read_text())
    config = metadata["configuration"]
    if config.get("version") != 1 or metadata.get("fingerprint") != _digest(config):
        raise ValueError("Saved convergence refit configuration has an invalid fingerprint/version")
    if config.get("source_fingerprint") != context["fingerprint"]:
        raise ValueError("Saved convergence refits belong to a different frozen context")
    selection = _selection(pd.read_csv(paths["selection.csv"], float_precision="round_trip"), context)
    if hashlib.sha256(selection.to_csv(index=False).encode()).hexdigest() != config.get("selection_sha256"):
        raise ValueError("Saved convergence selection does not match its configuration")
    bank_study = load_completed_convergence_banks(context, convergence_tag=convergence_tag)
    chosen = {}
    for replica in sorted({key[0] for key in bank_study["banks"]}):
        keys = sorted((key for key in bank_study["banks"] if key[0] == replica), key=lambda key: key[1])
        for key in keys[-largest_prefixes:]:
            chosen[key] = bank_study["banks"][key]
    prepared, edges = _validated_banks(context, {"banks": chosen}, selection, tuple(config["bin_counts"]))
    expected_banks = {(b["bank_kind"], b["replica"], b["generated_per_source"]): b for b in config["banks"]}
    for bank in prepared:
        key = bank["bank_kind"], bank["replica"], bank["generated_per_source"]
        saved = expected_banks.get(key)
        actual = dict(bank_kind=key[0], replica=key[1], generated_per_source=key[2], fingerprint=bank["fingerprint"],
                      rates=_array_digest(bank["rates"]),
                      templates={f"{n}:{mu}": _array_digest(value) for (n, mu), value in sorted(bank["templates"].items())})
        if saved != actual:
            raise ValueError("Saved convergence refits do not match the selected bank predictions")
    if any(config["edges"].get(str(n)) != value.tolist() for n, value in edges.items()):
        raise ValueError("Saved convergence bin boundaries differ from the frozen observable")
    source = pd.read_csv(paths["results.csv"], float_precision="round_trip")
    required = set(_RECORD_KEYS + ["n_bins", "q", "mu_hat", "valid", "message", "bank_fingerprint",
                                  "event_sha256", "n_observed", "count_reproduced", "mu_lower", "mu_upper",
                                  "is_random", "is_flagged", "branch_split"])
    if not required <= set(source) or source.duplicated(_RECORD_KEYS).any():
        raise ValueError("Saved convergence refits have missing columns or duplicate records")
    chosen_keys = {(b["bank_kind"], b["replica"], b["generated_per_source"]) for b in prepared}
    source = source.loc[[key in chosen_keys for key in zip(source.bank_kind, source.replica, source.generated_per_source)]].copy()
    models = [("analytic", 0)] + [(f"binned{n}", n) for n in config["bin_counts"]]
    expected = {(float(s.mu_test), b["bank_kind"], b["replica"], b["generated_per_source"], int(s.toy_id), model)
                for s in selection.itertuples() for b in prepared for model, _ in models}
    if set(source[_RECORD_KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError("Selected convergence refits are incomplete; finish section 6 before decomposing them")
    source["valid"] = _bool(source.valid)
    source["message"] = source.message.fillna("")
    for bank in prepared:
        subset = source.loc[(source.bank_kind == bank["bank_kind"]) & (source.replica == bank["replica"])
                            & (source.generated_per_source == bank["generated_per_source"])]
        if not (subset.bank_fingerprint == bank["fingerprint"]).all():
            raise ValueError("Saved fit rows reference a different integration bank")
    for key in ("is_random", "is_flagged", "branch_split"):
        values = source.merge(selection[["mu_test", "toy_id", key]], on=["mu_test", "toy_id"], suffixes=("", "_expected"), validate="many_to_one")
        actual = _bool(values[key]) if key.startswith("is_") else values[key]
        expected_values = _bool(values[f"{key}_expected"]) if key.startswith("is_") else values[f"{key}_expected"]
        if not (actual == expected_values).all():
            raise ValueError("Saved convergence fit labels disagree with the selected toy population")
    if (not (source.mu_lower == config["mu_bounds"][0]).all()
            or not (source.mu_upper == config["mu_bounds"][1]).all()
            or any(not (source.loc[source.model == model, "n_bins"] == n).all() for model, n in models)):
        raise ValueError("Saved convergence fit bounds/bin counts disagree with their configuration")
    source_files = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    return selection, source, prepared, edges, metadata, source_files


def run_likelihood_decomposition(context, *, convergence_tag="bank_convergence", decomposition_tag="fixed_pair",
                                 largest_prefixes=2, progress_every=10):
    """Decompose existing section-6 q values without refitting or making banks.

    This can run after only the notebook setup cells. Its completed section-6
    inputs are read-only. By default it uses the two largest saved prefixes per
    replica plus the original-bank control. A new tag is required if inputs
    change; checkpoints preserve failed source fits explicitly.
    """
    from .toy_sources import sample_simulator_poisson
    for tag in (convergence_tag, decomposition_tag):
        if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
            raise ValueError("Tags must contain only letters, digits, underscores or hyphens")
    for value, name in ((largest_prefixes, "largest_prefixes"), (progress_every, "progress_every")):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    selection, source, prepared, edges, source_metadata, source_files = _saved_inputs(context, convergence_tag, int(largest_prefixes))
    identity = dict(version=1, source_fingerprint=context["fingerprint"],
                    convergence_fingerprint=source_metadata["fingerprint"], source_files_sha256=source_files,
                    largest_prefixes=int(largest_prefixes), reference_mu=1.,
                    banks=[dict(bank_kind=b["bank_kind"], replica=b["replica"], generated_per_source=b["generated_per_source"],
                                fingerprint=b["fingerprint"]) for b in prepared],
                    q_reproduction_atol=1e-7, q_reproduction_rtol=1e-7)
    fingerprint = _digest(identity)
    output = Path(context["output_dir"]) / "convergence" / convergence_tag / "decomposition" / decomposition_tag
    config_path, results_path = output / "config.json", output / "records.csv"
    if config_path.exists():
        if json.loads(config_path.read_text()).get("fingerprint") != fingerprint:
            raise ValueError("Decomposition inputs changed; use a new DECOMPOSITION_TAG")
    elif results_path.exists():
        raise ValueError("Decomposition checkpoint is missing its configuration fingerprint")
    metadata = dict(configuration=identity, fingerprint=fingerprint, output_dir=str(output),
                    definition="D(mu)=-2 log[L_eta(mu)/L_eta(1)]; min_D=D(eta)-saved_q; delta_q=delta_pair-delta_min.",
                    sampling_note="Saved simulator toy IDs and event hashes are reproduced. No optimization, bank generation or ratio-network inference; the frozen selector is used to reconstruct events.",
                    uncertainty_note="Mean bars: paired toy-sampling SE conditional on banks and jointly valid rows; RMS bars use the delta method. Prefixes share events. Branch-stratum errors also condition on branch membership.",
                    failure_note="Failed source fits and nonfinite/reproduction failures remain explicit. Population summaries use random controls only; flagged extras appear only in per-toy tables.",
                    interpretation_note="Fixed-pair closure tests eta versus 1 for the frozen observable; the minimum term measures changes in the freely fitted denominator. Neither term alone establishes coverage.")
    save_json(config_path, metadata)
    records = pd.read_csv(results_path, float_precision="round_trip").to_dict("records") if results_path.exists() else []
    expected = set(source[_RECORD_KEYS].itertuples(index=False, name=None))
    cached_keys = [tuple(row[k] for k in _RECORD_KEYS) for row in records]
    if len(set(cached_keys)) != len(cached_keys) or not set(cached_keys) <= expected:
        raise ValueError("Decomposition checkpoint has duplicate or unexpected records")
    done = set(cached_keys)
    banks_by_key = {(b["bank_kind"], b["replica"], b["generated_per_source"]): b for b in prepared}
    config = context["configuration"]
    exposure = float(config["exposure_multiplier"])
    original = context["source"]["results"]
    original_counts = original.loc[original.case == "analytic_simulator"].set_index(["mu_test", "toy_id"]).n_observed
    for number, selected in enumerate(selection.itertuples(index=False), start=1):
        mu, toy_id = float(selected.mu_test), int(selected.toy_id)
        saved = source.loc[(source.mu_test == mu) & (source.toy_id == toy_id)]
        needed = saved.loc[[tuple(row) not in done for row in saved[_RECORD_KEYS].itertuples(index=False, name=None)]]
        if needed.empty:
            continue
        fields, counts = None, {}
        if needed.valid.any():
            x = sample_simulator_poisson(context["run"], mu, _rng(int(config["seed"]), mu, toy_id, 0),
                                         exposure=exposure, selector=context["selector"], threshold=context["threshold"])
            digest = _array_digest(x)
            successful = saved.loc[saved.valid]
            if (len(x) != original_counts.loc[mu, toy_id] or not (successful.n_observed == len(x)).all()
                    or not _bool(successful.count_reproduced).all()):
                raise ValueError("Reconstructed event count does not reproduce saved section-6 toys")
            if not (successful.event_sha256 == digest).all():
                raise ValueError("Reconstructed event coordinates do not match saved section-6 event hashes")
            previous = [row for row in records if row["mu_test"] == mu and row["toy_id"] == toy_id and row["valid"]]
            if any(row["event_sha256"] != digest for row in previous):
                raise ValueError("Regenerated event coordinates differ from the partial decomposition checkpoint")
            fields = context["run"].model.component_densities(x).T
            z = reference_observable(fields, context["rates"], mu, config["reference_power"])
            counts = {n: np.histogram(z, bins)[0] for n, bins in edges.items()}
        for item in needed.to_dict("records"):
            row = dict(item, D_eta=np.nan, D_at_muhat=np.nan, min_D=np.nan, q_recomputed=np.nan,
                       q_reproduction_error=np.nan, decomposition_valid=False, decomposition_message="")
            if not item["valid"]:
                row["decomposition_message"] = "Saved section-6 fit failed: " + str(item["message"])
            else:
                bank = banks_by_key[item["bank_kind"], item["replica"], item["generated_per_source"]]
                n = int(item["n_bins"])
                try:
                    if (not np.isfinite([item["q"], item["mu_hat"]]).all() or item["q"] < -1e-10
                            or not item["mu_lower"] <= item["mu_hat"] <= item["mu_upper"]):
                        raise ValueError("Saved fit q or fitted mu is invalid")
                    likelihood = (StatOnlyLikelihood(fields, bank["rates"], exposure=exposure) if n == 0 else
                                  StatOnlyLikelihood.from_binned(bank["templates"][n, mu], counts[n], exposure=exposure))
                    row["D_eta"] = float(likelihood.nll(mu, reference_mu=1.))
                    row["D_at_muhat"] = float(likelihood.nll(float(item["mu_hat"]), reference_mu=1.))
                    row["min_D"] = row["D_eta"] - float(item["q"])
                    row["q_recomputed"] = row["D_eta"] - row["D_at_muhat"]
                    row["q_reproduction_error"] = row["q_recomputed"] - float(item["q"])
                    if not np.isfinite([row[name] for name in ("D_eta", "D_at_muhat", "min_D", "q_recomputed")]).all():
                        raise ValueError("Fixed-reference likelihood is nonfinite at eta, muhat or reference mu=1")
                    if not np.isclose(row["q_recomputed"], item["q"], atol=identity["q_reproduction_atol"], rtol=identity["q_reproduction_rtol"]):
                        raise ValueError("Saved q is not reproduced by evaluating the likelihood at its saved muhat")
                    row["decomposition_valid"] = True
                except (ValueError, FloatingPointError, RuntimeError) as exc:
                    row["decomposition_message"] = str(exc)
            records.append(row)
            done.add(tuple(row[k] for k in _RECORD_KEYS))
        _atomic_csv(pd.DataFrame(records), results_path)
        if number % progress_every == 0 or number == len(selection):
            print(f"Fixed-pair decomposition: {number}/{len(selection)} saved experiments complete (no refits)", flush=True)
    records = pd.DataFrame(records).sort_values(_RECORD_KEYS).reset_index(drop=True)
    records["message"] = records.message.fillna("")
    records["decomposition_message"] = records.decomposition_message.fillna("")
    paired, summary, changes = _tables(records)
    for name, frame in (("paired", paired), ("summary", summary), ("changes", changes)):
        _atomic_csv(frame, output / f"{name}.csv")
    return dict(records=records, paired=paired, summary=summary, changes=changes, metadata=metadata)


def plot_likelihood_decomposition(study, output):
    """Signed decomposition terms stay on linear axes, including per-toy plots."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = study["summary"].loc[study["summary"].branch_pair == "all"]
    paired, changes = study["paired"], study["changes"]
    if summary.empty:
        raise ValueError("Decomposition plots require at least one saved random control")
    mus = sorted(summary.mu_test.unique())
    models = sorted(summary.model.unique(), key=lambda model: int(model.replace("binned", "")))
    colors = dict(zip(_TERMS, ("#0072b2", "#d55e00", "#009e73")))
    labels = dict(delta_pair=r"Fixed pair: $\Delta D(\eta)$", delta_min=r"Minimum: $\Delta\min D$", delta_q=r"Total: $\Delta q$")
    figures = {}
    note = "Random controls only; errors condition on the banks and valid rows. Prefixes share integration events."
    with plt.style.context(hep.style.ATLAS):
        for model in models:
            fig, axes = plt.subplots(2, len(mus), figsize=(7.4 * len(mus), 10.4), squeeze=False)
            for column, mu in enumerate(mus):
                frame = summary.loc[(summary.mu_test == mu) & (summary.model == model)]
                for row, statistic in enumerate(("mean", "rms")):
                    ax = axes[row, column]
                    for term in _TERMS:
                        for replica, part in frame.loc[frame.bank_kind == "independent"].groupby("replica"):
                            part = part.sort_values("generated_per_source")
                            ax.errorbar(part.generated_per_source, part[f"{statistic}_{term}"],
                                        yerr=part[f"{statistic}_{term}_se"], color=colors[term], capsize=2,
                                        fmt="o-" if int(replica) % 2 == 0 else "s--",
                                        label=f"{labels[term]}, replica {int(replica) + 1}")
                    ax.axhline(0., color=".65", linewidth=1)
                    ax.set(xscale="log", xlabel="Generated events per source", ylabel="Mean signed difference" if row == 0 else "RMS difference",
                           title=rf"$\eta={mu:g}$; {model.replace('binned', '')} bins")
                    ax.legend(fontsize=8, frameon=False)
            fig.text(.5, -.01, note + " RMS bars: delta-method SE.", ha="center", fontsize=9)
            fig.tight_layout()
            figures[f"decomposition_moments_{model}"] = fig
            fig, axes = plt.subplots(2, len(mus), figsize=(7.4 * len(mus), 10.4), squeeze=False)
            for column, mu in enumerate(mus):
                for row, kind in enumerate(("prefix", "replica")):
                    ax = axes[row, column]
                    frame = changes.loc[(changes.model == model) & (changes.mu_test == mu) & (changes.comparison_type == kind)]
                    for term in _TERMS:
                        for replica, part in frame.loc[frame.term == term].groupby("to_replica"):
                            part = part.sort_values("to_size")
                            ax.errorbar(part.to_size, part.mean_change, yerr=part.paired_se, color=colors[term], capsize=3,
                                        fmt="o-" if int(replica) % 2 == 0 else "s--",
                                        label=f"{labels[term]}, replica {int(replica) + 1}" if kind == "prefix" else labels[term])
                    ax.axhline(0., color=".6", linewidth=1)
                    ax.set(xscale="log", xlabel="Generated events per source", ylabel="Paired change in mean term",
                           title=rf"$\eta={mu:g}$; " + ("adjacent prefixes" if kind == "prefix" else "replicas: later minus first"))
                    ax.legend(fontsize=8, frameon=False)
            fig.text(.5, -.01, note, ha="center", fontsize=9)
            fig.tight_layout()
            figures[f"decomposition_bank_changes_{model}"] = fig
        branch_colors = {"low / low": "#0072b2", "high / high": "#009e73", "low / high": "#d55e00", "high / low": "#cc79a7"}
        independent = paired.loc[(paired.bank_kind == "independent") & _bool(paired.is_random) & _bool(paired.paired_valid)]
        for replica, frame in independent.groupby("replica"):
            size = int(frame.generated_per_source.max())
            frame = frame.loc[frame.generated_per_source == size]
            fig, axes = plt.subplots(len(models), len(mus), figsize=(7.4 * len(mus), 5.5 * len(models)), squeeze=False)
            for row, model in enumerate(models):
                for column, mu in enumerate(mus):
                    ax = axes[row, column]
                    part = frame.loc[(frame.model == model) & (frame.mu_test == mu)]
                    for branch, group in part.groupby("branch_pair"):
                        ax.scatter(group.delta_pair, group.delta_min, color=branch_colors[branch], s=20, alpha=.65,
                                   label=f"{branch} ({len(group)})")
                    finite = part[["delta_pair", "delta_min"]].to_numpy().ravel()
                    finite = finite[np.isfinite(finite)]
                    lower, upper = (float(finite.min()), float(finite.max())) if len(finite) else (-1., 1.)
                    pad = max(.05, .05 * (upper - lower))
                    limits = (lower - pad, upper + pad)
                    ax.plot(limits, limits, ":", color=".4", label=r"$\Delta q=0$")
                    ax.axhline(0., color=".75", linewidth=.7)
                    ax.axvline(0., color=".75", linewidth=.7)
                    ax.set(xlim=limits, ylim=limits, xlabel=r"Fixed-pair term $\Delta D(\eta)$", ylabel=r"Minimum term $\Delta\min D$",
                           title=rf"$\eta={mu:g}$; {model.replace('binned', '')} bins; replica {int(replica) + 1}")
                    # ATLAS-style right-aligned x labels can overlap the
                    # scientific-notation offset on these signed scatter axes.
                    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
                    ax.legend(title="Branch: analytical / binned", fontsize=8, title_fontsize=9, frameon=False)
            fig.text(.5, -.005, f"Largest bank: {size:,} generated/source. Each point is one random toy; vertical distance below diagonal equals Δq.", ha="center", fontsize=9)
            fig.tight_layout()
            figures[f"decomposition_toys_replica{int(replica) + 1}"] = fig
        for name, fig in figures.items():
            for extension in ("pdf", "png"):
                fig.savefig(output / f"12_{name}.{extension}", dpi=160, bbox_inches="tight")
    return figures
