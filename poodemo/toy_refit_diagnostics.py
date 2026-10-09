"""Controlled refits of the same simulator experiments from notebook 11.

No events are saved: each experiment is reconstructed from its original seed.
The observable and selected-rate normalizers are frozen while fit bounds,
optimizer resolution, bin resolution and integration precision are varied one
at a time. Population summaries use the randomly selected controls only;
deliberately selected boundary cases are diagnostic examples, not an ensemble.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import save_json
from .inference import histogram_components
from .toy_likelihood import StatOnlyLikelihood
from .toy_study import _array_digest, _atomic_csv, _failure, _fit, _rng, reference_observable


SIMULATOR_CASES = ("analytic_simulator", "learned_simulator", "binned_simulator")
CASE_LABELS = dict(zip(SIMULATOR_CASES, ("Analytical", "Learned", "Binned")))
VARIANT_LABELS = {
    "original": "Original settings", "dense": "Dense grid, original range",
    "wide": "Dense grid, wider range", "bins36": "36 bins, original bank",
    "integrated12": "12 bins, independent bank", "integrated36": "36 bins, independent bank",
    "analytic_integrated": "Analytical, independent rate integrals",
}


def _bool(values):
    """Read both pandas boolean columns and explicit CSV boolean strings."""
    if getattr(values, "dtype", None) == object:
        bad = ~values.isin([True, False, "True", "False", "true", "false", 0, 1])
        if bad.any():
            raise ValueError("Boolean fit flags contain missing or unrecognized values")
        return values.map(lambda value: value in (True, "True", "true", 1)).astype(bool)
    if values.isna().any():
        raise ValueError("Boolean fit flags contain missing values")
    return values.astype(bool)


def select_refit_toys(source, *, n_random=200, max_flagged=None, seed=120624,
                      branch_split=1.):
    """Select every upper-bound case plus independent random controls per mu.

    Random controls are sampled from all complete simulator-case records,
    including flagged toys. Overlap is retained as both flags. The optional
    ``max_flagged`` cap is for quick checks, not the production diagnostic.
    Failed original fits remain eligible when all three records are present.
    """
    for value, name in ((n_random, "n_random"), (seed, "seed")):
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if max_flagged is not None and (isinstance(max_flagged, bool)
                                   or int(max_flagged) != max_flagged or max_flagged < 0):
        raise ValueError("max_flagged must be None or a nonnegative integer")
    if not np.isfinite(branch_split) or branch_split <= 0:
        raise ValueError("branch_split must be finite and positive")
    n_random, seed = int(n_random), int(seed)
    results = source["results"]
    required = {"mu_test", "toy_id", "case", "at_upper", "mu_hat", "q"}
    if not required <= set(results):
        raise ValueError(f"Source results lack columns: {sorted(required - set(results))}")
    subset = results.loc[results.case.isin(SIMULATOR_CASES)].copy()
    if subset.duplicated(["mu_test", "toy_id", "case"]).any():
        raise ValueError("Duplicate original simulator-case records")
    subset["at_upper"] = _bool(subset.at_upper)
    rows = []
    for mu, group in subset.groupby("mu_test", sort=True):
        complete = group.groupby("toy_id").case.nunique()
        ids = np.sort(complete.index[complete == len(SIMULATOR_CASES)].to_numpy(dtype=int))
        group = group.loc[group.toy_id.isin(ids)]
        upper = group.groupby("toy_id").at_upper.any()
        flags = upper.index[upper].to_numpy(dtype=int)
        random_ids = _rng(seed, mu, 0, 12).choice(ids, size=min(n_random, len(ids)), replace=False)
        kept_flags = flags if max_flagged is None else flags[:int(max_flagged)]
        random_set, flagged_set = set(random_ids), set(kept_flags)
        for toy_id in sorted(random_set | flagged_set):
            original = group.loc[group.toy_id == toy_id].set_index("case")
            hats = original.mu_hat.to_numpy()
            finite_hats = hats[np.isfinite(hats)]
            disagree = bool(len(finite_hats) == 3 and np.any(finite_hats < branch_split)
                            and np.any(finite_hats >= branch_split))
            random, flagged = toy_id in random_set, bool(upper.loc[toy_id])
            rows.append(dict(mu_test=float(mu), toy_id=int(toy_id), is_random=random,
                             is_flagged=flagged, branch_disagreement=disagree,
                             selection_reason=("random + upper boundary" if random and flagged else
                                               "random" if random else "upper boundary"),
                             original_delta_q=float(original.loc["binned_simulator", "q"]
                                                    - original.loc["analytic_simulator", "q"]),
                             branch_split=float(branch_split), n_population=len(ids),
                             n_flagged_population=len(flags), n_flagged_selected=len(kept_flags)))
    columns = ["mu_test", "toy_id", "is_random", "is_flagged", "branch_disagreement",
               "selection_reason", "original_delta_q", "branch_split", "n_population",
               "n_flagged_population", "n_flagged_selected"]
    return pd.DataFrame(rows, columns=columns)


def _representatives(selection, n_scans):
    chosen = set()
    for mu, group in selection.groupby("mu_test", sort=True):
        ordered = group.assign(abs_delta=group.original_delta_q.abs().fillna(-1.)).sort_values(
            ["branch_disagreement", "is_flagged", "abs_delta", "toy_id"],
            ascending=[False, False, False, True])
        chosen.update((float(mu), int(toy_id)) for toy_id in ordered.head(n_scans).toy_id)
    return chosen


def _specifications(configuration, bounds, grid_size, bin_counts, integration):
    original_bins = int(configuration["n_bins"])
    specs = []
    for variant, fit_bounds, grid in (("original", configuration["mu_bounds"], configuration["grid_size"]),
                                      ("dense", configuration["mu_bounds"], grid_size),
                                      ("wide", bounds, grid_size)):
        for case in SIMULATOR_CASES:
            specs.append(dict(variant=variant, case=case, mu_bounds=list(fit_bounds), grid_size=int(grid),
                              n_bins=original_bins if case == "binned_simulator" else 0,
                              bank="original"))
    for n_bins in bin_counts:
        if n_bins != original_bins:
            specs.append(dict(variant=f"bins{n_bins}", case="binned_simulator", mu_bounds=list(bounds),
                              grid_size=int(grid_size), n_bins=n_bins, bank="original"))
        if integration is not None:
            specs.append(dict(variant=f"integrated{n_bins}", case="binned_simulator", mu_bounds=list(bounds),
                              grid_size=int(grid_size), n_bins=n_bins, bank="independent"))
    if integration is not None:
        specs.append(dict(variant="analytic_integrated", case="analytic_simulator", mu_bounds=list(bounds),
                          grid_size=int(grid_size), n_bins=0, bank="independent"))
    return specs


def _comparison_tables(refits, source, selection, configuration):
    keys = ["mu_test", "toy_id", "case"]
    original = source["results"].loc[source["results"].case.isin(SIMULATOR_CASES)]
    original = original[keys + ["q", "mu_hat", "valid", "at_upper", "n_observed"]].rename(
        columns={key: f"saved_{key}" for key in ("q", "mu_hat", "valid", "at_upper", "n_observed")})
    comparisons = refits.merge(original, on=keys, how="left", validate="many_to_one")
    comparisons["delta_q_saved"] = comparisons.q - comparisons.saved_q
    comparisons["delta_mu_hat_saved"] = comparisons.mu_hat - comparisons.saved_mu_hat
    comparisons["count_reproduced"] = comparisons.n_observed == comparisons.saved_n_observed
    good = _bool(comparisons.valid) & _bool(comparisons.saved_valid)
    comparisons["reproduction_pass"] = (
        good & comparisons.count_reproduced & np.isclose(comparisons.q, comparisons.saved_q, atol=1e-5, rtol=1e-7)
        & np.isclose(comparisons.mu_hat, comparisons.saved_mu_hat, atol=1e-5, rtol=1e-6))
    comparisons["reproduction_status"] = np.select(
        [comparisons.variant != "original", ~_bool(comparisons.saved_valid), comparisons.reproduction_pass],
        ["not original settings", "saved fit failed; no valid baseline", "matched"], default="mismatch")
    baseline = refits.loc[refits.variant == "wide", keys + ["q", "mu_hat", "valid"]].rename(
        columns={key: f"wide_{key}" for key in ("q", "mu_hat", "valid")})
    comparisons = comparisons.merge(baseline, on=keys, how="left", validate="many_to_one")
    comparisons["delta_q_wide"] = comparisons.q - comparisons.wide_q
    analytic = refits.loc[(refits.variant == "wide") & (refits.case == "analytic_simulator"),
                         ["mu_test", "toy_id", "q", "mu_hat"]].rename(
        columns={"q": "analytic_wide_q", "mu_hat": "analytic_wide_mu_hat"})
    comparisons = comparisons.merge(analytic, on=["mu_test", "toy_id"], how="left", validate="many_to_one")
    comparisons["delta_q_analytic_wide"] = comparisons.q - comparisons.analytic_wide_q
    improved = refits.loc[(refits.variant == "analytic_integrated") & (refits.case == "analytic_simulator"),
                          ["mu_test", "toy_id", "q", "mu_hat"]].rename(
        columns={"q": "analytic_integrated_q", "mu_hat": "analytic_integrated_mu_hat"})
    comparisons = comparisons.merge(improved, on=["mu_test", "toy_id"], how="left", validate="many_to_one")
    comparisons["delta_q_analytic_integrated"] = comparisons.q - comparisons.analytic_integrated_q
    comparisons["matched_analytic_q"] = np.where(comparisons.bank == "independent",
                                                 comparisons.analytic_integrated_q, comparisons.analytic_wide_q)
    comparisons["delta_q_matched_analytic"] = comparisons.q - comparisons.matched_analytic_q
    rows = []
    # Flags are deliberately over-represented, so never summarize all selected
    # experiments as an estimate of an unconditional ensemble probability.
    for (mu, case, variant), group in comparisons.loc[_bool(comparisons.is_random)].groupby(
            ["mu_test", "case", "variant"], sort=True):
        valid = group.loc[_bool(group.valid) & np.isfinite(group.q)]
        n = len(valid)
        branch_split = float(group.branch_split.iloc[0])
        upper = float(_bool(valid.at_upper).mean()) if n else np.nan
        rows.append(dict(mu_test=mu, case=case, variant=variant, n_random=len(group), n_valid=n,
                         n_failed=len(group) - n, mean_q=float(valid.q.mean()),
                         mean_delta_q_saved=float(valid.delta_q_saved.mean()),
                         mean_delta_q_analytic_wide=float(valid.delta_q_analytic_wide.mean()),
                         mean_delta_q_matched_analytic=float(valid.delta_q_matched_analytic.mean()),
                         upper_boundary_fraction=upper,
                         upper_boundary_se=float(np.sqrt(upper * (1 - upper) / n)) if n else np.nan,
                         upper_boundary_mu=float(group.mu_upper.iloc[0]),
                         high_branch_fraction=float((valid.mu_hat >= branch_split).mean()) if n else np.nan,
                         zero_q_fraction=float((valid.q <= 1e-10).mean()) if n else np.nan,
                         summary_population="random controls only"))
    return comparisons, pd.DataFrame(rows)


def run_refit_diagnostics(context, selection, *, integration=None, mu_bounds=(0., 3.),
                          grid_size=129, bin_counts=(12, 36), n_scans=3,
                          progress_every=10):
    """Reconstruct and refit a paired subset, checkpointing after each toy.

    ``integration['templates'][(n_bins, eta)]`` must hold the independent,
    highest-statistics replica-zero component yields in the *same frozen bins*.
    Their sums provide improved analytical rate integrals. Learned density
    fields and their original rate normalizers are never changed here.
    """
    from .toy_integration_diagnostics import diagnostic_bin_edges
    from .toy_sources import sample_simulator_poisson

    bounds = np.asarray(mu_bounds, float)
    config = context["configuration"]
    original_bounds = np.asarray(config["mu_bounds"], float)
    if (bounds.shape != (2,) or np.any(~np.isfinite(bounds)) or bounds[0] != original_bounds[0]
            or bounds[1] < original_bounds[1]):
        raise ValueError("Refit bounds must retain the lower bound and extend the original upper bound")
    for value, name, minimum in ((grid_size, "grid_size", int(config["grid_size"])),
                                 (n_scans, "n_scans", 0), (progress_every, "progress_every", 1)):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if any(isinstance(value, bool) or int(value) != value for value in bin_counts):
        raise ValueError("bin_counts must contain integers")
    bin_counts = tuple(int(value) for value in bin_counts)
    if int(config["n_bins"]) not in bin_counts or len(set(bin_counts)) != len(bin_counts):
        raise ValueError("Unique bin_counts must include the original bin count")
    if selection.empty or selection.duplicated(["mu_test", "toy_id"]).any():
        raise ValueError("Select at least one unique (mu_test, toy_id) before refitting")
    selection = selection.sort_values(["mu_test", "toy_id"]).reset_index(drop=True).copy()
    for key in ("is_random", "is_flagged", "branch_disagreement"):
        selection[key] = _bool(selection[key])
    mus = sorted(selection.mu_test.unique())
    if any(not bounds[0] <= mu <= bounds[1] for mu in mus):
        raise ValueError("Tested hypotheses must lie within both fit domains")
    edges = {n: diagnostic_bin_edges(n, base_bins=int(config["n_bins"])) for n in bin_counts}
    quad, rates = context["quad"], np.asarray(context["rates"])
    templates = {}
    for mu in mus:
        coordinates = reference_observable(quad["nominal"], rates, mu, config["reference_power"])
        for n, bins in edges.items():
            templates[(n, float(mu))] = histogram_components(coordinates, quad["nominal"], quad["weights"], bins)
    independent = integration["templates"] if integration is not None else {}
    if integration is not None:
        if integration.get("metadata", {}).get("source_fingerprint") != context["fingerprint"]:
            raise ValueError("Independent integration belongs to a different source context")
        for key, expected in templates.items():
            if key not in independent or np.asarray(independent[key]).shape != expected.shape:
                raise ValueError(f"Independent integration lacks compatible template {key}")
        if "edges" in integration:
            for n, bins in edges.items():
                if not np.array_equal(np.asarray(integration["edges"][n]), bins):
                    raise ValueError("Independent templates use different frozen bin edges")
    specs = _specifications(config, bounds, grid_size, bin_counts, integration)
    representatives = _representatives(selection, int(n_scans))
    identity = dict(version=1, context_fingerprint=context["fingerprint"],
                    selection_sha256=hashlib.sha256(selection.to_csv(index=False).encode()).hexdigest(),
                    specifications=specs, n_scans=int(n_scans), scan_points=321,
                    edges={str(n): bins.tolist() for n, bins in edges.items()},
                    original_templates={f"{n}:{mu}": _array_digest(value) for (n, mu), value in templates.items()},
                    independent_templates={f"{n}:{mu}": _array_digest(np.asarray(independent[n, mu]))
                                           for n, mu in templates} if integration is not None else {},
                    integration_fingerprint=(integration.get("metadata", {}).get("fingerprint")
                                             if integration is not None else None))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    output = Path(context["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    config_path, fit_path, scan_path = (output / f"refit_{suffix}" for suffix in ("config.json", "results.csv", "scans.csv"))
    if config_path.exists():
        if json.loads(config_path.read_text()).get("fingerprint") != fingerprint:
            raise ValueError("Cached refits use different inputs/selection/settings; choose a new diagnostic output tag")
    elif fit_path.exists() or scan_path.exists():
        raise ValueError("Refit checkpoint has no configuration fingerprint")
    metadata = dict(configuration=identity, fingerprint=fingerprint,
                    reconstruction_note="Original notebook 11 physical simulator stream 0; one event sample shared by all refits.",
                    normalization_note="Original reference-observable rates frozen for all banks and bins; learned normalization always frozen.",
                    scan_note="-2 log L(mu) + 2 log L(mu_test), allowed to be negative; eta fixed. Dense kappa grid is a numerical diagnostic, not proof that every minimum is resolved.",
                    sampling_note="All upper-bound cases plus independent random controls. Population summaries use is_random only.",
                    branch_note="Low/high mu_hat relative to the documented split; this operational label is not an inferred basin boundary.")
    save_json(config_path, metadata)
    _atomic_csv(selection, output / "refit_selection.csv")
    records = pd.read_csv(fit_path, keep_default_na=False).to_dict("records") if fit_path.exists() else []
    scan_records = pd.read_csv(scan_path).to_dict("records") if scan_path.exists() else []
    expected_keys = {(float(row.mu_test), int(row.toy_id), spec["case"], spec["variant"])
                     for row in selection.itertuples() for spec in specs}
    keys = [(float(row["mu_test"]), int(row["toy_id"]), row["case"], row["variant"]) for row in records]
    if len(set(keys)) != len(keys) or not set(keys) <= expected_keys:
        raise ValueError("Refit checkpoint contains duplicate or unexpected records")
    completed = set(keys)
    scan_completed = {(float(row["mu_test"]), int(row["toy_id"]), row["case"], row["variant"])
                      for row in scan_records}
    scan_variants = {"wide", f"bins{max(bin_counts)}", "analytic_integrated"}
    scan_variants.update(f"integrated{n}" for n in bin_counts)
    exposure = float(config["exposure_multiplier"])
    for number, selected in enumerate(selection.to_dict("records"), start=1):
        mu, toy_id = float(selected["mu_test"]), int(selected["toy_id"])
        needed = [spec for spec in specs if (mu, toy_id, spec["case"], spec["variant"]) not in completed]
        scan_needed = [spec for spec in specs if (mu, toy_id) in representatives
                       and spec["variant"] in scan_variants
                       and (mu, toy_id, spec["case"], spec["variant"]) not in scan_completed]
        if not needed and not scan_needed:
            continue
        likelihoods, errors, n_observed = {}, {}, np.nan
        try:
            x = sample_simulator_poisson(context["run"], mu, _rng(int(config["seed"]), mu, toy_id, 0),
                                         exposure=exposure, selector=context["selector"], threshold=context["threshold"])
            n_observed = len(x)
            analytic = context["run"].model.component_densities(x).T
            likelihoods[("analytic_simulator", 0, "original")] = StatOnlyLikelihood(analytic, rates, exposure=exposure)
            z = reference_observable(analytic, rates, mu, config["reference_power"])
            for n, bins in edges.items():
                counts = np.histogram(z, bins)[0]
                for bank, fields in (("original", templates), ("independent", independent)):
                    if (n, mu) in fields:
                        try:
                            likelihoods[("binned_simulator", n, bank)] = StatOnlyLikelihood.from_binned(
                                fields[n, mu], counts, exposure=exposure)
                        except (ValueError, FloatingPointError, RuntimeError) as error:
                            errors[("binned_simulator", n, bank)] = str(error)
            if integration is not None:
                improved_rates = np.asarray(independent[int(config["n_bins"]), mu]).sum(axis=1)
                likelihoods[("analytic_simulator", 0, "independent")] = StatOnlyLikelihood(
                    analytic, improved_rates, exposure=exposure)
            try:
                learned_events = context["workspace"].evaluate(x)[0]
                likelihoods[("learned_simulator", 0, "original")] = StatOnlyLikelihood(
                    learned_events, context["learned_rates"], exposure=exposure,
                    validity_intervals=context["learned_guard"].negative_intervals)
            except (ValueError, FloatingPointError, RuntimeError) as error:
                errors[("learned_simulator", 0, "original")] = str(error)
        except (ValueError, FloatingPointError, RuntimeError) as error:
            errors["generation"] = str(error)
        for spec in needed:
            key = (spec["case"], spec["n_bins"], spec["bank"])
            if key in likelihoods:
                result = _fit(likelihoods[key], mu, spec["mu_bounds"], spec["grid_size"])
            else:
                result = _failure(errors.get(key, errors.get("generation", "Likelihood construction failed")))
            records.append(dict(**selected, case=spec["case"], variant=spec["variant"], n_bins=spec["n_bins"],
                                bank=spec["bank"], mu_lower=spec["mu_bounds"][0], mu_upper=spec["mu_bounds"][1],
                                grid_size=spec["grid_size"], n_observed=n_observed, **result))
            completed.add((mu, toy_id, spec["case"], spec["variant"]))
        for spec in scan_needed:
            likelihood = likelihoods.get((spec["case"], spec["n_bins"], spec["bank"]))
            scan_mu = np.unique(np.r_[np.linspace(np.sqrt(bounds[0]), np.sqrt(bounds[1]), 321) ** 2, mu,
                                      original_bounds[1], bounds[1]])
            for point in scan_mu:
                try:
                    relative = likelihood.nll(float(point), reference_mu=mu) if likelihood is not None else np.nan
                except (ValueError, FloatingPointError, RuntimeError):
                    relative = np.nan
                scan_records.append(dict(mu_test=mu, toy_id=toy_id, case=spec["case"], variant=spec["variant"],
                                         mu=float(point), relative_nll=float(relative), finite=bool(np.isfinite(relative)),
                                         is_random=selected["is_random"], is_flagged=selected["is_flagged"]))
            scan_completed.add((mu, toy_id, spec["case"], spec["variant"]))
        # Writing scans first is safe if a process is interrupted between the
        # two files: missing fits regenerate, while completed scans are reused.
        if scan_records:
            _atomic_csv(pd.DataFrame(scan_records), scan_path)
        _atomic_csv(pd.DataFrame(records), fit_path)
        if number % progress_every == 0 or number == len(selection):
            print(f"Refit diagnostics: {number}/{len(selection)} paired experiments complete", flush=True)
    refits = pd.DataFrame(records)
    for column in ("q", "mu_hat", "nll", "n_observed"):
        refits[column] = pd.to_numeric(refits[column], errors="coerce")
    refits = refits.sort_values(["mu_test", "toy_id", "case", "variant"]).reset_index(drop=True)
    comparisons, summary = _comparison_tables(refits, context["source"], selection, config)
    mismatches = comparisons.loc[comparisons.reproduction_status == "mismatch"]
    if len(mismatches):
        print(f"WARNING: {len(mismatches)} original-setting fits do not reproduce notebook 11. "
              "Resolve these mismatches before interpreting changed-setting results.", flush=True)
    for suffix, frame in (("comparisons", comparisons), ("summary", summary)):
        _atomic_csv(frame, output / f"refit_{suffix}.csv")
    scans = pd.DataFrame(scan_records)
    return dict(refits=refits, comparisons=comparisons, scans=scans, summary=summary, metadata=metadata)


def plot_refit_diagnostics(study, output):
    """Plot paired effects; only random controls enter population summaries."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame, scans, summary = study["comparisons"], study["scans"], study["summary"]
    mus = sorted(frame.mu_test.unique())
    colors = dict(zip(SIMULATOR_CASES, ("black", "#d55e00", "#0072b2")))
    figures = {}
    with plt.style.context(hep.style.ATLAS):
        fig, axes = plt.subplots(1, len(mus), figsize=(7 * len(mus), 5.5), squeeze=False)
        for ax, mu in zip(axes[0], mus):
            for ci, case in enumerate(SIMULATOR_CASES):
                subset = frame.loc[(frame.mu_test == mu) & (frame.case == case)]
                pivot = subset.pivot(index="toy_id", columns="variant", values="q")
                for offset, a, b, marker in ((-.1, "dense", "original", "o"), (.1, "wide", "dense", "x")):
                    change = (pivot[a] - pivot[b]).dropna()
                    if len(change):
                        jitter = np.linspace(-.05, .05, len(change))
                        label = f"{CASE_LABELS[case]}: {'grid' if a == 'dense' else 'range'}"
                        ax.scatter(ci + offset + jitter, change, s=13, alpha=.6,
                                   color=colors[case], marker=marker, label=label)
            ax.axhline(0., color=".6", linestyle=":")
            ax.set(xticks=range(3), xticklabels=[CASE_LABELS[c] for c in SIMULATOR_CASES],
                   ylabel=r"Paired change in $q_\mu$", title=rf"$\mu_{{\rm test}}={mu:g}$; selected examples")
            ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        figures["fit_stability"] = fig

        fig, axes = plt.subplots(1, len(mus), figsize=(7 * len(mus), 5.8), squeeze=False)
        variants = [v for v in ("wide", "bins36", "integrated12", "integrated36") if v in set(frame.variant)]
        variant_colors = dict(zip(variants, ("#0072b2", "#56b4e9", "#e69f00", "#009e73")))
        for ax, mu in zip(axes[0], mus):
            for vi, variant in enumerate(variants):
                group = frame.loc[(frame.mu_test == mu) & (frame.case == "binned_simulator")
                                  & (frame.variant == variant) & _bool(frame.valid)]
                for random, marker in ((True, "o"), (False, "x")):
                    subset = group.loc[_bool(group.is_random) == random]
                    ax.scatter(subset.matched_analytic_q, subset.delta_q_matched_analytic,
                               s=15, alpha=.5, marker=marker, color=variant_colors[variant],
                               label=("12 bins, original bank" if variant == "wide" else VARIANT_LABELS.get(variant, variant))
                               if random else None)
            ax.axhline(0., color=".6", linestyle=":")
            ax.set(xlabel=r"Analytical $q_\mu$, matched rate integrals", ylabel=r"$q_\mu^{\rm binned}-q_\mu^{\rm analytical}$",
                   title=rf"$\mu_{{\rm test}}={mu:g}$; circles: random, crosses: flagged only")
            ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        figures["binning_integration"] = fig

        if not scans.empty:
            representatives = list(scans[["mu_test", "toy_id"]].drop_duplicates().itertuples(index=False, name=None))
            ncols = min(3, len(representatives))
            nrows = (len(representatives) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(6.3 * ncols, 4.8 * nrows), squeeze=False)
            for ax, (mu, toy_id) in zip(axes.flat, representatives):
                subset = scans.loc[(scans.mu_test == mu) & (scans.toy_id == toy_id)]
                for (case, variant), group in subset.groupby(["case", "variant"], sort=False):
                    group = group.sort_values("mu")
                    label = CASE_LABELS[case] if variant == "wide" else VARIANT_LABELS.get(variant, variant)
                    y = group.relative_nll.where(np.isfinite(group.relative_nll), np.nan)
                    ax.plot(group.mu, y, label=label, linewidth=1.5,
                            color=colors[case] if variant == "wide" else None,
                            linestyle="-" if variant == "wide" else "--")
                finite = subset.loc[np.isfinite(subset.relative_nll), "relative_nll"]
                lower = min(-1., float(finite.min()) * 1.1) if len(finite) else -1.
                # Show local and remote minima without letting far-away
                # hypotheses compress the diagnostically useful likelihood.
                ax.set_ylim(lower, max(8., -lower * .5))
                ax.axhline(0., color=".7", linestyle=":")
                ax.axvline(2., color=".7", linestyle=":", linewidth=1.)
                ax.set(xlabel=r"Fitted $\mu$ (fixed observable)",
                       ylabel=r"$-2\log[L(\mu)/L(\mu_{\rm test})]$",
                       title=rf"$\mu_{{\rm test}}={mu:g}$, toy {toy_id}")
                ax.legend(fontsize=8, frameon=False)
            for ax in list(axes.flat)[len(representatives):]:
                ax.set_visible(False)
            fig.tight_layout()
            figures["representative_scans"] = fig

        if not summary.empty:
            fig, axes = plt.subplots(1, len(mus), figsize=(8 * len(mus), 7.), squeeze=False)
            for ax, mu in zip(axes[0], mus):
                subset = summary.loc[summary.mu_test == mu].copy()
                labels = [f"{CASE_LABELS[row.case]} / {row.variant} (bound {row.upper_boundary_mu:g})"
                          for row in subset.itertuples()]
                ax.errorbar(subset.upper_boundary_fraction, np.arange(len(subset)),
                            xerr=subset.upper_boundary_se, fmt="o", color="black", markersize=4)
                ax.set(yticks=np.arange(len(subset)), yticklabels=labels,
                       xlabel="Upper-bound fraction, random controls only",
                       title=rf"$\mu_{{\rm test}}={mu:g}$")
                ax.tick_params(axis="y", labelsize=9)
                ax.invert_yaxis()
                extent = float((subset.upper_boundary_fraction + subset.upper_boundary_se).max())
                extent = max(.05, extent) if np.isfinite(extent) else .05
                ax.set_xlim(-.03 * extent, 1.1 * extent)
                ax.axvline(0., color=".75", linestyle=":", zorder=0)
            fig.tight_layout()
            figures["random_boundary_fractions"] = fig
        for name, fig in figures.items():
            for extension in ("png", "pdf"):
                fig.savefig(output / f"12_{name}.{extension}", dpi=160, bbox_inches="tight")
    return figures
