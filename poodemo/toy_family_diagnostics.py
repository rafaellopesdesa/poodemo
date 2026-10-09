"""Full-family and branch-preference controls for the saved simulator toys.

The two analytical coordinates I/(B+NI), S/(B+NI) retain the stat-only
parameter dependence before binning. Their binned likelihood remains a
finite-MC approximation; unsupported observations fail explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import save_json
from .toy_convergence_refits import _fraction_se, _mean_se
from .toy_likelihood import StatOnlyLikelihood
from .toy_likelihood_decomposition import _digest, _rms_se, _saved_inputs
from .toy_refit_diagnostics import _bool
from .toy_study import _array_digest, _atomic_csv, _rng, reference_observable

_BANK_KEYS = ["mu_test", "bank_kind", "replica", "generated_per_source"]
_PAIR_KEYS = _BANK_KEYS + ["toy_id"]
_RECORD_KEYS = _PAIR_KEYS + ["model"]
_TERMS = ("delta_G", "delta_common", "delta_refit", "delta_q")


def _family_fields(fields):
    """Nonnegative component representation of exactly 1 + kappa*u + kappa²*v.

    Dividing all event intensities by C=B+NI changes only a parameter-independent
    factor. Rates must remain those of the original selected intensity.
    """
    fields = np.asarray(fields, dtype=float)
    if fields.ndim != 2 or fields.shape[0] != 4 or not np.isfinite(fields).all() or (fields < 0).any():
        raise ValueError("Analytical component fields must be finite nonnegative (4,N) arrays")
    denominator = fields[2] + fields[3]
    if (denominator <= 0).any():
        raise ValueError("The full-family coordinates require B+NI > 0 at observed events")
    return np.stack((fields[0] / denominator, (fields[1] + fields[3]) / denominator,
                     np.ones_like(denominator), np.zeros_like(denominator)))


def _fit_branches(likelihood, mu_test, bounds, split, grid_size):
    """Both closed branches include the split; ties are labeled explicitly."""
    row = dict(q=np.nan, mu_hat=np.nan, D_eta=np.nan, D_low=np.nan, D_high=np.nan,
               mu_low=np.nan, mu_high=np.nan, G=np.nan, valid=False, message="",
               branch_label="invalid", ambiguous_branch=False, at_upper=False,
               optimizer_failures=0, low_at_split=False, high_at_split=False, two_interior_minima=False, both_away_from_split=False)
    try:
        if not bounds[0] < split < bounds[1]:
            raise ValueError("The branch split must lie inside the saved fit bounds")
        row["D_eta"] = float(likelihood.nll(mu_test, reference_mu=1.))
        if not np.isfinite(row["D_eta"]):
            raise ValueError("Test hypothesis has zero support or invalid intensity at observed events")
        low = likelihood.fit(mu_bounds=(bounds[0], split), grid_size=grid_size,
                             reference_mu=1., extra_mu=(mu_test, split))
        high = likelihood.fit(mu_bounds=(split, bounds[1]), grid_size=grid_size,
                              reference_mu=1., extra_mu=(mu_test, split))
        row.update(D_low=float(low["nll"]), D_high=float(high["nll"]),
                   mu_low=float(low["mu_hat"]), mu_high=float(high["mu_hat"]),
                   optimizer_failures=int(low["optimizer_failures"] + high["optimizer_failures"]))
        if not low["valid"] or not high["valid"]:
            raise ValueError(f"Branch minimization failed: low={low['message']}; high={high['message']}")
        row["low_at_split"] = bool(np.isclose(row["mu_low"], split, atol=1e-7, rtol=0.))
        row["high_at_split"] = bool(np.isclose(row["mu_high"], split, atol=1e-7, rtol=0.))
        row["both_away_from_split"] = not row["low_at_split"] and not row["high_at_split"]
        row["two_interior_minima"] = (row["both_away_from_split"]
                                       and bounds[0] + 1e-7 < row["mu_low"]
                                       and row["mu_high"] < bounds[1] - 1e-7)
        row["G"] = row["D_high"] - row["D_low"]
        winner = low if row["G"] >= 0 else high
        row["mu_hat"] = float(winner["mu_hat"])
        row["q"] = float(row["D_eta"] - min(row["D_low"], row["D_high"]))
        if row["q"] < -1e-7:
            raise ValueError("The branch union has a minimum above the explicitly included test point")
        row["q"] = max(0., row["q"])
        row["at_upper"] = bool(np.isclose(row["mu_hat"], bounds[1], atol=1e-7, rtol=0.))
        if np.isclose(row["mu_hat"], split, atol=1e-7, rtol=0.):
            row["branch_label"] = "shared_endpoint"
        elif abs(row["G"]) <= 1e-7:
            row["branch_label"] = "tie"
        else:
            row["branch_label"] = "high" if row["G"] < 0 else "low"
        row["ambiguous_branch"] = row["branch_label"] in ("tie", "shared_endpoint")
        row["valid"] = True
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        row["message"] = str(exc)
    return row


def _tables(records):
    """Paired branch-evidence decomposition; enriched extras never define rates."""
    names = ["G", "q", "mu_hat", "mu_low", "mu_high", "valid", "branch_label", "ambiguous_branch"]
    analytical = records.loc[records.model == "analytic", _PAIR_KEYS + names].rename(
        columns={name: f"analytic_{name}" for name in names})
    paired = records.loc[records.model != "analytic"].merge(analytical, on=_PAIR_KEYS, validate="many_to_one")
    paired["paired_valid"] = (_bool(paired.valid) & _bool(paired.analytic_valid)
                              & np.isfinite(paired.common_point_G) & np.isfinite(paired.G)
                              & np.isfinite(paired.analytic_G))
    good = paired.paired_valid
    paired["delta_G"] = (paired.G - paired.analytic_G).where(good)
    paired["delta_common"] = (paired.common_point_G - paired.analytic_G).where(good)
    paired["delta_refit"] = (paired.G - paired.common_point_G).where(good)
    paired["delta_q"] = (paired.q - paired.analytic_q).where(good)
    paired["decomposition_error"] = (paired.delta_G - paired.delta_common - paired.delta_refit).where(good)
    paired["branch_pair"] = "invalid"
    paired.loc[good, "branch_pair"] = paired.loc[good, "analytic_branch_label"] + " / " + paired.loc[good, "branch_label"]
    paired["branch_comparable"] = good & ~_bool(paired.ambiguous_branch) & ~_bool(paired.analytic_ambiguous_branch)
    paired["branch_disagreement"] = (paired.branch_label != paired.analytic_branch_label).where(paired.branch_comparable)
    summaries = []
    random = records.loc[_bool(records.is_random)]
    for key, group in random.groupby(_BANK_KEYS + ["model", "n_bins", "axis_bins"], sort=True):
        valid = group.loc[_bool(group.valid)]
        branch_valid = valid.loc[~_bool(valid.ambiguous_branch)]
        row = dict(zip(_BANK_KEYS + ["model", "n_bins", "axis_bins"], key))
        row.update(n_random=len(group), n_valid=len(valid), n_failed=len(group)-len(valid),
                   n_branch_valid=len(branch_valid), n_branch_ambiguous=len(valid)-len(branch_valid),
                   n_two_interior_minima=int(_bool(valid.two_interior_minima).sum()),
                   n_at_upper=int(_bool(valid.at_upper).sum()),
                   n_saved_fit_mismatch=int((_bool(group.saved_fit_available)
                                             & ~_bool(group.saved_fit_reproduced)).sum()),
                   n_empty_support=int((group.empty_observed_bins > 0).sum()),
                   max_family_identity_error=float(valid.family_identity_max_abs.max()) if len(valid) else np.nan,
                   population="random controls only; failures explicit; means conditional on valid rows")
        for name in ("q", "G"):
            row[f"mean_{name}"], row[f"mean_{name}_se"] = _mean_se(valid[name])
            row[f"rms_{name}"], row[f"rms_{name}_se"] = _rms_se(valid[name])
        row["high_branch_fraction"], row["high_branch_se"] = _fraction_se(branch_valid.branch_label == "high")
        row["zero_q_fraction"], row["zero_q_se"] = _fraction_se(valid.q <= 1e-10)
        if row["model"] != "analytic":
            match = paired.loc[_bool(paired.is_random)]
            for name in _BANK_KEYS + ["model"]:
                match = match.loc[match[name] == row[name]]
            joint = match.loc[match.paired_valid]
            row.update(n_paired_valid=len(joint), n_paired_failed=len(match)-len(joint))
            for name in _TERMS:
                row[f"mean_{name}"], row[f"mean_{name}_se"] = _mean_se(joint[name])
                row[f"rms_{name}"], row[f"rms_{name}_se"] = _rms_se(joint[name])
            comparable = match.loc[match.branch_comparable]
            row["n_branch_comparable"] = len(comparable)
            row["branch_disagreement_fraction"], row["branch_disagreement_se"] = _fraction_se(comparable.branch_disagreement)
        summaries.append(row)
    return paired, pd.DataFrame(summaries), _changes(paired)


def _changes(paired):
    rows = []
    for (mu, model), group in paired.loc[_bool(paired.is_random)].groupby(["mu_test", "model"]):
        keys = sorted(set(zip(group.replica, group.generated_per_source)))
        contrasts = []
        for replica in sorted({key[0] for key in keys}):
            chain = [key for key in keys if key[0] == replica]
            contrasts.extend(("prefix", a, b) for a, b in zip(chain[:-1], chain[1:]))
        for size in sorted({key[1] for key in keys}):
            chain = [key for key in keys if key[1] == size]
            contrasts.extend(("replica", chain[0], b) for b in chain[1:])
        for kind, left, right in contrasts:
            def subset(key):
                return group.loc[(group.replica == key[0]) & (group.generated_per_source == key[1]),
                                 ["toy_id", "paired_valid", *_TERMS]]
            joined = subset(left).merge(subset(right), on="toy_id", suffixes=("_left", "_right"), validate="one_to_one")
            good = joined.loc[_bool(joined.paired_valid_left) & _bool(joined.paired_valid_right)]
            for term in _TERMS:
                values = good[f"{term}_right"] - good[f"{term}_left"]
                mean, se = _mean_se(values)
                rms, rms_se = _rms_se(values)
                rows.append(dict(mu_test=float(mu), model=model, term=term, comparison_type=kind,
                                 from_replica=left[0], to_replica=right[0], from_size=left[1], to_size=right[1],
                                 n_random=len(joined), n_valid=len(good), n_failed=len(joined)-len(good),
                                 mean_change=mean, paired_se=se, rms_change=rms, rms_change_se=rms_se))
    return pd.DataFrame(rows)


def _scan_selection(source, n_scans):
    """Choose enriched illustrations from saved results, never summary samples."""
    if n_scans < 1:
        return pd.DataFrame(columns=["mu_test", "toy_id", "selection_reason"])
    source = source.loc[source.generated_per_source == source.generated_per_source.max()]
    a = source.loc[source.model == "analytic", _PAIR_KEYS + ["mu_hat", "q", "valid"]]
    b = source.loc[source.model == "binned36"].merge(a, on=_PAIR_KEYS, suffixes=("", "_analytic"))
    b = b.loc[_bool(b.valid) & _bool(b.valid_analytic)].copy()
    b["disagrees"] = (b.mu_hat >= b.branch_split) != (b.mu_hat_analytic >= b.branch_split)
    b["gap"] = (b.q - b.q_analytic).abs()
    rows = []
    for mu, group in b.groupby("mu_test", sort=True):
        used = set()
        switched = group.loc[group.disagrees].sort_values(["gap", "toy_id"], ascending=[False, True])
        for item in switched.itertuples():
            if item.toy_id in used:
                continue
            if len(used) >= max(0, n_scans-1):
                break
            rows.append(dict(mu_test=float(mu), toy_id=int(item.toy_id), selection_reason="large saved 1D-36 branch disagreement"))
            used.add(item.toy_id)
        same = group.loc[_bool(group.is_random) & ~group.disagrees].sort_values("toy_id")
        for item in same.itertuples():
            if item.toy_id not in used and len(used) < n_scans:
                rows.append(dict(mu_test=float(mu), toy_id=int(item.toy_id), selection_reason="random same-branch control"))
                used.add(item.toy_id)
                break
        for item in group.sort_values(["gap", "toy_id"], ascending=[False, True]).itertuples():
            if item.toy_id not in used and len(used) < n_scans:
                rows.append(dict(mu_test=float(mu), toy_id=int(item.toy_id), selection_reason="additional saved comparison"))
                used.add(item.toy_id)
    return pd.DataFrame(rows, columns=["mu_test", "toy_id", "selection_reason"])


def _prepare_inputs(context, family_banks, convergence_tag):
    largest = int(family_banks["metadata"]["configuration"]["largest_prefixes"])
    selection, source, prepared, edges, source_metadata, source_files = _saved_inputs(context, convergence_tag, largest)
    prepared = [bank for bank in prepared if bank["bank_kind"] == "independent"]
    source = source.loc[source.bank_kind == "independent"].copy()
    keys = {(b["replica"], b["generated_per_source"]) for b in prepared}
    if keys != set(family_banks["banks"]):
        raise ValueError("Family and completed section-6 banks must cover identical prefixes and replicas")
    axes = tuple(int(n) for n in family_banks["metadata"]["configuration"]["axis_bins"])
    for bank in prepared:
        family = family_banks["banks"][bank["replica"], bank["generated_per_source"]]
        meta = family["metadata"]
        if (meta.get("source_fingerprint") != context["fingerprint"]
                or meta.get("convergence_bank_fingerprint") != bank["fingerprint"]):
            raise ValueError("Family templates do not authenticate the corresponding section-6 bank")
        if not np.allclose(family["rates"], bank["rates"], rtol=2e-9, atol=1e-9):
            raise ValueError("Family templates and section-6 likelihood use different rates")
        for n in axes:
            template = np.asarray(family["templates"][n])
            if (template.shape != (4, n*n) or not np.isfinite(template).all() or (template < 0).any()
                    or not np.allclose(template.sum(axis=1), bank["rates"], rtol=2e-9, atol=1e-9)):
                raise ValueError("Family templates must be finite nonnegative arrays with matching rate sums")
    return selection, source, prepared, edges, source_metadata, source_files, axes


def run_family_diagnostics(context, family_banks, *, convergence_tag="bank_convergence",
                           family_tag="family_control", grid_size=129, n_scans=3, progress_every=10):
    """Refit both branches on saved toys with the scalar and full-family bins.

    Existing selected events, selector, observables, source fits and integration
    banks are authenticated. Checkpoints are atomic per toy. There is no learned
    ratio inference or NN retraining. Only random controls enter rate summaries.
    """
    from .toy_family_banks import family_bin_indices
    from .toy_sources import sample_simulator_poisson
    for tag in (convergence_tag, family_tag):
        if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
            raise ValueError("Tags must contain only letters, digits, underscores or hyphens")
    for value, name, minimum in ((grid_size, "grid_size", 5), (n_scans, "n_scans", 0), (progress_every, "progress_every", 1)):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    selection, source, prepared, edges, source_metadata, source_files, axes = _prepare_inputs(context, family_banks, convergence_tag)
    bounds = tuple(float(x) for x in source_metadata["configuration"]["mu_bounds"])
    scan_selection = _scan_selection(source, int(n_scans))
    scan_ids = {(float(r.mu_test), int(r.toy_id)): r.selection_reason for r in scan_selection.itertuples()}
    bank_identity = []
    for bank in prepared:
        family = family_banks["banks"][bank["replica"], bank["generated_per_source"]]
        bank_identity.append(dict(replica=bank["replica"], generated_per_source=bank["generated_per_source"],
                                  fingerprint=bank["fingerprint"], family_fingerprint=family["metadata"]["fingerprint"],
                                  templates={str(n): _array_digest(family["templates"][n]) for n in axes},
                                  edges={str(n): [_array_digest(e) for e in family["edges"][n]] for n in axes}))
    identity = dict(version=1, source_fingerprint=context["fingerprint"], source_files_sha256=source_files,
                    family_fingerprint=family_banks["metadata"]["fingerprint"], banks=bank_identity,
                    mu_bounds=list(bounds), grid_size=int(grid_size), reference_mu=1., axis_bins=list(axes),
                    n_scans=int(n_scans), scan_selection=scan_selection.to_dict("records"),
                    q_reproduction_atol=2e-6, q_reproduction_rtol=1e-7,
                    family_identity_atol=1e-7, family_identity_rtol=1e-9)
    fingerprint = _digest(identity)
    output = Path(context["output_dir"]) / "convergence" / convergence_tag / "family" / family_tag / "refits"
    config_path, record_path, scan_path = output / "config.json", output / "records.csv", output / "scans.csv"
    if config_path.exists():
        if json.loads(config_path.read_text()).get("fingerprint") != fingerprint:
            raise ValueError("Family diagnostic inputs changed; use a new FAMILY_TAG")
    elif record_path.exists() or scan_path.exists():
        raise ValueError("Family diagnostics are missing their configuration fingerprint")
    metadata = dict(configuration=identity, fingerprint=fingerprint, output_dir=str(output),
                    branch_definition="G=min_high D-min_low D; G<0 favors high. Both branches include the split; endpoint/ties are explicit.",
                    decomposition="delta_G=(D_bin(mu_high_analytic)-D_bin(mu_low_analytic)-G_analytic)+(G_bin-D_bin(mu_high_analytic)+D_bin(mu_low_analytic)).",
                    uncertainty_note="Paired toy SE conditional on fixed banks; not integration uncertainty. Prefixes share simulated events.",
                    population_note="Random controls only in population summaries. Representative scans deliberately include branch disagreements; they are not an unbiased population.",
                    failure_note="No prediction floor or empty-bin merging. Invalid support and saved-q disagreements remain explicit. New minima are reported, not forced to saved values.",
                    interpretation_note="The unbinned two-coordinate likelihood is exactly equivalent to the analytical full family; finite 2D binning and MC still lose precision. This is a branch diagnostic, not a coverage calibration.")
    save_json(config_path, metadata)
    _atomic_csv(scan_selection, output / "scan_selection.csv")
    records = pd.read_csv(record_path, float_precision="round_trip").to_dict("records") if record_path.exists() else []
    scans = pd.read_csv(scan_path, float_precision="round_trip").to_dict("records") if scan_path.exists() else []
    model_names = ["analytic", *(f"binned{n}" for n in edges), *(f"family{n}x{n}" for n in axes)]
    expected = {(float(s.mu_test), "independent", b["replica"], b["generated_per_source"], int(s.toy_id), model)
                for s in selection.itertuples() for b in prepared for model in model_names}
    keys = [tuple(row[k] for k in _RECORD_KEYS) for row in records]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError("Family checkpoint contains duplicate or unexpected records")
    done = set(keys)
    config = context["configuration"]
    exposure = float(config["exposure_multiplier"])
    original_counts = context["source"]["results"].query("case == 'analytic_simulator'").set_index(["mu_test", "toy_id"]).n_observed
    for number, selected in enumerate(selection.itertuples(index=False), start=1):
        mu, toy_id, split = float(selected.mu_test), int(selected.toy_id), float(selected.branch_split)
        wanted = {key for key in expected if key[0] == mu and key[4] == toy_id}
        scan_complete = True
        if (mu, toy_id) in scan_ids:
            # Require every scan point for every model, not merely one row per
            # model: interrupted/corrupted CSVs must not masquerade as complete.
            toy_records = [r for r in records if (r["mu_test"], r["toy_id"]) == (mu, toy_id)]
            toy_scans = [r for r in scans if (r["mu_test"], r["toy_id"]) == (mu, toy_id)]
            for bank in prepared:
                bank_records = [r for r in toy_records if (r["replica"], r["generated_per_source"])
                                == (bank["replica"], bank["generated_per_source"])]
                points = set(np.unique(np.r_[
                    np.linspace(np.sqrt(bounds[0]), np.sqrt(bounds[1]), int(grid_size)) ** 2,
                    mu, 1., split,
                    [r[k] for r in bank_records for k in ("mu_low", "mu_high") if np.isfinite(r[k])]]))
                for model in model_names:
                    subset = [r for r in toy_scans if (r["replica"], r["generated_per_source"], r["model"])
                              == (bank["replica"], bank["generated_per_source"], model)]
                    if (len(subset) != len(points) or {r["mu_scan"] for r in subset} != points
                            or any(np.isnan(r["D"]) for r in subset)):
                        scan_complete = False
        if wanted <= done and scan_complete:
            continue
        # Atomic per-toy restart also discards incomplete scan output for this toy.
        records = [r for r in records if (r["mu_test"], r["toy_id"]) != (mu, toy_id)]
        scans = [r for r in scans if (r["mu_test"], r["toy_id"]) != (mu, toy_id)]
        done -= wanted
        saved = source.loc[(source.mu_test == mu) & (source.toy_id == toy_id)]
        x = sample_simulator_poisson(context["run"], mu, _rng(int(config["seed"]), mu, toy_id, 0),
                                     exposure=exposure, selector=context["selector"], threshold=context["threshold"])
        event_digest = _array_digest(x)
        if (len(x) != original_counts.loc[mu, toy_id] or not (saved.n_observed == len(x)).all()
                or not _bool(saved.count_reproduced).all() or not (saved.event_sha256 == event_digest).all()):
            raise ValueError("Reconstructed simulator toy does not reproduce saved counts and coordinates")
        fields = context["run"].model.component_densities(x).T
        family_fields = _family_fields(fields)
        z = reference_observable(fields, context["rates"], mu, config["reference_power"])
        counts = {n: np.histogram(z, bins)[0] for n, bins in edges.items()}
        for bank in prepared:
            family = family_banks["banks"][bank["replica"], bank["generated_per_source"]]
            common = dict(mu_test=mu, toy_id=toy_id, bank_kind="independent", replica=bank["replica"],
                          generated_per_source=bank["generated_per_source"], bank_fingerprint=bank["fingerprint"],
                          family_fingerprint=family["metadata"]["fingerprint"], event_sha256=event_digest,
                          n_observed=len(x), is_random=bool(selected.is_random), is_flagged=bool(selected.is_flagged),
                          branch_split=split, mu_lower=bounds[0], mu_upper=bounds[1],
                          count_reproduced=True)
            likelihoods = {"analytic": StatOnlyLikelihood(fields, bank["rates"], exposure=exposure)}
            info = {"analytic": (0, 0, 0)}
            for n in edges:
                template = bank["templates"][n, mu]
                likelihoods[f"binned{n}"] = StatOnlyLikelihood.from_binned(template, counts[n], exposure=exposure)
                info[f"binned{n}"] = (n, n, int(np.sum((counts[n] > 0) & (template.sum(axis=0) == 0))))
            for n in axes:
                indices = family_bin_indices(fields, family["edges"][n])
                observed = np.bincount(indices, minlength=n*n)
                template = family["templates"][n]
                likelihoods[f"family{n}x{n}"] = StatOnlyLikelihood.from_binned(template, observed, exposure=exposure)
                info[f"family{n}x{n}"] = (n*n, n, int(np.sum((observed > 0) & (template.sum(axis=0) == 0))))
            bank_rows = []
            for model, likelihood in likelihoods.items():
                fit = _fit_branches(likelihood, mu, bounds, split, int(grid_size))
                n_bins, axis_bins, empty = info[model]
                row = dict(common, model=model, n_bins=n_bins, axis_bins=axis_bins, empty_observed_bins=empty,
                           common_point_G=np.nan, saved_q=np.nan, saved_q_error=np.nan,
                           saved_fit_available=False, saved_fit_reproduced=False, family_identity_max_abs=np.nan, **fit)
                if model in ("analytic", *(f"binned{n}" for n in edges)):
                    prior = saved.loc[(saved.replica == bank["replica"]) & (saved.generated_per_source == bank["generated_per_source"])
                                      & (saved.model == model)].iloc[0]
                    row["saved_fit_available"] = bool(prior.valid)
                    row["saved_q"] = float(prior.q)
                    if row["valid"] and prior.valid:
                        row["saved_q_error"] = row["q"] - float(prior.q)
                        row["saved_fit_reproduced"] = bool(np.isclose(row["q"], prior.q, atol=identity["q_reproduction_atol"],
                                                                     rtol=identity["q_reproduction_rtol"]))
                        if not row["saved_fit_reproduced"]:
                            row["valid"] = False
                            row["message"] = "Branch-union q differs from saved section-6 q; inspect potential newly resolved minimum"
                bank_rows.append(row)
            analytic_row = bank_rows[0]
            points = np.unique(np.r_[np.linspace(np.sqrt(bounds[0]), np.sqrt(bounds[1]), int(grid_size)) ** 2,
                                     mu, 1., split, [r[k] for r in bank_rows for k in ("mu_low", "mu_high") if np.isfinite(r[k])]])
            analytic_values = np.asarray([likelihoods["analytic"].nll(p, reference_mu=1.) for p in points])
            uv_likelihood = StatOnlyLikelihood(family_fields, bank["rates"], exposure=exposure)
            uv_values = np.asarray([uv_likelihood.nll(p, reference_mu=1.) for p in points])
            identity_error = float(np.max(np.abs(uv_values-analytic_values))) if len(points) else 0.
            if (not np.isfinite(analytic_values).all() or not np.isfinite(uv_values).all()
                    or not np.allclose(uv_values, analytic_values, atol=identity["family_identity_atol"], rtol=identity["family_identity_rtol"])):
                raise ValueError("Unbinned family-coordinate likelihood fails exact analytical equivalence")
            for row in bank_rows:
                row["family_identity_max_abs"] = identity_error
                if analytic_row["valid"] and row["valid"]:
                    likelihood = likelihoods[row["model"]]
                    try:
                        row["common_point_G"] = float(likelihood.nll(analytic_row["mu_high"], reference_mu=1.)
                                                       - likelihood.nll(analytic_row["mu_low"], reference_mu=1.))
                    except (ValueError, FloatingPointError):
                        row["common_point_G"] = np.nan
                records.append(row)
                done.add(tuple(row[k] for k in _RECORD_KEYS))
            if (mu, toy_id) in scan_ids:
                for model, likelihood in likelihoods.items():
                    for point in points:
                        try:
                            value = float(likelihood.nll(float(point), reference_mu=1.))
                        except (ValueError, FloatingPointError):
                            value = np.inf
                        scans.append(dict(common, model=model, mu_scan=float(point), D=value,
                                          selection_reason=scan_ids[mu, toy_id]))
        # Scans written first; a restart repeats a toy until its records commit.
        _atomic_csv(pd.DataFrame(scans, columns=list(scans[0]) if scans else [*_RECORD_KEYS, "mu_scan", "D", "selection_reason"]), scan_path)
        _atomic_csv(pd.DataFrame(records), record_path)
        if number % progress_every == 0 or number == len(selection):
            print(f"Full-family branch diagnostics: {number}/{len(selection)} saved experiments complete", flush=True)
    records = pd.DataFrame(records).sort_values(_RECORD_KEYS).reset_index(drop=True)
    records["message"] = records.message.fillna("")
    paired, summary, changes = _tables(records)
    for name, frame in (("paired", paired), ("summary", summary), ("changes", changes)):
        _atomic_csv(frame, output / f"{name}.csv")
    return dict(records=records, paired=paired, summary=summary, changes=changes,
                scans=pd.DataFrame(scans), scan_selection=scan_selection, metadata=metadata)
