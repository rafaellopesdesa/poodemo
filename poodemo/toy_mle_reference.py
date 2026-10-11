"""Toy-dependent MLE references with frozen mappings and freely fitted yields.

The reference is obtained from the analytical unbinned likelihood, independently
for each toy and integration bank. It stays fixed during the following binned
fit. This is a diagnostic statistic construction, not a conditional Poisson
sampling claim for a data-dependent partition.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.special import expit

from .data import save_json
from .inference import component_coefficients
from .toy_convergence_refits import _fraction_se, _mean_se
from .toy_family_diagnostics import _scan_selection
from .toy_likelihood import StatOnlyLikelihood
from .toy_likelihood_decomposition import _digest, _rms_se, _saved_inputs
from .toy_refit_diagnostics import _bool
from .toy_study import _array_digest, _atomic_csv, _rng, reference_observable

_BANK_KEYS = ["mu_test", "bank_kind", "replica", "generated_per_source"]
_PAIR_KEYS = _BANK_KEYS + ["toy_id"]
_RECORD_KEYS = _PAIR_KEYS + ["model"]
_TERMS = ("delta_pair", "refit_gain", "delta_q")
_CHUNK_SIZE = 200_000


def mle_reference_observable(fields, rates, eta, mu_reference, power):
    """Generalize the original shape-ratio map, with stable nearby hypotheses.

    The original selected-rate normalizers must be used for both hypotheses.
    A common positive eventwise factor in ``fields`` cancels; the same routine
    therefore acts on raw event densities and importance-weighted bank fields.
    Exact equal hypotheses return exactly 0.5, without perturbing the reference.
    """
    if mu_reference == 1.:
        return reference_observable(fields, rates, eta, power)
    fields, rates = np.asarray(fields, dtype=float), np.asarray(rates, dtype=float)
    if (not np.isfinite(eta) or eta < 0 or not np.isfinite(mu_reference) or mu_reference < 0
            or not np.isfinite(power) or power <= 0):
        raise ValueError("Hypotheses must be finite nonnegative and power finite positive")
    if fields.ndim != 2 or fields.shape[0] != 4 or rates.shape != (4,):
        raise ValueError("fields and frozen rates must have shapes (4,N) and (4,)")
    tested, reference = component_coefficients(eta), component_coefficients(mu_reference)
    density, base = tested @ fields, reference @ fields
    total, base_total = float(tested @ rates), float(reference @ rates)
    if (not np.isfinite(density).all() or (density < 0).any()
            or not np.isfinite(base).all() or (base <= 0).any()
            or not np.isfinite(total + base_total) or total <= 0 or base_total <= 0):
        raise ValueError("Reference observable requires physical rates and positive reference density")
    if eta == mu_reference:
        return np.full(fields.shape[1], .5)
    kappa, anchor = np.sqrt(eta), np.sqrt(mu_reference)
    dk = kappa - anchor
    interference = fields[1] - fields[0] - fields[2]
    relative = dk * (fields[0] * (kappa + anchor) + interference) / base
    rate_interference = rates[1] - rates[0] - rates[2]
    relative_rate = dk * (rates[0] * (kappa + anchor) + rate_interference) / base_total
    # log1p avoids cancellation near the reference. Farther away, direct logs
    # also handle exactly vanishing tested densities without rounding r to -1.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        log_ratio = np.log(density) - np.log(base)
        near = np.abs(relative) < .5
        log_ratio[near] = np.log1p(relative[near])
        log_rate = (np.log1p(relative_rate) if abs(relative_rate) < .5
                    else np.log(total) - np.log(base_total))
        return expit(power * (log_ratio - log_rate))


def _indices(z, edges):
    z, edges = np.asarray(z), np.asarray(edges)
    if (not np.isfinite(z).all() or (z < edges[0]).any() or (z > edges[-1]).any()
            or not (np.diff(edges) > 0).all()):
        raise ValueError("Observable values must lie inside strictly increasing bin edges")
    return np.minimum(np.searchsorted(edges, z, side="right") - 1, len(edges) - 2)


def _bank_templates(bank, original_rates, eta, reference, power, edges):
    """Bin the saved importance fields; do not renormalize their component rates."""
    templates = {n: np.zeros((4, n), dtype=float) for n in edges}
    occupancy = {n: np.zeros(n, dtype=np.int64) for n in edges}
    if eta == reference:
        # The exact limit is yield-only, even if observations contain one or
        # many events. No jitter or power change is made to avoid that limit.
        for n, bins in edges.items():
            index = _indices(np.array([.5]), bins)[0]
            templates[n][:, index] = bank["rates"]
            occupancy[n][index] = bank["fields"].shape[1]
        return templates, occupancy
    fields = bank["fields"]
    divisor = float(bank.get("normalization_divisor", bank["metadata"]["generated_per_source"]))
    for start in range(0, fields.shape[1], _CHUNK_SIZE):
        block = np.asarray(fields[:, start:start + _CHUNK_SIZE])
        z = mle_reference_observable(block, original_rates, eta, reference, power)
        for n, bins in edges.items():
            indices = _indices(z, bins)
            occupancy[n] += np.bincount(indices, minlength=n)
            for component in range(4):
                templates[n][component] += np.bincount(indices, weights=block[component], minlength=n)
    for template in templates.values():
        template /= divisor
        if not np.allclose(template.sum(axis=1), bank["rates"], rtol=2e-9, atol=1e-9):
            raise ValueError("MLE-reference templates do not preserve the matched integration-bank rates")
    return templates, occupancy


def _fit_branches(likelihood, mu_test, bounds, split, grid_size, extra_mu=()):
    """Free fit on both closed branches; include the analytical anchor explicitly."""
    row = dict(q=np.nan, mu_hat=np.nan, D_eta=np.nan, D_low=np.nan, D_high=np.nan,
               mu_low=np.nan, mu_high=np.nan, G=np.nan, valid=False, message="",
               branch_label="invalid", ambiguous_branch=False, at_upper=False,
               optimizer_failures=0, low_at_split=False, high_at_split=False,
               two_interior_minima=False, both_away_from_split=False)
    try:
        if not bounds[0] < split < bounds[1]:
            raise ValueError("The branch split must lie inside the saved fit bounds")
        row["D_eta"] = float(likelihood.nll(mu_test, reference_mu=1.))
        if not np.isfinite(row["D_eta"]):
            raise ValueError("Test hypothesis has zero support or invalid intensity at observed events")
        candidates = (mu_test, split, *extra_mu)
        low = likelihood.fit(mu_bounds=(bounds[0], split), grid_size=grid_size,
                             reference_mu=1., extra_mu=candidates)
        high = likelihood.fit(mu_bounds=(split, bounds[1]), grid_size=grid_size,
                              reference_mu=1., extra_mu=candidates)
        row.update(D_low=float(low["nll"]), D_high=float(high["nll"]),
                   mu_low=float(low["mu_hat"]), mu_high=float(high["mu_hat"]),
                   optimizer_failures=int(low["optimizer_failures"] + high["optimizer_failures"]))
        if not low["valid"] or not high["valid"]:
            raise ValueError(f"Branch minimization failed: low={low['message']}; high={high['message']}")
        row["low_at_split"] = bool(np.isclose(row["mu_low"], split, atol=1e-7, rtol=0.))
        row["high_at_split"] = bool(np.isclose(row["mu_high"], split, atol=1e-7, rtol=0.))
        row["both_away_from_split"] = not row["low_at_split"] and not row["high_at_split"]
        row["two_interior_minima"] = (row["both_away_from_split"] and bounds[0] + 1e-7 < row["mu_low"]
                                       and row["mu_high"] < bounds[1] - 1e-7)
        row["G"] = row["D_high"] - row["D_low"]
        winner = low if row["G"] >= 0 else high
        row["mu_hat"] = float(winner["mu_hat"])
        row["q"] = float(row["D_eta"] - min(row["D_low"], row["D_high"]))
        if row["q"] < -1e-7:
            raise ValueError("The free-fit minimum is above the explicitly included tested point")
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
    names = ["q", "G", "mu_hat", "valid", "branch_label", "ambiguous_branch"]
    analytical = records.loc[records.model == "analytic", _PAIR_KEYS + names].rename(
        columns={name: f"analytic_{name}" for name in names})
    # analytic_mu_hat already accompanies every row for reference diagnostics.
    paired = records.loc[records.model != "analytic"].drop(columns=["analytic_mu_hat"]).merge(
        analytical, on=_PAIR_KEYS, validate="many_to_one")
    paired["paired_valid"] = (_bool(paired.valid) & _bool(paired.analytic_valid)
                              & np.isfinite(paired.q_pair) & np.isfinite(paired.refit_gain))
    good = paired.paired_valid
    paired["delta_pair"] = (paired.q_pair - paired.analytic_q).where(good)
    paired["delta_q"] = (paired.q - paired.analytic_q).where(good)
    paired["decomposition_error"] = (paired.delta_q - paired.delta_pair - paired.refit_gain).where(good)
    paired["branch_pair"] = "invalid"
    paired.loc[good, "branch_pair"] = paired.loc[good, "analytic_branch_label"] + " / " + paired.loc[good, "branch_label"]
    paired["branch_comparable"] = good & ~_bool(paired.ambiguous_branch) & ~_bool(paired.analytic_ambiguous_branch)
    paired["branch_disagreement"] = (paired.branch_label != paired.analytic_branch_label).where(paired.branch_comparable)
    summaries = []
    random = records.loc[_bool(records.is_random)]
    for key, group in random.groupby(_BANK_KEYS + ["model", "n_bins"], sort=True):
        for stratum in ("all", "exact_degenerate", "nondegenerate"):
            exact = _bool(group.adaptive_exact_degenerate)
            part = group if stratum == "all" else group.loc[exact if stratum == "exact_degenerate" else ~exact]
            valid = part.loc[_bool(part.valid)]
            branch_valid = valid.loc[~_bool(valid.ambiguous_branch)]
            row = dict(zip(_BANK_KEYS + ["model", "n_bins"], key))
            row.update(stratum=stratum, n_random=len(part), n_random_all=len(group), n_valid=len(valid),
                       n_failed=len(part)-len(valid), n_branch_valid=len(branch_valid),
                       n_branch_ambiguous=len(valid)-len(branch_valid), n_at_upper=int(_bool(valid.at_upper).sum()),
                       n_exact_degenerate=int(_bool(part.exact_degenerate).sum()),
                       n_adaptive_exact_degenerate=int(_bool(part.adaptive_exact_degenerate).sum()),
                       n_empty_support=int((part.empty_observed_bins > 0).sum()),
                       n_saved_fit_mismatch=int((_bool(part.saved_fit_available) & ~_bool(part.saved_fit_reproduced)).sum()),
                       population="Random controls only; strata use eta == analytical MLE for all models; means conditional on valid rows")
            for name in ("q", "G", "q_pair", "refit_gain", "reference_separation_kappa", "occupied_toy_bins", "occupied_mc_bins"):
                row[f"mean_{name}"], row[f"mean_{name}_se"] = _mean_se(valid[name])
                row[f"rms_{name}"], row[f"rms_{name}_se"] = _rms_se(valid[name])
            row["high_branch_fraction"], row["high_branch_se"] = _fraction_se(branch_valid.branch_label == "high")
            row["exact_degenerate_fraction"], row["exact_degenerate_se"] = _fraction_se(_bool(part.exact_degenerate))
            row["zero_q_fraction"], row["zero_q_se"] = _fraction_se(valid.q <= 1e-10)
            if row["model"] != "analytic":
                match = paired.loc[_bool(paired.is_random)]
                for name in _BANK_KEYS + ["model"]:
                    match = match.loc[match[name] == row[name]]
                if stratum != "all":
                    exact_match = _bool(match.adaptive_exact_degenerate)
                    match = match.loc[exact_match if stratum == "exact_degenerate" else ~exact_match]
                joint = match.loc[match.paired_valid]
                row.update(n_paired_valid=len(joint), n_paired_failed=len(match)-len(joint))
                for name in _TERMS:
                    row[f"mean_{name}"], row[f"mean_{name}_se"] = _mean_se(joint[name])
                    row[f"rms_{name}"], row[f"rms_{name}_se"] = _rms_se(joint[name])
                comparable = match.loc[match.branch_comparable]
                row["n_branch_comparable"] = len(comparable)
                row["branch_disagreement_fraction"], row["branch_disagreement_se"] = _fraction_se(comparable.branch_disagreement)
                row["max_abs_decomposition_error"] = float(joint.decomposition_error.abs().max()) if len(joint) else np.nan
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
                                 ["toy_id", "paired_valid", "mu_reference", "analytic_mu_hat", *_TERMS]]
            joined = subset(left).merge(subset(right), on="toy_id", suffixes=("_left", "_right"), validate="one_to_one")
            good = joined.loc[_bool(joined.paired_valid_left) & _bool(joined.paired_valid_right)]
            for term in (*_TERMS, "mu_reference", "analytic_mu_hat"):
                values = good[f"{term}_right"] - good[f"{term}_left"]
                mean, se = _mean_se(values)
                rms, rms_se = _rms_se(values)
                rows.append(dict(mu_test=float(mu), model=model, term=term, comparison_type=kind,
                                 from_replica=left[0], to_replica=right[0], from_size=left[1], to_size=right[1],
                                 n_random=len(joined), n_valid=len(good), n_failed=len(joined)-len(good),
                                 mean_change=mean, paired_se=se, rms_change=rms, rms_change_se=rms_se,
                                 interpretation="Bank changes include movement of the analytical MLE and adaptive reference, as well as integrated yields"))
    return pd.DataFrame(rows)


def _prepare_inputs(context, reference_banks, convergence_tag):
    largest = int(reference_banks["metadata"]["configuration"]["largest_prefixes"])
    selection, source, prepared, edges, source_metadata, source_files = _saved_inputs(context, convergence_tag, largest)
    prepared = [bank for bank in prepared if bank["bank_kind"] == "independent"]
    source = source.loc[source.bank_kind == "independent"].copy()
    if {(b["replica"], b["generated_per_source"]) for b in prepared} != set(reference_banks["banks"]):
        raise ValueError("Reference caches and completed section-6 banks must cover identical prefixes and replicas")
    identities = []
    for bank in prepared:
        cached = reference_banks["banks"][bank["replica"], bank["generated_per_source"]]
        meta, fields = cached["metadata"], cached["fields"]
        divisor = float(cached.get("normalization_divisor", bank["generated_per_source"]))
        if (meta.get("source_fingerprint") != context["fingerprint"]
                or meta.get("convergence_bank_fingerprint") != bank["fingerprint"]):
            raise ValueError("Reference cache does not authenticate the corresponding section-6 bank")
        if fields.ndim != 2 or fields.shape[0] != 4 or divisor != bank["generated_per_source"]:
            raise ValueError("Reference bank must provide unnormalized (4,N) importance fields and its generated-per-source divisor")
        digest, summed = hashlib.sha256(), np.zeros(4)
        digest.update(str((fields.shape, str(fields.dtype))).encode())
        for start in range(0, fields.shape[1], _CHUNK_SIZE):
            block = np.asarray(fields[:, start:start + _CHUNK_SIZE])
            if not np.isfinite(block).all() or (block < 0).any():
                raise ValueError("Reference bank fields must be finite nonnegative importance weights")
            digest.update(np.ascontiguousarray(block).tobytes())
            summed += block.sum(axis=1)
        if (not np.allclose(cached["rates"], bank["rates"], rtol=2e-9, atol=1e-9)
                or not np.allclose(summed / divisor, bank["rates"], rtol=2e-9, atol=1e-9)):
            raise ValueError("Reference cache and saved likelihood must have matching component rates")
        identities.append(dict(replica=bank["replica"], generated_per_source=bank["generated_per_source"],
                               fingerprint=bank["fingerprint"], reference_fingerprint=meta["fingerprint"],
                               fields_sha256=digest.hexdigest(), normalization_divisor=divisor))
    return selection, source, prepared, edges, source_metadata, source_files, identities


def _scan_points(bounds, grid_size, mu, split, rows):
    return np.unique(np.r_[np.linspace(np.sqrt(bounds[0]), np.sqrt(bounds[1]), int(grid_size)) ** 2,
                           mu, 1., split, [r[k] for r in rows for k in ("mu_low", "mu_high", "analytic_mu_hat")
                                          if np.isfinite(r[k])]])


def _saved_check(row, prior, identity):
    row["saved_fit_available"] = bool(prior.valid)
    row["saved_q"] = float(prior.q)
    if row["valid"] and prior.valid:
        row["saved_q_error"] = row["q"] - float(prior.q)
        row["saved_fit_reproduced"] = bool(np.isclose(row["q"], prior.q,
                                                     atol=identity["q_reproduction_atol"], rtol=identity["q_reproduction_rtol"]))
        if not row["saved_fit_reproduced"]:
            row["valid"] = False
            row["message"] = "Branch-union q differs from saved section-6 q; inspect a possible newly resolved minimum"


def run_mle_reference_diagnostics(context, reference_banks, *, convergence_tag="bank_convergence",
                                  reference_tag="mle_reference", grid_size=129, n_scans=3, progress_every=10):
    """Compare fixed and per-toy MLE references with genuine freely fitted denominators.

    Only selected saved simulator toys are reconstructed. Both fit branches,
    matched integration banks, original normalizers/power/edges and physical
    observations are retained. New output is atomic per toy and resumable.
    """
    from .toy_sources import sample_simulator_poisson
    for tag in (convergence_tag, reference_tag):
        if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
            raise ValueError("Tags must contain only letters, digits, underscores or hyphens")
    for value, name, minimum in ((grid_size, "grid_size", 5), (n_scans, "n_scans", 0), (progress_every, "progress_every", 1)):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    selection, source, prepared, edges, source_metadata, source_files, bank_identity = _prepare_inputs(
        context, reference_banks, convergence_tag)
    bounds = tuple(float(x) for x in source_metadata["configuration"]["mu_bounds"])
    scan_selection = _scan_selection(source, int(n_scans))
    scan_ids = {(float(r.mu_test), int(r.toy_id)): r.selection_reason for r in scan_selection.itertuples()}
    identity = dict(version=1, source_fingerprint=context["fingerprint"], source_files_sha256=source_files,
                    reference_bank_fingerprint=reference_banks["metadata"]["fingerprint"], banks=bank_identity,
                    mu_bounds=list(bounds), grid_size=int(grid_size), bin_counts=list(edges),
                    normalizer_rates=_array_digest(context["rates"]), reference_power=context["configuration"]["reference_power"],
                    reference_policy="fixed 1 or analytical unbinned MLE, separately for each toy and bank; frozen during binned fit",
                    n_scans=int(n_scans), scan_selection=scan_selection.to_dict("records"),
                    q_reproduction_atol=2e-6, q_reproduction_rtol=1e-7, numerical_atol=1e-7)
    fingerprint = _digest(identity)
    output = Path(context["output_dir"]) / "convergence" / convergence_tag / "mle_reference" / reference_tag / "refits"
    config_path, record_path, scan_path = output / "config.json", output / "records.csv", output / "scans.csv"
    if config_path.exists():
        if json.loads(config_path.read_text()).get("fingerprint") != fingerprint:
            raise ValueError("MLE-reference diagnostic inputs changed; use a new REFERENCE_TAG")
    elif record_path.exists() or scan_path.exists():
        raise ValueError("MLE-reference diagnostics are missing their configuration fingerprint")
    metadata = dict(configuration=identity, fingerprint=fingerprint, output_dir=str(output),
                    decomposition="q_pair=D_bin(eta)-D_bin(analytical MLE); refit_gain=D_bin(analytical MLE)-min_mu D_bin(mu); delta_q=(q_pair-q_analytic)+refit_gain.",
                    uncertainty_note="Paired toy SE conditional on fixed banks; not integration uncertainty. Bank contrasts also include movement of the fitted analytical reference.",
                    population_note="Random controls only in population summaries; additional flagged toys and representative scans are diagnostic illustrations.",
                    degeneracy_note="Exact eta == analytical MLE is retained and separately summarized. Its adaptive observable is 0.5 and its free binned likelihood is yield-only; q_pair=0 does not force the freely fitted q to zero.",
                    failure_note="No prediction floor, pseudocount, empty-bin merging, reference jitter or anchored-q truncation. Invalid support and saved-q disagreement remain explicit.",
                    interpretation_note="The mapping is data-dependent but frozen within each fit. This defines an algorithmic statistic, not a conditional fixed-partition Poisson sampling model. Calibration must repeat unbinned fit, mapping and free binned refit inside every toy.")
    save_json(config_path, metadata)
    _atomic_csv(scan_selection, output / "scan_selection.csv")
    records = pd.read_csv(record_path, float_precision="round_trip").to_dict("records") if record_path.exists() else []
    scans = pd.read_csv(scan_path, float_precision="round_trip").to_dict("records") if scan_path.exists() else []
    model_names = ["analytic", *(f"fixed{n}" for n in edges), *(f"adaptive{n}" for n in edges)]
    expected = {(float(s.mu_test), "independent", b["replica"], b["generated_per_source"], int(s.toy_id), model)
                for s in selection.itertuples() for b in prepared for model in model_names}
    keys = [tuple(row[k] for k in _RECORD_KEYS) for row in records]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError("MLE-reference checkpoint contains duplicate or unexpected records")
    done, template_cache = set(keys), {}
    config = context["configuration"]
    exposure = float(config["exposure_multiplier"])
    original_counts = context["source"]["results"].query("case == 'analytic_simulator'").set_index(["mu_test", "toy_id"]).n_observed
    for number, selected in enumerate(selection.itertuples(index=False), start=1):
        mu, toy_id, split = float(selected.mu_test), int(selected.toy_id), float(selected.branch_split)
        wanted = {key for key in expected if key[0] == mu and key[4] == toy_id}
        scan_complete = True
        if (mu, toy_id) in scan_ids:
            toy_records = [r for r in records if (r["mu_test"], r["toy_id"]) == (mu, toy_id)]
            toy_scans = [r for r in scans if (r["mu_test"], r["toy_id"]) == (mu, toy_id)]
            for bank in prepared:
                bank_records = [r for r in toy_records if (r["replica"], r["generated_per_source"])
                                == (bank["replica"], bank["generated_per_source"])]
                points = set(_scan_points(bounds, grid_size, mu, split, bank_records))
                for model in model_names:
                    subset = [r for r in toy_scans if (r["replica"], r["generated_per_source"], r["model"])
                              == (bank["replica"], bank["generated_per_source"], model)]
                    if (len(subset) != len(points) or {r["mu_scan"] for r in subset} != points
                            or any(np.isnan(r["D"]) for r in subset)):
                        scan_complete = False
        if wanted <= done and scan_complete:
            continue
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
        fixed_z = reference_observable(fields, context["rates"], mu, config["reference_power"])
        fixed_counts = {n: np.histogram(fixed_z, bins)[0] for n, bins in edges.items()}
        for bank in prepared:
            cached = reference_banks["banks"][bank["replica"], bank["generated_per_source"]]
            analytical = StatOnlyLikelihood(fields, bank["rates"], exposure=exposure)
            analytical_fit = _fit_branches(analytical, mu, bounds, split, int(grid_size))
            prior_analytical = saved.loc[(saved.replica == bank["replica"])
                                         & (saved.generated_per_source == bank["generated_per_source"])
                                         & (saved.model == "analytic")].iloc[0]
            anchor_check = dict(analytical_fit, saved_q=np.nan, saved_q_error=np.nan,
                                saved_fit_available=False, saved_fit_reproduced=False)
            _saved_check(anchor_check, prior_analytical, identity)
            # An unverified analytical fit must not seed apparently valid
            # adaptive rows or their marginal branch-fraction summaries.
            analytical_fit["valid"] = anchor_check["valid"]
            analytical_fit["message"] = anchor_check["message"]
            anchor = analytical_fit["mu_hat"]
            common = dict(mu_test=mu, toy_id=toy_id, bank_kind="independent", replica=bank["replica"],
                          generated_per_source=bank["generated_per_source"], bank_fingerprint=bank["fingerprint"],
                          reference_fingerprint=cached["metadata"]["fingerprint"], event_sha256=event_digest,
                          n_observed=len(x), is_random=bool(selected.is_random), is_flagged=bool(selected.is_flagged),
                          branch_split=split, mu_lower=bounds[0], mu_upper=bounds[1], count_reproduced=True,
                          analytic_mu_hat=anchor, adaptive_exact_degenerate=bool(mu == anchor))
            likelihoods = {"analytic": analytical}
            observations, templates = {}, {}
            for n in edges:
                model = f"fixed{n}"
                observations[model], templates[model] = fixed_counts[n], bank["templates"][n, mu]
                likelihoods[model] = StatOnlyLikelihood.from_binned(templates[model], observations[model], exposure=exposure)
            adaptive_error = ""
            if analytical_fit["valid"]:
                try:
                    cache_key = (mu, anchor, bank["replica"], bank["generated_per_source"])
                    if cache_key not in template_cache:
                        template_cache[cache_key] = _bank_templates(cached, context["rates"], mu, anchor,
                                                                   config["reference_power"], edges)
                    adaptive_templates, _ = template_cache[cache_key]
                    adaptive_z = mle_reference_observable(fields, context["rates"], mu, anchor, config["reference_power"])
                    for n, bins in edges.items():
                        model = f"adaptive{n}"
                        observations[model] = np.histogram(adaptive_z, bins)[0]
                        templates[model] = adaptive_templates[n]
                        likelihoods[model] = StatOnlyLikelihood.from_binned(templates[model], observations[model], exposure=exposure)
                except (ValueError, FloatingPointError) as exc:
                    adaptive_error = str(exc)
            else:
                adaptive_error = "Analytical unbinned MLE is invalid, so no adaptive reference can be constructed"
            bank_rows = []
            for model in model_names:
                n_bins = 0 if model == "analytic" else int(model.removeprefix("fixed").removeprefix("adaptive"))
                ref = anchor if model.startswith("adaptive") else 1.
                fit = (analytical_fit.copy() if model == "analytic" else
                       _fit_branches(likelihoods[model], mu, bounds, split, int(grid_size), extra_mu=(anchor,))
                       if model in likelihoods else
                       dict(analytical_fit, valid=False, message=adaptive_error, q=np.nan,
                            mu_hat=np.nan, D_eta=np.nan, D_low=np.nan, D_high=np.nan, G=np.nan,
                            mu_low=np.nan, mu_high=np.nan, branch_label="invalid", ambiguous_branch=False))
                observed, template = observations.get(model), templates.get(model)
                row = dict(common, model=model, n_bins=n_bins, mu_reference=ref,
                           exact_degenerate=bool(n_bins and mu == ref),
                           reference_separation_kappa=float(abs(np.sqrt(mu)-np.sqrt(ref))) if np.isfinite(ref) else np.nan,
                           occupied_toy_bins=int(np.count_nonzero(observed)) if observed is not None else 0,
                           occupied_mc_bins=int(np.count_nonzero(template.sum(axis=0))) if template is not None else 0,
                           empty_observed_bins=int(np.sum((observed > 0) & (template.sum(axis=0) == 0))) if template is not None else 0,
                           D_anchor=np.nan, q_pair=np.nan, refit_gain=np.nan, yield_only_q_error=np.nan,
                           saved_q=np.nan, saved_q_error=np.nan, saved_fit_available=False, saved_fit_reproduced=False, **fit)
                if model == "analytic" or model.startswith("fixed"):
                    source_model = "analytic" if model == "analytic" else model.replace("fixed", "binned")
                    prior = saved.loc[(saved.replica == bank["replica"]) & (saved.generated_per_source == bank["generated_per_source"])
                                      & (saved.model == source_model)].iloc[0]
                    if model == "analytic":
                        row.update({name: anchor_check[name] for name in
                                    ("saved_q", "saved_q_error", "saved_fit_available", "saved_fit_reproduced")})
                    else:
                        _saved_check(row, prior, identity)
                if row["valid"] and analytical_fit["valid"]:
                    try:
                        row["D_anchor"] = float(likelihoods[model].nll(anchor, reference_mu=1.))
                        row["q_pair"] = row["D_eta"] - row["D_anchor"]
                        row["refit_gain"] = row["D_anchor"] - min(row["D_low"], row["D_high"])
                        if (not np.isfinite(row["q_pair"] + row["refit_gain"])
                                or row["refit_gain"] < -identity["numerical_atol"]
                                or abs(row["q"]-row["q_pair"]-row["refit_gain"]) > identity["numerical_atol"]):
                            raise ValueError("Anchored pair plus free-refit gain fails its likelihood identity")
                        if model.startswith("adaptive") and row["exact_degenerate"]:
                            yield_only = StatOnlyLikelihood.from_binned(bank["rates"][:, None], np.array([len(x)]), exposure=exposure)
                            yield_fit = _fit_branches(yield_only, mu, bounds, split, int(grid_size), extra_mu=(anchor,))
                            row["yield_only_q_error"] = row["q"] - yield_fit["q"]
                            if (not yield_fit["valid"] or abs(row["yield_only_q_error"]) > identity["numerical_atol"]
                                    or row["q_pair"] != 0.):
                                raise ValueError("Exact-degenerate adaptive fit does not reproduce the yield-only control")
                    except (ValueError, FloatingPointError) as exc:
                        row["valid"], row["message"] = False, str(exc)
                bank_rows.append(row)
            points = _scan_points(bounds, grid_size, mu, split, bank_rows)
            for row in bank_rows:
                records.append(row)
                done.add(tuple(row[k] for k in _RECORD_KEYS))
            if (mu, toy_id) in scan_ids:
                for row in bank_rows:
                    for point in points:
                        try:
                            value = float(likelihoods[row["model"]].nll(float(point), reference_mu=1.))
                        except (KeyError, ValueError, FloatingPointError):
                            value = np.inf
                        scans.append(dict(common, model=row["model"], mu_reference=row["mu_reference"],
                                          mu_scan=float(point), D=value, selection_reason=scan_ids[mu, toy_id]))
        _atomic_csv(pd.DataFrame(scans, columns=list(scans[0]) if scans else [*_RECORD_KEYS, "mu_scan", "D", "selection_reason"]), scan_path)
        _atomic_csv(pd.DataFrame(records), record_path)
        if number % progress_every == 0 or number == len(selection):
            print(f"MLE-reference diagnostics: {number}/{len(selection)} saved experiments complete", flush=True)
    records = pd.DataFrame(records).sort_values(_RECORD_KEYS).reset_index(drop=True)
    records["message"] = records.message.fillna("")
    paired, summary, changes = _tables(records)
    for name, frame in (("paired", paired), ("summary", summary), ("changes", changes)):
        _atomic_csv(frame, output / f"{name}.csv")
    return dict(records=records, paired=paired, summary=summary, changes=changes,
                scans=pd.DataFrame(scans), scan_selection=scan_selection, metadata=metadata)
