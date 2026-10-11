"""Fixed shifted ratio endpoints, with physical hypotheses and free fits unchanged.

The four maps are a controlled ablation: (eta,1), (eta+eps,1),
(eta,1+delta), and (eta+eps,1+delta). They are frozen before inspecting a
particular toy. The analytical MLE is used only to decompose the resulting
statistic; it is not the denominator endpoint of these observables.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import save_json
from .toy_convergence_refits import _mean_se
from .toy_family_diagnostics import _scan_selection
from .toy_likelihood import StatOnlyLikelihood
from .toy_likelihood_decomposition import _digest, _rms_se
from .toy_mle_reference import (
    _BANK_KEYS, _PAIR_KEYS, _RECORD_KEYS, _bank_templates, _fit_branches,
    _prepare_inputs, _saved_check, _scan_points, _tables, mle_reference_observable,
)
from .toy_refit_diagnostics import _bool
from .toy_study import _array_digest, _atomic_csv, _rng

_VARIANTS = ("baseline", "numerator", "denominator", "both")
_STRATA = {"all": "all", "exact_degenerate": "analytical_at_test",
           "nondegenerate": "analytical_away_from_test"}


def shifted_endpoints(eta, numerator_shift=.1, denominator_shift=.1):
    """Return map endpoints only; none of these values replaces physical eta."""
    values = np.asarray([eta, numerator_shift, denominator_shift], dtype=float)
    if not np.isfinite(values).all() or eta < 0 or numerator_shift < 0 or denominator_shift < 0:
        raise ValueError("eta and endpoint shifts must be finite and nonnegative")
    return {"baseline": (float(eta), 1.),
            "numerator": (float(eta + numerator_shift), 1.),
            "denominator": (float(eta), float(1. + denominator_shift)),
            "both": (float(eta + numerator_shift), float(1. + denominator_shift))}


def _contrasts(paired):
    """Direct paired changes relative to baseline, on common valid random toys.

    A negative mean change in absolute or squared delta-q error indicates
    improvement. The squared-error comparison also tests the direction of an
    RMS change without pretending that independent marginal errors apply.
    Ambiguous branches are excluded only from the branch-disagreement term;
    its explicit counts expose the denominator and the excluded failures.
    """
    rows = []
    random = paired.loc[_bool(paired.is_random)].copy()
    random["abs_delta_q"] = random.delta_q.abs()
    random["squared_delta_q"] = random.delta_q ** 2
    keys = _PAIR_KEYS + ["n_bins"]
    terms = ("delta_q", "abs_delta_q", "squared_delta_q", "refit_gain",
             "occupied_toy_bins", "occupied_mc_bins", "branch_disagreement")
    columns = keys + ["paired_valid", "branch_comparable", "adaptive_exact_degenerate", *terms]
    baseline = random.loc[random.variant == "baseline", columns]
    for key, group in random.loc[random.variant != "baseline"].groupby(
            _BANK_KEYS + ["variant", "model", "n_bins"], sort=True):
        matched = group.merge(baseline, on=keys, suffixes=("_shifted", "_baseline"), validate="one_to_one")
        identity = dict(zip(_BANK_KEYS + ["variant", "model", "n_bins"], key))
        for stratum in _STRATA.values():
            exact = _bool(matched.adaptive_exact_degenerate_shifted)
            part = matched if stratum == "all" else matched.loc[
                exact if stratum == "analytical_at_test" else ~exact]
            valid_mask = _bool(part.paired_valid_shifted) & _bool(part.paired_valid_baseline)
            good = part.loc[valid_mask]
            comparable = good.loc[_bool(good.branch_comparable_shifted) & _bool(good.branch_comparable_baseline)]
            rms_base, _ = _rms_se(good.delta_q_baseline)
            rms_shift, _ = _rms_se(good.delta_q_shifted)
            for term in terms:
                used = comparable if term == "branch_disagreement" else good
                left = pd.to_numeric(used[f"{term}_baseline"], errors="coerce").to_numpy(dtype=float)
                right = pd.to_numeric(used[f"{term}_shifted"], errors="coerce").to_numpy(dtype=float)
                finite = np.isfinite(left) & np.isfinite(right)
                left, right = left[finite], right[finite]
                mean, se = _mean_se(right - left)
                rows.append(dict(identity, stratum=stratum, term=term,
                                 n_random=len(part), n_valid=len(left), n_failed=len(part)-len(left),
                                 n_paired_valid=len(good), n_paired_failed=len(part)-len(good),
                                 n_branch_comparable=len(comparable), n_branch_ambiguous=len(good)-len(comparable),
                                 mean_baseline=float(left.mean()) if len(left) else np.nan,
                                 mean_shifted=float(right.mean()) if len(right) else np.nan,
                                 mean_change=mean, paired_se=se,
                                 rms_delta_q_baseline=rms_base, rms_delta_q_shifted=rms_shift,
                                 population="Random controls, same toy and bank, common valid baseline and shifted fits",
                                 interpretation="Shifted minus baseline; negative absolute/squared delta-q error is improvement; branch term uses jointly unambiguous fits"))
    return pd.DataFrame(rows)


def _shifted_tables(records):
    paired, summary, changes = _tables(records)
    summary["stratum"] = summary.stratum.map(_STRATA)
    summary["variant"] = summary.model.str.replace(r"\d+$", "", regex=True)
    endpoint_columns = records[["mu_test", "model", "mu_numerator", "mu_reference"]].drop_duplicates()
    summary = summary.merge(endpoint_columns, on=["mu_test", "model"], validate="many_to_one")
    summary["population"] = (
        "Random controls only; strata indicate eta == analytical MLE, not equality of shifted map endpoints; means conditional on valid rows")
    if len(changes):
        changes["interpretation"] = (
            "Same toys with changed integration banks; map endpoints remain fixed, while yields and analytical decomposition anchors change")
    return paired, summary, _contrasts(paired), changes


def run_shifted_reference_diagnostics(context, reference_banks, *, convergence_tag="bank_convergence",
                                      shift_tag="shifted_reference", numerator_shift=.1,
                                      denominator_shift=.1, grid_size=129, n_scans=3, progress_every=10):
    """Compare fixed endpoint ablations on authenticated existing simulator toys.

    Each bank/eta/map is histogrammed once and reused for all toys. The same
    physical toy is used across maps, 12/36 bins, and integration banks. All
    likelihoods are evaluated at physical eta and freely fitted on both saved
    branches. Normalizers, power and edges are frozen to the original study.
    Output is isolated, atomic per toy, resumable, and fingerprints all settings.
    """
    from .toy_sources import sample_simulator_poisson
    shifted_endpoints(0., numerator_shift, denominator_shift)
    for tag in (convergence_tag, shift_tag):
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
                    edges_sha256={str(n): _array_digest(bins) for n, bins in edges.items()},
                    normalizer_rates=_array_digest(context["rates"]), reference_power=context["configuration"]["reference_power"],
                    numerator_shift=float(numerator_shift), denominator_shift=float(denominator_shift),
                    endpoint_policy="baseline=(eta,1), numerator=(eta+epsilon,1), denominator=(eta,1+delta), both=(eta+epsilon,1+delta); frozen independently of toy",
                    n_scans=int(n_scans), scan_selection=scan_selection.to_dict("records"),
                    q_reproduction_atol=2e-6, q_reproduction_rtol=1e-7, numerical_atol=1e-7)
    fingerprint = _digest(identity)
    output = Path(context["output_dir"]) / "convergence" / convergence_tag / "shifted_reference" / shift_tag
    config_path, record_path, scan_path = output / "config.json", output / "records.csv", output / "scans.csv"
    if config_path.exists():
        if json.loads(config_path.read_text()).get("fingerprint") != fingerprint:
            raise ValueError("Shifted-reference diagnostic inputs changed; use a new SHIFT_TAG")
    elif record_path.exists() or scan_path.exists():
        raise ValueError("Shifted-reference diagnostics are missing their configuration fingerprint")
    metadata = dict(configuration=identity, fingerprint=fingerprint, output_dir=str(output),
                    decomposition="q_pair=D_bin(eta)-D_bin(analytical MLE); refit_gain=D_bin(analytical MLE)-min_mu D_bin(mu); delta_q=(q_pair-q_analytic)+refit_gain. The analytical anchor is NOT the observable denominator.",
                    uncertainty_note="Paired toy SE conditional on fixed banks; not integration uncertainty. Use independent-bank contrasts to assess bank sensitivity.",
                    population_note="Random controls only in summaries and contrasts; flagged toys and selected scans are diagnostic illustrations.",
                    degeneracy_note="exact_degenerate means equal map endpoints. adaptive_exact_degenerate is a compatibility field meaning physical eta == analytical MLE only, and does not imply a constant shifted map.",
                    failure_note="No prediction floor, pseudocount, empty-bin merging, endpoint jitter, anchored-q truncation, or power/edge retuning. Invalid support and saved-q disagreement remain explicit.",
                    interpretation_note="Only the observable changes. Toys and q_eta use physical eta; the likelihood denominator is a genuine free fit on both branches. Maps use analytical component ratios to isolate compression from learned-ratio calibration; endpoint settings are fixed before toy inspection.")
    save_json(config_path, metadata)
    _atomic_csv(scan_selection, output / "scan_selection.csv")
    records = pd.read_csv(record_path, float_precision="round_trip").to_dict("records") if record_path.exists() else []
    scans = pd.read_csv(scan_path, float_precision="round_trip").to_dict("records") if scan_path.exists() else []
    models = {f"{variant}{n}": (variant, n) for variant in _VARIANTS for n in edges}
    model_names = ["analytic", *models]
    expected = {(float(s.mu_test), "independent", b["replica"], b["generated_per_source"], int(s.toy_id), model)
                for s in selection.itertuples() for b in prepared for model in model_names}
    keys = [tuple(row[k] for k in _RECORD_KEYS) for row in records]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError("Shifted-reference checkpoint contains duplicate or unexpected records")
    done, template_cache = set(keys), {}
    config = context["configuration"]
    exposure, power = float(config["exposure_multiplier"]), float(config["reference_power"])
    original_counts = context["source"]["results"].query("case == 'analytic_simulator'").set_index(["mu_test", "toy_id"]).n_observed
    for number, selected in enumerate(selection.itertuples(index=False), start=1):
        mu, toy_id, split = float(selected.mu_test), int(selected.toy_id), float(selected.branch_split)
        endpoints = shifted_endpoints(mu, numerator_shift, denominator_shift)
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
        observations = {}
        for variant, (numerator, denominator) in endpoints.items():
            z = mle_reference_observable(fields, context["rates"], numerator, denominator, power)
            for n, bins in edges.items():
                observations[f"{variant}{n}"] = np.histogram(z, bins)[0]
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
            analytical_fit["valid"], analytical_fit["message"] = anchor_check["valid"], anchor_check["message"]
            anchor = analytical_fit["mu_hat"]
            common = dict(mu_test=mu, toy_id=toy_id, bank_kind="independent", replica=bank["replica"],
                          generated_per_source=bank["generated_per_source"], bank_fingerprint=bank["fingerprint"],
                          reference_fingerprint=cached["metadata"]["fingerprint"], event_sha256=event_digest,
                          n_observed=len(x), is_random=bool(selected.is_random), is_flagged=bool(selected.is_flagged),
                          branch_split=split, mu_lower=bounds[0], mu_upper=bounds[1], count_reproduced=True,
                          analytic_mu_hat=anchor, adaptive_exact_degenerate=bool(mu == anchor))
            likelihoods, templates, mc_occupancy = {"analytic": analytical}, {}, {}
            for variant, (numerator, denominator) in endpoints.items():
                cache_key = (mu, variant, bank["replica"], bank["generated_per_source"])
                if cache_key not in template_cache:
                    template_cache[cache_key] = _bank_templates(cached, context["rates"], numerator, denominator, power, edges)
                    if variant == "baseline":
                        for n, template in template_cache[cache_key][0].items():
                            if not np.allclose(template, bank["templates"][n, mu], rtol=2e-9, atol=1e-9):
                                raise ValueError("Replayed baseline histogram does not reproduce the saved section-6 templates")
                variant_templates, variant_occupancy = template_cache[cache_key]
                for n in edges:
                    model = f"{variant}{n}"
                    # Keep the previously evaluated baseline yields exactly;
                    # the recomputation above independently checks the mapping.
                    templates[model] = bank["templates"][n, mu] if variant == "baseline" else variant_templates[n]
                    mc_occupancy[model] = variant_occupancy[n]
                    likelihoods[model] = StatOnlyLikelihood.from_binned(templates[model], observations[model], exposure=exposure)
            bank_rows = []
            for model in model_names:
                variant, n_bins = ("analytic", 0) if model == "analytic" else models[model]
                numerator, denominator = (mu, 1.) if model == "analytic" else endpoints[variant]
                fit = (analytical_fit.copy() if model == "analytic" else
                       _fit_branches(likelihoods[model], mu, bounds, split, int(grid_size),
                                     extra_mu=(anchor,) if np.isfinite(anchor) else ()))
                observed, template = observations.get(model), templates.get(model)
                separation = float(abs(np.sqrt(numerator) - np.sqrt(denominator)))
                row = dict(common, model=model, variant=variant, n_bins=n_bins,
                           mu_numerator=numerator, mu_reference=denominator,
                           exact_degenerate=bool(n_bins and numerator == denominator),
                           endpoint_separation_kappa=separation, reference_separation_kappa=separation,
                           occupied_toy_bins=int(np.count_nonzero(observed)) if observed is not None else 0,
                           occupied_mc_bins=int(np.count_nonzero(template.sum(axis=0))) if template is not None else 0,
                           occupied_bank_event_bins=int(np.count_nonzero(mc_occupancy[model])) if model != "analytic" else 0,
                           empty_observed_bins=int(np.sum((observed > 0) & (template.sum(axis=0) == 0))) if template is not None else 0,
                           D_anchor=np.nan, q_pair=np.nan, refit_gain=np.nan, yield_only_q_error=np.nan,
                           saved_q=np.nan, saved_q_error=np.nan, saved_fit_available=False, saved_fit_reproduced=False, **fit)
                if model == "analytic" or variant == "baseline":
                    source_model = "analytic" if model == "analytic" else f"binned{n_bins}"
                    prior = saved.loc[(saved.replica == bank["replica"]) & (saved.generated_per_source == bank["generated_per_source"])
                                      & (saved.model == source_model)].iloc[0]
                    if model == "analytic":
                        row.update({name: anchor_check[name] for name in
                                    ("saved_q", "saved_q_error", "saved_fit_available", "saved_fit_reproduced")})
                    else:
                        _saved_check(row, prior, identity)
                if row["valid"] and not analytical_fit["valid"]:
                    row["valid"], row["message"] = False, "Analytical control is invalid; shifted comparison and decomposition cannot be authenticated"
                if row["valid"]:
                    try:
                        row["D_anchor"] = float(likelihoods[model].nll(anchor, reference_mu=1.))
                        row["q_pair"] = row["D_eta"] - row["D_anchor"]
                        row["refit_gain"] = row["D_anchor"] - min(row["D_low"], row["D_high"])
                        if (not np.isfinite(row["q_pair"] + row["refit_gain"])
                                or row["refit_gain"] < -identity["numerical_atol"]
                                or abs(row["q"] - row["q_pair"] - row["refit_gain"]) > identity["numerical_atol"]):
                            raise ValueError("Anchored pair plus free-refit gain fails its likelihood identity")
                        if row["exact_degenerate"]:
                            yield_only = StatOnlyLikelihood.from_binned(bank["rates"][:, None], np.array([len(x)]), exposure=exposure)
                            yield_fit = _fit_branches(yield_only, mu, bounds, split, int(grid_size), extra_mu=(anchor,))
                            row["yield_only_q_error"] = row["q"] - yield_fit["q"]
                            if not yield_fit["valid"] or abs(row["yield_only_q_error"]) > identity["numerical_atol"]:
                                raise ValueError("Equal-endpoint fit does not reproduce the yield-only control")
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
                        except (ValueError, FloatingPointError):
                            value = np.inf
                        scans.append(dict(common, model=row["model"], variant=row["variant"], n_bins=row["n_bins"],
                                          mu_numerator=row["mu_numerator"], mu_reference=row["mu_reference"],
                                          mu_scan=float(point), D=value, selection_reason=scan_ids[mu, toy_id]))
        _atomic_csv(pd.DataFrame(scans, columns=list(scans[0]) if scans else [*_RECORD_KEYS, "mu_scan", "D", "selection_reason"]), scan_path)
        _atomic_csv(pd.DataFrame(records), record_path)
        if number % progress_every == 0 or number == len(selection):
            print(f"Shifted-reference diagnostics: {number}/{len(selection)} saved experiments complete", flush=True)
    records = pd.DataFrame(records).sort_values(_RECORD_KEYS).reset_index(drop=True)
    records["message"] = records.message.fillna("")
    paired, summary, contrasts, changes = _shifted_tables(records)
    for name, frame in (("paired", paired), ("summary", summary), ("contrasts", contrasts), ("changes", changes)):
        _atomic_csv(frame, output / f"{name}.csv")
    return dict(records=records, paired=paired, summary=summary, contrasts=contrasts, changes=changes,
                scans=pd.DataFrame(scans), scan_selection=scan_selection, metadata=metadata)
