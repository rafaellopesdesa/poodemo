"""Replay authenticated integration banks for a full-family 2D control.

The raw (I/(B+NI), S/(B+NI)) coordinates retain the stat-only family before
binning. A fixed weighted-quantile grid is chosen from the original pilot bank,
independently of the diagnostic toys and integration replicas. These finite
histograms need not be sufficient, and empty cells are never regularized here.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .inference import component_coefficients
from .toy_convergence_banks import (
    _SOURCES, _accumulate_analytic, _combine, _digest, _file_digest,
    _integer, _load_state, _original_configuration,
    load_completed_convergence_banks,
)
from .toy_integration_diagnostics import _atomic_csv, _atomic_json, _atomic_npz, _moments


def family_coordinates(fields):
    """Return raw ``u=I/(B+NI), v=S/(B+NI)`` from S/SBI/B/NI intensities."""
    fields = np.asarray(fields, dtype=float)
    if fields.ndim != 2 or fields.shape[0] != 4 or not np.isfinite(fields).all():
        raise ValueError("Family coordinates require finite component fields of shape (4, n)")
    denominator = fields[2] + fields[3]
    if np.any(denominator <= 0):
        raise ValueError("Family coordinates require strictly positive B+NI")
    u = (fields[1] - fields[0] - fields[2]) / denominator
    v = fields[0] / denominator
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("Nonfinite full-family coordinates")
    return u, v


def _validate_edges(edges):
    if len(edges) != 2:
        raise ValueError("Family edges must contain the u and v boundary arrays")
    result = []
    for boundary in edges:
        boundary = np.asarray(boundary, dtype=float)
        if (boundary.ndim != 1 or len(boundary) < 3
                or boundary[0] != -np.inf or boundary[-1] != np.inf
                or not np.isfinite(boundary[1:-1]).all()
                or not np.all(np.diff(boundary) > 0)):
            raise ValueError("Family edges need strictly increasing finite interiors and infinite exterior edges")
        result.append(boundary)
    return tuple(result)


def family_bin_indices(fields, edges):
    """Return row-major cell indices ``u_bin * n_v + v_bin``; no clipping."""
    u_edges, v_edges = _validate_edges(edges)
    u, v = family_coordinates(fields)
    iu = np.searchsorted(u_edges, u, side="right") - 1
    iv = np.searchsorted(v_edges, v, side="right") - 1
    return iu * (len(v_edges) - 1) + iv


def _array_digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _pilot_edges(context, axis_bins):
    bins = tuple(_integer(n, "axis bin count", 2) for n in axis_bins)
    if not bins or bins != tuple(sorted(set(bins))):
        raise ValueError("axis_bins must be nonempty, unique, and increasing")
    finest = bins[-1]
    if any(finest % n for n in bins):
        raise ValueError("Each axis bin count must divide the finest grid for nested binning")
    fields = np.asarray(context["quad"]["nominal"], dtype=float)
    weights = np.asarray(context["quad"]["weights"], dtype=float)
    u, v = family_coordinates(fields)
    if weights.shape != u.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Pilot integration weights must be finite, nonnegative, and match the fields")
    weights = weights * (fields[2] + fields[3])
    positive = weights > 0
    if not np.any(positive) or not np.isfinite(weights).all():
        raise ValueError("The pilot B+NI weighted measure must be finite and positive")
    probabilities = np.arange(1, finest, dtype=float) / finest
    finest_edges = []
    for name, coordinates in (("u", u), ("v", v)):
        order = np.argsort(coordinates[positive], kind="stable")
        values, mass = coordinates[positive][order], weights[positive][order]
        # Interpolation of midpoint weighted ranks. The very same finest-grid
        # boundaries are sliced for coarser grids, making nesting exact.
        rank = (np.cumsum(mass) - .5 * mass) / np.sum(mass)
        interiors = np.interp(probabilities, rank, values)
        edges = np.r_[-np.inf, interiors, np.inf]
        if not np.all(np.diff(edges) > 0):
            raise ValueError(f"Pilot {name} weighted quantiles have tied boundaries; reduce axis_bins "
                             "or supply a larger pilot bank. Ties are not silently jittered or merged.")
        finest_edges.append(edges)
    edges = {n: tuple(boundaries[::finest // n].copy() for boundaries in finest_edges) for n in bins}
    for pair in edges.values():
        _validate_edges(pair)
    pilot_sha = _array_digest(dict(fields=fields, weights=np.asarray(context["quad"]["weights"], dtype=float)))
    return edges, pilot_sha


def _blank(specs, edges):
    state = dict(rate_sum=np.zeros(4), rate_second=np.zeros((4, 4)), selected=np.array(0))
    for k, spec in enumerate(specs):
        n = spec["n_bins"]
        state[f"bin_sum_{k}"] = np.zeros((4, n))
        state[f"bin_second_{k}"] = np.zeros((4, 4, n))
    for n in edges:
        state[f"family_sum_{n}"] = np.zeros((4, n * n))
        state[f"family_second_{n}"] = np.zeros((4, 4, n * n))
        state[f"family_count_{n}"] = np.zeros(n * n, dtype=np.int64)
    return state


def _accumulate(state, fields, inverse_proposal, specs, edges, rates, power):
    _accumulate_analytic(state, fields, inverse_proposal, specs, rates, power)
    values = fields * inverse_proposal
    for n, pair in edges.items():
        indices = family_bin_indices(fields, pair)
        state[f"family_count_{n}"] += np.bincount(indices, minlength=n * n)
        for p in range(4):
            state[f"family_sum_{n}"][p] += np.bincount(indices, weights=values[p], minlength=n * n)
            for q in range(p, 4):
                second = np.bincount(indices, weights=values[p] * values[q], minlength=n * n)
                state[f"family_second_{n}"][p, q] += second
                if p != q:
                    state[f"family_second_{n}"][q, p] += second


def _save(path, state, rng, completed, fingerprint):
    arrays = dict(state, completed=np.array(completed), rng_state=np.array(json.dumps(rng.bit_generator.state)))
    _atomic_npz(path, **arrays, fingerprint=np.array(fingerprint), payload_sha256=np.array(_array_digest(arrays)))


def _load(path, fingerprint, specs, edges, expected_size=None):
    blank = _blank(specs, edges)
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["fingerprint"]) != fingerprint:
            raise ValueError("Family replay checkpoint fingerprint changed; use another family_tag")
        state = {key: saved[key].copy() for key in blank}
        completed = int(saved["completed"])
        rng_state = json.loads(str(saved["rng_state"]))
        arrays = dict(state, completed=saved["completed"], rng_state=saved["rng_state"])
        if str(saved["payload_sha256"]) != _array_digest(arrays):
            raise ValueError(f"Family replay checkpoint payload changed: {path.name}")
    if completed < 1 or (expected_size is not None and completed != expected_size):
        raise ValueError(f"Invalid generated count in family replay checkpoint {path.name}")
    if any(value.shape != blank[key].shape or not np.isfinite(value).all() for key, value in state.items()):
        raise ValueError(f"Malformed family replay moments in {path.name}")
    selected = int(state["selected"])
    if not 0 <= selected <= completed:
        raise ValueError(f"Invalid selected count in family replay checkpoint {path.name}")
    for n in edges:
        count = state[f"family_count_{n}"]
        if np.any(count < 0) or not np.issubdtype(count.dtype, np.integer) or count.sum() != selected:
            raise ValueError(f"Invalid family occupancy in {path.name}")
        if not np.allclose(state[f"family_sum_{n}"].sum(axis=1), state["rate_sum"], rtol=1e-11, atol=1e-12):
            raise ValueError(f"Family template does not reproduce selected rates in {path.name}")
    return state, rng_state, completed


def _reproduction(state, rng, saved_state, saved_rng, *, replica, source, size):
    keys = list(saved_state)
    good = all(np.allclose(state[key], saved_state[key], rtol=2e-11, atol=2e-12) for key in keys)
    rng_match = rng.bit_generator.state == saved_rng
    selected_match = int(state["selected"]) == int(saved_state["selected"])
    if not good or not rng_match or not selected_match:
        raise ValueError(f"Exact-bank replay failed for replica {replica + 1}, {source}, {size}: "
                         f"moments={good}, RNG={rng_match}, selected count={selected_match}. "
                         "Do not interpret the 2D control until the frozen selector/model and numerical environment reproduce section 6.")
    max_rate = float(np.max(np.abs(state["rate_sum"] - saved_state["rate_sum"])))
    bin_keys = [key for key in keys if key.startswith("bin_sum_")]
    max_bin = max(float(np.max(np.abs(state[key] - saved_state[key]))) for key in bin_keys)
    return dict(replica=replica, source=source, generated_per_source=size,
                selected=int(state["selected"]), moments_reproduced=good,
                rng_reproduced=rng_match, selected_reproduced=selected_match,
                maximum_rate_sum_difference=max_rate, maximum_1d_sum_difference=max_bin)


def _combine_family(states, size, edges, expected, metadata, replica):
    bank = dict(rates=expected["rates"].copy(), rate_covariance=expected["rate_covariance"].copy(),
                templates={}, component_covariances={}, occupancy={}, edges=edges)
    for n in edges:
        template, covariance, count = np.zeros((4, n * n)), np.zeros((4, 4, n * n)), np.zeros(n * n, dtype=np.int64)
        for state in states:
            mean, second = _moments(state[f"family_sum_{n}"], state[f"family_second_{n}"], size)
            template += mean
            covariance += second
            count += state[f"family_count_{n}"]
        if not np.allclose(template.sum(axis=1), bank["rates"], rtol=2e-11, atol=2e-12):
            raise ValueError("Replayed family template sums do not match the same-bank analytical rates")
        bank["templates"][n], bank["component_covariances"][n], bank["occupancy"][n] = template, covariance, count
    finest = max(edges)
    for n in edges:
        factor = finest // n
        fine = bank["templates"][finest].reshape(4, n, factor, n, factor).sum(axis=(2, 4)).reshape(4, n * n)
        if not np.allclose(fine, bank["templates"][n], rtol=2e-11, atol=2e-12):
            raise ValueError("Family template refinements are not nested")
    arrays = {f"{kind}_{n}": bank[kind][n] for kind in ("templates", "component_covariances", "occupancy") for n in edges}
    item = dict(source_fingerprint=metadata["source_fingerprint"], family_fingerprint=metadata["fingerprint"],
                convergence_bank_fingerprint=expected["metadata"]["fingerprint"], replica=replica,
                generated_per_source=size, selected=sum(int(s["selected"]) for s in states),
                prediction_sha256=_array_digest(arrays))
    item["fingerprint"] = _digest(item)
    bank["metadata"] = item
    return bank


def _support(banks, configuration):
    rows = []
    exposure = configuration["exposure_multiplier"]
    for (replica, size), bank in banks.items():
        for n, template in bank["templates"].items():
            for mu in configuration["mu_values"]:
                coefficients = component_coefficients(mu)
                prediction = exposure * (coefficients @ template)
                variance = exposure**2 * np.einsum("i,ijb,j->b", coefficients, bank["component_covariances"][n], coefficients)
                # Roundoff from cancellation of component covariances may
                # make an exactly-zero variance slightly negative.
                scale = exposure**2 * np.einsum("i,ijb,j->b", np.abs(coefficients),
                    np.abs(bank["component_covariances"][n]), np.abs(coefficients))
                if np.any(variance < -1e-10 * np.maximum(scale, np.finfo(float).tiny)):
                    raise ValueError("Negative physical MC variance in family template")
                se = np.sqrt(np.maximum(variance, 0.))
                relative = np.divide(se, prediction, out=np.full_like(se, np.nan), where=prediction > 0)
                effective = np.divide(prediction**2, variance, out=np.full_like(se, np.nan), where=variance > 0)
                for cell in range(n * n):
                    rows.append(dict(replica=replica, generated_per_source=size, n_axis=n,
                        bin_index=cell, u_bin=cell // n, v_bin=cell % n, mu=float(mu),
                        occupancy=int(bank["occupancy"][n][cell]), predicted_yield=prediction[cell],
                        mc_se=se[cell], relative_mc_se=relative[cell], effective_entries=effective[cell],
                        empty_mc=bank["occupancy"][n][cell] == 0, nonpositive_yield=prediction[cell] <= 0))
    return pd.DataFrame(rows)


def prepare_family_banks(context, *, convergence_tag="bank_convergence", family_tag="family_control",
                         axis_bins=(12, 36), largest_prefixes=2, progress_every=10):
    """Replay section 6's exact streams and build matched finite 2D histograms.

    Replays all saved prefix boundaries, which is necessary to reproduce draws
    from component samplers whose RNG consumption depends on the chunk size.
    It evaluates only the frozen selector and analytical densities. Every saved
    prefix must reproduce its RNG state, selected count, rates, and 1D moments.
    The original snapshots are read-only. Per-chunk checkpoints make this extra
    proposal pass resumable; a completed call performs no new sampling.

    Returned ``banks`` and ``convergence_banks`` have matching (replica, size)
    keys for the largest requested prefixes. The former has square templates
    indexed by axis-bin count; the latter is the original section-6 1D control.
    """
    if not isinstance(family_tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", family_tag):
        raise ValueError("family_tag must contain only letters, digits, underscores or hyphens")
    largest_prefixes = _integer(largest_prefixes, "largest_prefixes")
    progress_every = _integer(progress_every, "progress_every")
    completed = load_completed_convergence_banks(context, convergence_tag=convergence_tag)
    edges, pilot_sha = _pilot_edges(context, axis_bins)
    original_dir = Path(context["output_dir"]) / "integration"
    convergence_dir = Path(context["output_dir"]) / "convergence" / convergence_tag / "integration"
    original, specs = _original_configuration(context, original_dir, completed["metadata"]["n_bins"])
    sizes = completed["metadata"]["sizes"]
    if largest_prefixes > len(sizes):
        raise ValueError("largest_prefixes exceeds the number of completed integration prefixes")
    chosen = sizes[-largest_prefixes:]
    expected = {key: bank for key, bank in completed["banks"].items() if key[1] in chosen}
    source_paths, source_hashes = {}, {}
    for replica in range(original["replicas"]):
        for size in sizes:
            for source in _SOURCES:
                is_original = size in original["sizes"]
                path = (original_dir if is_original else convergence_dir) / f"replica_{replica}_{source}_{size}.npz"
                source_paths[replica, source, size] = (path, is_original)
                source_hashes[f"{replica}:{source}:{size}"] = _file_digest(path)
    configuration = dict(convergence_tag=convergence_tag, family_tag=family_tag,
        axis_bins=list(edges), largest_prefixes=largest_prefixes, all_prefixes=sizes,
        selected_prefixes=chosen, replicas=original["replicas"], seed=original["seed"],
        chunk_size=original["chunk_size"], exposure_multiplier=original["exposure_multiplier"],
        mu_values=original["mu_values"])
    metadata = dict(version=1, source_fingerprint=context["fingerprint"], configuration=configuration,
        convergence_fingerprint=completed["metadata"]["fingerprint"],
        convergence_bank_fingerprints={f"{r}:{n}": b["metadata"]["fingerprint"] for (r, n), b in expected.items()},
        source_files_sha256=source_hashes, pilot_sha256=pilot_sha,
        # Null exteriors mean -infinity and +infinity, respectively; JSON never
        # contains NaN/Infinity. Runtime edge arrays retain actual infinities.
        edges={str(n): [[None] + pair[j][1:-1].tolist() + [None] for j in (0, 1)] for n, pair in edges.items()},
        note="Fixed original-pilot B+NI weighted quantiles; exact replay of section-6 generated streams. Raw u,v are sufficient before binning; finite 2D bins and finite MC require diagnostics. Empty cells are never smoothed.")
    metadata["fingerprint"] = _digest(metadata)
    directory = Path(context["output_dir"]) / "convergence" / convergence_tag / "family" / family_tag / "integration"
    manifest = directory / "configuration.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("Family bank configuration or saved inputs changed; use another family_tag")
    directory.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        _atomic_json(manifest, metadata)
    snapshots, closure_rows = {}, []
    model, selector = context["run"].model, context["selector"]
    chunks = 0
    for replica in range(original["replicas"]):
        for source_id, source in enumerate(_SOURCES):
            state, count = _blank(specs, edges), 0
            rng = np.random.default_rng(np.random.SeedSequence([original["seed"], 120524, replica, source_id]))
            work = directory / f"replica_{replica}_{source}_work.npz"
            for size in sizes:
                path = directory / f"replica_{replica}_{source}_{size}.npz"
                if path.exists():
                    state, rng_state, count = _load(path, metadata["fingerprint"], specs, edges, size)
                    rng.bit_generator.state = rng_state
                else:
                    if work.exists():
                        with np.load(work, allow_pickle=False) as saved:
                            work_size = int(saved["completed"])
                        if count <= work_size <= size:
                            state, rng_state, count = _load(work, metadata["fingerprint"], specs, edges)
                            rng.bit_generator.state = rng_state
                    while count < size:
                        n = min(original["chunk_size"], size - count)
                        x = model.sample_component(source, n, rng)
                        if selector is not None:
                            x = x[selector.predict_proba(x)[:, 0] >= context["threshold"]]
                        if len(x):
                            proposal = sum(model.component_pdf(x, p) for p in _SOURCES)
                            if np.any(proposal <= 0) or not np.isfinite(proposal).all():
                                raise ValueError("Invalid mixture proposal density during family replay")
                            _accumulate(state, model.component_densities(x).T, 1. / proposal, specs, edges,
                                        context["rates"], original["reference_power"])
                        count += n
                        _save(work, state, rng, count, metadata["fingerprint"])
                        chunks += 1
                        if chunks % progress_every == 0:
                            print(f"Family bank replay {replica + 1}/{original['replicas']}, {source}: {count:,}/{sizes[-1]:,} generated", flush=True)
                source_path, is_original = source_paths[replica, source, size]
                source_fingerprint = original["fingerprint"] if is_original else completed["metadata"]["fingerprint"]
                original_state, original_rng = _load_state(source_path, source_fingerprint, specs, size, original=is_original)
                closure_rows.append(_reproduction(state, rng, original_state, original_rng,
                    replica=replica, source=source, size=size))
                if not path.exists():
                    _save(path, state, rng, count, metadata["fingerprint"])
                if size in chosen:
                    snapshots[replica, source, size] = {key: value.copy() for key, value in state.items()}
    banks = {}
    for (replica, size), expected_bank in expected.items():
        states = [snapshots[replica, source, size] for source in _SOURCES]
        scalar = _combine(states, size, specs)
        if not np.allclose(scalar["rates"], expected_bank["rates"], rtol=2e-11, atol=2e-12):
            raise ValueError("Replayed analytical rates do not reproduce the saved bank")
        for key, prediction in scalar["templates"].items():
            if not np.allclose(prediction, expected_bank["templates"][key], rtol=2e-11, atol=2e-12):
                raise ValueError("Replayed 1D template does not reproduce the saved bank")
        banks[replica, size] = _combine_family(states, size, edges, expected_bank, metadata, replica)
    closure = pd.DataFrame(closure_rows)
    support = _support(banks, configuration)
    _atomic_csv(directory / "replay_closure.csv", closure)
    _atomic_csv(directory / "cell_support.csv", support)
    return dict(banks=banks, convergence_banks=expected, metadata=metadata, closure=closure, support=support)
