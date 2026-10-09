"""Continue notebook-12 integration banks without touching their saved results.

The original streams contain both analytical and learned moments. Only their
analytical sufficient statistics are continued here: no ratio-network inference
or re-normalization is needed. The original frozen selector and observable are
used throughout, and rejected events remain zeros in each fixed-size stratum.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .toy_integration_diagnostics import (
    _atomic_json, _atomic_npz, _bin_indices, _moments, diagnostic_bin_edges,
)
from .toy_study import reference_observable


_SOURCES = ("S", "B", "NI")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _original_configuration(context, directory, bin_counts):
    path = directory / "configuration.json"
    if not path.is_file():
        raise ValueError("Run notebook 12's original integration section before convergence refits")
    original = json.loads(path.read_text())
    payload = {k: v for k, v in original.items() if k != "fingerprint"}
    if original.get("version") != 1 or original.get("fingerprint") != _digest(payload):
        raise ValueError("Original integration configuration has an invalid fingerprint or version")
    if original["source_fingerprint"] != context["fingerprint"]:
        raise ValueError("Original integration bank belongs to a different frozen context")
    config = context["configuration"]
    for key in ("mu_values", "exposure_multiplier", "reference_power"):
        if original[key] != config[key]:
            raise ValueError(f"Original integration setting {key} does not match the frozen context")
    if not np.array_equal(original["frozen_observable_rates"], context["rates"]):
        raise ValueError("Original integration observable normalizers do not match the frozen context")
    if original.get("original_n_per_source") != context.get("original_n_per_source"):
        raise ValueError("Original integration generated count does not match the frozen context")
    for key, minimum in (("generated_per_source", 2), ("replicas", 1), ("chunk_size", 1), ("seed", 0)):
        _integer(original[key], key, minimum)
    sizes = [_integer(n, "cached size", 2) for n in original["sizes"]]
    if not sizes or sizes != sorted(set(sizes)):
        raise ValueError("Original integration sizes must be strictly increasing")
    if not bin_counts or len(set(bin_counts)) != len(bin_counts):
        raise ValueError("bin_counts must be nonempty and unique")
    requested = tuple(_integer(b, "bin count") for b in bin_counts)
    if not set(requested).issubset(original["n_bins"]):
        raise ValueError("Convergence bins must already exist in the original integration snapshots")
    # The checkpoint index follows the original (bins, mu) product, including
    # any bin counts not selected for the new convergence exercise.
    specs = []
    for b in original["n_bins"]:
        edges = diagnostic_bin_edges(b, config["n_bins"])
        for mu in original["mu_values"]:
            if b in requested:
                specs.append(dict(index=len(original["mu_values"]) * original["n_bins"].index(b)
                                  + original["mu_values"].index(mu),
                                  n_bins=int(b), mu=float(mu), edges=edges))
    return original, specs


def _load_state(path, fingerprint, specs, size, *, original=False):
    if not path.is_file():
        raise ValueError(f"Missing integration checkpoint {path.name}; complete the original integration section")
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["fingerprint"]) != fingerprint:
            raise ValueError(f"Integration checkpoint {path.name} has a different fingerprint")
        if not original and (int(saved["completed"]) != size or str(saved["state_kind"]) != "analytic_only_v1"):
            raise ValueError(f"Invalid analytical convergence checkpoint {path.name}")
        rng_state = json.loads(str(saved["rng_state"]))
        state = dict(rate_sum=saved["rate_sum"][:4].copy(),
                     rate_second=saved["rate_second"][:4, :4].copy(),
                     selected=saved["selected"].copy())
        for k, spec in enumerate(specs):
            index = spec["index"] if original else k
            state[f"bin_sum_{k}"] = saved[f"bin_sum_{index}"].copy()
            state[f"bin_second_{k}"] = saved[f"bin_second_{index}"].copy()
    expected = {"rate_sum": (4,), "rate_second": (4, 4), "selected": ()}
    for k, spec in enumerate(specs):
        expected[f"bin_sum_{k}"] = (4, spec["n_bins"])
        expected[f"bin_second_{k}"] = (4, 4, spec["n_bins"])
    if any(state[key].shape != shape or not np.isfinite(state[key]).all() for key, shape in expected.items()):
        raise ValueError(f"Malformed or nonfinite moments in {path.name}")
    if not 0 <= int(state["selected"]) <= size:
        raise ValueError(f"Invalid selected count in {path.name}")
    for k in range(len(specs)):
        if not np.allclose(state[f"bin_sum_{k}"].sum(axis=1), state["rate_sum"], rtol=1e-11, atol=1e-12):
            raise ValueError(f"Bin sums do not reproduce rate sums in {path.name}")
    return state, rng_state


def _combine(states, size, specs):
    rates = np.zeros(4)
    rate_covariance = np.zeros((4, 4))
    templates = {(s["n_bins"], s["mu"]): np.zeros((4, s["n_bins"])) for s in specs}
    covariance = {(s["n_bins"], s["mu"]): np.zeros((4, 4, s["n_bins"])) for s in specs}
    for state in states:
        mean, second = _moments(state["rate_sum"], state["rate_second"], size)
        rates += mean
        rate_covariance += second
        for k, spec in enumerate(specs):
            mean, second = _moments(state[f"bin_sum_{k}"], state[f"bin_second_{k}"], size)
            templates[spec["n_bins"], spec["mu"]] += mean
            covariance[spec["n_bins"], spec["mu"]] += second
    return dict(rates=rates, rate_covariance=rate_covariance, templates=templates,
                component_covariances=covariance,
                edges={s["n_bins"]: s["edges"].copy() for s in specs},
                selected=sum(int(state["selected"]) for state in states))


def _check_original_moments(path, bank, specs):
    """Check that saved summaries correspond to the saved stratum streams."""
    if not path.is_file():
        raise ValueError(f"Missing original integration summary {path.name}; complete the original integration section")
    with np.load(path, allow_pickle=False) as saved:
        checks = [(saved["rates"][:4], bank["rates"]),
                  (saved["rate_covariance"][:4, :4], bank["rate_covariance"])]
        for spec in specs:
            k, key = spec["index"], (spec["n_bins"], spec["mu"])
            if not np.array_equal(saved[f"bin_edges_{k}"], spec["edges"]):
                raise ValueError(f"Original bin boundaries changed in {path.name}")
            checks.extend([(saved[f"component_yields_{k}"], bank["templates"][key]),
                           (saved[f"component_covariance_{k}"], bank["component_covariances"][key])])
        if any(a.shape != b.shape or not np.allclose(a, b, rtol=1e-12, atol=1e-14) for a, b in checks):
            raise ValueError(f"Original moments do not match stratum checkpoints in {path.name}")


def _accumulate_analytic(state, fields, inverse_proposal, specs, rates, power):
    values = fields * inverse_proposal
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite analytical integration weight")
    state["rate_sum"] += values.sum(axis=1)
    state["rate_second"] += values @ values.T
    state["selected"] += fields.shape[1]
    for k, spec in enumerate(specs):
        z = reference_observable(fields, rates, spec["mu"], power)
        indices = _bin_indices(z, spec["edges"])
        b = spec["n_bins"]
        for p in range(4):
            state[f"bin_sum_{k}"][p] += np.bincount(indices, weights=values[p], minlength=b)
            for q in range(p, 4):
                moment = np.bincount(indices, weights=values[p] * values[q], minlength=b)
                state[f"bin_second_{k}"][p, q] += moment
                if p != q:
                    state[f"bin_second_{k}"][q, p] += moment


def _save_state(path, state, rng, completed, fingerprint):
    _atomic_npz(path, **state, fingerprint=np.array(fingerprint), completed=np.array(completed),
                state_kind=np.array("analytic_only_v1"), rng_state=np.array(json.dumps(rng.bit_generator.state)))


def _convergence_metadata(context, original, source_files, sizes, bin_counts, convergence_tag):
    metadata = dict(version=1, convergence_tag=convergence_tag, source_fingerprint=context["fingerprint"],
                    original_integration_fingerprint=original["fingerprint"],
                    original_files_sha256=source_files, original_sizes=original["sizes"],
                    sizes=sizes, replicas=original["replicas"],
                    generated_per_source=original["generated_per_source"],
                    seed=original["seed"], chunk_size=original["chunk_size"],
                    n_bins=list(map(int, bin_counts)), mu_values=original["mu_values"],
                    frozen_observable_rates=original["frozen_observable_rates"],
                    reference_power=original["reference_power"],
                    exposure_multiplier=original["exposure_multiplier"],
                    note="Analytical-only continuation of original RNG streams; fixed generated counts include rejected draws. Prefixes within a replica are correlated, replicas are independent. Original observable normalizers and selector remain frozen.")
    metadata["fingerprint"] = _digest(metadata)
    return metadata


def _finalize_banks(context, banks, metadata):
    for (replica, size), bank in banks.items():
        item = dict(source_fingerprint=context["fingerprint"],
                    integration_fingerprint=metadata["fingerprint"],
                    replica=int(replica), generated_per_source=int(size),
                    cached_original=size in metadata["original_sizes"], selected=bank.pop("selected"))
        # Include the actual predictions, so downstream caches cannot silently
        # reuse results if a checkpoint is manually replaced.
        item["prediction_sha256"] = _digest(dict(rates=bank["rates"].tolist(),
            templates={f"{b}:{mu}": value.tolist() for (b, mu), value in bank["templates"].items()}))
        item["fingerprint"] = _digest(item)
        bank["metadata"] = item
    return dict(banks=dict(sorted(banks.items())), metadata=metadata)


def load_completed_convergence_banks(context, *, convergence_tag="bank_convergence"):
    """Authenticate and load section 6's completed banks without any mutation.

    Settings come from its saved manifest. Every prefix and replica must already
    have its completed checkpoints; partial work snapshots are never continued.
    No directories, RNG draws, network evaluations, or files are created here.
    The return value is identical to ``prepare_convergence_banks`` for those
    saved settings, including prediction fingerprints used by the saved refits.
    """
    if not isinstance(convergence_tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", convergence_tag):
        raise ValueError("convergence_tag must contain only letters, digits, underscores or hyphens")

    def require(path):
        if not path.is_file():
            raise ValueError(f"Missing completed integration artifact {path.name}. "
                             "Run or complete notebook 12 section 6 and its prerequisite integration section.")
        return path

    original_dir = Path(context["output_dir"]) / "integration"
    directory = Path(context["output_dir"]) / "convergence" / convergence_tag / "integration"
    manifest = json.loads(require(directory / "configuration.json").read_text())
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if manifest.get("version") != 1 or manifest.get("fingerprint") != _digest(payload):
        raise ValueError("Convergence bank manifest has an invalid fingerprint or version")
    require(original_dir / "configuration.json")
    original, specs = _original_configuration(context, original_dir, manifest["n_bins"])
    original_sizes = original["sizes"]
    sizes = [_integer(size, "saved convergence size", 2) for size in manifest["sizes"]]
    if not sizes or sizes != sorted(set(sizes)) or not set(original_sizes).issubset(sizes):
        raise ValueError("Convergence bank manifest has invalid or missing prefix sizes")
    extra_sizes = sorted(set(sizes) - set(original_sizes))
    if any(size <= original_sizes[-1] or size % original["generated_per_source"] for size in extra_sizes):
        raise ValueError("Convergence bank manifest has an invalid extended prefix size")
    source_files = {"configuration.json": _file_digest(original_dir / "configuration.json")}
    banks = {}
    for replica in range(original["replicas"]):
        for size in original_sizes:
            states = []
            for source in _SOURCES:
                path = require(original_dir / f"replica_{replica}_{source}_{size}.npz")
                state, _ = _load_state(path, original["fingerprint"], specs, size, original=True)
                source_files[path.name] = _file_digest(path)
                states.append(state)
            bank = _combine(states, size, specs)
            path = require(original_dir / f"moments_replica_{replica}_{size}.npz")
            _check_original_moments(path, bank, specs)
            source_files[path.name] = _file_digest(path)
            banks[replica, size] = bank
    expected = _convergence_metadata(context, original, source_files, sizes, manifest["n_bins"], convergence_tag)
    if manifest != expected:
        raise ValueError("Convergence bank configuration or original snapshots changed; "
                         "complete section 6 with consistent saved inputs before this diagnostic")
    for replica in range(original["replicas"]):
        for size in extra_sizes:
            states = []
            for source in _SOURCES:
                path = require(directory / f"replica_{replica}_{source}_{size}.npz")
                state, _ = _load_state(path, manifest["fingerprint"], specs, size)
                states.append(state)
            banks[replica, size] = _combine(states, size, specs)
    return _finalize_banks(context, banks, manifest)


def prepare_convergence_banks(context, *, extra_multiples=(10,), bin_counts=(12, 36),
                              convergence_tag="bank_convergence"):
    """Return every original prefix/replica and extend its exact saved RNG stream.

    New multiples refer to ``generated_per_source`` in the original integration
    manifest. They must already be cached or exceed its largest prefix; an
    unavailable interior prefix cannot be reconstructed from sufficient moments.
    All original files are read-only. New checkpoints live under
    ``convergence/<convergence_tag>/integration`` and retain no invented
    learned-model moments. Use a new convergence tag when changing settings;
    the original diagnostic output tag must retain its completed snapshots.
    """
    if not isinstance(convergence_tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", convergence_tag):
        raise ValueError("convergence_tag must contain only letters, digits, underscores or hyphens")
    original_dir = Path(context["output_dir"]) / "integration"
    original, specs = _original_configuration(context, original_dir, bin_counts)
    multiples = tuple(_integer(m, "extra multiple") for m in extra_multiples)
    original_sizes = original["sizes"]
    extra_sizes = sorted(set(original["generated_per_source"] * m for m in multiples) - set(original_sizes))
    if any(n <= original_sizes[-1] for n in extra_sizes):
        raise ValueError("An uncached interior integration prefix cannot be recovered; request a larger multiple")
    all_sizes = sorted(set(original_sizes + extra_sizes))
    source_files = {"configuration.json": _file_digest(original_dir / "configuration.json")}
    banks, last_states, last_rng_states = {}, {}, {}
    # Read and validate every cached size, including both independent replicas,
    # before opening or creating the extension namespace.
    for replica in range(original["replicas"]):
        for size in original_sizes:
            states = []
            for source in _SOURCES:
                path = original_dir / f"replica_{replica}_{source}_{size}.npz"
                state, rng_state = _load_state(path, original["fingerprint"], specs, size, original=True)
                source_files[path.name] = _file_digest(path)
                states.append(state)
                if size == original_sizes[-1]:
                    last_states[replica, source] = state
                    last_rng_states[replica, source] = rng_state
            bank = _combine(states, size, specs)
            path = original_dir / f"moments_replica_{replica}_{size}.npz"
            _check_original_moments(path, bank, specs)
            source_files[path.name] = _file_digest(path)
            banks[replica, size] = bank
    metadata = _convergence_metadata(context, original, source_files, all_sizes, bin_counts, convergence_tag)
    directory = Path(context["output_dir"]) / "convergence" / convergence_tag / "integration"
    manifest = directory / "configuration.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("Convergence bank configuration or original snapshots changed; use another convergence_tag")
    directory.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        _atomic_json(manifest, metadata)
    snapshots = {}
    model, selector = context["run"].model, context["selector"]
    for replica in range(original["replicas"]):
        for source in _SOURCES:
            state = {k: v.copy() for k, v in last_states[replica, source].items()}
            completed = original_sizes[-1]
            rng = np.random.default_rng()
            rng.bit_generator.state = last_rng_states[replica, source]
            work = directory / f"replica_{replica}_{source}_work.npz"
            for size in extra_sizes:
                checkpoint = directory / f"replica_{replica}_{source}_{size}.npz"
                if checkpoint.exists():
                    state, rng_state = _load_state(checkpoint, metadata["fingerprint"], specs, size)
                    rng.bit_generator.state = rng_state
                    completed = size
                else:
                    if work.exists():
                        with np.load(work, allow_pickle=False) as saved:
                            work_size = int(saved["completed"])
                        if completed <= work_size <= size:
                            state, rng_state = _load_state(work, metadata["fingerprint"], specs, work_size)
                            rng.bit_generator.state = rng_state
                            completed = work_size
                    while completed < size:
                        count = min(original["chunk_size"], size - completed)
                        x = model.sample_component(source, count, rng)
                        if selector is not None:
                            x = x[selector.predict_proba(x)[:, 0] >= context["threshold"]]
                        if len(x):
                            # Equal proposal strata: f/(3q) with
                            # q=(pS+pB+pNI)/3; divide each stratum sum by n.
                            proposal = sum(model.component_pdf(x, p) for p in _SOURCES)
                            if np.any(proposal <= 0) or not np.isfinite(proposal).all():
                                raise ValueError("Invalid mixture proposal density")
                            _accumulate_analytic(state, model.component_densities(x).T, 1. / proposal,
                                                 specs, context["rates"], original["reference_power"])
                        completed += count
                        _save_state(work, state, rng, completed, metadata["fingerprint"])
                    _save_state(checkpoint, state, rng, completed, metadata["fingerprint"])
                    print(f"Convergence bank {replica + 1}/{original['replicas']}, {source}: "
                          f"{size:,} generated, {int(state['selected']):,} selected", flush=True)
                snapshots[replica, source, size] = {k: v.copy() for k, v in state.items()}
        for size in extra_sizes:
            banks[replica, size] = _combine([snapshots[replica, source, size] for source in _SOURCES], size, specs)
    return _finalize_banks(context, banks, metadata)
