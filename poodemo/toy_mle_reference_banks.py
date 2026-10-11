"""Authenticated event banks for the per-toy MLE-reference diagnostic.

Saved component fields are importance weights ``f_j(x)/sum_s p_s(x)``.
The S/B/NI strata each contain the same *generated* count, including rejected
draws. Sum the selected weights in a bin and divide by that count to obtain
its component yield. The common proposal factor cancels in density ratios.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .toy_convergence_banks import (
    _SOURCES, _accumulate_analytic, _combine, _digest, _file_digest,
    _integer, _load_state, _original_configuration,
    load_completed_convergence_banks,
)
from .toy_family_banks import _array_digest, _blank, _reproduction
from .toy_integration_diagnostics import _atomic_csv, _atomic_json, _atomic_npz


def _write_chunk(path, values):
    temporary = path.with_suffix(".npy.partial")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values, dtype=np.float64), allow_pickle=False)
    os.replace(temporary, path)
    return _file_digest(path)


def _save_checkpoint(path, state, rng, count, chunks, fingerprint):
    arrays = dict(state, completed=np.array(count),
                  rng_state=np.array(json.dumps(rng.bit_generator.state)),
                  chunks=np.array(json.dumps(chunks, sort_keys=True)))
    _atomic_npz(path, **arrays, fingerprint=np.array(fingerprint),
                payload_sha256=np.array(_array_digest(arrays)))


def _validate_chunks(directory, chunks, count, selected, verified):
    end, rows = 0, 0
    for chunk in chunks:
        if (set(chunk) != {"file", "start", "stop", "selected", "sha256"}
                or not re.fullmatch(r"replica_\d+_(S|B|NI)_chunk_\d+_\d+\.npy", chunk["file"])
                or chunk["start"] != end or chunk["stop"] <= end
                or not 0 <= chunk["selected"] <= chunk["stop"] - end):
            raise ValueError("Malformed MLE-reference chunk index")
        path = directory / chunk["file"]
        signature = (chunk["sha256"], chunk["selected"])
        if verified.get(path) != signature:
            if not path.is_file() or _file_digest(path) != chunk["sha256"]:
                raise ValueError(f"MLE-reference chunk payload changed or missing: {path.name}")
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if (values.dtype != np.dtype("float64") or values.shape != (4, chunk["selected"])
                    or not np.isfinite(values).all() or np.any(values < 0)):
                raise ValueError(f"Malformed MLE-reference component weights: {path.name}")
            verified[path] = signature
        end, rows = chunk["stop"], rows + chunk["selected"]
    if end != count or rows != selected:
        raise ValueError("MLE-reference chunks do not reproduce generated and selected counts")


def _load_checkpoint(path, fingerprint, specs, verified, expected_size=None):
    blank = _blank(specs, {})
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["fingerprint"]) != fingerprint:
            raise ValueError("MLE-reference checkpoint fingerprint changed; use another reference_tag")
        state = {key: saved[key].copy() for key in blank}
        arrays = dict(state, completed=saved["completed"], rng_state=saved["rng_state"], chunks=saved["chunks"])
        if str(saved["payload_sha256"]) != _array_digest(arrays):
            raise ValueError(f"MLE-reference checkpoint payload changed: {path.name}")
        count = int(saved["completed"])
        rng_state = json.loads(str(saved["rng_state"]))
        chunks = json.loads(str(saved["chunks"]))
    if count < 1 or (expected_size is not None and count != expected_size):
        raise ValueError(f"Invalid generated count in MLE-reference checkpoint {path.name}")
    if any(value.shape != blank[key].shape or not np.isfinite(value).all() for key, value in state.items()):
        raise ValueError(f"Malformed MLE-reference moments in {path.name}")
    selected = int(state["selected"])
    if not 0 <= selected <= count:
        raise ValueError(f"Invalid selected count in MLE-reference checkpoint {path.name}")
    _validate_chunks(path.parent, chunks, count, selected, verified)
    return state, rng_state, count, chunks


def _combined_bank(directory, replica, size, chunks, expected, metadata):
    """Concatenate once on disk; subsequent calls map the authenticated file."""
    path = directory / f"replica_{replica}_{size}_fields.npy"
    manifest_path = path.with_suffix(".json")
    selected = sum(chunk["selected"] for chunk in chunks)
    configuration = dict(source_fingerprint=metadata["source_fingerprint"],
        reference_fingerprint=metadata["fingerprint"],
        convergence_bank_fingerprint=expected["metadata"]["fingerprint"],
        replica=replica, generated_per_source=size, selected=selected,
        chunks_sha256=_digest(chunks), fields_file=path.name,
        weight_convention="component_density / sum(S_pdf, B_pdf, NI_pdf); divide bin sums by generated_per_source")
    if manifest_path.exists():
        item = json.loads(manifest_path.read_text())
        payload = {key: value for key, value in item.items() if key != "fingerprint"}
        if item.get("fingerprint") != _digest(payload):
            raise ValueError("MLE-reference combined bank manifest fingerprint changed")
        if any(item.get(key) != value for key, value in configuration.items()):
            raise ValueError("MLE-reference combined bank inputs changed; use another reference_tag")
        if not path.is_file() or _file_digest(path) != item.get("fields_sha256"):
            raise ValueError("MLE-reference combined bank payload changed or missing")
    else:
        temporary = path.with_suffix(".npy.partial")
        values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64, shape=(4, selected))
        start = 0
        for chunk in chunks:
            stop = start + chunk["selected"]
            values[:, start:stop] = np.load(directory / chunk["file"], mmap_mode="r", allow_pickle=False)
            start = stop
        values.flush()
        del values
        os.replace(temporary, path)
        item = dict(configuration, fields_sha256=_file_digest(path))
        item["fingerprint"] = _digest(item)
        _atomic_json(manifest_path, item)
    fields = np.load(path, mmap_mode="r", allow_pickle=False)
    if fields.shape != (4, selected) or fields.dtype != np.dtype("float64"):
        raise ValueError("Malformed MLE-reference combined bank array")
    if not np.allclose(fields.sum(axis=1) / size, expected["rates"], rtol=2e-11, atol=2e-12):
        raise ValueError("MLE-reference component weights do not reproduce the saved rates")
    return dict(fields=fields, fields_path=path, normalization_divisor=size,
                rates=expected["rates"].copy(), metadata=item)


def prepare_mle_reference_banks(context, *, convergence_tag="bank_convergence",
                                reference_tag="mle_reference", largest_prefixes=1,
                                progress_every=10):
    """Replay section 6 exactly, caching selected event weights for new maps.

    No density-ratio network is evaluated: only the original frozen selector.
    Every original prefix authenticates RNG state, selected count, analytical
    rates and all first/second 1D moments. Atomic per-chunk checkpoints resume
    interrupted work; a complete cache performs no simulation or selection.

    ``banks[(replica, generated_per_source)]['fields']`` is a read-only float64
    memory map with shape (4, n_selected), in S/SBI/B/NI order. It contains
    *unnormalized* importance weights, concatenating the three proposal strata.
    Divide bin sums by ``bank['metadata']['generated_per_source']``, never by
    the selected count or by three times the generated count. The proposal
    factor and this common normalization both cancel in per-event ratios.
    """
    if not isinstance(reference_tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", reference_tag):
        raise ValueError("reference_tag must contain only letters, digits, underscores or hyphens")
    largest_prefixes = _integer(largest_prefixes, "largest_prefixes")
    progress_every = _integer(progress_every, "progress_every")
    completed = load_completed_convergence_banks(context, convergence_tag=convergence_tag)
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
    configuration = dict(convergence_tag=convergence_tag, reference_tag=reference_tag,
        largest_prefixes=largest_prefixes, all_prefixes=sizes, selected_prefixes=chosen,
        replicas=original["replicas"], seed=original["seed"], chunk_size=original["chunk_size"],
        exposure_multiplier=original["exposure_multiplier"], mu_values=original["mu_values"])
    metadata = dict(version=1, source_fingerprint=context["fingerprint"], configuration=configuration,
        convergence_fingerprint=completed["metadata"]["fingerprint"],
        convergence_bank_fingerprints={f"{r}:{n}": b["metadata"]["fingerprint"] for (r, n), b in expected.items()},
        source_files_sha256=source_hashes,
        note="Exact section-6 stream replay; unnormalized component/proposal weights; analytical ratios only; frozen selector; no learned density network.")
    metadata["fingerprint"] = _digest(metadata)
    directory = Path(context["output_dir"]) / "convergence" / convergence_tag / "mle_reference" / reference_tag / "integration"
    manifest = directory / "configuration.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("MLE-reference bank configuration or saved inputs changed; use another reference_tag")
    directory.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        _atomic_json(manifest, metadata)
    snapshots, closure_rows, verified = {}, [], {}
    model, selector = context["run"].model, context["selector"]
    chunks_completed = 0
    for replica in range(original["replicas"]):
        for source_id, source in enumerate(_SOURCES):
            state, count, chunks = _blank(specs, {}), 0, []
            rng = np.random.default_rng(np.random.SeedSequence([original["seed"], 120524, replica, source_id]))
            work = directory / f"replica_{replica}_{source}_work.npz"
            for size in sizes:
                path = directory / f"replica_{replica}_{source}_{size}.npz"
                if path.exists():
                    state, rng_state, count, chunks = _load_checkpoint(path, metadata["fingerprint"], specs, verified, size)
                    rng.bit_generator.state = rng_state
                else:
                    if work.exists():
                        with np.load(work, allow_pickle=False) as saved:
                            work_size = int(saved["completed"])
                        if count <= work_size <= size:
                            state, rng_state, count, chunks = _load_checkpoint(work, metadata["fingerprint"], specs, verified)
                            rng.bit_generator.state = rng_state
                    while count < size:
                        take = min(original["chunk_size"], size - count)
                        x = model.sample_component(source, take, rng)
                        if selector is not None:
                            x = x[selector.predict_proba(x)[:, 0] >= context["threshold"]]
                        values = np.empty((4, 0), dtype=np.float64)
                        if len(x):
                            proposal = sum(model.component_pdf(x, p) for p in _SOURCES)
                            if np.any(proposal <= 0) or not np.isfinite(proposal).all():
                                raise ValueError("Invalid mixture proposal density during MLE-reference replay")
                            fields = model.component_densities(x).T
                            _accumulate_analytic(state, fields, 1. / proposal, specs,
                                                 context["rates"], original["reference_power"])
                            values = fields * (1. / proposal)
                        chunk_path = directory / f"replica_{replica}_{source}_chunk_{count}_{count + take}.npy"
                        digest = _write_chunk(chunk_path, values)
                        chunks.append(dict(file=chunk_path.name, start=count, stop=count + take,
                                           selected=len(x), sha256=digest))
                        verified[chunk_path] = (digest, len(x))
                        count += take
                        _save_checkpoint(work, state, rng, count, chunks, metadata["fingerprint"])
                        chunks_completed += 1
                        if chunks_completed % progress_every == 0:
                            print(f"MLE-reference bank replay {replica + 1}/{original['replicas']}, {source}: {count:,}/{sizes[-1]:,} generated", flush=True)
                source_path, is_original = source_paths[replica, source, size]
                source_fingerprint = original["fingerprint"] if is_original else completed["metadata"]["fingerprint"]
                original_state, original_rng = _load_state(source_path, source_fingerprint, specs, size, original=is_original)
                try:
                    closure_rows.append(_reproduction(state, rng, original_state, original_rng,
                        replica=replica, source=source, size=size))
                except ValueError as error:
                    raise ValueError(str(error).replace("2D control", "MLE-reference test")) from error
                if not path.exists():
                    _save_checkpoint(path, state, rng, count, chunks, metadata["fingerprint"])
                if size in chosen:
                    snapshots[replica, source, size] = ({key: value.copy() for key, value in state.items()}, list(chunks))
    banks = {}
    for (replica, size), expected_bank in expected.items():
        states = [snapshots[replica, source, size][0] for source in _SOURCES]
        scalar = _combine(states, size, specs)
        if not np.allclose(scalar["rates"], expected_bank["rates"], rtol=2e-11, atol=2e-12):
            raise ValueError("Replayed analytical rates do not reproduce the saved bank")
        for key, prediction in scalar["templates"].items():
            if not np.allclose(prediction, expected_bank["templates"][key], rtol=2e-11, atol=2e-12):
                raise ValueError("Replayed 1D template does not reproduce the saved bank")
        chunks = [chunk for source in _SOURCES for chunk in snapshots[replica, source, size][1]]
        banks[replica, size] = _combined_bank(directory, replica, size, chunks, expected_bank, metadata)
    closure = pd.DataFrame(closure_rows)
    _atomic_csv(directory / "replay_closure.csv", closure)
    return dict(banks=banks, convergence_banks=expected, metadata=metadata, closure=closure)
