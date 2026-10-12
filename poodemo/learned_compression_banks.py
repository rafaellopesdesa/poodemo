"""Independent simulation banks for the notebook-13 compression comparison.

The networks, their normalization constants, and the selected-rate constants
used by both observable maps are frozen on notebook 03's original quadrature.
Neither new bank is used to recalibrate those objects. Bank A supplies template
integrals and bank B supplies an independent expected-data/closure integral.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from .closure_validation import _selected_sample
from .data import PROCESSES, file_digest, save_json
from .pipeline import prepare_quadrature
from .toy_sources import load_frozen_selector, make_frozen_workspace


SOURCES = ("S", "B", "NI")
STREAM_NAMESPACE = 130013
_ARRAY_NAMES = ("x", "weights", "nominal", "down", "up", "learned_nominal",
                "truth", "stratum")


def _integer(value, name, *, minimum=1):
    try:
        valid = (not isinstance(value, (bool, np.bool_)) and np.isscalar(value)
                 and np.isfinite(value) and int(value) == value and value >= minimum)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _array_digest(array):
    array = np.asarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    # Fields have a small leading component axis. Flattening in bounded
    # buffers avoids materializing a full bank, including noncontiguous views.
    for block in np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                          order="C", buffersize=131072):
        digest.update(np.asarray(block).tobytes())
    return digest.hexdigest()


def _model_artifacts(run):
    names = {"model.pt", "ensemble.json", "calibrator.joblib", "normalization.json"}
    return {str(path.relative_to(run.root)): file_digest(path)
            for path in sorted(run.path("models").rglob("*"))
            if path.is_file() and path.name in names}


def _nominal_learned_fields(workspace, x):
    """Evaluate the three nominal ratio networks with the saved normalizers.

    This is the nominal part of ``FrozenWorkspaceFields.evaluate``. NI-up and
    NI-down networks are unnecessary when constructing the nominal compression
    map; nuisance templates are instead evaluated from simulator densities.
    """
    signal = workspace.run.model.component_density(x, "S")
    fields = np.empty((4, len(x)), dtype=float)
    fields[0] = signal
    for j, process in enumerate(PROCESSES[1:], 1):
        key = f"{process}_over_S"
        shape = signal / workspace.rates[0] * workspace.predictors[key].predict_ratio(x)
        normalization = workspace.normalizers[key]
        if not np.isfinite(normalization) or normalization <= 0:
            raise ValueError(f"Invalid frozen workspace normalizer for {key}")
        fields[j] = workspace.rates[j] * shape / normalization
    return fields


def _positive(values, description):
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{description} must be finite and strictly positive; no points were clipped or dropped")


def _load_bank(directory, fingerprint, stream_id, n):
    path = directory / "metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    payload = metadata.get("payload", {})
    expected_fingerprint = _digest({key: value for key, value in metadata.items()
                                   if key != "fingerprint"})
    if (metadata.get("parent_fingerprint") != fingerprint
            or metadata.get("stream_id") != stream_id
            or metadata.get("generated_per_source") != n
            or metadata.get("fingerprint") != expected_fingerprint
            or set(payload) != set(_ARRAY_NAMES)):
        raise ValueError(f"Stale compression-bank metadata in {directory}; choose a new OUTPUT_TAG")
    arrays = {}
    for name in _ARRAY_NAMES:
        item, file = payload[name], directory / f"{name}.npy"
        if not file.exists() or file_digest(file) != item["sha256"]:
            raise ValueError(f"Compression-bank payload changed: {file}; choose a new OUTPUT_TAG")
        arrays[name] = np.load(file, mmap_mode="r", allow_pickle=False)
        if list(arrays[name].shape) != item["shape"] or arrays[name].dtype.str != item["dtype"]:
            raise ValueError(f"Invalid compression-bank array {file}")
    return dict(**arrays, metadata=metadata, directory=directory)


def _build_bank(run, workspace, selector, threshold, directory, *, fingerprint,
                stream_id, n, seed, chunk):
    directory.mkdir(parents=True, exist_ok=True)
    sample_dir = directory / "source_draws"
    sample_dir.mkdir(exist_ok=True)
    selected = {}
    for source_id, source in enumerate(SOURCES):
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, STREAM_NAMESPACE, stream_id, source_id]))
        path = _selected_sample(run.model, selector, threshold, source, n, rng,
                                sample_dir, chunk)
        selected[source] = np.load(path, mmap_mode="r", allow_pickle=False)
        print(f"Compression {directory.name}: {source} {len(selected[source]):,} / {n:,} selected", flush=True)
    counts = {name: len(points) for name, points in selected.items()}
    size = sum(counts.values())
    if size < 2:
        raise ValueError("Compression bank has fewer than two selected events")
    shapes = dict(x=(size, 3), weights=(size,), nominal=(4, size),
                  down=(1, 4, size), up=(1, 4, size), learned_nominal=(4, size),
                  truth=(size,), stratum=(size,))
    arrays = {name: np.lib.format.open_memmap(
        directory / f"{name}.partial.npy", mode="w+", shape=shape,
        dtype=np.float32 if name == "x" else np.int8 if name == "stratum" else np.float64)
        for name, shape in shapes.items()}
    offset = 0
    for index, source in enumerate(SOURCES):
        count = counts[source]
        arrays["x"][offset:offset + count] = selected[source]
        arrays["stratum"][offset:offset + count] = index
        offset += count
    # The selected-event order retains source strata. Rejected draws remain
    # implicit zero contributions in the fixed generated-count denominator.
    for start in range(0, size, chunk):
        stop = min(start + chunk, size)
        x = arrays["x"][start:stop]
        proposal_sum = sum(run.model.component_pdf(x, source) for source in SOURCES)
        _positive(proposal_sum, "Proposal density")
        weights = 1. / (n * proposal_sum)
        nominal = run.model.component_densities(x).T
        down = run.model.component_densities(x, alpha_ni=-1.).T
        up = run.model.component_densities(x, alpha_ni=1.).T
        learned = _nominal_learned_fields(workspace, x)
        truth = run.model.intensity(x, run.config["asimov_mu"])
        for values, label in ((weights, "Importance weights"), (nominal, "Analytical components"),
                              (down, "Analytical down components"), (up, "Analytical up components"),
                              (learned, "Learned positive components"), (truth, "Physical truth intensity")):
            _positive(values, label)
        # Positivity of components does not imply positivity of their signed
        # interference combination. That check belongs to each tested map;
        # retain all events here, including any with invalid learned totals.
        arrays["weights"][start:stop] = weights
        arrays["nominal"][:, start:stop] = nominal
        arrays["down"][0, :, start:stop] = down
        arrays["up"][0, :, start:stop] = up
        arrays["learned_nominal"][:, start:stop] = learned
        arrays["truth"][start:stop] = truth
        print(f"Compression {directory.name}: evaluated {stop:,} / {size:,} selected events", flush=True)
    for array in arrays.values():
        array.flush()
    del array
    arrays.clear()
    payload = {}
    for name in _ARRAY_NAMES:
        path = directory / f"{name}.npy"
        (directory / f"{name}.partial.npy").replace(path)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        payload[name] = dict(shape=list(values.shape), dtype=values.dtype.str,
                             sha256=file_digest(path))
    metadata = dict(schema=1, parent_fingerprint=fingerprint, stream_id=stream_id,
                    sources=list(SOURCES), generated_per_source=n,
                    generated_counts={name: n for name in SOURCES}, selected_counts=counts,
                    selected_events=size, payload=payload,
                    proposal="equal stratified nominal S/B/NI mixture before selection",
                    weight_formula="1 / (n_generated_per_source * (pdf_S + pdf_B + pdf_NI))",
                    normalization="Raw importance integrals; no yield or shape renormalization on this bank")
    metadata["fingerprint"] = _digest(metadata)
    save_json(directory / "metadata.json", metadata)
    selected.clear()
    for source in SOURCES:
        (sample_dir / f"{source}.npy").unlink()
    sample_dir.rmdir()
    return _load_bank(directory, fingerprint, stream_id, n)


def prepare_compression_banks(run, *, n_per_source=2_000_000, seed=None,
                              output_tag="13_learned_compression_36_floor02", chunk_size=None):
    """Prepare independent template and validation banks for learned compression.

    ``n_per_source`` is the number generated BEFORE selection in each of S, B,
    and NI, separately for each bank. Memory-mapped caches are authenticated
    against the physics, selector, ratio artifacts, original quadrature, and
    every saved array. An interrupted, incomplete bank is rebuilt with the
    same stream; a complete matching bank is reused without new NN inference.
    """
    n = _integer(n_per_source, "n_per_source", minimum=2)
    seed = _integer(run.config["seed"] if seed is None else seed, "seed", minimum=0)
    chunk = _integer(run.config["generation_chunk"] if chunk_size is None else chunk_size, "chunk_size")
    if not isinstance(output_tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", output_tag):
        raise ValueError("output_tag must be a simple filename tag")
    truth_mu = float(run.config["asimov_mu"])
    if not np.isfinite(truth_mu) or truth_mu < 0:
        raise ValueError("asimov_mu must be finite and nonnegative")
    quad = prepare_quadrature(run)
    selector, threshold, selection = load_frozen_selector(run)
    rates = np.asarray(quad["nominal"]) @ np.asarray(quad["weights"])
    _positive(rates, "Frozen selected component rates")
    configuration = dict(schema=1, seed=seed, stream_namespace=STREAM_NAMESPACE,
                         streams={"template": 0, "validation": 1}, sources=list(SOURCES),
                         n_per_source=n, chunk_size=chunk, numpy_version=np.__version__,
                         asimov_mu=truth_mu, physics=run.model.to_dict(), selection=selection,
                         model_artifacts=_model_artifacts(run),
                         original_quadrature_hashes={name: _array_digest(quad[name])
                                                    for name in ("x", "weights", "nominal", "down", "up")},
                         frozen_rates=rates.tolist(),
                         normalization_source="Original notebook-03 quadrature only")
    directory = run.path("results", output_tag, "banks")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "provenance.json"
    workspace = None
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        if (metadata.get("configuration") != configuration
                or metadata.get("fingerprint") != _digest({key: value for key, value in metadata.items()
                                                           if key != "fingerprint"})):
            raise ValueError("Compression-bank settings/models/original quadrature changed; choose a new OUTPUT_TAG")
    else:
        if any((directory / name / "metadata.json").exists() for name in ("template", "validation")):
            raise ValueError("Compression banks exist without provenance; choose a new OUTPUT_TAG")
        workspace = make_frozen_workspace(run, quad)
        metadata = dict(configuration=configuration,
                        frozen_normalizers={key: float(value) for key, value in workspace.normalizers.items()})
        _positive(np.asarray(list(metadata["frozen_normalizers"].values())), "Frozen learned normalizers")
        metadata["fingerprint"] = _digest(metadata)
        save_json(manifest, metadata)
    banks = {}
    for stream_id, name in enumerate(("template", "validation")):
        bank_dir = directory / name
        banks[name] = _load_bank(bank_dir, metadata["fingerprint"], stream_id, n)
        if banks[name] is None:
            if workspace is None:
                workspace = make_frozen_workspace(run, quad)
                normalizers = {key: float(value) for key, value in workspace.normalizers.items()}
                if normalizers != metadata["frozen_normalizers"]:
                    raise ValueError("Frozen workspace normalizers changed; choose a new OUTPUT_TAG")
            banks[name] = _build_bank(
                run, workspace, selector, threshold, bank_dir,
                fingerprint=metadata["fingerprint"], stream_id=stream_id,
                n=n, seed=seed, chunk=chunk)
        else:
            print(f"Compression {name}: reusing {banks[name]['metadata']['selected_events']:,} authenticated events", flush=True)
    return dict(**banks, rates=rates, metadata=metadata, directory=directory)
