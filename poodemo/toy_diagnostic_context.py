"""Authenticate notebook 11 inputs before reconstructing its experiments.

No original model, simulation bank, or notebook 11 result is changed.  Learned
normalizations and the observable's reference rates stay on the original bank.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from .data import file_digest, save_json
from .pipeline import observable_bin_edges, prepare_quadrature
from .toy_diagnostics import load_toy_study
from .toy_likelihood import StatOnlyLikelihood
from .toy_sources import load_frozen_selector, make_frozen_workspace
from .toy_study import _array_digest


def _same(actual, expected, name):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"Notebook 11 {name} changed. Restore its matching inputs before running notebook 12.")


def _original_generated_per_source(run, quad):
    """Recover generated (not accepted) stratum size from the MC weights."""
    x, weights = np.asarray(quad["x"]), np.asarray(quad["weights"])
    if not len(x) or weights.shape != (len(x),) or np.any(weights <= 0):
        raise ValueError("Original quadrature must contain positive weights")
    indices = np.linspace(0, len(x)-1, min(128, len(x)), dtype=int)
    points = x[indices]
    proposal = sum(run.model.component_pdf(points, p) for p in ("S", "B", "NI"))/3.
    inferred = 1./(3.*weights[indices]*proposal)
    if np.any(~np.isfinite(inferred)):
        raise ValueError("Cannot infer original generated proposal size")
    n = int(round(float(np.median(inferred))))
    if n < 2 or not np.allclose(inferred, n, rtol=1e-9, atol=1e-7):
        raise ValueError("Original quadrature is not an equal-size stratified S/B/NI bank")
    strata = np.asarray(quad["stratum"])
    if strata.shape != weights.shape or not np.isin(strata, (0, 1, 2)).all():
        raise ValueError("Original quadrature has invalid stratum labels")
    if any(np.sum(strata == s) > n for s in range(3)):
        raise ValueError("Accepted stratum count exceeds original generated size")
    diagnostics = run.path("quadrature", "diagnostics.json")
    if diagnostics.exists():
        proposed = json.loads(diagnostics.read_text()).get("proposed_events")
        if proposed != 3*n:
            raise ValueError("Original quadrature diagnostics disagree with its MC weights")
    return n


def prepare_toy_diagnostic_context(run, source_tag="toy_study_direct", output_tag="toy_diagnostics"):
    """Load frozen NN inputs only after matching notebook 11's provenance.

    The independent integration and paired-refit modules share this context.
    Changed source results or model inputs cannot silently reuse checkpoints.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output_tag):
        raise ValueError("output_tag must contain only letters, digits, underscores or hyphens")
    source = load_toy_study(run, source_tag)
    config = source["metadata"]["configuration"]
    _same(run.model.to_dict(), config["physics"], "physics model")
    dependencies = {}
    for path in sorted(run.path("models").rglob("*")):
        if path.is_file() and path.name in {"model.pt", "ensemble.json", "calibrator.joblib"}:
            dependencies[str(path.relative_to(run.root))] = file_digest(path)
    _same(dependencies, config["model_artifacts"], "model artifacts")
    selector, threshold, selection = load_frozen_selector(run)
    _same(selection, config["selection"], "selection")
    _same(observable_bin_edges(config["n_bins"], "reference_ratio").tolist(),
          config["bin_edges"], "bin edges")
    # Require the saved bank: diagnostics must not regenerate missing inputs.
    names = ("x", "weights", "nominal", "down", "up", "truth", "stratum", "half")
    missing = [name for name in names if not run.path("quadrature", name+".npy").exists()]
    if missing:
        raise FileNotFoundError(f"Restore notebook 11's original quadrature; missing {missing}")
    quad = prepare_quadrature(run)
    bank_hashes = {key: _array_digest(quad[key]) for key in ("x", "weights", "nominal")}
    _same(bank_hashes, config["bank_hashes"], "integration bank")
    n = _original_generated_per_source(run, quad)
    workspace = make_frozen_workspace(run, quad)
    learned = np.asarray(workspace.learned_quadrature[0])
    learned_hash = _array_digest(learned)
    _same(learned_hash, config["learned_bank_sha256"], "learned bank evaluation")
    rates = np.asarray(quad["nominal"]) @ quad["weights"]
    learned_rates = learned @ quad["weights"]
    guard = StatOnlyLikelihood(np.empty((4, 0)), learned_rates,
                              exposure=config["exposure_multiplier"], validity_fields=learned)
    identity = dict(version=1, source_tag=source_tag,
                    source_results_sha256=source["results_sha256"],
                    source_config_sha256=source["config_sha256"],
                    source_fingerprint=source["metadata"]["fingerprint"],
                    original_n_per_source=n,
                    stratum_sha256=_array_digest(quad["stratum"]))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    output = run.path("results", "12_toy_diagnostics", output_tag)
    manifest = output / "context.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("fingerprint") != fingerprint:
        raise ValueError("Diagnostic source/input fingerprint changed. Use a new OUTPUT_TAG.")
    save_json(manifest, dict(fingerprint=fingerprint, identity=identity,
                            normalization_note="Original observable and learned-rate normalizers are frozen; fresh MC only changes the explicitly compared predictions."))
    return dict(run=run, source=source, configuration=config, quad=quad,
                workspace=workspace, selector=selector, threshold=threshold,
                rates=rates, learned_rates=learned_rates, learned_guard=guard,
                original_n_per_source=n, fingerprint=fingerprint, output_dir=output)
