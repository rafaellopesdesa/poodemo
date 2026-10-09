"""Provenance and generated-size checks before reconstructing saved toys."""
from types import SimpleNamespace

import numpy as np
import pytest

from poodemo import toy_diagnostic_context as diagnostics
from poodemo.data import file_digest, save_json
from poodemo.physics import PhysicsModel
from poodemo.pipeline import observable_bin_edges
from poodemo.toy_study import _array_digest


def _fixture(tmp_path, monkeypatch):
    model = PhysicsModel()
    run = SimpleNamespace(root=tmp_path, model=model,
                          path=lambda *parts: tmp_path.joinpath(*parts))
    rng = np.random.default_rng(52617)
    n = 40
    x = np.concatenate([model.sample_component(p, n, rng) for p in ("S", "B", "NI")])
    strata = np.repeat(np.arange(3), n)
    keep = x[:, 0] > .4
    x, strata = x[keep], strata[keep]
    nominal = model.component_densities(x).T
    quad = dict(x=x, weights=1. / (n * sum(model.component_pdf(x, p) for p in ("S", "B", "NI"))),
                nominal=nominal, down=nominal[None], up=nominal[None],
                truth=model.intensity(x, 1.), stratum=strata, half=np.zeros(len(x), dtype=int))
    run.path("quadrature").mkdir()
    for key, value in quad.items():
        np.save(run.path("quadrature", key + ".npy"), value)
    save_json(run.path("quadrature", "diagnostics.json"), dict(proposed_events=3*n))
    checkpoint = run.path("models", "preselection", "model.pt")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen-checkpoint")
    selection = dict(selector_sha256="frozen", selection=dict(threshold=.5))
    workspace = SimpleNamespace(learned_quadrature=(nominal * 1.02,))
    config = dict(physics=model.to_dict(),
                  model_artifacts={str(checkpoint.relative_to(tmp_path)): file_digest(checkpoint)},
                  selection=selection.copy(), n_bins=12,
                  bin_edges=observable_bin_edges(12, "reference_ratio").tolist(),
                  bank_hashes={key: _array_digest(quad[key]) for key in ("x", "weights", "nominal")},
                  learned_bank_sha256=_array_digest(workspace.learned_quadrature[0]),
                  exposure_multiplier=1.)
    source = dict(metadata=dict(configuration=config, fingerprint="source-fingerprint"),
                  results_sha256="source-results", config_sha256="source-config")
    calls = dict(quadrature=0, workspace=0)

    def prepare(_):
        calls["quadrature"] += 1
        return quad

    def frozen(_, supplied_quad):
        assert supplied_quad is quad
        calls["workspace"] += 1
        return workspace

    monkeypatch.setattr(diagnostics, "load_toy_study", lambda *args: source)
    monkeypatch.setattr(diagnostics, "load_frozen_selector", lambda *args: (object(), .5, selection))
    monkeypatch.setattr(diagnostics, "prepare_quadrature", prepare)
    monkeypatch.setattr(diagnostics, "make_frozen_workspace", frozen)
    return run, quad, source, selection, workspace, calls, n


def test_context_recovers_generated_counts_and_preserves_frozen_rates(tmp_path, monkeypatch):
    run, quad, source, _, workspace, _, n = _fixture(tmp_path, monkeypatch)
    assert len(quad["x"]) < 3*n
    ctx = diagnostics.prepare_toy_diagnostic_context(run)
    assert ctx["original_n_per_source"] == n
    np.testing.assert_array_equal(ctx["rates"], quad["nominal"] @ quad["weights"])
    np.testing.assert_array_equal(ctx["learned_rates"], workspace.learned_quadrature[0] @ quad["weights"])
    assert ctx["workspace"] is workspace
    assert ctx["source"] is source
    assert ctx["output_dir"] != run.path("results")
    assert diagnostics.prepare_toy_diagnostic_context(run)["fingerprint"] == ctx["fingerprint"]


@pytest.mark.parametrize("changed", ("model", "selection", "bank", "learned"))
def test_context_rejects_changed_inputs_before_refitting(tmp_path, monkeypatch, changed):
    run, quad, source, selection, workspace, calls, _ = _fixture(tmp_path, monkeypatch)
    if changed == "model":
        run.path("models", "preselection", "model.pt").write_bytes(b"different-checkpoint")
        match = "model artifacts"
    elif changed == "selection":
        # Replace the returned selection provenance, preserving the saved one.
        monkeypatch.setattr(diagnostics, "load_frozen_selector",
                            lambda *args: (object(), .6, dict(selection, selector_sha256="changed")))
        match = "selection"
    elif changed == "bank":
        quad["weights"][0] *= 1.01
        match = "integration bank"
    else:
        workspace.learned_quadrature = (workspace.learned_quadrature[0] * 1.001,)
        match = "learned bank evaluation"
    with pytest.raises(ValueError, match=match):
        diagnostics.prepare_toy_diagnostic_context(run)
    if changed in {"model", "selection"}:
        assert calls == dict(quadrature=0, workspace=0)
    if changed == "bank":
        assert calls["workspace"] == 0
    assert not run.path("results", "12_toy_diagnostics").exists()


def test_context_never_regenerates_a_missing_saved_bank(tmp_path, monkeypatch):
    run, _, _, _, _, calls, _ = _fixture(tmp_path, monkeypatch)
    missing = run.path("quadrature", "weights.npy")
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="Restore notebook 11"):
        diagnostics.prepare_toy_diagnostic_context(run)
    assert calls == dict(quadrature=0, workspace=0)
    assert not missing.exists()


def test_changed_source_records_cannot_reuse_existing_diagnostics(tmp_path, monkeypatch):
    run, _, source, _, _, _, _ = _fixture(tmp_path, monkeypatch)
    first = diagnostics.prepare_toy_diagnostic_context(run)
    manifest = first["output_dir"] / "context.json"
    original = manifest.read_bytes()
    source["results_sha256"] = "extended-or-modified-results"
    with pytest.raises(ValueError, match="source/input fingerprint changed"):
        diagnostics.prepare_toy_diagnostic_context(run)
    assert manifest.read_bytes() == original
    second = diagnostics.prepare_toy_diagnostic_context(run, output_tag="extended_source")
    assert second["fingerprint"] != first["fingerprint"]


@pytest.mark.parametrize("problem", ("manifest", "weights", "strata"))
def test_generated_count_inference_rejects_inconsistent_bank(tmp_path, monkeypatch, problem):
    run, quad, _, _, _, _, n = _fixture(tmp_path, monkeypatch)
    if problem == "manifest":
        save_json(run.path("quadrature", "diagnostics.json"), dict(proposed_events=3*n+3))
        match = "diagnostics disagree"
    elif problem == "weights":
        quad["weights"][0] *= 2.
        match = "equal-size stratified"
    else:
        quad["stratum"][0] = 7
        match = "invalid stratum"
    with pytest.raises(ValueError, match=match):
        diagnostics._original_generated_per_source(run, quad)
