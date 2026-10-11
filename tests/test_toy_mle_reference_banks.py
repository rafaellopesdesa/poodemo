"""Event-level replay preserves saved templates, source streams and resumes."""
import json

import numpy as np
import pytest

from test_toy_convergence_banks import _context, _original, _hashes, GuardedModel
from poodemo.toy_convergence_banks import prepare_convergence_banks
from poodemo.toy_integration_diagnostics import _bin_indices
from poodemo.toy_mle_reference_banks import prepare_mle_reference_banks
from poodemo.toy_study import reference_observable


def _prepared(path):
    context = _context(path)
    _original(context)
    prepare_convergence_banks(context, extra_multiples=(10,))
    context.pop("workspace")
    return context


def _directory(context, tag="mle_reference"):
    return context["output_dir"] / "convergence/bank_convergence/mle_reference" / tag / "integration"


def test_event_weights_reproduce_templates_and_cache_reuses_every_draw(tmp_path):
    context = _prepared(tmp_path)
    original_hashes = _hashes(tmp_path / "integration")
    convergence_hashes = _hashes(tmp_path / "convergence/bank_convergence/integration")
    study = prepare_mle_reference_banks(context, largest_prefixes=2)
    assert set(study["banks"]) == {(r, n) for r in (0, 1) for n in (400, 800)}
    assert len(study["closure"]) == 2 * 4 * 3
    assert study["closure"][["moments_reproduced", "rng_reproduced", "selected_reproduced"]].all().all()
    for key, bank in study["banks"].items():
        fields, divisor = bank["fields"], bank["normalization_divisor"]
        expected = study["convergence_banks"][key]
        assert isinstance(fields, np.memmap) and not fields.flags.writeable
        assert fields.shape == (4, expected["metadata"]["selected"])
        assert divisor == key[1] == bank["metadata"]["generated_per_source"]
        assert bank["metadata"]["convergence_bank_fingerprint"] == expected["metadata"]["fingerprint"]
        np.testing.assert_array_equal(bank["rates"], expected["rates"])
        np.testing.assert_allclose(fields.sum(axis=1) / divisor, bank["rates"], rtol=1e-13)
        for (n, mu), template in expected["templates"].items():
            z = reference_observable(fields, context["rates"], mu, context["configuration"]["reference_power"])
            indices = _bin_indices(z, expected["edges"][n])
            actual = np.array([np.bincount(indices, weights=row, minlength=n) for row in fields]) / divisor
            np.testing.assert_allclose(actual, template, rtol=1e-12, atol=1e-13)
    # A complete call must use no new sampling or selector evaluations.
    context["run"].model = GuardedModel(context["run"].model, fail_after=0)
    class NoSelector:
        def predict_proba(self, x):
            raise AssertionError("Selector should not run for completed event cache")
    context["selector"] = NoSelector()
    repeated = prepare_mle_reference_banks(context, largest_prefixes=2)
    for key in study["banks"]:
        np.testing.assert_array_equal(study["banks"][key]["fields"], repeated["banks"][key]["fields"])
        assert study["banks"][key]["metadata"] == repeated["banks"][key]["metadata"]
    assert context["run"].model.calls == 0
    assert _hashes(tmp_path / "integration") == original_hashes
    assert _hashes(tmp_path / "convergence/bank_convergence/integration") == convergence_hashes


def test_interrupted_replay_resumes_chunk_schedule_exactly(tmp_path):
    context = _prepared(tmp_path / "resume")
    model = context["run"].model
    context["run"].model = GuardedModel(model, fail_after=5)
    with pytest.raises(RuntimeError, match="Injected"):
        prepare_mle_reference_banks(context)
    work = _directory(context) / "replica_0_S_work.npz"
    with np.load(work) as saved:
        assert int(saved["completed"]) == 233  # 73+7, 73+7, 73.
        assert len(json.loads(str(saved["chunks"]))) == 5
    context["run"].model = GuardedModel(model)
    resumed = prepare_mle_reference_banks(context)
    fresh = prepare_mle_reference_banks(_prepared(tmp_path / "fresh"))
    assert set(resumed["banks"]) == {(0, 800), (1, 800)}
    for key in resumed["banks"]:
        np.testing.assert_array_equal(resumed["banks"][key]["fields"], fresh["banks"][key]["fields"])
    # Six streams, each with all original prefix/chunk boundaries preserved.
    assert context["run"].model.calls == 6 * 14 - 5


@pytest.mark.parametrize("target", ["chunk", "checkpoint", "combined", "manifest"])
def test_modified_cache_payload_is_rejected(tmp_path, target):
    context = _prepared(tmp_path)
    prepare_mle_reference_banks(context)
    directory = _directory(context)
    if target in ("chunk", "combined"):
        path = directory / ("replica_0_S_chunk_0_73.npy" if target == "chunk" else "replica_0_800_fields.npy")
        values = np.load(path)
        values[0, 0] += 1.
        np.save(path, values)
    elif target == "checkpoint":
        path = directory / "replica_0_S_80.npz"
        with np.load(path) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        arrays["rate_sum"][0] += 1.
        np.savez_compressed(path, **arrays)
    else:
        path = directory / "replica_0_800_fields.json"
        item = json.loads(path.read_text())
        item["selected"] += 1
        path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="payload changed|fingerprint changed"):
        prepare_mle_reference_banks(context)


def test_drifted_selector_and_changed_configuration_are_rejected(tmp_path):
    context = _prepared(tmp_path)
    prepare_mle_reference_banks(context)
    with pytest.raises(ValueError, match="reference_tag"):
        prepare_mle_reference_banks(context, reference_tag="../bad")
    with pytest.raises(ValueError, match="largest_prefixes"):
        prepare_mle_reference_banks(context, largest_prefixes=5)
    with pytest.raises(ValueError, match="configuration or saved inputs changed"):
        prepare_mle_reference_banks(context, largest_prefixes=2)
    context["selector"] = None
    with pytest.raises(ValueError, match="Exact-bank replay failed"):
        prepare_mle_reference_banks(context, reference_tag="drift")


def test_chunk_written_before_checkpoint_commit_is_safe_to_replay(tmp_path, monkeypatch):
    from poodemo import toy_mle_reference_banks as module
    context = _prepared(tmp_path / "interrupted")
    save = module._save_checkpoint
    called = 0
    def interrupt_after_chunk(*args, **kwargs):
        nonlocal called
        called += 1
        if called == 2:
            raise RuntimeError("Interrupted between chunk and checkpoint")
        return save(*args, **kwargs)
    monkeypatch.setattr(module, "_save_checkpoint", interrupt_after_chunk)
    with pytest.raises(RuntimeError, match="between chunk"):
        prepare_mle_reference_banks(context)
    monkeypatch.setattr(module, "_save_checkpoint", save)
    resumed = prepare_mle_reference_banks(context)
    fresh = prepare_mle_reference_banks(_prepared(tmp_path / "fresh"))
    for key in resumed["banks"]:
        np.testing.assert_array_equal(resumed["banks"][key]["fields"], fresh["banks"][key]["fields"])
