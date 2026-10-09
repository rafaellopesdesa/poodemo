"""Read-only prefix reuse and exact RNG continuation for notebook 12."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr

from poodemo.physics import PhysicsModel
from poodemo.toy_convergence_banks import prepare_convergence_banks
from poodemo.toy_integration_diagnostics import run_integration_diagnostics


class AnalyticWorkspace:
    def __init__(self, model):
        self.model = model

    def evaluate(self, x):
        fields = self.model.component_densities(x).T
        return fields, fields, fields


class RejectingSelector:
    def predict_proba(self, x):
        return (x[:, :1] >= .4).astype(float)


class GuardedModel:
    def __init__(self, model, fail_after=None):
        self.model, self.fail_after, self.calls = model, fail_after, 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def sample_component(self, source, n, rng):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("Injected interruption")
        self.calls += 1
        return self.model.sample_component(source, n, rng)


def _context(path):
    model = PhysicsModel(lambda_s=4., lambda_b=15., lambda_ni=35., exposure=1.)
    n, rng = 300, np.random.default_rng(77219)
    x = np.concatenate([model.sample_component(p, n, rng) for p in ("S", "B", "NI")])
    fields = model.component_densities(x).T
    weights = 1. / (n * sum(model.component_pdf(x, p) for p in ("S", "B", "NI")))
    rates = fields @ weights
    return dict(run=SimpleNamespace(model=model), rates=rates, learned_rates=rates,
                workspace=AnalyticWorkspace(model), selector=RejectingSelector(), threshold=.5,
                quad=dict(x=x, nominal=fields, weights=weights), original_n_per_source=n,
                configuration=dict(mu_values=[0., 1.4], exposure_multiplier=2., reference_power=100., n_bins=12),
                fingerprint="frozen-test-context", output_dir=path)


def _original(context, multiples=(1, 2, 5)):
    return run_integration_diagnostics(context, generated_per_source=80, multiples=multiples,
                                       replicas=2, n_bins=(12, 36), seed=82721, chunk_size=73)


def _hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}


def _assert_banks_equal(a, b):
    assert set(a["banks"]) == set(b["banks"])
    for key in a["banks"]:
        for name in ("rates", "rate_covariance"):
            np.testing.assert_array_equal(a["banks"][key][name], b["banks"][key][name])
        for template in a["banks"][key]["templates"]:
            np.testing.assert_array_equal(a["banks"][key]["templates"][template], b["banks"][key]["templates"][template])


def test_all_cached_prefixes_and_replicas_reused_without_sampling_or_networks(tmp_path):
    context = _context(tmp_path)
    original = _original(context)
    directory = tmp_path / "integration"
    hashes = _hashes(directory)
    context.pop("workspace")  # Any accidental ratio evaluation now fails.
    context["run"].model = GuardedModel(context["run"].model, fail_after=0)
    banks = prepare_convergence_banks(context, extra_multiples=())
    assert set(banks["banks"]) == {(r, n) for r in (0, 1) for n in (80, 160, 400)}
    np.testing.assert_array_equal(banks["banks"][0, 400]["rates"], original["rates"])
    for key, template in original["templates"].items():
        np.testing.assert_array_equal(banks["banks"][0, 400]["templates"][key], template)
    assert _hashes(directory) == hashes
    assert context["run"].model.calls == 0


def test_extension_matches_fresh_same_prefix_schedule_and_preserves_original(tmp_path):
    context = _context(tmp_path / "extend")
    _original(context)
    original_hashes = _hashes(context["output_dir"] / "integration")
    context.pop("workspace")
    extended = prepare_convergence_banks(context, extra_multiples=(10,))
    assert len(extended["banks"]) == 8
    fresh = _context(tmp_path / "fresh")
    _original(fresh, multiples=(1, 2, 5, 10))
    for replica in (0, 1):
        bank = extended["banks"][replica, 800]
        with np.load(fresh["output_dir"] / "integration" / f"moments_replica_{replica}_800.npz") as saved:
            np.testing.assert_array_equal(bank["rates"], saved["rates"][:4])
            np.testing.assert_allclose(bank["rate_covariance"], saved["rate_covariance"][:4, :4], rtol=1e-14)
            for k, key in enumerate(bank["templates"]):
                np.testing.assert_array_equal(bank["templates"][key], saved[f"component_yields_{k}"])
                np.testing.assert_array_equal(bank["component_covariances"][key], saved[f"component_covariance_{k}"])
        # Exact continued RNG state proves that fresh generated points were
        # identical, including points rejected by the frozen selector.
        for source in ("S", "B", "NI"):
            extended_path = context["output_dir"] / "convergence" / "bank_convergence" / "integration" / f"replica_{replica}_{source}_800.npz"
            with np.load(extended_path) as a, np.load(fresh["output_dir"] / "integration" / extended_path.name) as b:
                assert str(a["rng_state"]) == str(b["rng_state"])
                assert str(a["state_kind"]) == "analytic_only_v1"
                assert a["rate_sum"].shape == (4,)
        assert bank["metadata"]["selected"] < 3 * 800
        # The selector is an exact half-space, so these three selected yields
        # have known Gaussian integrals independent of either MC workflow.
        for index, source in ((0, "S"), (2, "B"), (3, "NI")):
            model = context["run"].model
            sigma = np.sqrt(np.asarray(getattr(model, f"cov_{source.lower()}"))[0, 0])
            expected = model.component_yield(source) * ndtr((model.mean(source)[0] - .4) / sigma)
            uncertainty = np.sqrt(bank["rate_covariance"][index, index])
            assert abs(bank["rates"][index] - expected) < 5 * uncertainty
        for (bins, mu), template in bank["templates"].items():
            np.testing.assert_allclose(template.sum(axis=1), bank["rates"], rtol=1e-13)
        for mu in (0., 1.4):
            np.testing.assert_allclose(bank["templates"][36, mu].reshape(4, 12, 3).sum(axis=2), bank["templates"][12, mu], rtol=1e-13)
    # Re-running uses checkpoints, without ratio evaluations or new draws.
    context["run"].model = GuardedModel(context["run"].model, fail_after=0)
    _assert_banks_equal(extended, prepare_convergence_banks(context, extra_multiples=(10,)))
    assert _hashes(context["output_dir"] / "integration") == original_hashes


def test_interrupted_chunk_resume_is_exact_and_uses_fixed_generated_count(tmp_path):
    context = _context(tmp_path / "interrupted")
    _original(context)
    context.pop("workspace")
    model = context["run"].model
    context["run"].model = GuardedModel(model, fail_after=2)
    with pytest.raises(RuntimeError, match="Injected"):
        prepare_convergence_banks(context)
    path = context["output_dir"] / "convergence" / "bank_convergence" / "integration" / "replica_0_S_work.npz"
    with np.load(path) as checkpoint:
        assert checkpoint["completed"] == 400 + 2 * 73
    context["run"].model = GuardedModel(model)
    resumed = prepare_convergence_banks(context)
    uninterrupted = _context(tmp_path / "uninterrupted")
    _original(uninterrupted)
    result = prepare_convergence_banks(uninterrupted)
    _assert_banks_equal(resumed, result)
    # Recover a rate from the three stratum sums, not selected counts.
    rate = np.zeros(4)
    for source in ("S", "B", "NI"):
        file = path.parent / f"replica_0_{source}_800.npz"
        with np.load(file) as checkpoint:
            rate += checkpoint["rate_sum"] / 800
    np.testing.assert_array_equal(resumed["banks"][0, 800]["rates"], rate)


def test_rejects_changed_context_settings_and_original_snapshot(tmp_path):
    context = _context(tmp_path)
    _original(context)
    prepare_convergence_banks(context, extra_multiples=())
    with pytest.raises(ValueError, match="convergence_tag"):
        prepare_convergence_banks(context, extra_multiples=(10,))
    with pytest.raises(ValueError, match="interior"):
        prepare_convergence_banks(context, extra_multiples=(3,), convergence_tag="different")
    with pytest.raises(ValueError, match="already exist"):
        prepare_convergence_banks(context, bin_counts=(12, 108))
    with pytest.raises(ValueError, match="convergence_tag"):
        prepare_convergence_banks(context, convergence_tag="../outside")
    rates = context["rates"].copy()
    context["rates"] = rates * 1.01
    with pytest.raises(ValueError, match="normalizers"):
        prepare_convergence_banks(context, extra_multiples=())
    context["rates"] = rates
    checkpoint = tmp_path / "integration" / "replica_1_NI_160.npz"
    with np.load(checkpoint) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays["rng_state"] = np.array(json.dumps(np.random.default_rng(8).bit_generator.state))
    np.savez_compressed(checkpoint, **arrays)
    with pytest.raises(ValueError, match="snapshots changed"):
        prepare_convergence_banks(context, extra_multiples=())
