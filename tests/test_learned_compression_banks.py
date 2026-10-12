"""Independent weighted banks with frozen neural normalization and safe reuse."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr

from poodemo.closure_validation import FrozenWorkspaceFields
from poodemo.data import Run
from poodemo.inference import component_coefficients
from poodemo.learned_compression_banks import (prepare_compression_banks,
                                              _nominal_learned_fields)
from poodemo.physics import PhysicsModel


class HalfSpaceSelector:
    def predict_proba(self, x):
        return (x[:, :1] >= .25).astype(float)


class ExactRatio:
    def __init__(self, model, numerator, denominator, alpha=0.):
        self.model, self.numerator, self.denominator, self.alpha = model, numerator, denominator, alpha
        self.calls = 0

    def predict_ratio(self, x):
        self.calls += 1
        return (self.model.component_density(x, self.numerator, alpha_ni=self.alpha)
                / self.model.component_density(x, self.denominator))


@pytest.fixture
def context(tmp_path, monkeypatch):
    from poodemo import learned_compression_banks as module
    model = PhysicsModel(lambda_s=4., lambda_b=15., lambda_ni=35., exposure=1.)
    run = Run(tmp_path, dict(seed=1627, generation_chunk=131, asimov_mu=1.), model)
    run.path("models", "ratios").mkdir(parents=True)
    run.path("models", "ratios", "model.pt").write_bytes(b"checkpoint-a")
    rng, n = np.random.default_rng(77219), 1000
    x = np.concatenate([model.sample_component(p, n, rng) for p in ("S", "B", "NI")])
    x = x[x[:, 0] >= .25]
    weights = 1. / (n * sum(model.component_pdf(x, p) for p in ("S", "B", "NI")))
    quad = dict(x=x, weights=weights, nominal=model.component_densities(x).T,
                down=model.component_densities(x, -1.).T[None],
                up=model.component_densities(x, 1.).T[None])
    predictors = {f"{name}_over_S": ExactRatio(model, name, "S") for name in ("SBI", "B", "NI")}
    predictors.update({f"NI_{name}_over_NI": ExactRatio(model, "NI", "NI", alpha)
                       for name, alpha in (("up", 1.), ("down", -1.))})
    workspace = FrozenWorkspaceFields(run, quad, predictors)
    monkeypatch.setattr(module, "prepare_quadrature", lambda run: quad)
    monkeypatch.setattr(module, "load_frozen_selector", lambda run: (HalfSpaceSelector(), .5, {"fixed": True}))
    monkeypatch.setattr(module, "make_frozen_workspace", lambda run, quad: workspace)
    return SimpleNamespace(run=run, quad=quad, workspace=workspace, module=module)


def test_three_nominal_networks_match_full_frozen_evaluator(context):
    workspace = context.workspace
    x = context.run.model.sample_component("B", 20, np.random.default_rng(19))
    expected = workspace.evaluate(x)[0]
    variation_calls = {key: predictor.calls for key, predictor in workspace.predictors.items() if "NI_over" not in key and "NI_" in key}
    np.testing.assert_array_equal(_nominal_learned_fields(workspace, x), expected)
    for key, calls in variation_calls.items():
        assert workspace.predictors[key].calls == calls


def test_independent_strata_raw_weights_and_frozen_rates(context):
    n = 4500
    result = prepare_compression_banks(context.run, n_per_source=n, chunk_size=700)
    np.testing.assert_array_equal(result["rates"], context.quad["nominal"] @ context.quad["weights"])
    assert result["metadata"]["frozen_normalizers"] == context.workspace.normalizers
    assert not np.array_equal(result["template"]["x"], result["validation"]["x"])
    for name in ("template", "validation"):
        bank, model = result[name], context.run.model
        assert isinstance(bank["nominal"], np.memmap)
        assert np.all(bank["x"][:, 0] >= .25)
        assert bank["metadata"]["generated_counts"] == dict(S=n, B=n, NI=n)
        assert bank["metadata"]["selected_events"] < 3 * n
        expected = 1. / (n * sum(model.component_pdf(bank["x"], p) for p in ("S", "B", "NI")))
        np.testing.assert_array_equal(bank["weights"], expected)
        np.testing.assert_array_equal(bank["nominal"], model.component_densities(bank["x"]).T)
        np.testing.assert_array_equal(bank["down"][0], model.component_densities(bank["x"], -1.).T)
        np.testing.assert_allclose(bank["learned_nominal"], bank["nominal"], rtol=2e-14)
        # No post-hoc equalization to the original integration-bank rates.
        assert not np.allclose(bank["nominal"] @ bank["weights"], result["rates"], rtol=1e-5)
        # Known half-space rates test the generated-count denominator, including
        # rejected points as zero contributions to each stratum's MC variance.
        for process_index, process in ((0, "S"), (2, "B"), (3, "NI")):
            contributions = bank["nominal"][process_index] * bank["weights"]
            variance = sum(n / (n - 1) * (np.sum(contributions[bank["stratum"] == h] ** 2)
                                          - np.sum(contributions[bank["stratum"] == h]) ** 2 / n)
                           for h in range(3))
            sigma = np.sqrt(np.asarray(getattr(model, f"cov_{process.lower()}"))[0, 0])
            exact = model.component_yield(process) * ndtr((model.mean(process)[0] - .25) / sigma)
            assert abs(contributions.sum() - exact) < 5 * np.sqrt(variance)


def test_cache_reuses_without_network_inference_and_rejects_payload_change(context, monkeypatch):
    kwargs = dict(n_per_source=200, chunk_size=73)
    first = prepare_compression_banks(context.run, **kwargs)
    monkeypatch.setattr(context.module, "make_frozen_workspace",
                        lambda *a: (_ for _ in ()).throw(AssertionError("Network rerun")))
    second = prepare_compression_banks(context.run, **kwargs)
    assert second["metadata"] == first["metadata"]
    np.testing.assert_array_equal(first["template"]["x"], second["template"]["x"])
    path = first["validation"]["directory"] / "learned_nominal.npy"
    tampered = np.load(path, mmap_mode="r+")
    tampered[0, 0] *= 1.01
    tampered.flush()
    with pytest.raises(ValueError, match="payload changed"):
        prepare_compression_banks(context.run, **kwargs)


@pytest.mark.parametrize("change", ["seed", "chunk", "count", "checkpoint", "quadrature", "truth"])
def test_rejects_changed_source_provenance(context, change):
    kwargs = dict(n_per_source=100, chunk_size=73)
    prepare_compression_banks(context.run, **kwargs)
    if change == "seed":
        kwargs["seed"] = 982
    elif change == "chunk":
        kwargs["chunk_size"] += 1
    elif change == "count":
        kwargs["n_per_source"] += 1
    elif change == "checkpoint":
        context.run.path("models", "ratios", "model.pt").write_bytes(b"checkpoint-b")
    elif change == "quadrature":
        context.quad["weights"][0] *= 1.01
    elif change == "truth":
        context.run.config["asimov_mu"] = .2
    with pytest.raises(ValueError, match="settings/models/original quadrature changed"):
        prepare_compression_banks(context.run, **kwargs)


def test_negative_combined_learned_intensity_is_retained_for_map_diagnostics(context, monkeypatch):
    def positive_components(workspace, x):
        signal = workspace.run.model.component_density(x, "S")
        return np.asarray([signal, .001 * signal, .001 * signal, .001 * signal])
    monkeypatch.setattr(context.module, "_nominal_learned_fields", positive_components)
    result = prepare_compression_banks(context.run, n_per_source=100)
    for name in ("template", "validation"):
        bank = result[name]
        assert np.all(bank["learned_nominal"] > 0)
        assert np.all(component_coefficients(.25) @ bank["learned_nominal"] < 0)
        assert len(bank["weights"]) == sum(bank["metadata"]["selected_counts"].values())


@pytest.mark.parametrize("kwargs", [{"n_per_source": True}, {"n_per_source": 1},
                                    {"chunk_size": 0}, {"seed": -1}, {"output_tag": "../outside"}])
def test_input_validation_before_sampling(context, kwargs):
    with pytest.raises(ValueError):
        prepare_compression_banks(context.run, **kwargs)
