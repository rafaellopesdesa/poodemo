"""Cross-module checks for score normalization and stale selection protection."""
import json
from pathlib import Path

import numpy as np
import pytest

from poodemo.data import Run, SPLITS, create_run, generate_samples, materialize_selection
from poodemo.physics import PhysicsModel
from poodemo.pipeline import RATIO_TASKS, _score, mu_fit_starts, mu_grid, spline_eta_grid


def test_disjoint_stage_partitions_and_cached_selected_score(tmp_path):
    intervals = sorted(SPLITS.values())
    assert intervals[0][0] == 0 and intervals[-1][1] == 1
    assert all(a[1] == b[0] for a, b in zip(intervals[:-1], intervals[1:]))
    model = PhysicsModel()
    rng = np.random.default_rng(42)
    x = np.concatenate([model.sample_component(p, 500, rng) for p in ("S", "B", "NI")])
    proposal = sum(model.component_pdf(x, p) for p in ("S", "B", "NI"))/3
    selected = x[:, 0] + x[:, 1] > 0.5
    weights = (1/(len(x)*proposal))[selected]
    x = x[selected]
    nominal = model.component_densities(x).T
    yields = dict(zip(("S", "SBI", "B", "NI"), nominal @ weights))
    quad = {"x": x, "nominal": nominal, "weights": weights}
    run = Run(tmp_path, {}, model)
    for eta in (.1, 1., 3.):
        np.testing.assert_allclose(_score(run, quad, eta),
                                   model.score(x, eta, selected_yields=yields), atol=1e-13)
        probabilities = weights * model.intensity(x, eta)
        probabilities /= probabilities.sum()
        np.testing.assert_allclose(np.sum(probabilities * _score(run, quad, eta)), 0., atol=1e-14)


def test_changed_selector_cannot_reuse_cached_events(tmp_path):
    (tmp_path / "selected").mkdir()
    (tmp_path / "models" / "preselection").mkdir(parents=True)
    (tmp_path / "models" / "preselection" / "model.pt").write_bytes(b"new weights")
    (tmp_path / "selected" / "selection_state.json").write_text(json.dumps({"selection_fingerprint": "old"}))
    run = Run(tmp_path, {"generation_chunk": 10}, PhysicsModel())
    with pytest.raises(ValueError, match="Preselection changed"):
        materialize_selection(run, None, .4)


def test_ni_only_generation_and_training_inventory(tmp_path):
    run = create_run(tmp_path, "smoke", {"n_per_sample": 64, "generation_chunk": 32})
    records = generate_samples(run)
    samples = {"S", "SBI", "B", "NI", "NI_up", "NI_down"}
    assert {r["sample"] for r in records} == samples
    assert {p.stem for p in run.path("raw").glob("*.npy")} == samples
    assert set(RATIO_TASKS) == {("SBI", "S"), ("B", "S"), ("NI", "S"),
                               ("NI_up", "NI"), ("NI_down", "NI")}
    for sample in samples:
        x = np.load(run.path("raw", sample + ".npy"))
        assert x.shape == (64, 3) and np.all(np.isfinite(x))

    # An old cached nuisance layout must not silently supply extra parameters.
    manifest_path = run.path("run.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["schema_version"] = 3
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Choose a new RUN_NAME"):
        create_run(tmp_path, "smoke", {"n_per_sample": 64, "generation_chunk": 32})


def test_physics_overrides_are_recorded_and_cannot_reuse_another_model(tmp_path):
    overrides = {"physics_overrides": {"shift_fraction_ni": .4}}
    run = create_run(tmp_path, "smoke", overrides)
    assert run.model.shift_fraction_ni == .4
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["physics"]["shift_fraction_ni"] == .4
    assert create_run(tmp_path, "smoke", overrides).config == run.config
    with pytest.raises(ValueError, match="Choose a new RUN_NAME"):
        create_run(tmp_path, "smoke")
    starts = mu_fit_starts(run)
    assert run.config["asimov_mu"] in starts
    assert min(starts) == run.config["mu_fit_bounds"][0]
    assert max(starts) == run.config["mu_fit_bounds"][1]
    assert any(.3 < value < .7 for value in starts)


def test_production_spline_comparison_uses_genuine_held_out_etas(tmp_path):
    run = create_run(tmp_path)
    rates = np.array([run.model.component_yield(p) for p in ("S", "SBI", "B", "NI")])
    quad = {"nominal": rates[:, None], "weights": np.ones(1)}
    anchors = spline_eta_grid(run, quad)
    scan = mu_grid(run)
    at_anchor = np.any(np.isclose(scan[:, None], anchors[None, :], rtol=0., atol=1e-12), axis=1)
    # Aligned grids made a previous interpolation check agree by construction.
    assert np.mean(~at_anchor) > .9
    assert np.all(np.diff(anchors) > 0)
    assert anchors[0] == run.config["mu_min"]
    assert anchors[-1] == run.config["mu_max"]
    assert run.config["asimov_mu"] in anchors
    turnover = ((rates[0] + rates[2] - rates[1]) / (2 * rates[0]))**2
    nearby = anchors[np.abs(anchors - turnover) < .05]
    assert np.max(np.diff(nearby)) < .0002


def test_same_sample_ratio_normalization_cancels_arbitrary_predictor_scales(tmp_path, monkeypatch):
    """All five ratios use the final reference masses, including NI anchors."""
    from poodemo import training
    from poodemo.pipeline import _learned_fields

    run = create_run(tmp_path, "smoke")
    x = np.linspace(-1., 1., 31)
    weights = np.linspace(.02, .07, len(x))
    nominal = np.stack([np.exp(.1*x), 4*np.exp(.2*x),
                        3*np.exp(-.3*x), 12*np.exp(-.1*x)])
    up, down = nominal[None].copy(), nominal[None].copy()
    up[0, 3] *= np.exp(.2*x)
    down[0, 3] *= np.exp(-.2*x)
    quad = dict(x=x[:, None], weights=weights, nominal=nominal, up=up, down=down)
    names = [f"{num}_over_{den}" for num, den in RATIO_TASKS]
    scales = dict.fromkeys(names, 1.)

    class Predictor:
        def __init__(self, name):
            self.name = name

        def predict_ratio(self, values):
            slope = .1*(names.index(self.name)+1)
            return scales[self.name]*np.exp(slope*values[:, 0])

    monkeypatch.setattr(training, "load_predictor", lambda path: Predictor(path.name))
    fields = _learned_fields(run, quad)
    for actual, exact in zip(fields, (nominal, down, up)):
        np.testing.assert_allclose(actual @ weights, exact @ weights, rtol=1e-13)
    scales.update(zip(names, [1e-3, 7., 1e4, .02, 85.]))
    rescaled = _learned_fields(run, quad)
    for actual, expected in zip(rescaled, fields):
        np.testing.assert_allclose(actual, expected, rtol=1e-13)


@pytest.mark.parametrize("bad", [0., -1., np.inf, np.nan])
def test_model_asimov_rejects_invalid_total_intensities(bad):
    from poodemo.pipeline import _model_asimov_truth
    nominal = np.ones((4, 3))
    nominal[1, 1], nominal[3, 1] = bad, 0.
    with pytest.raises(ValueError, match="1 total event intensities"):
        _model_asimov_truth(nominal, 1.)
