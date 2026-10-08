"""Training contracts: real toolkit checkpoint roundtrip and held-out selection."""
import importlib.util
import json

import numpy as np
import pytest

from poodemo.training import (TrainingConfig, choose_preselection_threshold,
                             fit_density_ratio, load_predictor, normalize_ratio,
                             Predictor, RatioEnsemble, training_metadata)


class KnownPreselection:
    labels = ("S", "B", "NI")
    output_dir = None
    def predict_proba(self, x):
        s = x[:, 0]
        return np.column_stack((s, (1 - s) / 2, (1 - s) / 2))


def test_threshold_uses_physical_yields_and_nonempty_denominator():
    samples = {"S": np.linspace(0.6, 1, 10001)[:, None],
               "NI": np.linspace(0, 0.7, 10001)[:, None]}
    result = choose_preselection_threshold(KnownPreselection(), samples,
                                          {"S": 1, "NI": 100}, min_ni_events=20)
    assert result["achieved_ratio"] >= 0.1
    assert result["selected_counts"]["NI"] >= 20
    assert result["efficiencies"]["S"] > 0.5
    ratio = .01 * result["efficiencies"]["S"] / result["efficiencies"]["NI"]
    assert result["achieved_ratio"] == pytest.approx(ratio)


def test_threshold_does_not_treat_empty_ni_as_success():
    samples = {"S": np.linspace(0, 1, 1001)[:, None],
               "NI": np.linspace(0, 1, 1001)[:, None]}
    with pytest.raises(ValueError, match="No statistically populated cut"):
        choose_preselection_threshold(KnownPreselection(), samples, {"S": 1, "NI": 100})


HAS_TRAINING = all(importlib.util.find_spec(x) for x in
                   ("torch", "pytorch_lightning", "nsbi_common_utils"))


def test_raw_ratios_bypass_postprocessing_including_ensemble_members():
    # The raw diagnostics must not silently inherit a member's calibration or
    # normalization. Build light predictors to exercise this without torch.
    class FixedCalibration:
        def cali_pred(self, score):
            return np.full_like(score, .75)

    members = []
    for value, normalization in [(2.0, 2.0), (8.0, 4.0)]:
        predictor = Predictor.__new__(Predictor)
        predictor.kind, predictor.labels = "ratio", ("den", "num")
        predictor.calibration = FixedCalibration()
        predictor.log_normalization = np.log(normalization)
        predictor._raw = lambda x, batch_size, v=value: np.full((len(x), 1), np.log(v))
        members.append(predictor)
    x = np.zeros((5, 3))
    np.testing.assert_allclose(members[0].predict_raw_ratio(x), 2)
    np.testing.assert_allclose(members[0].predict_ratio(x), 1.5)
    ensemble = RatioEnsemble(members, log_normalization=np.log(3))
    np.testing.assert_allclose(ensemble.predict_raw_ratio(x), 5)
    np.testing.assert_allclose(ensemble.predict_raw_log_ratio(x), np.log(5))
    np.testing.assert_allclose(ensemble.predict_ratio(x), (1.5 + .75) / 2 / 3)


@pytest.mark.skipif(not HAS_TRAINING, reason="optional toolkit training dependencies")
def test_pinned_classifier_is_standard_bce_without_penalties_or_dropout():
    import torch
    from nsbi_common_utils.lightning_tools import DensityRatioLightning
    module = DensityRatioLightning(input_dim=3, n_hidden=3, n_neurons=128,
                                    activation="swish", use_log_loss=True,
                                    learning_rate=.001, callback_factor=.5,
                                    callback_patience=15)
    predictor = Predictor(module, np.zeros(3), np.ones(3), "ratio", ("S", "B"), device="cpu")
    report = training_metadata(predictor)
    assert report["architecture"]["trainable_parameters"] == 33665
    assert [(x["input_features"], x["output_features"]) for x in
            report["architecture"]["layers"] if x["type"] == "Linear"] == [
                (3, 128), (128, 128), (128, 128), (128, 1)]
    assert report["architecture"]["hidden_activation"] == "swish"
    assert not report["architecture"]["dropout_layers"]
    assert not report["architecture"]["batch_normalization"]
    assert report["optimizer"]["type"] == "NAdam"
    assert report["optimizer"]["parameter_group_weight_decay"] == [0.0]
    assert report["scheduler"]["type"] == "StepLR"
    assert report["scheduler"]["step_size"] == 15
    assert report["scheduler"]["gamma"] == .5
    # Compare the actual toolkit training step against ordinary BCE, including
    # its gradient. This would detect an added penalty or class weighting.
    x = torch.randn(12, 3)
    target = torch.arange(12) % 2
    module.log = lambda *args, **kwargs: None
    loss = module.training_step((x, target, torch.ones(12)), 0)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        module(x).ravel(), target.float())
    torch.testing.assert_close(loss, expected)
    actual_grad = torch.autograd.grad(loss, tuple(module.parameters()))
    expected_grad = torch.autograd.grad(expected, tuple(module.parameters()))
    for actual, wanted in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, wanted)


@pytest.mark.skipif(not HAS_TRAINING, reason="optional toolkit training dependencies")
def test_toolkit_model_roundtrip_and_independent_normalization(tmp_path):
    rng = np.random.default_rng(41)
    denominator = rng.normal(size=(1200, 3)).astype(np.float32)
    numerator = rng.normal(.4, size=(1200, 3)).astype(np.float32)
    config = TrainingConfig.smoke(epochs=2, max_train_per_class=400,
                                  max_validation_per_class=100, accelerator="cpu")
    callbacks = []
    model = fit_density_ratio(numerator[:600], denominator[:600],
                              numerator[600:800], denominator[600:800], tmp_path, config,
                              member_callback=lambda predictor, index, total:
                                  callbacks.append((predictor, index, total)))
    assert callbacks == [(model, 0, 1)]
    before = model.predict_ratio(denominator[1000:])
    assert np.all(np.isfinite(before)) and np.all(before > 0)
    restored = load_predictor(tmp_path, device="cpu")
    np.testing.assert_array_equal(before, restored.predict_ratio(denominator[1000:]))
    report = normalize_ratio(restored, denominator[800:1000])
    assert restored.predict_ratio(denominator[800:1000]).mean() == pytest.approx(1, abs=1e-12)
    assert 0 < report["effective_sample_size"] <= 200.00001
    assert report["relative_standard_error"] >= 0
    again = normalize_ratio(restored, denominator[800:1000])
    assert again["normalization"] == pytest.approx(report["normalization"])
    metadata = json.loads((tmp_path / "training.json").read_text())
    assert metadata["complete"]
    assert metadata["class_prior"] == [0.5, 0.5]
    assert metadata["architecture"]["hidden_layers"] == config.depth
    assert metadata["architecture"]["hidden_width"] == config.width
    assert metadata["loss"]["name"] == "binary cross entropy"
    assert metadata["regularization"]["weight_decay"] == [0.0]
    assert metadata["architecture"]["output_activation"].startswith("sigmoid")
    assert metadata["architecture"]["layers"][-1]["type"] == "Sigmoid"
    assert not metadata["checkpoint_selection"]["early_stopping_enabled"]
    # History reports the learning rate actually used for each training epoch.
    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history) == config.epochs
    assert history[0]["learning_rate"] == pytest.approx(config.learning_rate)
    assert history[-1]["learning_rate"] == pytest.approx(1e-9)
    assert metadata["epochs_completed"] == config.epochs
    assert metadata["lowest_learning_rate_used"] == 1e-9
    assert metadata["final_learning_rate_used"] == 1e-9
    assert metadata["preprocessing"]["fit_partition"].startswith("training only")
    # Mimic an older completed run lacking descriptive architecture metadata.
    # Loading it must enrich the JSON while preserving weights and signature.
    model_bytes = (tmp_path / "model.pt").read_bytes()
    old_signature = metadata["signature"]
    for key in ("architecture", "loss", "optimizer", "scheduler", "regularization",
                "checkpoint_selection", "preprocessing", "architecture_metadata_version"):
        metadata.pop(key)
    (tmp_path / "training.json").write_text(json.dumps(metadata))
    # Completed-model reuse must recover the same held-out calibration too.
    reused = fit_density_ratio(numerator[:600], denominator[:600],
                               numerator[600:800], denominator[600:800], tmp_path, config)
    np.testing.assert_array_equal(reused.predict_ratio(denominator[1000:]),
                                  restored.predict_ratio(denominator[1000:]))
    enriched = json.loads((tmp_path / "training.json").read_text())
    assert enriched["signature"] == old_signature
    assert enriched["architecture"]["hidden_width"] == config.width
    assert (tmp_path / "model.pt").read_bytes() == model_bytes


@pytest.mark.skipif(not HAS_TRAINING, reason="optional toolkit training dependencies")
def test_explicit_sigmoid_classifier_has_stable_bce_and_ratio_tails():
    import torch
    from poodemo.ratio_network import SigmoidDensityRatioLightning
    module = SigmoidDensityRatioLightning(n_hidden=0, input_dim=3)
    with torch.no_grad():
        module.out.weight.zero_()
        module.out.weight[0, 0] = 1.
        module.out.bias.zero_()
    x = torch.tensor([[-100., 0., 0.], [0., 0., 0.], [100., 0., 0.]])
    probabilities = module(x)
    torch.testing.assert_close(probabilities, torch.sigmoid(module.forward_logits(x)))
    assert isinstance(module.sigmoid, torch.nn.Sigmoid)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    predictor = Predictor(module, np.zeros(3), np.ones(3), "ratio", ("den", "num"), device="cpu")
    np.testing.assert_allclose(predictor.predict_raw_log_ratio(x.numpy()), [-100., 0., 100.])
    ratios = predictor.predict_raw_ratio(x.numpy())
    assert np.all(np.isfinite(ratios)) and np.all(ratios > 0)
    np.testing.assert_allclose(ratios, np.exp([-100., 0., 100.]))
    # Deliberately misclassified saturated outputs must retain finite gradients.
    target = torch.tensor([1., 1., 0.])
    module.log = lambda *args, **kwargs: None
    loss = module.training_step((x, target, torch.ones(3)), 0)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        module.forward_logits(x).ravel(), target)
    torch.testing.assert_close(loss, expected)
    actual_grad = torch.autograd.grad(loss, tuple(module.parameters()))
    expected_grad = torch.autograd.grad(expected, tuple(module.parameters()))
    for actual, wanted in zip(actual_grad, expected_grad):
        assert torch.all(torch.isfinite(actual))
        torch.testing.assert_close(actual, wanted)
    assert actual_grad[0][0, 0] > 60


@pytest.mark.skipif(not HAS_TRAINING, reason="optional toolkit training dependencies")
@pytest.mark.parametrize("epochs", [1, 2, 20, 80])
def test_ratio_learning_rate_floor_is_used_during_training(epochs):
    import torch
    from poodemo.ratio_network import LogLinearFloorLR
    optimizer = torch.optim.NAdam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3, weight_decay=0.)
    scheduler = LogLinearFloorLR(optimizer, min_lr=1e-9, total_epochs=epochs, decay_fraction=.9)
    used_rates = []
    for _ in range(epochs):
        used_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert used_rates[-1] == 1e-9
    assert np.all(np.diff(used_rates) <= 0)
    assert all(rate >= 1e-9 for rate in used_rates)
    assert used_rates[scheduler.decay_epochs:] == [1e-9] * (epochs - scheduler.decay_epochs)
    if epochs > 1:
        assert used_rates[0] == pytest.approx(1e-3)


@pytest.mark.skipif(not HAS_TRAINING, reason="optional toolkit training dependencies")
def test_production_ratio_architecture_and_legacy_checkpoint_loading(tmp_path):
    import torch
    from nsbi_common_utils.lightning_tools import DensityRatioLightning
    from poodemo.ratio_network import SigmoidDensityRatioLightning
    config = TrainingConfig()
    assert config.batch_size == 1024
    assert config.width == 1024
    module = SigmoidDensityRatioLightning()
    predictor = Predictor(module, np.zeros(3), np.ones(3), "ratio", ("den", "num"), device="cpu")
    report = training_metadata(predictor)
    assert report["architecture"]["trainable_parameters"] == 2104321
    assert [(x["input_features"], x["output_features"]) for x in report["architecture"]["layers"]
            if x["type"] == "Linear"] == [(3, 1024), (1024, 1024), (1024, 1024), (1024, 1)]
    assert report["architecture"]["layers"][-1]["type"] == "Sigmoid"
    assert report["optimizer"]["parameter_group_weight_decay"] == [0.0]
    assert not report["architecture"]["dropout_layers"]
    assert not report["architecture"]["batch_normalization"]
    assert report["scheduler"]["min_learning_rate"] == 1e-9
    legacy = Predictor(DensityRatioLightning(input_dim=3, n_neurons=4, n_hidden=1, use_log_loss=True),
                       np.zeros(3), np.ones(3), "ratio", ("den", "num"), device="cpu")
    legacy.save(tmp_path)
    payload = torch.load(tmp_path / "model.pt", weights_only=False)
    payload.pop("model_type")  # Original version did not write this field.
    torch.save(payload, tmp_path / "model.pt")
    x = np.random.default_rng(18).normal(size=(20, 3)).astype(np.float32)
    np.testing.assert_array_equal(load_predictor(tmp_path, device="cpu").predict_ratio(x),
                                  legacy.predict_ratio(x))


@pytest.mark.parametrize("ensemble_size", [1, 3])
def test_member_validation_callback_runs_before_next_training(monkeypatch, tmp_path, ensemble_size):
    from pathlib import Path
    from types import SimpleNamespace
    import poodemo.training as training
    events = []

    def fake_fit(train_arrays, validation_arrays, output_dir, config, kind, labels, force):
        index = len([e for e in events if e[0] == "fit"])
        events.append(("fit", index))
        return SimpleNamespace(output_dir=Path(output_dir).resolve(), labels=tuple(labels), index=index)

    def callback(predictor, index, total):
        assert total == ensemble_size and predictor.index == index
        events.append(("validation", index))

    monkeypatch.setattr(training, "_fit", fake_fit)
    x = np.zeros((4, 3))
    fit_density_ratio(x, x, x, x, tmp_path, TrainingConfig.smoke(ensemble_size=ensemble_size),
                      member_callback=callback)
    assert events == [(event, i) for i in range(ensemble_size) for event in ("fit", "validation")]


@pytest.mark.skipif(not HAS_TRAINING, reason="optional toolkit training dependencies")
def test_interrupted_training_preserves_history_and_reports_used_floor(monkeypatch, tmp_path):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    rng = np.random.default_rng(21)
    denominator = rng.normal(size=(120, 3)).astype(np.float32)
    numerator = rng.normal(.3, size=(120, 3)).astype(np.float32)
    config = TrainingConfig.smoke(epochs=4, max_train_per_class=80,
                                  max_validation_per_class=40, accelerator="cpu")
    real_trainer = pl.Trainer

    class InterruptAfterTwoEpochs(Callback):
        def on_train_epoch_start(self, trainer, module):
            if trainer.current_epoch == 2:
                raise RuntimeError("deliberate training interruption")

    class InterruptedTrainer(real_trainer):
        def __init__(self, **kwargs):
            kwargs["callbacks"] = [*kwargs["callbacks"], InterruptAfterTwoEpochs()]
            super().__init__(**kwargs)

    monkeypatch.setattr(pl, "Trainer", InterruptedTrainer)
    arguments = (numerator[:80], denominator[:80], numerator[80:], denominator[80:], tmp_path, config)
    with pytest.raises(RuntimeError, match="deliberate training interruption"):
        fit_density_ratio(*arguments)
    first_rows = json.loads((tmp_path / "history.json").read_text())
    assert [row["epoch"] for row in first_rows] == [0, 1]
    # Mimic history being written before a checkpoint at a replayed epoch.
    (tmp_path / "history.json").write_text(json.dumps([
        *first_rows, {"epoch": 2, "learning_rate": 999., "val_loss": 999.}]))
    monkeypatch.setattr(pl, "Trainer", real_trainer)
    fit_density_ratio(*arguments)
    rows = json.loads((tmp_path / "history.json").read_text())
    assert rows[:2] == first_rows
    assert [row["epoch"] for row in rows] == [0, 1, 2, 3]
    np.testing.assert_allclose([row["learning_rate"] for row in rows], [1e-3, 1e-5, 1e-7, 1e-9])
    metadata = json.loads((tmp_path / "training.json").read_text())
    assert metadata["complete"] and metadata["epochs_completed"] == 4
    assert metadata["lowest_learning_rate_used"] == 1e-9
    assert metadata["final_learning_rate_used"] == 1e-9
    assert "earlier epoch" in metadata["checkpoint_selection"]["learning_rate_summary"]
