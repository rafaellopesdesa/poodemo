"""Training contracts: real toolkit checkpoint roundtrip and held-out selection."""
import importlib.util
import json

import numpy as np
import pytest

from poodemo.training import (TrainingConfig, choose_preselection_threshold,
                             fit_density_ratio, load_predictor, normalize_ratio)


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


@pytest.mark.skipif(not HAS_TRAINING, reason="optional toolkit training dependencies")
def test_toolkit_model_roundtrip_and_independent_normalization(tmp_path):
    rng = np.random.default_rng(41)
    denominator = rng.normal(size=(1200, 3)).astype(np.float32)
    numerator = rng.normal(.4, size=(1200, 3)).astype(np.float32)
    config = TrainingConfig.smoke(epochs=2, max_train_per_class=400,
                                  max_validation_per_class=100, accelerator="cpu")
    model = fit_density_ratio(numerator[:600], denominator[:600],
                              numerator[600:800], denominator[600:800], tmp_path, config)
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
    # Completed-model reuse must recover the same held-out calibration too.
    reused = fit_density_ratio(numerator[:600], denominator[:600],
                               numerator[600:800], denominator[600:800], tmp_path, config)
    np.testing.assert_array_equal(reused.predict_ratio(denominator[1000:]),
                                  restored.predict_ratio(denominator[1000:]))
