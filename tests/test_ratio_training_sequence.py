"""Immediate independent diagnostics must finish before the next classifier starts."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from poodemo import ratio_validation, training


@pytest.mark.parametrize("ensemble_size", [1, 3])
def test_each_trained_or_reused_member_is_validated_before_next_fit(tmp_path, monkeypatch,
                                                                ensemble_size):
    events = []
    predictors = []

    def fake_fit(train, validation, output_dir, config, kind, labels, force=False):
        member = SimpleNamespace(labels=labels, output_dir=Path(output_dir).resolve())
        predictors.append(member)
        events.append(("fit", len(predictors) - 1))
        return member

    def validate(member, index, total):
        assert member is predictors[index]
        assert total == ensemble_size
        events.append(("validate", index))

    monkeypatch.setattr(training, "_fit", fake_fit)
    sample = np.zeros((10, 3))
    config = training.TrainingConfig.smoke(ensemble_size=ensemble_size)
    training.fit_density_ratio(sample, sample, sample, sample, tmp_path, config,
                               member_callback=validate)
    assert events == [(event, index) for index in range(ensemble_size)
                      for event in ("fit", "validate")]


def test_failed_member_validation_stops_before_training_next_member(tmp_path, monkeypatch):
    fits = []

    def fake_fit(train, validation, output_dir, config, kind, labels, force=False):
        fits.append(output_dir)
        return SimpleNamespace(labels=labels, output_dir=Path(output_dir).resolve())

    def validate(member, index, total):
        raise RuntimeError("diagnostic prediction is nonfinite")

    monkeypatch.setattr(training, "_fit", fake_fit)
    sample = np.zeros((10, 3))
    with pytest.raises(RuntimeError, match="nonfinite"):
        training.fit_density_ratio(sample, sample, sample, sample, tmp_path,
                                   training.TrainingConfig.smoke(ensemble_size=3),
                                   member_callback=validate)
    assert len(fits) == 1


def test_single_member_plots_use_raw_scores_without_normalizing_validation_bank(tmp_path,
                                                                               monkeypatch):
    class RawPredictor:
        def predict_raw_log_ratio(self, x):
            return np.full(len(x), np.log(2.))

        def predict_log_ratio(self, x):
            raise AssertionError("Individual-member checks must use raw network only")

    seen = []
    def capture_plot(table, numerator, denominator, **kwargs):
        seen.append((table.copy(), kwargs))
        return object()

    monkeypatch.setattr(ratio_validation, "plot_reweighting", capture_plot)
    monkeypatch.setattr(ratio_validation, "plot_calibration", capture_plot)
    sample = np.zeros((100, 3))
    result = ratio_validation.plot_ratio_task(
        "B", "S", {"B": sample, "S": sample}, RawPredictor(),
        tmp_path / "member_00", results=tmp_path / "tables" / "member_00",
        edges=[-1., 1.], score_bins=5, stages=("raw",), label="member 1/3")
    assert len(seen) == 2
    assert all(kwargs["label"] == "member 1/3" for _, kwargs in seen)
    assert set(result["reweighting"].stage) == {"raw"}
    np.testing.assert_allclose(result["reweighting"].reweighted_over_target, 2.)
    valid = result["calibration"].query("calibration_valid")
    np.testing.assert_allclose(valid.predicted, 2. / 3)
    np.testing.assert_allclose(valid.residual, .5 - 2. / 3)
    assert (tmp_path / "tables/member_00/03_reweighting_B_over_S.csv").exists()
    assert (tmp_path / "tables/member_00/03_calibration_B_over_S.csv").exists()


def test_pipeline_displays_independent_member_and_ensemble_checks_before_next_task(tmp_path,
                                                                                 monkeypatch):
    from poodemo import pipeline
    from poodemo.data import Run, save_json

    tasks = [("SBI", "S"), ("B", "S")]
    events = []
    diagnostic_bank = {name: np.full((20, 3), 9.) for name in ("S", "SBI", "B")}
    split_samples = {split: np.full((10, 3), index) for index, split in enumerate(
        ("train", "validation", "calibration", "integration"))}
    run = Run(tmp_path, {"seed": 100}, None)
    save_json(run.path("selected", "manifest.json"), {"records": [
        {"sample": name, "split": "integration", "expected_yield": 1.}
        for name in diagnostic_bank]})
    run.path("results").mkdir()

    def fake_fit(train, validation, output_dir, config, kind, labels, force=False):
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        index = int(output_dir.name.split("_")[-1])
        key = f"{labels[1]}_over_{labels[0]}"
        assert all(x is split_samples["train"] for x in train)
        assert all(x is split_samples["validation"] for x in validation)
        events.append((key, "fit", index))
        return SimpleNamespace(labels=labels, output_dir=output_dir)

    def normalize(predictor, sample):
        # Normalization is fitted once for the ensemble on its reserved split;
        # neither individual members nor the independent bank are normalized.
        assert isinstance(predictor, training.RatioEnsemble)
        assert sample is split_samples["calibration"]
        predictor.normalized_for_test = True
        events.append((predictor.output_dir.name, "normalize", None))
        return {"normalization": 1.}

    def plot_task(num, den, samples, predictor, output, **options):
        assert samples is diagnostic_bank
        assert options["score_bins"] == 8
        stage = Path(output).name
        assert str(options["results"]).endswith(f"{num}_over_{den}/{stage}")
        if isinstance(predictor, training.RatioEnsemble):
            assert predictor.normalized_for_test
            assert options["stages"] == ("raw", "normalized")
            index = None
        else:
            assert options["stages"] == ("raw",)
            index = int(stage.split("_")[-1]) - 1
        events.append((f"{num}_over_{den}", "plot", index))
        return {"reweighting": None, "calibration": None, "figures": {}}

    def display(report):
        assert report["stage"] == ("ensemble" if report["member_index"] is None
                                   else f"member_{report['member_index'] + 1:02d}")
        events.append((report["task"], "display", report["member_index"]))

    def diagnostics(predictor, numerator, denominator, **kwargs):
        assert numerator is split_samples["integration"]
        assert denominator is split_samples["integration"]
        return {"reliability": {"edges": [0., 1.], "predicted": [.5], "observed": [.5]}}

    monkeypatch.setattr(pipeline, "RATIO_TASKS", tasks)
    monkeypatch.setattr(pipeline, "_training_config", lambda run: training.TrainingConfig.smoke(ensemble_size=2))
    monkeypatch.setattr(pipeline, "selected_split", lambda run, name, split: split_samples[split])
    monkeypatch.setattr(training, "_fit", fake_fit)
    monkeypatch.setattr(training, "normalize_ratio", normalize)
    monkeypatch.setattr(training, "ratio_diagnostics", diagnostics)
    monkeypatch.setattr(ratio_validation, "plot_ratio_task", plot_task)
    result = pipeline.train_ratios(run, validation_samples=diagnostic_bank,
                                  on_validation=display, validation_options={"score_bins": 8})
    expected = []
    for num, den in tasks:
        key = f"{num}_over_{den}"
        for index in range(2):
            expected.extend((key, event, index) for event in ("fit", "plot", "display"))
        expected.extend((key, event, None) for event in ("normalize", "plot", "display"))
    assert events == expected
    assert result.ratio.tolist() == ["SBI/S", "B/S"]
