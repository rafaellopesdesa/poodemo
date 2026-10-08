"""Memory-conscious training with the pinned NSBI toolkit's Lightning models.

The toolkit owns both network architectures, losses and optimizers.  This adapter
changes data loading (balanced, streamed minibatches rather than several pandas
copies), checkpoints the best validation model, fits preprocessing on training
events only, and fits ratio calibration on the separate calibration partition.
Workspace construction may subsequently integrate the learned model on independent
quadrature nodes to normalize its probability density. That numerical integration
is not a fit of network weights or of the calibration map.

All public training inputs are NumPy arrays or memory-mapped ``.npy`` views with
shape (n_events, 3).  Split the generated events *before* applying preselection.
PyTorch and the NSBI toolkit are imported only by functions that need them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import math
from typing import Mapping

import numpy as np
from scipy.special import expit, logsumexp, softmax

TOOLKIT_REVISION = "fc09848fc6540fd32310faebbe9db6eea7ecd17b"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 80
    patience: int = 15
    batch_size: int = 8192
    width: int = 128
    depth: int = 3
    learning_rate: float = 1e-3
    lr_factor: float = 0.5
    lr_patience: int = 15
    max_train_per_class: int | None = 2_000_000
    max_validation_per_class: int | None = 200_000
    scaler_per_class: int = 100_000
    ensemble_size: int = 1
    seed: int = 20261006
    accelerator: str = "auto"
    progress_bar: bool = True
    precision: str = "32-true"

    @classmethod
    def smoke(cls, **kwargs):
        """Small executable test, deliberately not a production quality fit."""
        return cls(**({"epochs": 2, "patience": 2, "batch_size": 256,
                       "width": 24, "depth": 2, "max_train_per_class": 1500,
                       "max_validation_per_class": 600, "progress_bar": False,
                       "ensemble_size": 1} | kwargs))


def _json_write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def _validate_arrays(arrays):
    dim = None
    for x in arrays:
        if len(x.shape) != 2 or len(x) < 2:
            raise ValueError("Each class needs at least two events in a 2D array.")
        if dim is None:
            dim = x.shape[1]
        if x.shape[1] != dim:
            raise ValueError("All classes must have the same feature dimension.")
    return dim


def _signature(arrays, config, kind, labels):
    """Catch stale completed checkpoints without hashing gigabytes of data."""
    h = hashlib.sha256()
    h.update(json.dumps({"config": asdict(config), "kind": kind, "labels": labels,
                         "toolkit_revision": TOOLKIT_REVISION}, sort_keys=True).encode())
    for x in arrays:
        h.update(str(x.shape).encode())
        # These are iid data partitions; a sample fingerprint identifies a run.
        for block in (x[:16], x[-16:]):
            h.update(np.asarray(block, dtype=np.float32).tobytes())
    return h.hexdigest()


def _feature_scaling(arrays, max_per_class):
    """Equal class mixture, estimated using training events only."""
    mean, second = [], []
    for x in arrays:
        n = min(len(x), max_per_class)
        index = np.linspace(0, len(x) - 1, n, dtype=np.int64)
        a = np.asarray(x[index], dtype=np.float64)
        if not np.all(np.isfinite(a)):
            raise ValueError("Training features contain nonfinite values.")
        mean.append(a.mean(axis=0))
        second.append(np.square(a).mean(axis=0))
    m = np.mean(mean, axis=0)
    s = np.sqrt(np.maximum(np.mean(second, axis=0) - m * m, 1e-12))
    return m.astype(np.float32), s.astype(np.float32)


def _network_metadata(model, kind, labels, mean, scale, config=None):
    """Describe the instantiated pinned-toolkit network, not a guessed diagram.

    Constructing its optimizer/scheduler does not fit or change network weights.
    The loss descriptions follow the pinned toolkit's ``training_step``; our
    data loader supplies unit event weights and equal class minibatches.
    """
    import torch
    layers = []
    for name, layer in model.named_modules():
        if not name or any(layer.children()):
            continue
        record = {"name": name, "type": type(layer).__name__}
        if isinstance(layer, torch.nn.Linear):
            record.update(input_features=layer.in_features, output_features=layer.out_features,
                          bias=layer.bias is not None)
        elif isinstance(layer, torch.nn.SiLU):
            record["function"] = "x * sigmoid(x) (Swish / SiLU)"
        elif isinstance(layer, torch.nn.ReLU):
            record["function"] = "max(0, x)"
        elif isinstance(layer, torch.nn.Tanh):
            record["function"] = "tanh(x)"
        if isinstance(layer, torch.nn.modules.dropout._DropoutNd):
            record["probability"] = float(layer.p)
        layers.append(record)
    optimizers = model.configure_optimizers()
    optimizer = optimizers["optimizer"]
    scheduler = optimizers["lr_scheduler"]["scheduler"]
    # JSON roundtrip converts tuple-valued optimizer defaults, e.g. betas.
    defaults = json.loads(json.dumps(optimizer.defaults, allow_nan=False))
    dropout = [r for r in layers if "probability" in r]
    batch_norm = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                     for m in model.modules())
    binary = kind == "ratio"
    scheduler_details = {"type": type(scheduler).__name__, "interval": "epoch",
                         "frequency": 1}
    if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        scheduler_details.update(step_size=int(scheduler.step_size), gamma=float(scheduler.gamma),
                                 rule="lr(epoch) = initial_lr * gamma ** floor(epoch / step_size)")
    elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler_details.update(monitor="val_loss", mode=scheduler.mode,
                                 factor=float(scheduler.factor), patience=int(scheduler.patience),
                                 min_lr=list(scheduler.min_lrs))
    config = asdict(config) if isinstance(config, TrainingConfig) else (config or {})
    return {
        "architecture": {
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "type": "fully connected feed-forward multilayer perceptron",
            "input_features": int(len(mean)), "layers": layers,
            "hidden_layers": int(model.hparams.n_hidden),
            "hidden_width": int(model.hparams.n_neurons),
            "hidden_activation": str(model.hparams.activation),
            "output_activation": "linear (raw logit)" if binary else "linear (class logits)",
            "probability_transform": "sigmoid(logit)" if binary else "softmax(class logits)",
            "raw_ratio_transform": "exp(logit), numerator / denominator" if binary else None,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "dropout_layers": dropout, "batch_normalization": batch_norm,
        },
        "loss": {
            "name": "binary cross entropy" if binary else "multiclass cross entropy",
            "implementation": ("torch.nn.functional.binary_cross_entropy_with_logits" if binary
                               else "torch.nn.functional.cross_entropy"),
            "reduction": "arithmetic mean over the minibatch (all event weights are one)",
            "label_order": list(labels), "class_prior": [1 / len(labels)] * len(labels),
            "label_smoothing": 0.0,
        },
        "optimizer": {"type": type(optimizer).__name__, "defaults": defaults,
                      "parameter_group_weight_decay": [float(g.get("weight_decay", 0))
                                                       for g in optimizer.param_groups]},
        "scheduler": scheduler_details,
        "regularization": {
            "dropout": bool(dropout), "batch_normalization": batch_norm,
            "weight_decay": [float(g.get("weight_decay", 0)) for g in optimizer.param_groups],
            "explicit_loss_penalties": "none (no L1 or L2 penalty)",
        },
        "checkpoint_selection": {"monitor": "val_loss", "mode": "min",
                                 "restore": "best validation checkpoint",
                                 "early_stopping_patience": config.get("patience"),
                                 "max_epochs": config.get("epochs")},
        "preprocessing": {"type": "featurewise standardization: (x - mean) / scale",
                          "fit_partition": "training only, equal mixture of classes",
                          "mean": np.asarray(mean).tolist(), "scale": np.asarray(scale).tolist()},
    }


def training_metadata(predictor, *, persist=True):
    """Inspect architecture/training details, including reused saved models.

    ``persist=True`` enriches an existing ``training.json`` without changing its
    signature, completion state, checkpoint, calibration, or model weights.
    Ensemble reports include each member and the ratio-averaging convention.
    """
    if isinstance(predictor, RatioEnsemble):
        return {"kind": "ratio ensemble", "ensemble_size": len(predictor.members),
                "aggregation": "arithmetic mean of member density ratios, not logits",
                "members": [training_metadata(m, persist=persist) for m in predictor.members]}
    path = predictor.output_dir / "training.json" if predictor.output_dir else None
    previous = json.loads(path.read_text()) if path is not None and path.exists() else {}
    report = dict(previous)
    report.update(_network_metadata(predictor.model, predictor.kind, predictor.labels,
                                    predictor.mean, predictor.scale, previous.get("config")))
    report["architecture_metadata_version"] = 1
    if persist and path is not None and path.exists() and report != previous:
        _json_write(path, report)
    return report


class Predictor:
    """Best-validation toolkit model, including training-only standardization.

    ``predict_proba`` returns preselection class probabilities in ``labels`` order.
    ``predict_log_ratio`` and ``predict_ratio`` return numerator/denominator ratios
    for binary models. Equal class minibatches imply a 1:1 training prior, so no
    yield prior enters the conversion ``r = exp(logit)``. Physical process yields
    must be supplied separately to the extended likelihood.
    """
    def __init__(self, model, mean, scale, kind, labels, output_dir=None,
                 log_normalization=0.0, calibration=None, device="auto"):
        import torch
        self.model = model.eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.kind = kind
        self.labels = tuple(labels)
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.log_normalization = float(log_normalization)
        self.calibration = calibration
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model.to(self.device)

    def _raw(self, x, batch_size=65536):
        import torch
        output = []
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                a = (np.asarray(x[start:start + batch_size], dtype=np.float32) - self.mean) / self.scale
                v = self.model(torch.from_numpy(np.ascontiguousarray(a)).to(self.device))
                output.append(v.detach().cpu().numpy())
        shape = (0, len(self.labels)) if self.kind == "preselection" else (0, 1)
        return np.concatenate(output) if output else np.empty(shape, dtype=np.float32)

    def predict_proba(self, x, batch_size=65536):
        raw = self._raw(x, batch_size)
        if self.kind == "preselection":
            return softmax(raw, axis=1)
        return expit(raw[:, 0])

    def predict_raw_log_ratio(self, x, batch_size=65536):
        """Raw BCE logit, before isotonic calibration or scalar normalization."""
        if self.kind != "ratio":
            raise TypeError("This is a multiclass preselection model, not a ratio model.")
        result = self._raw(x, batch_size)[:, 0].astype(np.float64)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("The network produced nonfinite raw log density ratios.")
        return result

    def predict_raw_ratio(self, x, batch_size=65536):
        """Exponentiated raw BCE logit, before all post-training adjustments."""
        result = self.predict_raw_log_ratio(x, batch_size)
        if np.any(np.abs(result) > 700):
            raise FloatingPointError("Extreme predicted raw log ratio; inspect the network.")
        return np.exp(result)

    def predict_log_ratio(self, x, batch_size=65536, normalized=True):
        if self.kind != "ratio":
            raise TypeError("This is a multiclass preselection model, not a ratio model.")
        result = self._raw(x, batch_size)[:, 0].astype(np.float64)
        if self.calibration is not None:
            # The optional toolkit isotonic map is fitted on a separate partition.
            score = self.calibration.cali_pred(expit(result))
            score = np.clip(score, 1e-9, 1 - 1e-9)
            result = np.log(score) - np.log1p(-score)
        if normalized:
            result -= self.log_normalization
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("The network produced nonfinite log density ratios.")
        return result

    def predict_ratio(self, x, batch_size=65536, normalized=True):
        lr = self.predict_log_ratio(x, batch_size, normalized)
        if np.any(np.abs(lr) > 700):
            raise FloatingPointError("Extreme predicted log ratio; inspect calibration before fitting.")
        return np.exp(lr)

    def predict(self, x, batch_size=65536):
        if self.kind == "preselection":
            return self.predict_proba(x, batch_size)
        return self.predict_ratio(x, batch_size)

    __call__ = predict

    def save(self, output_dir=None):
        import torch
        import joblib
        directory = Path(output_dir or self.output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"kind": self.kind, "labels": self.labels,
                   "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                   "log_normalization": self.log_normalization,
                   "hparams": dict(self.model.hparams),
                   "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                   "toolkit_revision": TOOLKIT_REVISION}
        temporary = directory / "model.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(directory / "model.pt")
        if self.calibration is not None:
            joblib.dump(self.calibration, directory / "calibrator.joblib")
        elif (directory / "calibrator.joblib").exists():
            (directory / "calibrator.joblib").unlink()
        self.output_dir = directory
        return directory

    @classmethod
    def load(cls, directory, device="auto"):
        import torch
        import joblib
        from nsbi_common_utils.lightning_tools import DensityRatioLightning, MultiClassLightning
        directory = Path(directory).resolve()
        # This checkpoint is generated by this repository; only load trusted files.
        p = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
        model_class = MultiClassLightning if p["kind"] == "preselection" else DensityRatioLightning
        model = model_class(**p["hparams"])
        model.load_state_dict(p["state_dict"])
        cal = joblib.load(directory / "calibrator.joblib") if (directory / "calibrator.joblib").exists() else None
        return cls(model, p["mean"], p["scale"], p["kind"], p["labels"], directory,
                   p["log_normalization"], cal, device)

    def export_onnx(self, path=None):
        """Optional portable export, outputs logits with embedded standardization.

        Main notebook inference uses the native checkpoint so ONNX is optional.
        For ratio models the output is the raw network log ratio, before any
        isotonic or mean-normalization correction; those remain in metadata.
        """
        import torch
        path = Path(path or self.output_dir / "model.onnx")
        class Wrapped(torch.nn.Module):
            def __init__(self, predictor):
                super().__init__()
                self.network = predictor.model
                self.register_buffer("mean", torch.tensor(predictor.mean, device=predictor.device))
                self.register_buffer("scale", torch.tensor(predictor.scale, device=predictor.device))
            def forward(self, x):
                return self.network((x - self.mean) / self.scale)
        torch.onnx.export(Wrapped(self).eval(), torch.zeros(1, len(self.mean), device=self.device),
                          str(path), input_names=["features"], output_names=["logits"],
                          dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)
        return path


class RatioEnsemble:
    """Arithmetic mean of independently trained density ratios (not logits)."""
    def __init__(self, members, output_dir=None, log_normalization=0.0):
        self.members = list(members)
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.log_normalization = float(log_normalization)
        self.kind = "ratio"
        self.labels = self.members[0].labels

    def predict_raw_log_ratio(self, x, batch_size=65536):
        """Log arithmetic mean of raw member ratios, before all adjustments."""
        values = np.stack([m.predict_raw_log_ratio(x, batch_size) for m in self.members])
        return logsumexp(values, axis=0) - math.log(len(self.members))

    def predict_raw_ratio(self, x, batch_size=65536):
        result = self.predict_raw_log_ratio(x, batch_size)
        if np.any(np.abs(result) > 700):
            raise FloatingPointError("Extreme ensemble raw log ratio; inspect the networks.")
        return np.exp(result)

    def predict_log_ratio(self, x, batch_size=65536, normalized=True):
        values = np.stack([m.predict_log_ratio(x, batch_size) for m in self.members])
        result = logsumexp(values, axis=0) - math.log(len(self.members))
        return result - self.log_normalization if normalized else result

    def predict_ratio(self, x, batch_size=65536, normalized=True):
        lr = self.predict_log_ratio(x, batch_size, normalized)
        if np.any(np.abs(lr) > 700):
            raise FloatingPointError("Extreme ensemble ratio; inspect calibration.")
        return np.exp(lr)

    predict = predict_ratio
    __call__ = predict_ratio

    def save(self, output_dir=None):
        directory = Path(output_dir or self.output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        _json_write(directory / "ensemble.json", {"members": [str(m.output_dir.relative_to(directory))
            for m in self.members], "log_normalization": self.log_normalization})
        self.output_dir = directory
        return directory

    @classmethod
    def load(cls, directory, device="auto"):
        directory = Path(directory).resolve()
        p = json.loads((directory / "ensemble.json").read_text())
        return cls([Predictor.load(directory / name, device) for name in p["members"]],
                   directory, p["log_normalization"])


def load_predictor(directory, device="auto"):
    directory = Path(directory).resolve()
    return (RatioEnsemble.load(directory, device) if (directory / "ensemble.json").exists()
            else Predictor.load(directory, device))


def _fit(train_arrays, validation_arrays, output_dir, config, kind, labels, force=False):
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, IterableDataset
    from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
    from nsbi_common_utils.lightning_tools import DensityRatioLightning, MultiClassLightning

    dimension = _validate_arrays([*train_arrays, *validation_arrays])
    if config.batch_size < len(train_arrays):
        raise ValueError("batch_size must be at least the number of classes.")
    if config.epochs < 1:
        raise ValueError("epochs must be positive.")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    signature = _signature([*train_arrays, *validation_arrays], config, kind, labels)
    metadata_path = directory / "training.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["signature"] != signature and not force:
            raise ValueError(f"Training inputs/config changed in {directory}. Use a new model directory or force=True.")
        if previous["signature"] == signature and previous.get("complete") and not force:
            predictor = Predictor.load(directory, device="cpu" if config.accelerator == "cpu" else "auto")
            training_metadata(predictor)
            return predictor
    else:
        previous = {}
    if force:
        # Only reset checkpoints belonging to this one explicitly retrained model.
        for p in (directory / "checkpoints").glob("*.ckpt"):
            p.unlink()
    pl.seed_everything(config.seed, workers=True)
    mean, scale = _feature_scaling(train_arrays, config.scaler_per_class)
    nclass = len(train_arrays)
    per_class_batch = max(config.batch_size // nclass, 1)

    class BalancedBatches(IterableDataset):
        def __init__(self, arrays, training):
            self.arrays, self.training, self.epoch = arrays, training, 0
            cap = config.max_train_per_class if training else config.max_validation_per_class
            count = max(map(len, arrays)) if training else min(map(len, arrays))
            self.count = min(count, cap) if cap is not None else count
            self.n_batches = math.ceil(self.count / per_class_batch)
        def __len__(self):
            return self.n_batches
        def __iter__(self):
            rng = np.random.default_rng(config.seed + self.epoch * 1009)
            if self.training:
                self.epoch += 1
            for step in range(self.n_batches):
                count = min(per_class_batch, self.count - step * per_class_batch)
                xx, yy = [], []
                for label, a in enumerate(self.arrays):
                    if self.training:
                        # Equal class draws implement a balanced prior even when
                        # postselection acceptances differ substantially.
                        index = rng.integers(0, len(a), count)
                    else:
                        positions = np.arange(step * per_class_batch, step * per_class_batch + count)
                        index = np.minimum((positions * len(a) / self.count).astype(np.int64), len(a) - 1)
                    xx.append((np.asarray(a[index], dtype=np.float32) - mean) / scale)
                    yy.append(np.full(count, label, dtype=np.int64))
                x, y = np.concatenate(xx), np.concatenate(yy)
                if self.training:
                    perm = rng.permutation(len(x))
                    x, y = x[perm], y[perm]
                yield torch.from_numpy(x), torch.from_numpy(y), torch.ones(len(y))

    class History(Callback):
        def __init__(self):
            self.rows = []
        def on_validation_epoch_end(self, trainer, module):
            if trainer.sanity_checking:
                return
            record = {"epoch": int(trainer.current_epoch),
                      "learning_rate": float(trainer.optimizers[0].param_groups[0]["lr"])}
            for name, value in trainer.callback_metrics.items():
                if value.numel() == 1:
                    record[name] = float(value.detach().cpu())
            self.rows.append(record)
            _json_write(directory / "history.json", self.rows)

    kwargs = dict(n_hidden=config.depth, n_neurons=config.width, input_dim=dimension,
                  learning_rate=config.learning_rate, callback_factor=config.lr_factor,
                  callback_patience=config.lr_patience, activation="swish")
    model_class = MultiClassLightning if kind == "preselection" else DensityRatioLightning
    model = model_class(**kwargs, **({"num_classes": nclass} if kind == "preselection" else {"use_log_loss": True}))
    checkpoint = ModelCheckpoint(dirpath=directory / "checkpoints", filename="best",
                                 monitor="val_loss", mode="min", save_top_k=1, save_last=True)
    history = History()
    trainer = pl.Trainer(accelerator=config.accelerator, devices=1, max_epochs=config.epochs,
                         precision=config.precision, logger=False,
                         callbacks=[checkpoint, EarlyStopping(monitor="val_loss", patience=config.patience), history],
                         enable_progress_bar=config.progress_bar, enable_model_summary=False,
                         default_root_dir=directory, num_sanity_val_steps=0, deterministic=True)
    _json_write(metadata_path, {"signature": signature, "complete": False,
                               "kind": kind, "labels": labels, "config": asdict(config),
                               "class_prior": [1 / nclass] * nclass,
                               "training_counts": list(map(len, train_arrays)),
                               "validation_counts": list(map(len, validation_arrays)),
                               "toolkit_revision": TOOLKIT_REVISION,
                               "architecture_metadata_version": 1,
                               **_network_metadata(model, kind, labels, mean, scale, config),
                               "sampling": "equal class minibatches, training draws with replacement"})
    last_checkpoint = directory / "checkpoints" / "last.ckpt"
    resume = str(last_checkpoint) if last_checkpoint.exists() and not force and previous.get("signature") == signature else None
    trainer.fit(model,
                train_dataloaders=DataLoader(BalancedBatches(train_arrays, True), batch_size=None, num_workers=0),
                val_dataloaders=DataLoader(BalancedBatches(validation_arrays, False), batch_size=None, num_workers=0),
                ckpt_path=resume)
    if not checkpoint.best_model_path:
        raise RuntimeError("Training produced no finite best-validation checkpoint.")
    best = model_class.load_from_checkpoint(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    predictor = Predictor(best, mean, scale, kind, labels, directory,
                          device="cpu" if config.accelerator == "cpu" else "auto")
    predictor.save()
    metadata = json.loads(metadata_path.read_text())
    metadata.update(complete=True, best_validation_loss=float(checkpoint.best_model_score),
                    best_checkpoint=str(Path(checkpoint.best_model_path).relative_to(directory)))
    _json_write(metadata_path, metadata)
    return predictor


def fit_preselection(train: Mapping[str, np.ndarray], validation: Mapping[str, np.ndarray],
                     output_dir, config=None, force=False):
    """Train balanced S/B/NI multiclass cross entropy using MultiClassLightning."""
    config = config or TrainingConfig()
    labels = ["S", "B", "NI"]
    return _fit([train[k] for k in labels], [validation[k] for k in labels],
                output_dir, replace(config, ensemble_size=1), "preselection", labels, force)


def fit_density_ratio(numerator_train, denominator_train, numerator_validation,
                      denominator_validation, output_dir, config=None,
                      numerator_label="numerator", denominator_label="denominator", force=False):
    """Train p_numerator / p_denominator, using DensityRatioLightning BCE logits.

    The denominator is label 0 and numerator is label 1, matching the toolkit.
    A three-member production ensemble can be requested via ``ensemble_size=3``.
    Calibration/normalization is an explicit separate call on separate events.
    """
    config = config or TrainingConfig()
    if config.ensemble_size < 1:
        raise ValueError("ensemble_size must be positive.")
    labels = [denominator_label, numerator_label]
    if config.ensemble_size == 1:
        return _fit([denominator_train, numerator_train], [denominator_validation, numerator_validation],
                    output_dir, config, "ratio", labels, force)
    directory = Path(output_dir).resolve()
    members = []
    for i in range(config.ensemble_size):
        member_config = replace(config, seed=config.seed + 100003 * i, ensemble_size=1)
        members.append(_fit([denominator_train, numerator_train], [denominator_validation, numerator_validation],
                            directory / f"member_{i:02d}", member_config, "ratio", labels, force))
    ensemble = RatioEnsemble(members, directory)
    # Normalize the ensemble explicitly after this call. Recomputing its scalar
    # correction on the held-out partition is inexpensive and avoids stale values.
    ensemble.save()
    return ensemble


def normalize_ratio(predictor, denominator_calibration, batch_size=65536):
    """Enforce E_den[r]=1 on calibration events, without claiming full calibration.

    Returns a normalization uncertainty/ESS diagnostic. Integration events remain
    independent of this calibration step and provide a separate closure check.
    Workspace construction can still numerically integrate model normalization
    on its independent quadrature nodes. Repeated calls
    replace (rather than compound) the same scalar correction.
    """
    if len(denominator_calibration) < 2:
        raise ValueError("Need at least two denominator calibration events.")
    log_sum, log_sum_sq = -np.inf, -np.inf
    for start in range(0, len(denominator_calibration), batch_size):
        lr = predictor.predict_log_ratio(denominator_calibration[start:start + batch_size], normalized=False)
        log_sum = np.logaddexp(log_sum, logsumexp(lr))
        log_sum_sq = np.logaddexp(log_sum_sq, logsumexp(2 * lr))
    n = len(denominator_calibration)
    predictor.log_normalization = float(log_sum - math.log(n))
    ess = float(np.exp(2 * log_sum - log_sum_sq))
    relative_se = math.sqrt(max(n / ess - 1, 0) / (n - 1))
    result = {"n_calibration": n, "log_normalization": predictor.log_normalization,
              "normalization": float(np.exp(predictor.log_normalization)),
              "relative_standard_error": relative_se, "effective_sample_size": ess,
              "note": "Mean normalization is necessary but does not ensure ratio calibration."}
    predictor.save()
    _json_write(predictor.output_dir / "normalization.json", result)
    return result


def calibrate_isotonic(predictor, numerator_calibration, denominator_calibration,
                       max_per_class=200_000):
    """Optional toolkit calibration; separate from the default neural ratio.

    Isotonic calibration can flatten tails and is not automatically better. Keep
    its diagnostics alongside the raw model. Fits equal total class weights.
    """
    if not isinstance(predictor, Predictor) or predictor.kind != "ratio":
        raise TypeError("Isotonic calibration applies to a single binary Predictor.")
    from nsbi_common_utils.calibration import IsotonicCalibrator
    scores, labels, weights = [], [], []
    for label, x in enumerate([denominator_calibration, numerator_calibration]):
        index = np.linspace(0, len(x) - 1, min(len(x), max_per_class), dtype=np.int64)
        raw = predictor._raw(x[index])[:, 0]
        scores.append(expit(raw))
        labels.append(np.full(len(index), label))
        weights.append(np.full(len(index), 1 / len(index)))
    predictor.calibration = IsotonicCalibrator(np.concatenate(scores), np.concatenate(labels), np.concatenate(weights))
    predictor.log_normalization = 0.0
    return normalize_ratio(predictor, denominator_calibration)


def choose_preselection_threshold(predictor, calibration: Mapping[str, np.ndarray],
                                  yields: Mapping[str, float], target_ratio=0.1,
                                  n_thresholds=2001, min_ni_events=20):
    """Choose the loosest S-softmax cut reaching the requested physical S/NI.

    Threshold selection uses the calibration partition only. At least
    ``min_ni_events`` must survive, so an empty denominator cannot manufacture an
    apparently successful cut. If no stable cut reaches the target, raise and
    request more calibration events or a revised model rather than silently
    returning a misleading selection.
    """
    if target_ratio <= 0 or yields["S"] <= 0 or yields["NI"] <= 0:
        raise ValueError("Target ratio and process yields must be positive.")
    index_s = list(predictor.labels).index("S")
    scores = {k: np.sort(predictor.predict_proba(calibration[k])[:, index_s]) for k in ("S", "NI")}
    candidates = np.unique(np.concatenate(([0.0], np.quantile(scores["S"], np.linspace(0, 0.999, n_thresholds)))))
    counts = {k: len(v) - np.searchsorted(v, candidates, side="left") for k, v in scores.items()}
    eff = {k: counts[k] / len(scores[k]) for k in scores}
    ratio = np.divide(yields["S"] * eff["S"], yields["NI"] * eff["NI"],
                      out=np.full(len(candidates), np.inf), where=eff["NI"] > 0)
    viable = np.flatnonzero((ratio >= target_ratio) & (counts["NI"] >= min_ni_events) & (counts["S"] > 0))
    if not len(viable):
        valid = counts["NI"] >= min_ni_events
        best = float(np.max(ratio[valid])) if np.any(valid) else 0.0
        raise ValueError(f"No statistically populated cut reaches S/NI={target_ratio:g}; best is {best:g}. "
                         "Increase calibration statistics, train longer, or revise the toy separation.")
    i = int(viable[0])  # ascending threshold => largest signal acceptance
    result = {"threshold": float(candidates[i]), "target_ratio": float(target_ratio),
              "achieved_ratio": float(ratio[i]),
              "efficiencies": {k: float(eff[k][i]) for k in eff},
              "selected_counts": {k: int(counts[k][i]) for k in counts},
              "calibration_counts": {k: len(scores[k]) for k in scores},
              "rule": "signal_softmax >= threshold", "signal_class_index": index_s}
    if predictor.output_dir:
        _json_write(predictor.output_dir / "selection.json", result)
    return result


def ratio_diagnostics(predictor, numerator_evaluation, denominator_evaluation,
                      exact_log_ratio=None, max_events=100_000, bins=30):
    """Independent closure and balanced-class reliability table for notebook plots.

    ``exact_log_ratio`` is a callable for the *selected, normalized* distributions,
    including acceptance factors. Never use this evaluation set for fitting.
    """
    inputs, logs = [], []
    for x in (denominator_evaluation, numerator_evaluation):
        index = np.linspace(0, len(x) - 1, min(len(x), max_events), dtype=np.int64)
        sample = x[index]
        inputs.append(sample)
        logs.append(predictor.predict_log_ratio(sample))
    r_den = np.exp(logs[0])
    report = {"denominator_mean_ratio": float(r_den.mean()),
              "denominator_mean_ratio_se": float(r_den.std(ddof=1) / math.sqrt(len(r_den))),
              "denominator_ratio_ess": float(r_den.sum() ** 2 / np.square(r_den).sum()),
              "evaluation_counts": [len(v) for v in logs]}
    edges = np.linspace(0, 1, bins + 1)
    score0, score1 = expit(logs[0]), expit(logs[1])
    h0, _ = np.histogram(score0, edges)
    h1, _ = np.histogram(score1, edges)
    f0, f1 = h0 / len(score0), h1 / len(score1)
    empirical = np.divide(f1, f0 + f1, out=np.full(bins, np.nan), where=f0 + f1 > 0)
    total_w = np.concatenate((np.full(len(score0), 1 / len(score0)), np.full(len(score1), 1 / len(score1))))
    all_scores = np.concatenate((score0, score1))
    predicted_sum, _ = np.histogram(all_scores, edges, weights=total_w * all_scores)
    predicted = np.divide(predicted_sum, f0 + f1, out=np.full(bins, np.nan), where=f0 + f1 > 0)
    report["reliability"] = {"edges": edges, "predicted": predicted, "empirical": empirical,
                             "denominator_counts": h0, "numerator_counts": h1}
    if exact_log_ratio is not None:
        delta = np.concatenate([lr - exact_log_ratio(x) for x, lr in zip(inputs, logs)])
        report["log_ratio_rmse"] = float(np.sqrt(np.mean(delta * delta)))
        report["log_ratio_bias"] = float(np.mean(delta))
    return report
