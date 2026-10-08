"""Reproducible, resumable disk storage with independent statistical splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import numpy as np

TOOLKIT_COMMIT = "fc09848fc6540fd32310faebbe9db6eea7ecd17b"
PROCESSES = ("S", "SBI", "B", "NI")
SAMPLES = {
    "S": ("S", 0.), "SBI": ("SBI", 0.),
    "B": ("B", 0.), "NI": ("NI", 0.),
    "NI_up": ("NI", 1.), "NI_down": ("NI", -1.),
}
# Stable streams preserve the retained samples when nuisance layouts change.
SAMPLE_STREAMS = {"S": 0, "SBI": 1, "B": 2, "NI": 3, "NI_up": 8, "NI_down": 9}
SPLITS = {"preselection_train": (0., .20), "preselection_validation": (.20, .25),
          "selection_calibration": (.25, .30), "train": (.30, .70),
          "validation": (.70, .80), "calibration": (.80, .85), "integration": (.85, 1.)}
ANALYSIS_SPLITS = ("train", "validation", "calibration", "integration")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def save_array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(tmp, path)


@dataclass
class Run:
    root: Path
    config: dict
    model: object

    def path(self, *parts):
        return self.root.joinpath(*parts)


def default_config(mode="production"):
    if mode not in ("production", "smoke"):
        raise ValueError("mode must be production or smoke")
    small = mode == "smoke"
    return dict(schema_version=6, mode=mode, seed=20261006,
                physics_overrides={},
                toolkit_commit=TOOLKIT_COMMIT,
                n_per_sample=12_000 if small else 5_000_000,
                generation_chunk=10_000 if small else 250_000,
                preselection_train_cap=5_000 if small else 1_000_000,
                ratio_train_cap=6_000 if small else 2_000_000,
                epochs=5 if small else 100, patience=5 if small else 20,
                preselection_epochs=20 if small else 100,
                batch_size=512 if small else 8192,
                n_hidden=2 if small else 3, n_neurons=32 if small else 128,
                ensemble_size=1 if small else 3,
                quadrature_per_process=1500 if small else 250_000,
                asimov_mu=1., target_s_over_ni=.1,
                score_scale=.04, mu_min=.02, mu_max=1.40,
                # The production scan is deliberately not aligned with the
                # uniform spline grid, so it tests interpolation between knots.
                mu_points=17 if small else 82,
                bin_counts=[8, 16, 32] if small else [8, 16, 32, 64, 128, 256, 512],
                spline_anchors=241 if small else 2401,
                spline_focus_anchors=121 if small else 1201,
                spline_focus_halfwidth=.1,
                nuisance_bounds=[-3., 3.], mu_fit_bounds=[.001, 2.])


def create_run(root, mode="production", overrides=None):
    from .physics import PhysicsModel
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = default_config(mode)
    overrides = {} if overrides is None else dict(overrides)
    unknown = set(overrides)-set(config)
    protected = set(overrides) & {"schema_version", "mode", "toolkit_commit"}
    if unknown or protected:
        raise ValueError(f"Unsupported configuration overrides: {sorted(unknown | protected)}")
    config.update(overrides)
    for name in ("n_per_sample", "generation_chunk", "preselection_train_cap", "ratio_train_cap",
                 "epochs", "preselection_epochs", "patience", "batch_size", "n_hidden", "n_neurons",
                 "ensemble_size", "quadrature_per_process", "mu_points", "spline_anchors"):
        if not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config["spline_anchors"] < 2 or config["mu_points"] < 2:
        raise ValueError("At least two spline anchors and scan points are required")
    focused = config["spline_focus_anchors"]
    if not isinstance(focused, int) or focused < 0 or focused == 1:
        raise ValueError("spline_focus_anchors must be zero or an integer >= 2")
    if not np.isfinite(config["spline_focus_halfwidth"]) or config["spline_focus_halfwidth"] <= 0:
        raise ValueError("spline_focus_halfwidth must be finite and positive")
    if config["score_scale"] <= 0 or not 0 < config["mu_min"] < config["mu_max"]:
        raise ValueError("Use a positive score scale and positive ordered eta scan bounds")
    if not config["mu_min"] <= config["asimov_mu"] <= config["mu_max"]:
        raise ValueError("The generating Asimov mu must lie within the scan/anchor range")
    if not config["bin_counts"] or any(not isinstance(b, int) or b < 2 for b in config["bin_counts"]):
        raise ValueError("bin_counts must be a nonempty list of integers >= 2")
    if not isinstance(config["physics_overrides"], dict):
        raise ValueError("physics_overrides must be a dictionary of PhysicsModel parameters")
    model = PhysicsModel(**config["physics_overrides"])
    manifest = {"config": config, "physics": model.to_dict(),
                "splits": {k: list(v) for k, v in SPLITS.items()}}
    path = root / "run.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != manifest:
            raise ValueError("Run configuration differs from saved run.json. Choose a new RUN_NAME; existing artifacts were not changed.")
    else:
        save_json(path, manifest)
    for name in ("raw", "selected", "models", "quadrature", "workspaces", "results", "plots"):
        (root / name).mkdir(exist_ok=True)
    # Record installed distributions without importing heavyweight runtimes.
    from importlib.metadata import version, PackageNotFoundError
    import platform
    versions = {}
    for name in ("numpy", "scipy", "pandas", "torch", "pytorch-lightning", "jax", "jaxlib",
                 "jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda13-plugin", "jax-cuda13-pjrt",
                 "iminuit", "nsbi-common-utils"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed as a distribution"
    save_json(root / "runtime_environment.json", {"python": platform.python_version(), "packages": versions})
    return Run(root, config, model)


def generate_samples(run):
    """Generate exact iid samples in bounded memory; resume complete files only."""
    records = []
    n = run.config["n_per_sample"]
    for name, (component, alpha_ni) in SAMPLES.items():
        path = run.path("raw", name + ".npy")
        if path.exists():
            x = np.load(path, mmap_mode="r", allow_pickle=False)
            if x.shape != (n, 3) or x.dtype != np.float32:
                raise ValueError(f"Invalid cached sample {path}")
            del x
        else:
            rng = np.random.default_rng(np.random.SeedSequence([run.config["seed"], SAMPLE_STREAMS[name]]))
            tmp = path.with_suffix(".partial.npy")
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype="float32", shape=(n, 3))
            for start in range(0, n, run.config["generation_chunk"]):
                stop = min(n, start + run.config["generation_chunk"])
                out[start:stop] = run.model.sample_component(component, stop-start, rng,
                                                          alpha_ni=alpha_ni)
            out.flush()
            del out
            os.replace(tmp, path)
        records.append(dict(sample=name, events=n,
                            expected_yield=run.model.component_yield(component, alpha_ni=alpha_ni),
                            megabytes=path.stat().st_size/1e6))
        print(f"{name}: {n:,} events ready", flush=True)
    save_json(run.path("raw", "manifest.json"), {"samples": records, "splits": SPLITS})
    return records


def raw_split(run, sample, split):
    x = np.load(run.path("raw", sample + ".npy"), mmap_mode="r", allow_pickle=False)
    a, b = SPLITS[split]
    return x[int(a*len(x)):int(b*len(x))]


def selected_split(run, sample, split):
    return np.load(run.path("selected", f"{sample}_{split}.npy"), mmap_mode="r", allow_pickle=False)


def materialize_selection(run, predictor, threshold):
    """Apply one frozen decision to every process and variation, split by split."""
    records = []
    batch = run.config["generation_chunk"]
    fingerprint = hashlib.sha256((file_digest(run.path("models", "preselection", "model.pt"))
                                  + repr(float(threshold))).encode()).hexdigest()
    state_path = run.path("selected", "selection_state.json")
    if state_path.exists():
        if json.loads(state_path.read_text())["selection_fingerprint"] != fingerprint:
            raise ValueError("Preselection changed after selected samples were saved. Choose a new run; stale arrays were not reused.")
    elif any(run.path("selected").glob("*.npy")):
        raise ValueError("Selected arrays lack a model fingerprint. Choose a new run directory.")
    else:
        save_json(state_path, {"selection_fingerprint": fingerprint, "threshold": float(threshold)})
    for name, (component, an) in SAMPLES.items():
        for split in ANALYSIS_SPLITS:
            path = run.path("selected", f"{name}_{split}.npy")
            x = raw_split(run, name, split)
            # Memmapped output is allocated in a single pass; the final file has no unused rows.
            if not path.exists():
                chunks = []
                try:
                    for k, start in enumerate(range(0, len(x), batch)):
                        block = np.asarray(x[start:start+batch])
                        prob = predictor.predict_proba(block)[:, 0]
                        part = path.with_suffix(f".chunk{k}.npy")
                        save_array(part, block[prob >= threshold])
                        chunks.append(part)
                    counts = [len(np.load(p, mmap_mode="r")) for p in chunks]
                    tmp = path.with_suffix(".partial.npy")
                    output = np.lib.format.open_memmap(tmp, mode="w+", dtype="float32", shape=(sum(counts), 3))
                    offset = 0
                    for part, count in zip(chunks, counts):
                        output[offset:offset+count] = np.load(part, mmap_mode="r")
                        offset += count
                    output.flush()
                    del output
                    os.replace(tmp, path)
                finally:
                    for part in chunks:
                        part.unlink(missing_ok=True)
            accepted = len(np.load(path, mmap_mode="r"))
            records.append(dict(sample=name, split=split, generated=len(x), accepted=accepted,
                                efficiency=accepted/len(x),
                                expected_yield=run.model.component_yield(component, alpha_ni=an)*accepted/len(x)))
    save_json(run.path("selected", "manifest.json"), {"threshold": float(threshold),
              "selection_fingerprint": fingerprint, "records": records})
    return records


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
